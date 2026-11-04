"""Two-stage pipeline: perception (frames -> observations) and analysis
(observations -> alerts).

Keeping the stages separate is what lets the tests drive the whole analysis
stack with a scripted observation source, no camera and no model loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

from .alerts import Alert, AlertManager, ConditionSignal
from .behaviors import (
    DistractionMonitor,
    DrinkingMonitor,
    DrowsinessMonitor,
    EatingMonitor,
    SeatbeltMonitor,
    SmokingMonitor,
)
from .config import AppConfig
from .face import FaceObservation
from .face_onnx import create_face_analyzer
from .objects import (
    PHONE_LABELS,
    BackgroundDetector,
    Detection,
    DetectorGate,
    YoloDetector,
    merge_detections,
)


@dataclass
class FrameObservation:
    timestamp: float
    face: FaceObservation
    detections: list[Detection] = field(default_factory=list)
    # Seconds between the detector pass that produced `detections` and this
    # frame. On a Pi 3 that is several seconds, so a box drawn at face level
    # can belong to a pose the driver left long ago.
    det_age: float | None = None


class Perception:
    """Camera frame -> facial metrics plus (slower, threaded) object detections."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        if cfg.opencv_threads > 0:
            cv2.setNumThreads(int(cfg.opencv_threads))
        self.face = create_face_analyzer(cfg.face)
        self.detectors = [YoloDetector(det_cfg) for det_cfg in cfg.detectors]
        # Serializing is only useful when there is more than one model. The
        # detector order is the priority order: cigarette is listed first in
        # config_pi.json, so it gets the next free pass before phone.
        self._gate = DetectorGate() if cfg.serialize_detectors else None
        self._detectors_paused = False
        self.workers = [
            BackgroundDetector(
                detector,
                detector.cfg.target_hz,
                gate=self._gate,
                priority=index,
            )
            for index, detector in enumerate(self.detectors)
            if detector.enabled
        ]

    @property
    def detectors_paused(self) -> bool:
        return self._detectors_paused

    def set_detectors_paused(self, paused: bool) -> None:
        """Face-first switch: stop starting detector passes while true."""
        paused = bool(paused)
        if paused == self._detectors_paused:
            return
        self._detectors_paused = paused
        for worker in self.workers:
            worker.set_paused(paused)

    @property
    def labels(self) -> list[str]:
        labels: set[str] = set()
        for detector in self.detectors:
            if detector.enabled:
                labels.update(label for label in detector.labels if label)
        return sorted(labels)

    def self_test(self) -> list[str]:
        """Run every enabled stage once on a blank frame before the loop.

        Loading a model only proves the file parsed. A backend that cannot
        execute the graph (unsupported operator, wrong CPU target) fails at
        forward time, and inside the background thread that exception is
        swallowed, so the driver only ever sees "no detections". One pass per
        stage here prints the failure and its cost up front.
        """
        frame = np.zeros(
            (self.cfg.capture_height, self.cfg.capture_width, 3), dtype=np.uint8
        )
        lines: list[str] = [
            f"runtime: OpenCV {cv2.__version__}, numpy {np.__version__}"
        ]

        started = time.perf_counter()
        try:
            face = self.face.process(frame, 0)
        except Exception as exc:
            lines.append(f"face opencv: FAILED {type(exc).__name__}: {exc}")
        else:
            elapsed = (time.perf_counter() - started) * 1000.0
            lines.append(
                f"face opencv: ok in {elapsed:.0f} ms "
                f"(face found on a blank frame: {'yes' if face.present else 'no'})"
            )

        for detector in self.detectors:
            if not detector.enabled:
                lines.append(f"{detector.name}: disabled")
                continue
            started = time.perf_counter()
            try:
                found = detector.detect(frame)
            except Exception as exc:
                lines.append(
                    f"{detector.name} ({detector.source}): FAILED "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            elapsed = (time.perf_counter() - started) * 1000.0
            names = ", ".join(label for label in detector.labels if label)
            if len(names) > 100:
                names = names[:97] + "..."
            lines.append(
                f"{detector.name} [{detector.cfg.model_path}]: "
                f"{elapsed:.0f} ms for one pass, "
                f"{len(found)} boxes on a blank frame, "
                f"{len([label for label in detector.labels if label])} classes "
                f"[{names}]"
            )
            emitted = detector.num_classes
            if emitted is not None and emitted != len(detector.labels):
                lines.append(
                    f"{detector.name}: WARNING the graph emits {emitted} classes "
                    f"but the labels file lists {len(detector.labels)} — "
                    "class names are misaligned"
                )
        return lines

    def start(self) -> None:
        for worker in self.workers:
            worker.start()

    def analyze(self, frame_bgr: np.ndarray, now: float) -> FrameObservation:
        for worker in self.workers:
            worker.submit(frame_bgr)
        face = self.face.process(frame_bgr, int(now * 1000.0))
        detections: list[Detection] = []
        ages: list[float] = []
        for worker in self.workers:
            results, published = worker.snapshot()
            detections.extend(results)
            if published is not None:
                ages.append(max(0.0, time.monotonic() - published))
        return FrameObservation(
            timestamp=now,
            face=face,
            detections=merge_detections(detections),
            det_age=max(ages) if ages else None,
        )

    def close(self) -> None:
        # Wake every worker before joining: one may be queued on the gate
        # while another is inside a multi-second pass, and the stop event
        # alone does not wake a condition wait.
        for worker in self.workers:
            worker.request_stop()
        if self._gate is not None:
            self._gate.close()
        for worker in self.workers:
            worker.stop()
        for detector in self.detectors:
            detector.close()
        self.face.close()


class Analyzer:
    """Observations -> condition signals -> de-duplicated alerts."""

    def __init__(
        self,
        cfg: AppConfig,
        detector_labels: Iterable[str] | None = None,
        sinks: Iterable[Callable[[Alert], None]] | None = None,
    ):
        self.cfg = cfg
        th = cfg.thresholds
        labels = None if detector_labels is None else set(detector_labels)

        self.drowsiness = DrowsinessMonitor(th)
        self.smoking = SmokingMonitor(
            th, detector_has_cigarette=labels is None or "cigarette" in labels
        )
        self.drinking = DrinkingMonitor(th)
        self.eating = EatingMonitor(th)
        self.seatbelt = SeatbeltMonitor(
            th, detector_has_seatbelt=labels is None or "seatbelt" in labels
        )
        self.distraction = DistractionMonitor(th)

        self.notes: list[str] = []
        if labels is not None and "cigarette" not in labels:
            self.notes.append(
                "detector declares no 'cigarette' class: smoking detection is disabled"
            )
        if labels is not None and "drinking" not in labels:
            self.notes.append(
                "detector declares no 'drinking' class: drinking detection is disabled"
            )
        if labels is not None and "eating" not in labels:
            self.notes.append("detector declares no 'eating' class: eating detection is disabled")
        if labels is not None and "seatbelt" not in labels:
            self.notes.append(
                "detector declares no 'seatbelt' class: seatbelt detection is disabled"
            )
        if labels is not None and not (labels & PHONE_LABELS):
            self.notes.append("detector declares no phone class: phone escalation is disabled")

        self.alerts = AlertManager(
            sinks=sinks or [], cooldown_s=cfg.alert_cooldown_s
        )
        self.active_signals: list[ConditionSignal] = []

    def update(self, observation: FrameObservation) -> list[Alert]:
        now = observation.timestamp
        signals: list[ConditionSignal] = []
        signals += self.drowsiness.update(now, observation.face)
        signals += self.smoking.update(now, observation.face, observation.detections)
        signals += self.drinking.update(now, observation.face, observation.detections)
        signals += self.eating.update(now, observation.face, observation.detections)
        signals += self.seatbelt.update(now, observation.face, observation.detections)
        signals += self.distraction.update(
            now, observation.face, observation.detections
        )
        self.active_signals = signals
        return self.alerts.update(now, signals)

    def active_messages(self) -> list[ConditionSignal]:
        return [s for s in self.active_signals if s.active]

    def close(self) -> None:
        pass
