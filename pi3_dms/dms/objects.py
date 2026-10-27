"""Optional object-detection module.

The face core does not import or start anything in this module unless the user
enables objects explicitly.  Two backends are supported:

* ``mediapipe``: a local TFLite detector loaded through MediaPipe Tasks.  The
  shipped ``efficientdet_lite0`` model is an official COCO control model and is
  restricted to phone/bottle classes.  It has no cigarette class.
* ``http``: a sidecar detector.  This keeps a heavy or differently licensed
  model out of the face process.  The sidecar receives JPEG frames and returns
  timestamped detections.

Object detections are evidence only.  Behaviour is decided in
``dms.fusion``; a bottle can never be relabelled as a cigarette.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .camera import preprocess_frame

CANONICAL_LABELS = {"phone", "bottle", "cigarette"}

_LABEL_ALIASES = {
    "cell phone": "phone",
    "cellphone": "phone",
    "mobile phone": "phone",
    "mobile_phone": "phone",
    "phone": "phone",
    "bottle": "bottle",
    "water bottle": "bottle",
    "cigarette": "cigarette",
    "cigarette butt": "cigarette",
    "cigarette_butt": "cigarette",
}


def canonical_label(raw: str) -> str | None:
    """Map only explicit detector labels; never infer a class from geometry."""
    if not raw:
        return None
    return _LABEL_ALIASES.get(str(raw).strip().lower())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # normalized x1, y1, x2, y2
    timestamp: float
    source: str
    raw_label: str = ""
    frame_id: int | None = None

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        age_ms = None
        if now is not None:
            age_ms = round(max(0.0, now - self.timestamp) * 1000.0, 1)
        return {
            "label": self.label,
            "raw_label": self.raw_label or self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 4) for v in self.bbox],
            "age_ms": age_ms,
            "source": self.source,
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class ObjectSnapshot:
    status: str
    enabled: bool
    detections: tuple[ObjectDetection, ...] = ()
    timestamp: float | None = None
    latency_ms: float | None = None
    model: str | None = None
    classes: tuple[str, ...] = ()
    error: str | None = None
    inference_count: int = 0
    stale_ms: float = 1500.0

    @classmethod
    def disabled(cls) -> "ObjectSnapshot":
        return cls(
            status="disabled",
            enabled=False,
            classes=(),
            error=None,
        )

    @classmethod
    def unavailable(cls, reason: str, *, model: str | None = None) -> "ObjectSnapshot":
        return cls(
            status="unavailable",
            enabled=True,
            model=model,
            classes=(),
            error=reason,
        )

    def effective_status(self, now: float) -> str:
        if not self.enabled:
            return "disabled"
        if self.status in {"error", "unavailable", "starting"}:
            return self.status
        if self.timestamp is None:
            return "starting"
        if (now - self.timestamp) * 1000.0 > self.stale_ms:
            return "stale"
        return "running"

    def to_dict(self, now: float, *, stale_ms: float | None = None) -> dict[str, Any]:
        configured_stale = float(self.stale_ms if stale_ms is None else stale_ms)
        status = self.effective_status(now)
        age_ms = (
            None
            if self.timestamp is None
            else round(max(0.0, now - self.timestamp) * 1000.0, 1)
        )
        detections = (
            [d.to_dict(now) for d in self.detections]
            if status == "running"
            else []
        )
        return {
            "status": status,
            "enabled": self.enabled,
            "model": self.model,
            "classes": list(self.classes),
            "error": self.error,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "inference_count": self.inference_count,
            "age_ms": age_ms,
            "stale_ms": configured_stale,
            "detections": detections,
        }


class ObjectDetector(Protocol):
    model_name: str

    def detect(
        self, frame_bgr: np.ndarray, timestamp: float, frame_id: int | None = None
    ) -> list[ObjectDetection]: ...

    def close(self) -> None: ...


class MediaPipeObjectDetector:
    """Local TFLite detector through MediaPipe Tasks.

    The detector returns only canonical labels in ``allowed_classes``.  A
    generic COCO model can never produce a cigarette because it has no such
    class and because that label is filtered out.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        allowed_classes: set[str],
        min_confidence: float = 0.35,
        max_results: int = 10,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"object model not found: {model_path}")
        self.model_path = model_path
        self.model_name = model_path.name
        self.allowed_classes = set(allowed_classes)
        self.min_confidence = float(min_confidence)
        self.max_results = int(max_results)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._vision = vision
        self._mp = None
        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=self.min_confidence,
            max_results=self.max_results,
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def detect(
        self, frame_bgr: np.ndarray, timestamp: float, frame_id: int | None = None
    ) -> list[ObjectDetection]:
        if self._mp is None:
            import mediapipe as mp

            self._mp = mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        height, width = frame_bgr.shape[:2]
        output: list[ObjectDetection] = []
        for detection in result.detections:
            if not detection.categories:
                continue
            category = detection.categories[0]
            raw_label = str(category.category_name or "")
            label = canonical_label(raw_label)
            if label is None or label not in self.allowed_classes:
                continue
            score = float(category.score)
            if score < self.min_confidence:
                continue
            box = detection.bounding_box
            x1 = _clamp01(box.origin_x / float(max(1, width)))
            y1 = _clamp01(box.origin_y / float(max(1, height)))
            x2 = _clamp01((box.origin_x + box.width) / float(max(1, width)))
            y2 = _clamp01((box.origin_y + box.height) / float(max(1, height)))
            if x2 <= x1 or y2 <= y1:
                continue
            output.append(
                ObjectDetection(
                    label=label,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    timestamp=float(timestamp),
                    source="mediapipe",
                    raw_label=raw_label,
                    frame_id=frame_id,
                )
            )
        return output


class HttpObjectDetector:
    """Client for a separate object-inference service.

    Request JSON:
        {"timestamp": <seconds>, "frame_id": <int>, "image_b64": <jpeg>}

    Response JSON:
        {"detections": [
            {"label": "phone", "confidence": 0.81,
             "bbox": [x1, y1, x2, y2], "timestamp": <optional seconds>}
        ]}

    ``bbox`` must be normalized to the submitted frame.  The service is
    responsible for its own model licence and accuracy; this client does not
    invent scores or labels.
    """

    def __init__(self, url: str, *, timeout_s: float = 2.0, allowed_classes: set[str] | None = None):
        if not url:
            raise ValueError("object service URL is empty")
        self.url = str(url)
        self.timeout_s = float(timeout_s)
        self.allowed_classes = None if allowed_classes is None else set(allowed_classes)
        self.model_name = "http-object-service"

    def close(self) -> None:
        return None

    def detect(
        self, frame_bgr: np.ndarray, timestamp: float, frame_id: int | None = None
    ) -> list[ObjectDetection]:
        ok, encoded = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 78]
        )
        if not ok:
            raise RuntimeError("object frame JPEG encoding failed")
        payload = json.dumps(
            {
                "timestamp": float(timestamp),
                "frame_id": frame_id,
                "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("object service returned a non-object JSON body")
        raw_detections = body.get("detections", [])
        if not isinstance(raw_detections, list):
            raise ValueError("object service returned a non-list detections field")
        output: list[ObjectDetection] = []
        for item in raw_detections:
            if not isinstance(item, dict):
                continue
            label = canonical_label(str(item.get("label", "")))
            if label is None:
                continue
            if self.allowed_classes is not None and label not in self.allowed_classes:
                continue
            try:
                score = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or not (0.0 <= score <= 1.0):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            x1, y1, x2, y2 = (_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                detection_timestamp = float(item.get("timestamp", timestamp))
            except (TypeError, ValueError):
                detection_timestamp = float(timestamp)
            if not math.isfinite(detection_timestamp):
                detection_timestamp = float(timestamp)
            output.append(
                ObjectDetection(
                    label=label,
                    confidence=score,
                    bbox=(x1, y1, x2, y2),
                    timestamp=detection_timestamp,
                    source="http",
                    raw_label=str(item.get("label", "")),
                    frame_id=frame_id,
                )
            )
        return output


def build_object_detector(args: Any) -> ObjectDetector:
    """Build the configured detector, raising a useful error if unavailable."""
    backend = getattr(args, "object_backend", "none")
    if not getattr(args, "objects", False) or backend == "none":
        raise ValueError("object module is disabled")
    classes = set(getattr(args, "object_classes", ["phone", "bottle"]))
    unknown = classes - CANONICAL_LABELS
    if unknown:
        raise ValueError(f"unsupported object classes: {sorted(unknown)}")
    if backend == "http":
        return HttpObjectDetector(
            getattr(args, "object_service_url", ""),
            timeout_s=float(getattr(args, "object_timeout_s", 2.0)),
            allowed_classes=classes,
        )
    if backend == "mediapipe":
        return MediaPipeObjectDetector(
            Path(getattr(args, "object_model", "")),
            allowed_classes=classes,
            min_confidence=float(getattr(args, "object_min_confidence", 0.35)),
            max_results=int(getattr(args, "object_max_results", 10)),
        )
    raise ValueError(f"unknown object backend: {backend}")


class ObjectWorker(threading.Thread):
    """Periodic object inference on the newest shared frame.

    There is no queue: if inference is slower than the desired interval, frames
    are skipped rather than accumulated.  This is the bounded-work contract for
    a Pi 3.
    """

    def __init__(
        self,
        store: Any,
        detector: ObjectDetector,
        *,
        interval_ms: float = 500.0,
        stale_ms: float = 1500.0,
        classes: tuple[str, ...] = (),
        color_mode: str = "off",
    ):
        super().__init__(name="object-worker", daemon=True)
        self.store = store
        self.detector = detector
        self.interval_s = max(0.05, float(interval_ms) / 1000.0)
        self.stale_ms = float(stale_ms)
        self.classes = tuple(classes)
        self.color_mode = color_mode
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = ObjectSnapshot(
            status="starting",
            enabled=True,
            model=getattr(detector, "model_name", type(detector).__name__),
            classes=self.classes,
            stale_ms=self.stale_ms,
        )
        self._last_sequence = -1
        self._inference_count = 0

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> ObjectSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(self, snapshot: ObjectSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def run(self) -> None:
        while not self.stop_event.is_set():
            packet, _info, _error, _fps = self.store.latest()
            if packet is None:
                self.stop_event.wait(0.05)
                continue
            if packet.sequence == self._last_sequence:
                self.stop_event.wait(0.03)
                continue
            started = time.monotonic()
            self._last_sequence = packet.sequence
            try:
                object_frame = preprocess_frame(packet.process_bgr, self.color_mode)
                detections = self.detector.detect(
                    object_frame, packet.timestamp, packet.sequence
                )
                latency_ms = (time.monotonic() - started) * 1000.0
                self._inference_count += 1
                self._publish(
                    ObjectSnapshot(
                        status="running",
                        enabled=True,
                        detections=tuple(detections),
                        timestamp=packet.timestamp,
                        latency_ms=latency_ms,
                        model=getattr(self.detector, "model_name", type(self.detector).__name__),
                        classes=self.classes,
                        stale_ms=self.stale_ms,
                        inference_count=self._inference_count,
                    )
                )
            except Exception as exc:
                self._publish(
                    ObjectSnapshot(
                        status="error",
                        enabled=True,
                        detections=(),
                        timestamp=None,
                        model=getattr(self.detector, "model_name", type(self.detector).__name__),
                        classes=self.classes,
                        error=f"{type(exc).__name__}: {exc}",
                        inference_count=self._inference_count,
                        stale_ms=self.stale_ms,
                    )
                )
                print(
                    f"[objects] {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self.stop_event.wait(0.5)
                continue
            elapsed = time.monotonic() - started
            remaining = self.interval_s - elapsed
            if remaining > 0:
                self.stop_event.wait(remaining)

    def close(self) -> None:
        self.stop()
        self.detector.close()
