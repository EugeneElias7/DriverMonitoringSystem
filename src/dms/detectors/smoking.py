"""Cigarette/smoking detector with face/mouth proximity and persistence."""

from __future__ import annotations

import time
from typing import Any

from ..backends.tflite import TFLiteDetectorBackend
from ..backends.yolo_onnx import YoloOnnxBackend
from ..core.base import HealthMixin, validate_frame
from ..core.events import EventManager
from ..core.geometry import bbox_area, normalize_label, point_to_bbox_distance
from ..core.types import Detection, FaceObservation, FeatureResult


class SmokingDetector(HealthMixin):
    """Two-stage smoking heuristic.

    A ``cigarette`` detection is not enough: it must be close to the mouth/face
    and persist across frames. The ``smoke`` class never triggers an event, and
    a face/mouth reference is required for confirmation.
    """

    name = "smoking"

    def __init__(self, config: Any, backend_override: Any = None):
        super().__init__(self.name)
        self.config = config
        self.backend = backend_override or self._make_backend()
        self.events = EventManager(float(config.get("smoking.event_cooldown_ms", 3000)))
        self._reset_state()

    def _make_backend(self) -> Any:
        backend_name = str(self.config.get("smoking.backend", "none")).lower()
        if backend_name == "none":
            return None
        if backend_name == "yolo_onnx":
            return YoloOnnxBackend(
                "smoking_yolo_onnx",
                self.config,
                self.config.resolve_path(self.config.get("smoking.model_path")),
            )
        if backend_name == "tflite_ssd":
            return TFLiteDetectorBackend(
                "smoking_tflite_ssd",
                self.config,
                self.config.resolve_path(self.config.get("smoking.tflite_model_path")),
                self.config.resolve_path(self.config.get("smoking.tflite_labelmap_path")),
                input_range=str(self.config.get("smoking.tflite_input_range", "unit")),
                score_threshold=float(self.config.get("smoking.tflite_score_threshold", 0.3)),
                known_issue=bool(self.config.get("smoking.known_tflite_issue", False)),
                labelmap_offset=int(self.config.get("smoking.tflite_labelmap_offset", 0)),
            )
        raise ValueError(f"unknown smoking backend: {backend_name}")

    def _reset_state(self) -> None:
        self._near_since: float | None = None
        self._last_near_t: float | None = None
        self._hits = 0
        self._near_active = False
        self._confirmed = False
        self._last_inference_t: float | None = None
        self._last_inference_monotonic = 0.0
        self._cached_detections: list[Detection] = []
        self._cached_at: float | None = None
        self._state = "unknown"
        self._last_face_t: float | None = None
        self._cigarette_episode = False

    def load(self) -> None:
        if self.backend is None:
            self._mark_disabled("smoking backend disabled (config smoking.backend=none)")
            return
        started = time.perf_counter()
        try:
            self.backend.load()
            self._validate_classes()
            if self.backend.health.state == "degraded":
                self.health.ready = True
                self._loaded = True
                self.health.state = "degraded"
                self.health.message = self.backend.health.message
            else:
                self._mark_ready(
                    f"{self.backend.name}: {self.backend.class_names}; target="
                    f"{self.config.get('smoking.expected_target_class')}"
                )
        except Exception as exc:
            self._mark_error(f"{type(exc).__name__}: {exc}")
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.health.last_latency_ms = elapsed
            self.health.mean_latency_ms = elapsed

    def _validate_classes(self) -> None:
        names = [normalize_label(v) for v in getattr(self.backend, "class_names", [])]
        if not names:
            raise ValueError("smoking backend exposed no class names")
        target = normalize_label(str(self.config.get("smoking.expected_target_class", "cigarette")))
        if target not in names:
            raise ValueError(
                f"target class {target!r} is absent from checkpoint labels {names!r}; "
                "refusing to guess a mapping"
            )
        triggers = {normalize_label(v) for v in self.config.get("smoking.trigger_classes", [])}
        if target not in triggers:
            raise ValueError("expected_target_class must be in trigger_classes")
        # Explicitly confirm that smoke is not a trigger in this configuration.
        nontrigger = {normalize_label(v) for v in self.config.get("smoking.non_trigger_classes", [])}
        overlap = triggers & nontrigger
        if overlap:
            raise ValueError(f"classes are both triggers and non-triggers: {sorted(overlap)}")

    def detect(self, frame: Any, timestamp: float, face: FaceObservation | None = None) -> FeatureResult:
        if not self._loaded:
            return FeatureResult(
                feature=self.name,
                state="unavailable" if self.backend is not None else "disabled",
                timestamp=timestamp,
                valid=False,
                message=self.health.message,
                measurements={"backend": self.backend.name if self.backend is not None else None},
                events=self.events.take(),
            )
        try:
            validate_frame(frame)
        except Exception as exc:
            return FeatureResult(
                feature=self.name,
                state="error",
                timestamp=timestamp,
                valid=False,
                message=f"invalid frame: {exc}",
                events=self.events.take(),
            )
        started = time.perf_counter()
        error: Exception | None = None
        detections: list[Detection] = []
        try:
            if face is not None:
                self._last_face_t = float(timestamp)
            detections = self._infer(frame, timestamp)
            result = self._update_state(detections, face, float(timestamp))
        except Exception as exc:
            error = exc
            result = FeatureResult(
                feature=self.name,
                state="error",
                timestamp=timestamp,
                valid=False,
                message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.health.last_latency_ms = elapsed_ms
            self._latencies.append(elapsed_ms)
            self.health.mean_latency_ms = sum(self._latencies) / len(self._latencies)
            self.health.frames_processed += 1
            if error is not None:
                self.health.failures += 1
                self.health.state = "error"
                self.health.last_error = f"{type(error).__name__}: {error}"
                self.health.message = self.health.last_error
            elif self.health.state == "error":
                self.health.state = "ok"
        return result

    def _infer(self, frame: Any, timestamp: float) -> list[Detection]:
        interval = int(self.config.get("smoking.inference_interval", 1))
        now = time.monotonic()
        if interval > 1 and self._cached_at is not None:
            # Reuse only briefly; stale boxes must not be treated as current.
            age_ms = (now - self._last_inference_monotonic) * 1000.0
            if age_ms < float(self.config.get("smoking.stale_after_ms", 250)):
                cached = [
                    Detection(**{**d.__dict__, "metadata": {**d.metadata, "cached": True}})
                    for d in self._cached_detections
                ]
                return cached
        detections = self.backend.detect(frame, timestamp)
        self._cached_detections = detections
        self._last_inference_t = timestamp
        self._last_inference_monotonic = now
        self._cached_at = timestamp
        return detections

    def _update_state(
        self,
        detections: list[Detection],
        face: FaceObservation | None,
        timestamp: float,
    ) -> FeatureResult:
        target_name = normalize_label(str(self.config.get("smoking.expected_target_class", "cigarette")))
        triggers = {normalize_label(v) for v in self.config.get("smoking.trigger_classes", [])}
        masks = {normalize_label(v) for v in self.config.get("smoking.mask_classes", [])}
        non_triggers = {normalize_label(v) for v in self.config.get("smoking.non_trigger_classes", [])}
        targets = [
            d
            for d in detections
            if normalize_label(d.label) in triggers and normalize_label(d.label) not in masks
        ]
        visible_smoke = [d for d in detections if normalize_label(d.label) in non_triggers]
        all_result_detections = list(detections)

        if face is None or face.mouth_center is None or face.face_width is None or face.face_width <= 0:
            lost_ms = (
                None
                if self._last_face_t is None
                else max(0.0, (timestamp - self._last_face_t) * 1000.0)
            )
            if lost_ms is None or lost_ms > float(self.config.get("smoking.miss_tolerance_ms", 600)):
                self._clear_near()
            self._state = "cigarette_visible_no_face" if targets else "no_face"
            return FeatureResult(
                feature=self.name,
                state=self._state,
                timestamp=timestamp,
                valid=False,
                measurements={
                    "backend": getattr(self.backend, "name", None),
                    "face_visible": False,
                    "target_count": len(targets),
                    "smoke_count": len(visible_smoke),
                    "confirmed": False,
                    "note": "face/mouth reference required; no smoking confirmation without it",
                    "lost_face_ms": None if lost_ms is None else round(lost_ms, 1),
                },
                detections=[self._decorate(d, None) for d in all_result_detections],
                events=self.events.take(),
            )

        mouth = face.mouth_center
        max_distance = float(self.config.get("smoking.mouth_distance_face_width", 0.65))
        max_area_ratio = float(self.config.get("smoking.max_target_face_area_ratio", 0.5))
        face_area = max(1.0, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        near_detections: list[Detection] = []
        for det in targets:
            distance = point_to_bbox_distance(mouth, det.bbox)
            distance_ratio = distance / face.face_width
            area_ratio = bbox_area(det.bbox) / face_area
            near = distance_ratio <= max_distance and area_ratio <= max_area_ratio
            decorated = self._decorate(
                det,
                {
                    "near_mouth": near,
                    "mouth_distance_px": round(distance, 2),
                    "mouth_distance_face_width": round(distance_ratio, 4),
                    "target_face_area_ratio": round(area_ratio, 4),
                },
            )
            # Replace metadata in the output list.
            all_result_detections = [
                decorated if existing is det else existing for existing in all_result_detections
            ]
            if near:
                near_detections.append(decorated)

        if visible_smoke:
            # Keep auxiliary smoke detections visible but never use them as a trigger.
            all_result_detections = [
                self._decorate(d, {"auxiliary": True, "trigger": False}) if d in visible_smoke else d
                for d in all_result_detections
            ]

        if not targets:
            if self._near_active and self._last_near_t is not None:
                gap = (timestamp - self._last_near_t) * 1000.0
                if gap > float(self.config.get("smoking.miss_tolerance_ms", 600)):
                    self._clear_near()
            if self._confirmed:
                self._confirmed = False
                self.events.emit(
                    timestamp=timestamp,
                    feature=self.name,
                    kind="smoking_ended",
                    severity="info",
                    message="No cigarette detection near the mouth in the current frame; confirmation ended",
                )
            self._state = "no_cigarette"
            return FeatureResult(
                feature=self.name,
                state=self._state,
                timestamp=timestamp,
                valid=True,
                confidence=None,
                measurements={
                    "backend": getattr(self.backend, "name", None),
                    "face_visible": True,
                    "target_count": 0,
                    "smoke_count": len(visible_smoke),
                    "confirmed": False,
                    "near_ms": 0.0,
                    "hits": self._hits,
                },
                detections=all_result_detections,
                events=self.events.take(),
            )

        if not near_detections:
            gap_ms = (
                None if self._last_near_t is None else (timestamp - self._last_near_t) * 1000.0
            )
            if gap_ms is None or gap_ms > float(self.config.get("smoking.miss_tolerance_ms", 600)):
                self._clear_near()
            self._state = "cigarette_visible"
            best = max(targets, key=lambda d: d.confidence)
            return FeatureResult(
                feature=self.name,
                state=self._state,
                timestamp=timestamp,
                valid=True,
                measurements={
                    "backend": getattr(self.backend, "name", None),
                    "face_visible": True,
                    "target_count": len(targets),
                    "near_count": 0,
                    "smoke_count": len(visible_smoke),
                    "max_cigarette_confidence": round(best.confidence, 4),
                    "max_mouth_distance_face_width": round(
                        max(
                            d.metadata.get("mouth_distance_face_width", 999)
                            for d in all_result_detections
                            if normalize_label(d.label) == target_name
                        ),
                        4,
                    )
                    if targets
                    else None,
                    "confirmed": False,
                    "note": "cigarette visible but not close enough to the mouth/face",
                },
                detections=all_result_detections,
                events=self.events.take(),
            )

        # Near-mouth evidence: accumulate across frames, with a miss tolerance.
        gap_ms = None if self._last_near_t is None else (timestamp - self._last_near_t) * 1000.0
        if gap_ms is not None and gap_ms > float(self.config.get("smoking.miss_tolerance_ms", 600)):
            self._clear_near()
        if self._near_since is None:
            self._near_since = timestamp
            self._hits = 0
            self._near_active = True
            self._state = "smoking_suspected"
            self.events.emit(
                timestamp=timestamp,
                feature=self.name,
                kind="cigarette_near_mouth",
                severity="info",
                message="Cigarette detection is near the mouth; awaiting temporal confirmation",
                confidence=max(d.confidence for d in near_detections),
                data={"hits": self._hits + 1},
            )
        self._last_near_t = timestamp
        self._hits += 1
        near_ms = max(0.0, (timestamp - self._near_since) * 1000.0)
        required_ms = float(self.config.get("smoking.confirmation_ms", 900))
        min_hits = int(self.config.get("smoking.min_hits", 2))
        confirmed = near_ms >= required_ms and self._hits >= min_hits
        if not self._cigarette_episode:
            self._cigarette_episode = True
            self.events.emit(
                timestamp=timestamp,
                feature=self.name,
                kind="cigarette_detected",
                severity="info",
                message="Cigarette-class detection near the driver's mouth",
                confidence=max(d.confidence for d in near_detections),
                data={"near_ms": round(near_ms, 1)},
            )
        if confirmed and not self._confirmed:
            self._confirmed = True
            self._state = "smoking_confirmed"
            self.events.emit(
                timestamp=timestamp,
                feature=self.name,
                kind="smoking_confirmed",
                severity="warning",
                message=(
                    f"Cigarette near mouth for {near_ms:.0f} ms across {self._hits} hits "
                    "(temporal heuristic; validate on target data)"
                ),
                confidence=max(d.confidence for d in near_detections),
                data={
                    "near_ms": round(near_ms, 1),
                    "hits": self._hits,
                    "backend": getattr(self.backend, "name", None),
                },
            )
        elif not confirmed:
            self._state = "smoking_suspected"
        best = max(near_detections, key=lambda d: d.confidence)
        return FeatureResult(
            feature=self.name,
            state=self._state,
            timestamp=timestamp,
            valid=True,
            confidence=None,
            measurements={
                "backend": getattr(self.backend, "name", None),
                "face_visible": True,
                "target_count": len(targets),
                "near_count": len(near_detections),
                "smoke_count": len(visible_smoke),
                "max_cigarette_confidence": round(best.confidence, 4),
                "near_ms": round(near_ms, 1),
                "hits": self._hits,
                "min_hits": min_hits,
                "confirmation_ms": required_ms,
                "confirmed": confirmed,
                "rule_score": round(min(1.0, near_ms / required_ms) if required_ms > 0 else 0.0, 4),
                "note": "visual smoking heuristic; not evidence of tobacco use or impairment",
            },
            detections=all_result_detections,
            events=self.events.take(),
        )

    def _decorate(self, detection: Detection, metadata: dict[str, Any] | None) -> Detection:
        return Detection(
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox,
            class_id=detection.class_id,
            source=detection.source,
            metadata={**detection.metadata, **(metadata or {})},
        )

    def _clear_near(self) -> None:
        self._near_since = None
        self._last_near_t = None
        self._hits = 0
        self._near_active = False
        self._cigarette_episode = False

    def reset(self) -> None:
        self._reset_state()
        self.events.reset()
        if self.backend is not None:
            self.backend.reset()

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
        super().close()

    def get_status(self) -> Any:
        return self.health
