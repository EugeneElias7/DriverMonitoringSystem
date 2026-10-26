"""Shared camera-frame pipeline: one decode, one face pass, reused landmarks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .core.base import validate_frame
from .core.types import Detection, Event, FaceObservation, FeatureResult, utc_iso
from .detectors.drowsiness import DrowsinessDetector
from .detectors.face import FaceAnalyzer
from .detectors.head_pose import DistractionDetector, HeadPoseDetector
from .detectors.smoking import SmokingDetector


class DMSPipeline:
    """Compose the face, drowsiness, head-pose, distraction and smoking modules.

    The pipeline processes one camera frame at a time. It does not re-decode or
    re-detect faces for individual features; every feature reuses the same
    :class:`FaceObservation`.
    """

    def __init__(self, config: Any, mode: str = "video"):
        self.config = config
        self.mode = mode.lower()
        configured_mode = str(config.get("face.mode", "auto")).lower()
        if configured_mode == "auto":
            face_mode = "VIDEO" if self.mode in {"video", "webcam"} else "IMAGE"
        else:
            face_mode = configured_mode.upper()
        self.face = FaceAnalyzer(
            config,
            mode=face_mode,
            expected_sha256=config.get("face.model_sha256") or None,
        )
        self.drowsiness = DrowsinessDetector(config)
        self.head_pose = HeadPoseDetector(config)
        self.distraction = DistractionDetector(config)
        self.smoking = SmokingDetector(config)
        self._loaded = False
        self._frame_index = 0
        self._started_monotonic = time.monotonic()
        self._last_timestamp: float | None = None
        self._face_errors = 0

    def load(self) -> None:
        """Load every module independently so one failure does not abort all."""
        if self._loaded:
            return
        self._safe_load(self.face)
        self._safe_load(self.drowsiness)
        self._safe_load(self.head_pose)
        self._safe_load(self.distraction)
        self._safe_load(self.smoking)
        self._loaded = True

    @staticmethod
    def _safe_load(module: Any) -> None:
        try:
            module.load()
        except Exception as exc:
            health = module.get_status()
            health.state = "error"
            health.ready = False
            health.last_error = f"{type(exc).__name__}: {exc}"
            health.message = health.last_error

    def process(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
        frame_index: int | None = None,
        roi_hint: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        validate_frame(frame)
        if not self._loaded:
            self.load()
        timestamp = self._resolve_timestamp(timestamp)
        if frame_index is None:
            frame_index = self._frame_index
        self._frame_index = int(frame_index) + 1
        height, width = frame.shape[:2]

        face: FaceObservation | None = None
        try:
            face = self.face.analyze(frame, timestamp, roi_hint=roi_hint)
        except Exception:
            # analyze already isolates errors; this is a final guard.
            self._face_errors += 1

        drowsy = self._safe_update(self.drowsiness, face, timestamp, "drowsiness")
        pose = self._safe_update(self.head_pose, face, timestamp, "head_pose")
        distract = self._safe_update(self.distraction, pose, timestamp, "distraction")
        try:
            smoke = self.smoking.detect(frame, timestamp, face=face)
        except Exception as exc:
            smoke = FeatureResult(
                feature="smoking",
                state="error",
                timestamp=timestamp,
                valid=False,
                message=f"{type(exc).__name__}: {exc}",
            )

        events: list[Event] = []
        for result in (drowsy, pose, distract, smoke):
            events.extend(result.events if result is not None else [])

        detections: list[Detection] = []
        if face is not None:
            detections.append(
                Detection(
                    label="face",
                    confidence=1.0,
                    bbox=face.bbox,
                    source="face_landmarker",
                    metadata={"pose_source": face.pose_source},
                )
            )
        detections.extend(smoke.detections if smoke is not None else [])

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        health = {
            "face": self.face.get_status().to_dict(),
            "drowsiness": self.drowsiness.get_status().to_dict(),
            "head_pose": self.head_pose.get_status().to_dict(),
            "distraction": self.distraction.get_status().to_dict(),
            "smoking": self.smoking.get_status().to_dict(),
        }
        if getattr(self.smoking, "backend", None) is not None:
            health["smoking_backend"] = self.smoking.backend.get_status().to_dict()

        return {
            "schema_version": 1,
            "frame_index": int(frame_index),
            "timestamp": round(float(timestamp), 6),
            "iso_time": utc_iso(timestamp),
            "image_size": [width, height],
            "face": {"present": False, "reason": self.face.health.message}
            if face is None
            else face.compact_dict(),
            "features": {
                "drowsiness": drowsy.to_dict() if drowsy else None,
                "head_pose": pose.to_dict() if pose else None,
                "distraction": distract.to_dict() if distract else None,
                "smoking": smoke.to_dict() if smoke else None,
            },
            "detections": [d.to_dict() for d in detections],
            "events": [e.to_dict() for e in events],
            "health": health,
            "timing_ms": {
                "total": round(elapsed_ms, 3),
                "face": _round_health(health["face"]),
                "drowsiness": _round_health(health["drowsiness"]),
                "head_pose": _round_health(health["head_pose"]),
                "distraction": _round_health(health["distraction"]),
                "smoking": _round_health(health["smoking"]),
            },
        }

    def _safe_update(self, module: Any, face: Any, timestamp: float, name: str) -> FeatureResult:
        try:
            return module.update(face, timestamp)
        except Exception as exc:
            return FeatureResult(
                feature=name,
                state="error",
                timestamp=timestamp,
                valid=False,
                message=f"{type(exc).__name__}: {exc}",
            )

    def _resolve_timestamp(self, timestamp: float | None) -> float:
        if timestamp is None:
            timestamp = time.time()
        timestamp = float(timestamp)
        # Modules need monotonically non-decreasing time. A caller-supplied
        # video timestamp is preserved; only pathological backwards jumps are
        # nudged forward.
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            timestamp = self._last_timestamp
        self._last_timestamp = timestamp
        return timestamp

    def reset(self) -> None:
        self.face.reset()
        for module in (self.drowsiness, self.head_pose, self.distraction, self.smoking):
            module.reset()
        self._frame_index = 0
        self._last_timestamp = None

    def close(self) -> None:
        for module in (self.face, self.smoking):
            try:
                module.close()
            except Exception:
                pass
        self._loaded = False

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize once for processing, preserving aspect ratio."""
        width = int(self.config.get("pipeline.resize_width", 0) or 0)
        if width <= 0 or frame.shape[1] <= width:
            return frame
        scale = width / frame.shape[1]
        import cv2

        return cv2.resize(
            frame,
            (width, max(2, int(round(frame.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )


def _round_health(health: dict[str, Any]) -> float | None:
    value = health.get("last_latency_ms")
    return None if value is None else round(float(value), 3)
