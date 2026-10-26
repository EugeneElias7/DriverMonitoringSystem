"""Shared structured types.

The JSON contract is deliberately plain: datetimes are ISO-8601 UTC strings,
boxes are ``[x1, y1, x2, y2]`` in pixels of the processed frame, and rule-based
states carry measurements rather than fabricated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


def utc_iso(timestamp: float) -> str:
    """Return an ISO-8601 UTC timestamp with a ``Z`` suffix."""
    return (
        datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class Detection:
    """One object detection in processed-frame pixel coordinates."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    class_id: int | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "bbox": [round(float(v), 3) for v in self.bbox],
            "class_id": self.class_id,
            "source": self.source,
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class Event:
    """A structured, throttled state transition or warning."""

    id: str
    timestamp: float
    feature: str
    kind: str
    severity: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": round(float(self.timestamp), 6),
            "iso_time": utc_iso(self.timestamp),
            "feature": self.feature,
            "kind": self.kind,
            "severity": self.severity,
            "confidence": None if self.confidence is None else round(float(self.confidence), 6),
            "message": self.message,
            "data": _jsonable(self.data),
        }


@dataclass
class DetectorHealth:
    """Runtime health for one module or model backend."""

    name: str
    ready: bool = False
    state: str = "uninitialized"
    message: str = ""
    frames_processed: int = 0
    failures: int = 0
    last_latency_ms: float | None = None
    mean_latency_ms: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.ready,
            "state": self.state,
            "message": self.message,
            "frames_processed": self.frames_processed,
            "failures": self.failures,
            "last_latency_ms": None if self.last_latency_ms is None else round(self.last_latency_ms, 3),
            "mean_latency_ms": None if self.mean_latency_ms is None else round(self.mean_latency_ms, 3),
            "last_error": self.last_error,
        }


@dataclass
class FaceObservation:
    """Landmark-derived face measurements, all in processed-frame pixels."""

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    image_size: tuple[int, int]
    timestamp: float
    ear_left: float | None = None
    ear_right: float | None = None
    ear: float | None = None
    ear_max: float | None = None
    mar: float | None = None
    mouth_center: tuple[float, float] | None = None
    face_width: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    pose_source: str = "none"
    transform: np.ndarray | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    roi: tuple[int, int, int, int] | None = None
    roi_upscaled: bool = False

    def compact_dict(self) -> dict[str, Any]:
        return {
            "present": True,
            "bbox": [round(float(v), 3) for v in self.bbox],
            "ear": _round_opt(self.ear, 4),
            "ear_left": _round_opt(self.ear_left, 4),
            "ear_right": _round_opt(self.ear_right, 4),
            "ear_max": _round_opt(self.ear_max, 4),
            "mar": _round_opt(self.mar, 4),
            "mouth_center": None
            if self.mouth_center is None
            else [round(float(v), 2) for v in self.mouth_center],
            "face_width": _round_opt(self.face_width, 2),
            "pitch": _round_opt(self.pitch, 2),
            "yaw": _round_opt(self.yaw, 2),
            "roll": _round_opt(self.roll, 2),
            "pose_source": self.pose_source,
            "roi": None if self.roi is None else list(self.roi),
            "roi_upscaled": self.roi_upscaled,
            "blendshapes": {k: round(float(v), 4) for k, v in self.blendshapes.items()},
        }


def _round_opt(value: float | None, ndigits: int) -> float | None:
    return None if value is None else round(float(value), ndigits)


@dataclass
class PoseSample:
    """One head-pose sample used by the distraction rule engine."""

    timestamp: float
    valid: bool
    state: str = "unknown"
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "state": self.state,
            "pitch": _round_opt(self.pitch, 2),
            "yaw": _round_opt(self.yaw, 2),
            "roll": _round_opt(self.roll, 2),
            "source": self.source,
        }


@dataclass
class FeatureResult:
    """Output of one feature module for one frame."""

    feature: str
    state: str
    timestamp: float
    valid: bool = True
    confidence: float | None = None
    measurements: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "state": self.state,
            "valid": self.valid,
            "confidence": None
            if self.confidence is None
            else round(float(self.confidence), 6),
            "measurements": _jsonable(self.measurements),
            "detections": [d.to_dict() for d in self.detections],
            "message": self.message,
        }
