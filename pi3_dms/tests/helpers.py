"""Shared deterministic fixtures for simulated DMS tests."""

from __future__ import annotations

import numpy as np

from dms.core import FaceObservation


def make_face(
    *,
    ear: float = 0.30,
    mar: float = 0.20,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    quality: str = "good",
    quality_detail: str = "face and eye geometry available",
    face_width: float = 120.0,
    bbox: tuple[float, float, float, float] = (120.0, 50.0, 360.0, 330.0),
    mouth_center: tuple[float, float] = (240.0, 250.0),
) -> FaceObservation:
    # MeasurementEngine does not read the raw points; they exist so overlay
    # and future landmark consumers receive the same object shape.
    points = np.zeros((478, 2), dtype=np.float32)
    points[33] = (120.0, 150.0)
    points[263] = (360.0, 150.0)
    points[13] = (mouth_center[0], mouth_center[1] - 8.0)
    points[14] = (mouth_center[0], mouth_center[1] + 8.0)
    points[61] = (mouth_center[0] - 35.0, mouth_center[1])
    points[291] = (mouth_center[0] + 35.0, mouth_center[1])
    return FaceObservation(
        points=points,
        bbox=bbox,
        face_width=face_width,
        ear_left=ear,
        ear_right=ear,
        ear=ear,
        mar=mar,
        mouth_center=mouth_center,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        quality=quality,
        quality_detail=quality_detail,
    )


def update(engine, face, timestamp: float):
    return engine.update(
        face,
        timestamp,
        source_frame_width=480,
        source_frame_height=270,
    )
