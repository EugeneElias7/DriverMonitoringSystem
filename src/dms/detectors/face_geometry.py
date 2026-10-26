"""Landmark geometry: EAR, MAR, face boxes and head-pose estimation."""

from __future__ import annotations

import math

import cv2
import numpy as np

# MediaPipe Face Mesh indices used by the conventional EAR/MAR formulas.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FACE_LEFT = 33
FACE_RIGHT = 263

# Approximate canonical points in millimetres, with y pointing down in the
# image/camera convention. These are only used for a relative fallback pose.
PNP_INDICES = [1, 152, 33, 263, 61, 291]
PNP_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, 63.6, -12.5),       # chin
        (-43.3, -32.7, -26.0),    # image-left eye outer corner
        (43.3, -32.7, -26.0),     # image-right eye outer corner
        (-28.9, 28.9, -24.1),     # image-left mouth corner
        (28.9, 28.9, -24.1),      # image-right mouth corner
    ],
    dtype=np.float32,
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def eye_aspect_ratio(points: np.ndarray, indices: list[int]) -> float:
    p = np.asarray(points, dtype=np.float64)[indices]
    denominator = 2.0 * _dist(p[0], p[3])
    if denominator <= 1e-6:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / denominator


def mouth_aspect_ratio(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    width = _dist(p[MOUTH_LEFT], p[MOUTH_RIGHT])
    if width <= 1e-6:
        return 0.0
    return _dist(p[MOUTH_TOP], p[MOUTH_BOTTOM]) / width


def mouth_center(points: np.ndarray) -> tuple[float, float]:
    p = np.asarray(points, dtype=np.float64)
    center = (p[MOUTH_TOP] + p[MOUTH_BOTTOM]) / 2.0
    return (float(center[0]), float(center[1]))


def face_width(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    return _dist(p[FACE_LEFT], p[FACE_RIGHT])


def face_bbox(points: np.ndarray, width: int, height: int, padding_ratio: float = 0.02) -> tuple[float, float, float, float]:
    p = np.asarray(points, dtype=np.float64)
    x1, y1 = p[:, 0].min(), p[:, 1].min()
    x2, y2 = p[:, 0].max(), p[:, 1].max()
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    x1 = max(0.0, x1 - pad_x)
    y1 = max(0.0, y1 - pad_y)
    x2 = min(float(width), x2 + pad_x)
    y2 = min(float(height), y2 + pad_y)
    return (float(x1), float(y1), float(x2), float(y2))


def rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return pitch, yaw, roll in degrees.

    Convention used throughout this project:
    * pitch > 0 means the face is looking downwards in our camera setup
    * yaw sign is image-left positive for MediaPipe's transform matrix, but the
      semantic label is configurable because it is easy to get backwards.
    """
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape[0] < 3 or r.shape[1] < 3:
        raise ValueError("rotation matrix must be at least 3x3")
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def pose_from_transform(matrix: np.ndarray) -> tuple[float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"expected 4x4 transformation matrix, got {m.shape}")
    return rotation_to_euler(m[:3, :3])


def pose_from_pnp(points: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float] | None:
    """Approximate solvePnP fallback, relative rather than calibrated."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= max(PNP_INDICES):
        return None
    image_points = points[PNP_INDICES].astype(np.float32)
    if not np.isfinite(image_points).all():
        return None
    width, height = image_size
    focal = 1.2 * max(width, height)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    success, rvec, _ = cv2.solvePnP(
        PNP_MODEL_POINTS,
        image_points,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(rotation)


def map_normalized_landmarks_to_pixels(
    landmarks: list[object],
    crop_x1: int,
    crop_y1: int,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    """Map MediaPipe normalized landmarks from a crop back to full-frame pixels."""
    return np.asarray(
        [
            (crop_x1 + float(lm.x) * crop_width, crop_y1 + float(lm.y) * crop_height)
            for lm in landmarks
        ],
        dtype=np.float32,
    )
