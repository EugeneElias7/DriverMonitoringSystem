"""Facial geometry: eye/mouth aspect ratios and PnP head pose.

All functions take landmarks as an (N, 2) float array in pixel coordinates.
The default indices are the MediaPipe FaceMesh layout; the OpenCV 2d106 face
backend passes its own index tuples, so the defaults are only a fallback.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# EAR uses six points per eye: outer corner, two upper lids, inner corner,
# two lower lids.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

MOUTH_CORNERS = (61, 291)
MOUTH_LIPS = (13, 14)  # inner upper lip, inner lower lip

# nose tip, chin, outer eye corners, mouth corners
POSE_INDICES = (1, 152, 33, 263, 61, 291)

# Generic 3D face model in millimetres, camera-style axes: +x right, +y down,
# +z forward. Taken from the well-known OpenCV head-pose tutorial model, which
# uses +y up / +z toward the viewer; both y and z are negated here, which is a
# 180 degree rotation about x and therefore preserves handedness.
_S = 1.0 / 5.5
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),                    # nose tip
        (0.0, 330.0 * _S, 65.0 * _S),       # chin
        (-225.0 * _S, -170.0 * _S, 135.0 * _S),  # subject's right eye outer
        (225.0 * _S, -170.0 * _S, 135.0 * _S),   # subject's left eye outer
        (-150.0 * _S, 150.0 * _S, 125.0 * _S),   # subject's right mouth corner
        (150.0 * _S, 150.0 * _S, 125.0 * _S),    # subject's left mouth corner
    ],
    dtype=np.float64,
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, indices=RIGHT_EYE) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(
    points: np.ndarray, lips: tuple = MOUTH_LIPS, corners: tuple = MOUTH_CORNERS
) -> float:
    upper, lower = points[lips[0]], points[lips[1]]
    left, right = points[corners[0]], points[corners[1]]
    width = _dist(left, right)
    if width < 1e-6:
        return 0.0
    return _dist(upper, lower) / width


def camera_matrix(image_size: tuple[int, int], focal_scale: float = 1.0) -> np.ndarray:
    h, w = image_size
    f = float(max(w, h)) * focal_scale
    return np.array(
        [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose R = Rz*Ry*Rx into pitch (x), yaw (y), roll (z), in degrees."""
    r = rotation
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-7:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def head_pose(
    points: np.ndarray,
    image_size: tuple[int, int],
    matrix: np.ndarray | None = None,
    pose_indices: tuple = POSE_INDICES,
) -> tuple[float, float, float]:
    """Solve head orientation from 2D landmarks. Returns (pitch, yaw, roll) degrees.

    Sign convention, verified by the test suite: yaw > 0 means the face turned
    toward the +x image axis, pitch > 0 means the nose tilted toward +y (down).

    Six loose face points are close to planar, so PnP has a mirrored solution
    that fits almost as well and leaves the face rolled ~180 degrees. Every
    solver is tried and mirrored candidates are dropped, leaving the least
    error among physically possible poses.
    """
    uv = np.array([points[i] for i in pose_indices], dtype=np.float64)
    k = matrix if matrix is not None else camera_matrix(image_size)
    distortion = np.zeros((4, 1), dtype=np.float64)
    best: tuple[float, tuple[float, float, float]] | None = None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, uv, k, distortion, flags=flag)
        except cv2.error:
            continue
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        angles = rotation_matrix_to_euler(rotation)
        if abs(angles[2]) > 90.0:
            continue
        projected, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, k, distortion)
        err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - uv, axis=1)))
        if best is None or err < best[0]:
            best = (err, angles)
    if best is None:
        return 0.0, 0.0, 0.0
    return best[1]


def eye_center(points: np.ndarray, indices=RIGHT_EYE) -> np.ndarray:
    return np.mean(points[list(indices)], axis=0)


def mouth_center(
    points: np.ndarray, lips: tuple = MOUTH_LIPS, corners: tuple = MOUTH_CORNERS
) -> np.ndarray:
    return np.mean(points[list(corners) + list(lips)], axis=0)
