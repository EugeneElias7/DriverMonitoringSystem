"""Face landmarks -> per-frame driver metrics.

The face path is OpenCV only: YuNet face detection plus insightface 2d106
landmarks, both through OpenCV DNN. MediaPipe is deliberately not used: its
prebuilt ARM binary needs ARM AES instructions that the Raspberry Pi 3
Cortex-A53 lacks, and it dies with "This binary was compiled with aes enabled"
on first use. The OpenCV path runs on both the PC and the Pi 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as geo
from .config import FaceConfig


@dataclass
class FaceObservation:
    """Per-frame facial metrics. `landmarks` is optional so scripted or
    landmark-free sources can drive the analysers directly."""

    present: bool
    landmarks: np.ndarray | None = None
    bbox: tuple[float, float, float, float] | None = None
    ear_left: float = math.nan
    ear_right: float = math.nan
    ear: float = math.nan
    mar: float = math.nan
    pitch: float = math.nan
    yaw: float = math.nan
    roll: float = math.nan
    # Mouth anchor in source pixels, set by the active landmark backend. The
    # property below is a fallback for landmark-free callers.
    mouth: tuple[float, float] | None = None

    @property
    def mouth_point(self) -> tuple[float, float] | None:
        if self.mouth is not None:
            return self.mouth
        if self.landmarks is None:
            return None
        cx, cy = geo.mouth_center(self.landmarks)
        return float(cx), float(cy)

    @property
    def box_area(self) -> float:
        if self.bbox is None:
            return 0.0
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _observe(
    points: np.ndarray,
    size: tuple[int, int],
    matrix: np.ndarray,
    right_eye: tuple,
    left_eye: tuple,
    mouth_lips: tuple,
    mouth_corners: tuple,
    pose_indices: tuple,
) -> FaceObservation:
    ear_left = geo.eye_aspect_ratio(points, left_eye)
    ear_right = geo.eye_aspect_ratio(points, right_eye)
    pitch, yaw, roll = geo.head_pose(points, size, matrix, pose_indices)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    mx, my = geo.mouth_center(points, mouth_lips, mouth_corners)
    return FaceObservation(
        present=True,
        landmarks=points,
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        ear_left=ear_left,
        ear_right=ear_right,
        ear=(ear_left + ear_right) / 2.0,
        mar=geo.mouth_aspect_ratio(points, mouth_lips, mouth_corners),
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        mouth=(float(mx), float(my)),
    )


class OpenCvFaceAnalyzer:
    """YuNet detection + 2d106 landmarks, everything through OpenCV DNN."""

    # 106-point mark-up (insightface 2d106det), matching the face SDK layout
    # pictured in nttstar/insightface-resources alignment/images/2d106markup.jpg
    # and verified point-by-point on a labelled capture:
    #   0-32   face contour, 0 = chin bottom centre
    #   33-42  image-left eye (subject's right), 38 = eye centre
    #   43-51  image-left brow
    #   52-71  mouth: corners 52/61, inner-lip ring 65/66/62/70/69/57/60/54
    #   72-86  nose, 85 = tip (lowest point of the nose base)
    #   87-96  image-right eye (subject's left), 88 = eye centre
    #   97-105 image-right brow
    RIGHT_EYE = (39, 40, 42, 35, 37, 33)  # inner, upper, upper, outer, lower, lower
    LEFT_EYE = (89, 95, 94, 93, 87, 90)  # inner, upper, upper, outer, lower, lower
    MOUTH_LIPS = (62, 60)  # central inner upper / inner lower lip
    MOUTH_CORNERS = (52, 61)
    POSE_INDICES = (85, 0, 35, 93, 52, 61)  # nose tip, chin, eye outer corners, mouth corners

    LANDMARK_INPUT = 192

    def __init__(self, cfg: FaceConfig):
        self._det = cv2.FaceDetectorYN.create(
            cfg.detector_model_path,
            "",
            (320, 320),
            cfg.min_detection_confidence,
            0.3,
            5000,
        )
        self._net = cv2.dnn.readNetFromONNX(cfg.landmark_model_path)
        self._size: tuple[int, int] | None = None
        self._matrix: np.ndarray | None = None
        self._eye_base = None
        self._mouth_base = None
        self._base_n = 0
        self._base_max = 30

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceObservation:
        h, w = frame_bgr.shape[:2]
        if self._size != (h, w):
            self._size = (h, w)
            self._det.setInputSize((w, h))
            self._matrix = geo.camera_matrix((h, w))

        _, faces = self._det.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return FaceObservation(present=False)

        face = faces[0]  # x, y, w, h, 5 landmarks, score
        points = self._landmarks(frame_bgr, float(face[0]), float(face[1]),
                                 float(face[2]), float(face[3]))
        if points is None:
            return FaceObservation(present=False)
        observation = _observe(
            points,
            (h, w),
            self._matrix,
            self.RIGHT_EYE,
            self.LEFT_EYE,
            self.MOUTH_LIPS,
            self.MOUTH_CORNERS,
            self.POSE_INDICES,
        )
        # Brightness heuristic disabled: unreliable, caused false alerts.
        # Drowsiness uses head-pose nodding only.
        return observation

    def _landmarks(
        self, frame_bgr: np.ndarray, x: float, y: float, w: float, h: float
    ) -> np.ndarray | None:
        matrix, scale = _align_params(x, y, w, h)
        crop = cv2.warpAffine(
            frame_bgr, matrix, (self.LANDMARK_INPUT, self.LANDMARK_INPUT),
            borderMode=cv2.BORDER_REPLICATE,
        )
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / 128.0, (self.LANDMARK_INPUT, self.LANDMARK_INPUT),
            (127.5, 127.5, 127.5), swapRB=True,
        )
        self._net.setInput(blob)
        pred = self._net.forward()
        return _decode_106(pred, matrix, scale)

    def _eyes_closed(self, frame_bgr, face):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        vals = []
        for xi, yi in ((4, 5), (6, 7)):
            ex = int(face[xi]); ey = int(face[yi])
            if not (0 <= ex < w and 0 <= ey < h):
                continue
            half = max(6, int(face[2] * 0.10))
            x1 = max(0, ex - half); x2 = min(w, ex + half)
            y1 = max(0, ey - half // 2); y2 = min(h, ey + half // 2)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            vals.append(float(roi.mean()))
        if not vals:
            return False
        brightness = sum(vals) / len(vals)
        if self._base_n < self._base_max:
            if self._eye_base is None:
                self._eye_base = brightness
            else:
                self._eye_base = 0.9 * self._eye_base + 0.1 * brightness
            self._base_n += 1
            return False
        if self._eye_base is None or self._eye_base < 1.0:
            return False
        return brightness > self._eye_base * 1.2

    def _mouth_open(self, frame_bgr, face):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mx = int((face[8] + face[10]) / 2)
        my = int((face[9] + face[11]) / 2)
        if not (0 <= mx < w and 0 <= my < h):
            return False
        half_x = max(8, int(face[2] * 0.12))
        half_y = max(4, int(face[2] * 0.06))
        x1 = max(0, mx - half_x); x2 = min(w, mx + half_x)
        y1 = max(0, my - half_y); y2 = min(h, my + half_y)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        brightness = float(roi.mean())
        if self._base_n < self._base_max:
            if self._mouth_base is None:
                self._mouth_base = brightness
            else:
                self._mouth_base = 0.9 * self._mouth_base + 0.1 * brightness
            return False
        if self._mouth_base is None or self._mouth_base < 1.0:
            return False
        return brightness < self._mouth_base * 0.80

    def close(self) -> None:
        pass

    def __enter__(self) -> "OpenCvFaceAnalyzer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _align_params(x: float, y: float, w: float, h: float) -> tuple[np.ndarray, float]:
    """Affine matrix + scale mapping a 1.5x face box into the 192 landmark grid.

    Mirrors insightface's model_zoo/landmark.py: crop side = 1.5 * max(w, h)
    centred on the box, resized to 192, no rotation.
    """
    side = 1.5 * max(w, h)
    cx, cy = x + w / 2.0, y + h / 2.0
    scale = OpenCvFaceAnalyzer.LANDMARK_INPUT / side
    matrix = np.array(
        [
            [scale, 0.0, OpenCvFaceAnalyzer.LANDMARK_INPUT / 2.0 - cx * scale],
            [0.0, scale, OpenCvFaceAnalyzer.LANDMARK_INPUT / 2.0 - cy * scale],
        ],
        dtype=np.float64,
    )
    return matrix, scale


def _decode_106(pred: np.ndarray, matrix: np.ndarray, scale: float) -> np.ndarray | None:
    """2d106 output -> 106 full-frame pixel points.

    The model emits points on a [-1, 1] grid (insightface decodes them with
    (pred + 1) * input / 2); the crop affine `u = scale * f + t` inverts as
    `f = (u - t) / scale`.
    """
    flat = np.asarray(pred).reshape(-1)
    if flat.size < 212:
        return None
    pts = flat[:212].reshape(106, 2).astype(np.float64)
    pts192 = (pts + 1.0) * (OpenCvFaceAnalyzer.LANDMARK_INPUT / 2.0)
    return (pts192 - matrix[:, 2]) / scale


# The face path has one implementation on purpose. Keeping a backend factory
# would invite a MediaPipe fallback that cannot run on the Pi 3.
FaceAnalyzer = OpenCvFaceAnalyzer
