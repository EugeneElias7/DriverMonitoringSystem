"""Face analysis using YuNet detection + ONNX face landmark model.

The landmark model is Google's face_landmarks (converted from MediaPipe's
face_landmarks_detector.tflite to ONNX). It provides 478 3D landmarks that
track eye and mouth expressions, unlike the static 2d106 alignment model.

Runs on OpenCV DNN (no MediaPipe runtime, no ARM AES dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from .config import FaceConfig
from .face import FaceObservation, OpenCvFaceAnalyzer


class OnnxFaceAnalyzer(OpenCvFaceAnalyzer):
    """YuNet face detection + ONNX face landmark model (478 points).

    Replaces the static 2d106 alignment model with Google's face landmark
    model, which tracks eye and mouth expressions for accurate drowsiness
    detection. Runs on OpenCV DNN (no MediaPipe, no ARM AES).
    """

    # MediaPipe Face Mesh landmark indices.
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)
    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    MOUTH_LIPS = (13, 14)
    MOUTH_CORNERS = (61, 291)
    POSE_INDICES = (1, 152, 33, 263, 61, 291)

    def __init__(self, cfg: FaceConfig):
        super().__init__(cfg)
        model_path = (
            Path(__file__).resolve().parent.parent / "models" / "face_landmarks.onnx"
        )
        self._lm_net = cv2.dnn.readNetFromONNX(str(model_path))
        self._lm_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def process(self, frame_bgr, timestamp_ms):
        h, w = frame_bgr.shape[:2]
        if self._size != (h, w):
            self._size = (h, w)
            self._det.setInputSize((w, h))
            self._matrix = geo.camera_matrix((h, w))

        _, faces = self._det.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return FaceObservation(present=False)

        face = faces[0]
        x, y, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])

        # Crop face region with margin, resize to 256x256.
        margin = 0.25
        cx, cy = x + fw / 2, y + fh / 2
        side = max(fw, fh) * (1 + margin)
        x1 = max(0, int(cx - side / 2))
        y1 = max(0, int(cy - side / 2))
        x2 = min(w, int(cx + side / 2))
        y2 = min(h, int(cy + side / 2))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return FaceObservation(present=False)

        crop_resized = cv2.resize(crop, (256, 256))
        blob = (crop_resized.astype(np.float32) - 127.5) / 127.5
        blob = blob[None]  # NHWC: [1, 256, 256, 3]

        self._lm_net.setInput(blob)
        out = self._lm_net.forward(["Identity", "Identity_1", "Identity_2"])
        landmarks_raw = out[0].reshape(-1)  # 1434 = 478 * 3
        if landmarks_raw.size < 478 * 3:
            return FaceObservation(present=False)

        pts = landmarks_raw[:478 * 3].reshape(478, 3)
        # The graph emits pixel coordinates in the 256x256 network input, not
        # normalised [0, 1] values. Scale back to the crop's own pixels, then
        # add the crop origin to land in full-frame coordinates.
        crop_h, crop_w = crop.shape[:2]
        points = np.zeros((478, 2), dtype=np.float64)
        points[:, 0] = pts[:, 0] * (crop_w / 256.0) + x1
        points[:, 1] = pts[:, 1] * (crop_h / 256.0) + y1

        ear_left = geo.eye_aspect_ratio(points, self.LEFT_EYE)
        ear_right = geo.eye_aspect_ratio(points, self.RIGHT_EYE)
        pitch, yaw, roll = geo.head_pose(points, (h, w), self._matrix, self.POSE_INDICES)
        mx, my = geo.mouth_center(points, self.MOUTH_LIPS, self.MOUTH_CORNERS)

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        return FaceObservation(
            present=True,
            landmarks=points,
            bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
            ear_left=float(ear_left),
            ear_right=float(ear_right),
            ear=float((ear_left + ear_right) / 2.0),
            mar=float(geo.mouth_aspect_ratio(points, self.MOUTH_LIPS, self.MOUTH_CORNERS)),
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
            mouth=(float(mx), float(my)),
        )

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def create_face_analyzer(cfg):
    """Use ONNX face landmarks if available, else fall back to 2d106."""
    model_path = (
        Path(__file__).resolve().parent.parent / "models" / "face_landmarks.onnx"
    )
    if model_path.exists():
        try:
            return OnnxFaceAnalyzer(cfg)
        except Exception as e:
            print(f"ONNX face analyzer failed: {e}", flush=True)
    return OpenCvFaceAnalyzer(cfg)
