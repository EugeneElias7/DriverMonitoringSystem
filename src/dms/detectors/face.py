"""MediaPipe Face Landmarker wrapper with ROI retry and pose fallback."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.base import HealthMixin, validate_frame
from ..core.types import Detection, FaceObservation
from .face_geometry import (
    LEFT_EYE,
    RIGHT_EYE,
    eye_aspect_ratio,
    face_bbox,
    face_width as compute_face_width,
    map_normalized_landmarks_to_pixels,
    mouth_aspect_ratio,
    mouth_center,
    pose_from_pnp,
    pose_from_transform,
)


class FaceAnalyzer(HealthMixin):
    """Shared face/landmark component.

    The landmark model is used once per processed frame. Drowsiness, head pose
    and smoking proximity reuse the resulting observation instead of running
    extra face networks.
    """

    name = "face_landmarker"

    def __init__(self, config: Any, mode: str = "IMAGE", expected_sha256: str | None = None):
        super().__init__(self.name)
        self.config = config
        self.mode = mode.upper()
        if self.mode not in {"IMAGE", "VIDEO"}:
            raise ValueError("face mode must be IMAGE or VIDEO")
        self.expected_sha256 = expected_sha256
        self._landmarker = None
        self._fallback_landmarker = None
        self._last_timestamp_ms = 0
        self._last_observation: FaceObservation | None = None
        self.model_path = config.resolve_path(config.get("face.model_path"))
        self._hash_checked = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"face landmarker not found: {self.model_path}")
            if self.expected_sha256:
                digest = _sha256(self.model_path)
                if digest.lower() != self.expected_sha256.lower():
                    raise ValueError(
                        f"face landmarker SHA256 mismatch: {digest} != {self.expected_sha256}"
                    )
                self._hash_checked = True
            self._landmarker = self._create_landmarker(self.mode)
            self._mark_ready(
                f"MediaPipe Face Landmarker ({self.mode}); 478 landmarks; model hash verified"
                if self._hash_checked
                else f"MediaPipe Face Landmarker ({self.mode})"
            )
        except Exception as exc:
            self._mark_error(f"{type(exc).__name__}: {exc}")
            raise

    def _create_landmarker(self, mode: str):
        # Imported lazily so tools that do not use the face model do not pay
        # MediaPipe's import/initialisation cost.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO if mode == "VIDEO" else vision.RunningMode.IMAGE,
            num_faces=int(self.config.get("face.num_faces", 1)),
            min_face_detection_confidence=float(
                self.config.get("face.min_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                self.config.get("face.min_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(self.config.get("face.min_tracking_confidence", 0.5)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _run(self, image_bgr: np.ndarray, timestamp_ms: int):
        if self._landmarker is None:
            raise RuntimeError("face landmarker is not loaded")
        import mediapipe as mp

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "VIDEO":
            # MediaPipe requires strictly increasing integer timestamps.
            ts = max(int(timestamp_ms), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = ts
            return self._landmarker.detect_for_video(mp_image, ts)
        return self._landmarker.detect(mp_image)

    def _run_fallback(self, image_bgr: np.ndarray):
        if self._fallback_landmarker is None:
            self._fallback_landmarker = self._create_landmarker("IMAGE")
        import mediapipe as mp

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._fallback_landmarker.detect(mp_image)

    def analyze(
        self,
        frame: np.ndarray,
        timestamp: float,
        roi_hint: tuple[int, int, int, int] | None = None,
    ) -> FaceObservation | None:
        """Return a face observation for this frame, or None if no face is found."""
        started = time.perf_counter()
        error: Exception | None = None
        observation: FaceObservation | None = None
        try:
            if not self._loaded:
                return None
            validate_frame(frame)
            height, width = frame.shape[:2]
            timestamp_ms = int(round(float(timestamp) * 1000.0))
            result = self._run(frame, timestamp_ms)
            if not result.face_landmarks:
                roi = self._select_roi(roi_hint, width, height)
                if roi is not None:
                    observation = self._try_roi(frame, roi, timestamp)
                    if observation is not None:
                        observation.pose_source = "pnp_roi" if observation.pitch is not None else "none"
                        observation.roi_upscaled = True
            if observation is None and result.face_landmarks:
                observation = self._to_observation(
                    result=result,
                    frame_shape=frame.shape,
                    timestamp=timestamp,
                    roi=None,
                    used_roi=False,
                )
            self._last_observation = observation
            if observation is None:
                self.health.state = "ok"
                self.health.message = "no face in frame"
            else:
                self.health.state = "ok"
                self.health.message = ""
        except Exception as exc:  # one bad frame must not kill the pipeline
            error = exc
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
        return observation

    def _select_roi(
        self,
        roi_hint: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        configured = self.config.get("face.roi")
        if roi_hint is None and configured:
            roi_hint = tuple(int(v) for v in configured)  # type: ignore[assignment]
        if roi_hint is not None:
            x1, y1, x2, y2 = roi_hint
            pad = float(self.config.get("face.roi_padding", 0.15))
            w, h = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - w * pad))
            y1 = max(0, int(y1 - h * pad))
            x2 = min(width, int(x2 + w * pad))
            y2 = min(height, int(y2 + h * pad))
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                return (x1, y1, x2, y2)
            return None
        if not bool(self.config.get("face.auto_roi_retry", True)):
            return None
        # Conservative upper-centre retry for a dash/window-mounted camera.
        x1 = int(width * 0.18)
        y1 = int(height * 0.03)
        x2 = int(width * 0.82)
        y2 = int(height * 0.78)
        if x2 - x1 >= 32 and y2 - y1 >= 32:
            return (x1, y1, x2, y2)
        return None

    def _try_roi(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
        timestamp: float,
    ) -> FaceObservation | None:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        min_side = int(self.config.get("face.roi_min_side", 512))
        short_side = min(crop.shape[:2])
        if short_side < min_side:
            scale = min(3.0, min_side / max(short_side, 1))
            crop_for_model = cv2.resize(
                crop,
                (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            crop_for_model = crop
        try:
            result = self._run_fallback(crop_for_model)
        except Exception:
            # If the optional fallback path fails, report no observation rather
            # than turning a no-face frame into a module error.
            return None
        if not result.face_landmarks:
            return None
        return self._to_observation(
            result=result,
            frame_shape=frame.shape,
            timestamp=timestamp,
            roi=(x1, y1, x2, y2),
            used_roi=True,
            crop_origin=(x1, y1),
            crop_size=(crop.shape[1], crop.shape[0]),
        )

    def _to_observation(
        self,
        *,
        result: Any,
        frame_shape: tuple[int, ...],
        timestamp: float,
        roi: tuple[int, int, int, int] | None,
        used_roi: bool,
        crop_origin: tuple[int, int] = (0, 0),
        crop_size: tuple[int, int] | None = None,
    ) -> FaceObservation:
        height, width = frame_shape[:2]
        if used_roi:
            cw, ch = crop_size or (width, height)
            points = map_normalized_landmarks_to_pixels(
                result.face_landmarks[0], crop_origin[0], crop_origin[1], cw, ch
            )
        else:
            points = np.asarray(
                [(float(lm.x) * width, float(lm.y) * height) for lm in result.face_landmarks[0]],
                dtype=np.float32,
            )
        box = face_bbox(points, width, height)
        ear_l = eye_aspect_ratio(points, LEFT_EYE)
        ear_r = eye_aspect_ratio(points, RIGHT_EYE)
        mar = mouth_aspect_ratio(points)
        mcenter = mouth_center(points)
        fwidth = compute_face_width(points)

        pitch = yaw = roll = None
        pose_source = "none"
        transform = None
        if getattr(result, "facial_transformation_matrixes", None):
            try:
                transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
                if not used_roi:
                    pitch, yaw, roll = pose_from_transform(transform)
                    pose_source = "transform"
                else:
                    # The transform was estimated on an upscaled crop. Use the
                    # landmarks in original-frame geometry for the fallback pose.
                    pose = pose_from_pnp(points, (width, height))
                    if pose is not None:
                        pitch, yaw, roll = pose
                        pose_source = "pnp_roi"
            except Exception:
                pitch = yaw = roll = None
        if pitch is None and bool(self.config.get("face.compute_pnp_fallback", True)):
            pose = pose_from_pnp(points, (width, height))
            if pose is not None:
                pitch, yaw, roll = pose
                pose_source = "pnp"

        blendshapes: dict[str, float] = {}
        if getattr(result, "face_blendshapes", None):
            for category in result.face_blendshapes[0]:
                name = getattr(category, "category_name", None)
                score = getattr(category, "score", None)
                if name is not None and score is not None:
                    blendshapes[str(name)] = float(score)
        return FaceObservation(
            bbox=box,
            landmarks=points,
            image_size=(width, height),
            timestamp=float(timestamp),
            ear_left=ear_l,
            ear_right=ear_r,
            ear=(ear_l + ear_r) / 2.0,
            # Averaging both eyes is fragile under yaw: the far eye can be
            # foreshortened while the near eye is open. The maximum is used for
            # closure logic; both values stay in the output for inspection.
            ear_max=max(ear_l, ear_r),
            mar=mar,
            mouth_center=mcenter,
            face_width=fwidth,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            pose_source=pose_source,
            transform=transform,
            blendshapes=blendshapes,
            roi=roi,
            roi_upscaled=used_roi,
        )

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        observation = self.analyze(frame, timestamp)
        if observation is None:
            return []
        return [
            Detection(
                label="face",
                confidence=1.0,
                bbox=observation.bbox,
                source=self.name,
                metadata={"pose_source": observation.pose_source},
            )
        ]

    @property
    def last_observation(self) -> FaceObservation | None:
        return self._last_observation

    def reset(self) -> None:
        self._last_timestamp_ms = 0
        self._last_observation = None
        if self.mode == "VIDEO" and self._loaded:
            # A new video sequence needs a fresh tracker.
            try:
                if self._landmarker is not None:
                    self._landmarker.close()
                self._landmarker = self._create_landmarker(self.mode)
            except Exception as exc:
                self.health.state = "error"
                self.health.last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        for landmarker in (self._landmarker, self._fallback_landmarker):
            try:
                if landmarker is not None:
                    landmarker.close()
            except Exception:
                pass
        self._landmarker = None
        self._fallback_landmarker = None
        super().close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
