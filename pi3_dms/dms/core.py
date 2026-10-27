"""Face-landmark DMS core.

This module is deliberately independent of the web dashboard and of the
optional object detector.  It contains the MediaPipe face model, landmark
geometry and the temporal measurement engine for eyes, mouth, head pose and
tracking confidence.

The calculator is a research aid.  It is not a medical device and the
``tracking_confidence`` value is a transparent heuristic, not a probability
reported by MediaPipe.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# MediaPipe Face Mesh indices.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FACE_LEFT = 33
FACE_RIGHT = 263
NOSE_TIP = 1

# State names used by the core and the dashboard.
NO_FACE = "no_face"
UNCERTAIN = "uncertain"
ALERT = "alert"
EYES_CLOSED = "eyes_closed"
PROLONGED_CLOSURE = "prolonged_closure"
MICROSLEEP = "microsleep"
YAWNING = "yawning"


@dataclass
class FaceObservation:
    """One face-landmark result in inference-frame pixels."""

    points: np.ndarray
    bbox: tuple[float, float, float, float]
    face_width: float
    ear_left: float
    ear_right: float
    ear: float
    mar: float
    mouth_center: tuple[float, float]
    yaw: float | None
    pitch: float | None
    roll: float | None
    quality: str
    quality_detail: str


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        )
    )


def _ear(points: np.ndarray, indices: list[int]) -> float:
    p = np.asarray(points, dtype=np.float64)[indices]
    denominator = 2.0 * _dist(p[0], p[3])
    if denominator <= 1e-6:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / denominator


def _mar(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    width = _dist(p[MOUTH_LEFT], p[MOUTH_RIGHT])
    if width <= 1e-6:
        return 0.0
    return _dist(p[MOUTH_TOP], p[MOUTH_BOTTOM]) / width


def _rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(rotation, dtype=np.float64)
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


class FaceModel:
    """MediaPipe Face Landmarker in VIDEO mode."""

    def __init__(
        self,
        model_path: Path,
        *,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"face model not found: {model_path}")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = None
        self._last_timestamp_ms = -1
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=int(num_faces),
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=0.5,
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        *,
        min_face_width_px: float = 40.0,
    ) -> FaceObservation | None:
        if self._mp is None:
            import mediapipe as mp

            self._mp = mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        stamp = max(int(timestamp_s * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = stamp
        result = self._landmarker.detect_for_video(image, stamp)
        if not result.face_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        landmarks = result.face_landmarks[0]
        points = np.asarray(
            [(float(lm.x) * width, float(lm.y) * height) for lm in landmarks],
            dtype=np.float32,
        )
        if points.shape[0] <= max(RIGHT_EYE):
            return None

        face_w = _dist(points[FACE_LEFT], points[FACE_RIGHT])
        x1, y1 = points[:, 0].min(), points[:, 1].min()
        x2, y2 = points[:, 0].max(), points[:, 1].max()
        pad_x = max(2.0, (x2 - x1) * 0.03)
        pad_y = max(2.0, (y2 - y1) * 0.03)
        bbox = (
            float(max(0.0, x1 - pad_x)),
            float(max(0.0, y1 - pad_y)),
            float(min(float(width), x2 + pad_x)),
            float(min(float(height), y2 + pad_y)),
        )

        yaw = pitch = roll = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
            if matrix.shape == (4, 4):
                pitch, yaw, roll = _rotation_to_euler(matrix[:3, :3])

        ear_left = _ear(points, LEFT_EYE)
        ear_right = _ear(points, RIGHT_EYE)
        # Using the more open eye avoids calling a one-eye landmark failure a
        # closure.  A genuine bilateral closure is still below threshold in
        # both eyes.
        ear = max(ear_left, ear_right)
        mar = _mar(points)
        quality = "good"
        detail = "face and eye geometry available"
        if face_w < min_face_width_px:
            quality = "too_small"
            detail = "face too small for a precise eye measurement"
        elif not (0.01 <= ear <= 0.9):
            quality = "invalid_ear"
            detail = "eye geometry is not reliable"
        elif yaw is not None and abs(yaw) > 55.0:
            quality = "low_yaw"
            detail = "head turned too far for eye measurement"
        elif pitch is not None and abs(pitch) > 40.0:
            quality = "low_pitch"
            detail = "head angle too steep for eye measurement"

        center = (points[MOUTH_TOP] + points[MOUTH_BOTTOM]) / 2.0
        return FaceObservation(
            points=points,
            bbox=bbox,
            face_width=face_w,
            ear_left=ear_left,
            ear_right=ear_right,
            ear=ear,
            mar=mar,
            mouth_center=(float(center[0]), float(center[1])),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            quality=quality,
            quality_detail=detail,
        )


def _head_state(yaw: float | None, pitch: float | None) -> str:
    if yaw is None or pitch is None:
        return "unknown"
    # A small dead zone prevents a camera mounted slightly off-centre from
    # reporting a permanent left/right turn.  Normal mirror glances stay inside
    # this band; a clear cabin-facing turn does not.
    if yaw > 30.0:
        return "left"
    if yaw < -30.0:
        return "right"
    if pitch > 25.0:
        return "down"
    if pitch < -20.0:
        return "up"
    return "forward"


def _tracking_confidence(
    obs: FaceObservation,
    *,
    source_frame_width: int,
    source_frame_height: int,
    presence_streak: int,
) -> tuple[float, str]:
    """Return a transparent reliability heuristic, not a model probability.

    The score combines face size in the inference frame, pose validity and
    recent face continuity.  It is deliberately conservative: a small or
    highly rotated face never reaches the top of the range.
    """
    if obs.quality != "good":
        return 0.0, "face quality gate failed"
    if obs.face_width < 40.0:
        return 0.0, "face too small for a reliable tracking score"
    target_width = max(80.0, float(source_frame_width) * 0.25)
    size_score = min(1.0, max(0.0, obs.face_width / target_width))
    pose_penalty = 0.0
    if obs.yaw is not None:
        pose_penalty += min(1.0, max(0.0, abs(obs.yaw) - 20.0) / 35.0) * 0.55
    if obs.pitch is not None:
        pose_penalty += min(1.0, max(0.0, abs(obs.pitch) - 15.0) / 30.0) * 0.45
    pose_score = max(0.0, 1.0 - pose_penalty)
    continuity = min(1.0, max(0.0, presence_streak / 5.0))
    score = 0.55 * size_score + 0.30 * pose_score + 0.15 * continuity
    return round(min(1.0, max(0.0, score)), 2), "heuristic: size, pose, continuity"


def _safe_float(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


class MeasurementEngine:
    """Smooth, threshold and time-limit the face measurements.

    Eye and mouth measurements are independent.  A yawn does not require the
    eyes to be open and an eye closure does not erase a yawn episode.  The
    dashboard can therefore display simultaneous conditions without one
    overwriting another.
    """

    def __init__(
        self,
        *,
        ear_smoothing_frames: int = 3,
        mar_smoothing_frames: int = 5,
        perclos_window_s: float = 60.0,
        perclos_min_window_s: float = 10.0,
        prolonged_ms: float = 900.0,
        microsleep_ms: float = 1500.0,
        blink_max_ms: float = 500.0,
        yawn_mar: float = 0.60,
        yawn_release_mar: float | None = None,
        yawn_min_ms: float = 800.0,
        distracted_ms: float = 1500.0,
    ):
        self.ear_history: deque[float] = deque(maxlen=max(1, ear_smoothing_frames))
        self.mar_history: deque[float] = deque(maxlen=max(1, mar_smoothing_frames))
        self.baseline = 0.26
        self.samples = 0
        self.presence_streak = 0
        self.last_timestamp: float | None = None

        self.eye_state = "unknown"
        self.closed_since: float | None = None
        self.closure_candidate: float | None = None
        self.raw_closure_start: float | None = None
        self.closure_ms = 0.0
        self.prolonged_emitted = False
        self.microsleep_emitted = False

        self.yawn_since: float | None = None
        self.yawn_ms = 0.0
        self.yawn_emitted = False
        self.yawn_state = "clear"

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.perclos: deque[tuple[float, bool]] = deque()
        self.perclos_window_s = perclos_window_s
        self.perclos_min_window_s = perclos_min_window_s
        self.prolonged_ms = prolonged_ms
        self.microsleep_ms = microsleep_ms
        self.blink_max_ms = blink_max_ms
        self.yawn_mar = yawn_mar
        # A lower release threshold prevents flicker at the yawn boundary.
        # It does not lower the threshold for starting a new yawn episode.
        self.yawn_release_mar = (
            yawn_mar * 0.82 if yawn_release_mar is None else yawn_release_mar
        )
        self.yawn_min_ms = yawn_min_ms
        self.distracted_ms = distracted_ms
        self.away_since: float | None = None
        self.away_ms = 0.0
        self.last_state = "unknown"

    def reset_tracking(self) -> None:
        """Clear continuous face measurements, preserving counters and baseline."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.eye_state = "unknown"
        self.closed_since = None
        self.closure_candidate = None
        self.raw_closure_start = None
        self.closure_ms = 0.0
        self.prolonged_emitted = False
        self.microsleep_emitted = False
        self.yawn_since = None
        self.yawn_ms = 0.0
        self.yawn_emitted = False
        self.yawn_state = "clear"
        self.away_since = None
        self.away_ms = 0.0
        self.presence_streak = 0

    def _threshold(self, pitch: float | None) -> float:
        ratio = 0.55 if pitch is not None and pitch > 25.0 else 0.65
        # Do not let a low personal baseline drive the threshold so low that a
        # real closure is missed.  0.15 is a floor for this landmark model; the
        # 0.20 cap prevents a high baseline from calling an open eye closed.
        return min(0.20, max(0.15, self.baseline * ratio))

    def _adapt_baseline(self, ear: float, obs: FaceObservation) -> None:
        if obs.yaw is not None and abs(obs.yaw) > 30.0:
            return
        if obs.pitch is not None and abs(obs.pitch) > 20.0:
            return
        alpha = 0.05 if self.samples < 30 else 0.01
        self.baseline = (1.0 - alpha) * self.baseline + alpha * ear
        self.baseline = min(0.40, max(0.16, self.baseline))
        self.samples += 1

    def _append_perclos(self, timestamp: float, closed: bool) -> None:
        self.perclos.append((timestamp, closed))
        cutoff = timestamp - self.perclos_window_s
        while self.perclos and self.perclos[0][0] < cutoff:
            self.perclos.popleft()

    def _perclos_value(self) -> tuple[float | None, float]:
        if not self.perclos:
            return None, 0.0
        window = self.perclos[-1][0] - self.perclos[0][0]
        if window < self.perclos_min_window_s:
            return None, window
        closed = sum(1 for _, value in self.perclos if value)
        return closed / len(self.perclos), window

    def update(
        self,
        obs: FaceObservation | None,
        timestamp: float,
        *,
        source_frame_width: int,
        source_frame_height: int,
    ) -> dict[str, Any]:
        # Reject duplicate or out-of-order observations.  The service uses
        # capture timestamps; a stale repeat must not advance a closure timer.
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return self._result(
                state=UNCERTAIN,
                eye_state="unknown",
                quality="stale",
                quality_detail="observation timestamp is not newer than the previous one",
                ear_raw=None,
                ear_smoothed=None,
                threshold=None,
                mar_raw=None,
                mar_smoothed=None,
                head_state="unknown",
                obs=None,
                distraction_state="unknown",
                tracking_confidence=0.0,
                source_frame_width=source_frame_width,
                source_frame_height=source_frame_height,
            )
        self.last_timestamp = timestamp

        if obs is None:
            self.reset_tracking()
            self.last_state = NO_FACE
            return self._result(
                state=NO_FACE,
                eye_state="unknown",
                quality="no_face",
                quality_detail="face not detected",
                ear_raw=None,
                ear_smoothed=None,
                threshold=None,
                mar_raw=None,
                mar_smoothed=None,
                head_state="unknown",
                obs=None,
                distraction_state="unknown",
                tracking_confidence=None,
                source_frame_width=source_frame_width,
                source_frame_height=source_frame_height,
            )

        if obs.quality != "good":
            self.reset_tracking()
            self.last_state = UNCERTAIN
            return self._result(
                state=UNCERTAIN,
                eye_state="unknown",
                quality=obs.quality,
                quality_detail=obs.quality_detail,
                ear_raw=obs.ear,
                ear_smoothed=None,
                threshold=None,
                mar_raw=obs.mar,
                mar_smoothed=None,
                head_state=_head_state(obs.yaw, obs.pitch),
                obs=obs,
                distraction_state="unknown",
                tracking_confidence=0.0,
                source_frame_width=source_frame_width,
                source_frame_height=source_frame_height,
            )

        self.presence_streak += 1
        confidence, confidence_basis = _tracking_confidence(
            obs,
            source_frame_width=source_frame_width,
            source_frame_height=source_frame_height,
            presence_streak=self.presence_streak,
        )

        threshold = self._threshold(obs.pitch)
        raw_ear = float(obs.ear)
        if self.eye_state == "closed" and raw_ear >= threshold:
            # Reopening is never delayed by the closing median.
            self.ear_history.clear()
        self.ear_history.append(raw_ear)
        ear_smoothed = float(statistics.median(self.ear_history))
        if raw_ear <= threshold:
            if self.closure_candidate is None:
                self.closure_candidate = timestamp
            if self.raw_closure_start is None:
                self.raw_closure_start = timestamp
        else:
            # Count a one- or two-frame raw blink that the closing median never
            # promoted to a smoothed closure.  Longer closures are counted when
            # the smoothed state reopens, so they are not double-counted.
            if self.raw_closure_start is not None:
                raw_closed_ms = max(0.0, (timestamp - self.raw_closure_start) * 1000.0)
                if self.closed_since is None and raw_closed_ms < self.blink_max_ms:
                    self.blinks += 1
                self.raw_closure_start = None
            # Clear the start time as soon as the raw eye is open.  Keeping it
            # through a marginal open frame can later report a long closure
            # that never happened.
            self.closure_candidate = None

        self.mar_history.append(float(obs.mar))
        mar_smoothed = float(statistics.median(self.mar_history))

        if self.eye_state == "closed" and raw_ear >= threshold:
            eye_state = "open"
        else:
            eye_state = "closed" if ear_smoothed <= threshold else "open"
        self.eye_state = eye_state

        if eye_state == "closed":
            if self.closed_since is None:
                self.closed_since = self.closure_candidate or timestamp
                self.prolonged_emitted = False
                self.microsleep_emitted = False
            self.closure_ms = max(0.0, (timestamp - self.closed_since) * 1000.0)
            self._append_perclos(timestamp, True)
            if self.closure_ms >= self.microsleep_ms:
                if not self.microsleep_emitted:
                    self.microsleeps += 1
                    self.microsleep_emitted = True
                    self.prolonged_emitted = True
                state = MICROSLEEP
            elif self.closure_ms >= self.prolonged_ms:
                self.prolonged_emitted = True
                state = PROLONGED_CLOSURE
            else:
                state = EYES_CLOSED
        else:
            if self.closed_since is not None:
                closed_ms = max(0.0, (timestamp - self.closed_since) * 1000.0)
                if closed_ms < self.blink_max_ms:
                    self.blinks += 1
                self.closed_since = None
                self.closure_ms = 0.0
                self.prolonged_emitted = False
                self.microsleep_emitted = False
            self._adapt_baseline(ear_smoothed, obs)
            self._append_perclos(timestamp, False)
            state = ALERT

        # Yawning is evaluated independently of the eye state.  A lower
        # release threshold gives a single, stable episode instead of a state
        # that flickers around the exact MAR boundary.
        if mar_smoothed >= self.yawn_mar:
            if self.yawn_since is None:
                self.yawn_since = timestamp
                self.yawn_emitted = False
            self.yawn_ms = max(0.0, (timestamp - self.yawn_since) * 1000.0)
            if self.yawn_ms >= self.yawn_min_ms:
                self.yawn_state = "yawning"
                if not self.yawn_emitted:
                    self.yawns += 1
                    self.yawn_emitted = True
            else:
                self.yawn_state = "mouth_open"
        elif self.yawn_state == "yawning" and mar_smoothed >= self.yawn_release_mar:
            self.yawn_ms = max(0.0, (timestamp - (self.yawn_since or timestamp)) * 1000.0)
            self.yawn_state = "yawning"
        else:
            self.yawn_since = None
            self.yawn_ms = 0.0
            self.yawn_emitted = False
            self.yawn_state = "clear"

        if state == ALERT and self.yawn_state == "yawning":
            state = YAWNING

        head_state = _head_state(obs.yaw, obs.pitch)
        if head_state == "unknown":
            self.away_since = None
            self.away_ms = 0.0
            distraction_state = "unknown"
        elif head_state == "forward":
            self.away_since = None
            self.away_ms = 0.0
            distraction_state = "forward"
        else:
            if self.away_since is None:
                self.away_since = timestamp
            self.away_ms = max(0.0, (timestamp - self.away_since) * 1000.0)
            distraction_state = "distracted" if self.away_ms >= self.distracted_ms else "glance"

        self.last_state = state
        return self._result(
            state=state,
            eye_state=eye_state,
            quality=obs.quality,
            quality_detail=obs.quality_detail,
            ear_raw=obs.ear,
            ear_smoothed=ear_smoothed,
            threshold=threshold,
            mar_raw=obs.mar,
            mar_smoothed=mar_smoothed,
            head_state=head_state,
            obs=obs,
            distraction_state=distraction_state,
            tracking_confidence=confidence,
            tracking_confidence_basis=confidence_basis,
            source_frame_width=source_frame_width,
            source_frame_height=source_frame_height,
        )

    def _result(
        self,
        *,
        state: str,
        eye_state: str,
        quality: str,
        quality_detail: str,
        ear_raw: float | None,
        ear_smoothed: float | None,
        threshold: float | None,
        mar_raw: float | None,
        mar_smoothed: float | None,
        head_state: str,
        obs: FaceObservation | None,
        distraction_state: str,
        tracking_confidence: float | None,
        source_frame_width: int,
        source_frame_height: int,
        tracking_confidence_basis: str = "",
    ) -> dict[str, Any]:
        perclos: float | None = None
        window = 0.0
        if quality == "good" and eye_state in {"open", "closed"}:
            perclos, window = self._perclos_value()
        drowsiness_state = "unknown"
        if quality == "good" and eye_state in {"open", "closed"}:
            if state in {PROLONGED_CLOSURE, MICROSLEEP}:
                drowsiness_state = "drowsy"
            elif eye_state == "closed":
                drowsiness_state = "eyes_closed"
            else:
                drowsiness_state = "alert"
        attention_state = "unknown"
        if distraction_state in {"forward", "glance", "distracted"}:
            attention_state = distraction_state
        face_bbox = None if obs is None else [round(v, 1) for v in obs.bbox]
        mouth_center = None if obs is None else [round(v, 1) for v in obs.mouth_center]
        return {
            "state": state,
            "eye_state": eye_state,
            "quality": quality,
            "quality_detail": quality_detail,
            "ear_raw": ear_raw,
            "ear_smoothed": ear_smoothed,
            "threshold": threshold,
            "closure_ms": self.closure_ms,
            "mar_raw": mar_raw,
            "mar_smoothed": mar_smoothed,
            "mouth_state": self.yawn_state,
            "mouth_open_ms": self.yawn_ms,
            "yawns": self.yawns,
            "blinks": self.blinks,
            "microsleeps": self.microsleeps,
            "head_state": head_state,
            "yaw": None if obs is None else obs.yaw,
            "pitch": None if obs is None else obs.pitch,
            "roll": None if obs is None else obs.roll,
            "away_ms": self.away_ms,
            "distraction_state": distraction_state,
            "attention_state": attention_state,
            "drowsiness_state": drowsiness_state,
            "perclos": perclos,
            "perclos_window_s": window,
            "face_width": None if obs is None else obs.face_width,
            "face_bbox": face_bbox,
            "mouth_center": mouth_center,
            "source_frame_width": source_frame_width,
            "source_frame_height": source_frame_height,
            "tracking_confidence": tracking_confidence,
            "tracking_confidence_basis": tracking_confidence_basis,
            "blink_max_ms": self.blink_max_ms,
            "prolonged_ms": self.prolonged_ms,
            "microsleep_ms": self.microsleep_ms,
            "distracted_ms": self.distracted_ms,
            "yawn_mar": self.yawn_mar,
            "yawn_min_ms": self.yawn_min_ms,
        }


def driver_state(measurements: dict[str, Any]) -> tuple[str, str]:
    """Return the single large badge state and its reason.

    This is a display priority only.  Drowsiness, attention, yawning and the
    optional behaviours remain independent in the status payload.
    """
    good_tracking = (
        measurements["quality"] == "good"
        and measurements["eye_state"] in {"open", "closed"}
        and measurements["tracking_confidence"] is not None
        and float(measurements["tracking_confidence"]) > 0.0
    )
    if measurements["state"] in {NO_FACE, UNCERTAIN} or not good_tracking:
        return "TRACKING UNCERTAIN", measurements["quality_detail"]
    if measurements["state"] in {PROLONGED_CLOSURE, MICROSLEEP}:
        return (
            "DROWSY",
            f"eyes closed {measurements['closure_ms']:.0f} ms; "
            f"smoothed EAR {_safe_float(measurements['ear_smoothed'], 3)}",
        )
    if measurements["attention_state"] == "unknown" or measurements["head_state"] == "unknown":
        return "TRACKING UNCERTAIN", "head pose is unavailable; attentive state is not claimed"
    if measurements["distraction_state"] == "distracted":
        return (
            "DISTRACTED",
            f"head {measurements['head_state']} for {measurements['away_ms']:.0f} ms",
        )
    if measurements["state"] == YAWNING:
        return (
            "YAWNING",
            f"mouth open {measurements['mouth_open_ms']:.0f} ms; "
            f"smoothed MAR {_safe_float(measurements['mar_smoothed'], 2)}",
        )
    if measurements["eye_state"] == "closed":
        if measurements["closure_ms"] < measurements["blink_max_ms"]:
            return (
                "BLINK",
                f"brief blink; closure {measurements['closure_ms']:.0f} ms",
            )
        return (
            "EYES CLOSED",
            f"closure {measurements['closure_ms']:.0f} ms; sustained alert not reached",
        )
    if measurements["distraction_state"] == "glance":
        return (
            "GLANCE",
            f"head {measurements['head_state']} for {measurements['away_ms']:.0f} ms",
        )
    return "ATTENTIVE", "face tracked; eyes open; head forward"
