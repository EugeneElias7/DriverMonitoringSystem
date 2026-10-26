"""Eye/mouth landmark drowsiness state machine.

This is a rule-based research module, not a medical or safety-certified fatigue
detector. It distinguishes short blinks from sustained closure and reports the
measurements behind every state.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..core.base import BaseFeature
from ..core.events import EventManager
from ..core.types import FaceObservation, FeatureResult


class DrowsinessDetector(BaseFeature):
    feature = "drowsiness"

    def __init__(self, config: Any):
        super().__init__("drowsiness", config)
        self.events = EventManager(float(config.get("drowsiness.event_cooldown_ms", 3000)))
        self._reset_state()

    def _reset_state(self) -> None:
        self.baseline = float(self.config.get("drowsiness.ear_baseline_initial", 0.26))
        self._calibration_samples = 0
        self._samples = 0
        self._perclos: deque[tuple[float, bool]] = deque()
        self._eye_state = "unknown"
        self._closed_since: float | None = None
        self._prolonged_emitted = False
        self._microsleep_emitted = False
        self._microsleeps = 0
        self._blinks = 0
        self._yawns = 0
        self._yawn_since: float | None = None
        self._yawn_emitted = False
        self._last_face_t: float | None = None
        self._last_face_lost_event = False
        self._recent_microsleep_until: float | None = None
        self._state = "unknown"
        self._last_measurements: dict[str, Any] = {}

    def load(self) -> None:
        self._reset_state()
        self._mark_ready("EAR/MAR landmark state machine (no extra model)")

    def reset(self) -> None:
        self.events.reset()
        self._reset_state()

    def _update(self, face: FaceObservation | None, timestamp: float) -> FeatureResult:
        timestamp = float(timestamp)
        if face is None:
            return self._handle_missing_face(timestamp)

        if self._last_face_t is not None and timestamp - self._last_face_t > float(
            self.config.get("drowsiness.face_loss_grace_ms", 600)
        ) / 1000.0:
            # Do not bridge a long tracking gap into a closure duration.
            self._closed_since = None
            self._prolonged_emitted = False
            self._microsleep_emitted = False
            self._eye_state = "unknown"
        self._last_face_t = timestamp
        self._last_face_lost_event = False

        ear = face.ear_max if face.ear_max is not None else face.ear
        yaw = face.yaw
        pitch = face.pitch
        quality = "good"
        if ear is None or not (0.01 <= float(ear) <= 0.9):
            quality = "invalid_ear"
        if yaw is not None and abs(yaw) > float(self.config.get("drowsiness.max_yaw_deg", 55.0)):
            quality = "low_yaw"
        if pitch is not None and abs(pitch) > float(self.config.get("drowsiness.max_pitch_deg", 40.0)):
            quality = "low_pitch"

        if quality != "good":
            self._eye_state = "unknown"
            state = "unknown"
            self._state = state
            measurements = self._measurements(face, ear, quality)
            self._last_measurements = measurements
            return FeatureResult(
                feature=self.feature,
                state=state,
                timestamp=timestamp,
                valid=True,
                measurements=measurements,
                events=self.events.take(),
                message=f"eye geometry quality: {quality}",
            )

        threshold = self._closed_threshold(pitch)
        open_threshold = self.baseline * float(self.config.get("drowsiness.open_ratio", 0.82))
        eye_state = self._classify_eye(float(ear), threshold, open_threshold)
        self._eye_state = eye_state
        closure_ms = 0.0

        if eye_state == "closed":
            if self._closed_since is None:
                self._closed_since = timestamp
                self._prolonged_emitted = False
                self._microsleep_emitted = False
            closure_ms = max(0.0, (timestamp - self._closed_since) * 1000.0)
            if closure_ms >= float(self.config.get("drowsiness.microsleep_ms", 1500)):
                state = "microsleep"
                if not self._microsleep_emitted:
                    self._microsleeps += 1
                    self._microsleep_emitted = True
                    self._recent_microsleep_until = timestamp + float(
                        self.config.get("drowsiness.recent_microsleep_ms", 30000)
                    ) / 1000.0
                    event = self.events.emit(
                        timestamp=timestamp,
                        feature=self.feature,
                        kind="microsleep",
                        severity="warning",
                        message=f"Eyes closed for {closure_ms:.0f} ms; possible microsleep (rule-based, not safety-certified)",
                        data={"closure_ms": round(closure_ms, 1), "baseline_ear": round(self.baseline, 4)},
                    )
                    if event:
                        events.append(event)
            elif closure_ms >= float(self.config.get("drowsiness.prolonged_ms", 900)):
                state = "prolonged_closure"
                if not self._prolonged_emitted:
                    self._prolonged_emitted = True
                    event = self.events.emit(
                        timestamp=timestamp,
                        feature=self.feature,
                        kind="prolonged_closure",
                        severity="warning",
                        message=f"Prolonged eye closure ({closure_ms:.0f} ms)",
                        data={"closure_ms": round(closure_ms, 1)},
                    )
                    if event:
                        events.append(event)
            else:
                state = "eyes_closed"
            self._append_perclos(timestamp, True)
        else:
            if self._closed_since is not None:
                closed_ms = max(0.0, (timestamp - self._closed_since) * 1000.0)
                if closed_ms < float(self.config.get("drowsiness.blink_max_ms", 500)):
                    self._blinks += 1
                    state = "blink"
                else:
                    state = "alert"
                if closed_ms >= float(self.config.get("drowsiness.prolonged_ms", 900)) and not self._microsleep_emitted:
                    event = self.events.emit(
                        timestamp=timestamp,
                        feature=self.feature,
                        kind="closure_ended",
                        severity="info",
                        message=f"Eye closure ended after {closed_ms:.0f} ms",
                        data={"closure_ms": round(closed_ms, 1)},
                    )
                    if event:
                        events.append(event)
                self._closed_since = None
                self._prolonged_emitted = False
                self._microsleep_emitted = False
            else:
                state = "alert"

            if self._eye_state == "open":
                self._adapt_baseline(float(ear), face)
            self._append_perclos(timestamp, False)

        # Yawning is a secondary sign and never overrides a microsleep.
        yawning = self._update_yawn(face, timestamp, state, events)
        if yawning and state not in {"microsleep", "prolonged_closure", "eyes_closed"}:
            state = "yawning"

        perclos = self._compute_perclos(timestamp)
        if (
            state in {"alert", "blink", "yawning"}
            and perclos is not None
            and perclos >= float(self.config.get("drowsiness.perclos_drowsy", 0.30))
        ):
            state = "drowsy"
            event = self.events.emit(
                timestamp=timestamp,
                feature=self.feature,
                kind="perclos_high",
                severity="warning",
                message=f"PERCLOS {perclos:.2f} exceeds configured threshold (sample-weighted)",
                data={"perclos": round(perclos, 4)},
            )
            if event:
                events.append(event)
        elif (
            state in {"alert", "blink", "yawning"}
            and self._recent_microsleep_until is not None
            and timestamp < self._recent_microsleep_until
        ):
            state = "drowsy"

        measurements = self._measurements(face, ear, quality)
        measurements.update(
            {
                "baseline_ear": round(self.baseline, 4),
                "closed_threshold": round(threshold, 4),
                "open_threshold": round(open_threshold, 4),
                "perclos": None if perclos is None else round(perclos, 4),
                "closure_ms": round(closure_ms, 1),
                "eye_state": eye_state,
                "blinks": self._blinks,
                "microsleeps": self._microsleeps,
                "yawns": self._yawns,
            }
        )
        rule_score = 0.0
        if closure_ms > 0:
            rule_score = min(1.0, closure_ms / float(self.config.get("drowsiness.microsleep_ms", 1500)))
        if perclos is not None:
            rule_score = max(rule_score, min(1.0, perclos / float(self.config.get("drowsiness.perclos_drowsy", 0.30))))
        measurements["drowsiness_rule_score"] = round(rule_score, 4)
        self._state = state
        self._last_measurements = measurements
        return FeatureResult(
            feature=self.feature,
            state=state,
            timestamp=timestamp,
            valid=True,
            measurements=measurements,
            events=self.events.take(),
            message="" if quality == "good" else f"eye geometry quality: {quality}",
        )

    def _handle_missing_face(self, timestamp: float) -> FeatureResult:
        if self._last_face_t is None:
            self._state = "no_face"
            return FeatureResult(
                feature=self.feature,
                state="no_face",
                timestamp=timestamp,
                valid=False,
                measurements={"face_visible": False},
                events=self.events.take(),
            )
        lost_ms = max(0.0, (timestamp - self._last_face_t) * 1000.0)
        grace = float(self.config.get("drowsiness.face_loss_grace_ms", 600))
        if lost_ms < grace:
            state = "face_lost"
        else:
            state = "no_face"
            # A disappearance must not be turned into a closure duration.
            self._closed_since = None
            self._eye_state = "unknown"
        if lost_ms >= float(self.config.get("drowsiness.face_lost_event_ms", 1500)) and not self._last_face_lost_event:
            self._last_face_lost_event = True
            self.events.emit(
                timestamp=timestamp,
                feature=self.feature,
                kind="face_lost",
                severity="warning",
                message=f"No face detected for {lost_ms:.0f} ms; drowsiness state unavailable",
                data={"lost_ms": round(lost_ms, 1)},
            )
            self._perclos.clear()
        self._state = state
        return FeatureResult(
            feature=self.feature,
            state=state,
            timestamp=timestamp,
            valid=False,
            measurements={"face_visible": False, "lost_ms": round(lost_ms, 1)},
            events=self.events.take(),
            message="face missing or tracker lost",
        )

    def _closed_threshold(self, pitch: float | None) -> float:
        ratio = float(self.config.get("drowsiness.closed_ratio", 0.70))
        if pitch is not None and pitch > float(self.config.get("drowsiness.pitch_guard_deg", 25.0)):
            ratio = float(self.config.get("drowsiness.guarded_closed_ratio", 0.55))
        return self.baseline * ratio

    def _classify_eye(self, ear: float, closed_threshold: float, open_threshold: float) -> str:
        if ear <= closed_threshold:
            return "closed"
        if ear >= open_threshold:
            return "open"
        # Hysteresis band: keep the previous decision if there is one.
        if self._eye_state in {"open", "closed"}:
            return self._eye_state
        return "unknown"

    def _adapt_baseline(self, ear: float, face: FaceObservation) -> None:
        yaw = face.yaw
        pitch = face.pitch
        if yaw is not None and abs(yaw) > 30.0:
            return
        if pitch is not None and abs(pitch) > 20.0:
            return
        calibration = int(self.config.get("drowsiness.calibration_samples", 30))
        alpha = float(
            self.config.get("drowsiness.calibration_alpha", 0.05)
            if self._samples < calibration
            else self.config.get("drowsiness.ear_adapt_alpha", 0.01)
        )
        self.baseline = (1.0 - alpha) * self.baseline + alpha * ear
        self.baseline = min(
            float(self.config.get("drowsiness.ear_baseline_max", 0.42)),
            max(float(self.config.get("drowsiness.ear_baseline_min", 0.18)), self.baseline),
        )
        self._samples += 1

    def _append_perclos(self, timestamp: float, closed: bool) -> None:
        self._perclos.append((timestamp, bool(closed)))
        window_s = float(self.config.get("drowsiness.perclos_window_ms", 60000)) / 1000.0
        cutoff = timestamp - window_s
        while self._perclos and self._perclos[0][0] < cutoff:
            self._perclos.popleft()

    def _compute_perclos(self, timestamp: float) -> float | None:
        if not self._perclos:
            return None
        window_ms = float(self.config.get("drowsiness.perclos_window_ms", 60000))
        min_ms = float(self.config.get("drowsiness.perclos_min_window_ms", 10000))
        span_ms = (timestamp - self._perclos[0][0]) * 1000.0
        if span_ms < min_ms and len(self._perclos) < 3:
            return None
        # If the clip is shorter than the minimum window, still report the
        # sample fraction once there are enough samples; label it provisional.
        if span_ms < min_ms:
            return None
        closed = sum(1 for _, is_closed in self._perclos if is_closed)
        return closed / len(self._perclos)

    def _update_yawn(
        self,
        face: FaceObservation,
        timestamp: float,
        state: str,
        events: list[Any],
    ) -> bool:
        mar = face.mar
        threshold = float(self.config.get("drowsiness.yawn_mar", 0.55))
        if mar is not None and mar >= threshold:
            if self._yawn_since is None:
                self._yawn_since = timestamp
                self._yawn_emitted = False
            duration = max(0.0, (timestamp - self._yawn_since) * 1000.0)
            if duration >= float(self.config.get("drowsiness.yawn_min_ms", 400)):
                if not self._yawn_emitted:
                    self._yawns += 1
                    self._yawn_emitted = True
                    event = self.events.emit(
                        timestamp=timestamp,
                        feature=self.feature,
                        kind="yawning",
                        severity="info",
                        message=f"Yawn-like mouth opening (MAR {mar:.2f}) for {duration:.0f} ms",
                        data={"mar": round(float(mar), 4), "duration_ms": round(duration, 1)},
                    )
                    if event:
                        events.append(event)
                return state not in {"microsleep", "prolonged_closure", "eyes_closed"}
        else:
            self._yawn_since = None
            self._yawn_emitted = False
        return False

    def _measurements(self, face: FaceObservation, ear: float | None, quality: str) -> dict[str, Any]:
        return {
            "face_visible": True,
            "ear": None if face.ear is None else round(float(face.ear), 4),
            "ear_max": None if ear is None else round(float(ear), 4),
            "ear_left": None if face.ear_left is None else round(float(face.ear_left), 4),
            "ear_right": None if face.ear_right is None else round(float(face.ear_right), 4),
            "mar": None if face.mar is None else round(float(face.mar), 4),
            "pitch": None if face.pitch is None else round(float(face.pitch), 2),
            "yaw": None if face.yaw is None else round(float(face.yaw), 2),
            "quality": quality,
            "face_bbox": [round(float(v), 2) for v in face.bbox],
            "mouth_center": None
            if face.mouth_center is None
            else [round(float(v), 2) for v in face.mouth_center],
        }
