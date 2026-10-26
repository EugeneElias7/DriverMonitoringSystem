"""Head-pose estimation from MediaPipe landmarks and a distraction rule engine."""

from __future__ import annotations

from typing import Any

from ..core.base import BaseFeature
from ..core.events import EventManager
from ..core.types import FaceObservation, FeatureResult, PoseSample


class HeadPoseDetector(BaseFeature):
    feature = "head_pose"

    def __init__(self, config: Any):
        super().__init__("head_pose", config)
        self._ema: tuple[float, float, float] | None = None
        self._raw: tuple[float, float, float] | None = None
        self._last_face_t: float | None = None

    def load(self) -> None:
        self._ema = None
        self._raw = None
        self._last_face_t = None
        self._mark_ready("MediaPipe transform-matrix Euler angles; solvePnP fallback")

    def reset(self) -> None:
        self._ema = None
        self._raw = None
        self._last_face_t = None

    def _update(self, face: FaceObservation | None, timestamp: float) -> FeatureResult:
        timestamp = float(timestamp)
        if face is None or face.pitch is None or face.yaw is None or face.roll is None:
            self._ema = None
            self._raw = None
            sample = PoseSample(timestamp=timestamp, valid=False, state="unknown", source="none")
            return FeatureResult(
                feature=self.feature,
                state="unknown",
                timestamp=timestamp,
                valid=False,
                measurements={"pose": sample.to_dict(), "face_visible": face is not None},
                message="head pose unavailable",
            )
        self._last_face_t = timestamp
        raw = (float(face.pitch), float(face.yaw), float(face.roll))
        alpha = float(self.config.get("head_pose.smoothing_alpha", 0.45))
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = tuple(
                alpha * new + (1.0 - alpha) * old for new, old in zip(raw, self._ema)
            )
        self._raw = raw
        pitch, yaw, roll = self._ema
        state = self._classify(pitch, yaw)
        sample = PoseSample(
            timestamp=timestamp,
            valid=True,
            state=state,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            source=face.pose_source,
        )
        yaw_threshold = float(self.config.get("head_pose.yaw_threshold_deg", 15.0))
        pitch_threshold = float(self.config.get("head_pose.pitch_threshold_deg", 15.0))
        return FeatureResult(
            feature=self.feature,
            state=state,
            timestamp=timestamp,
            valid=True,
            measurements={
                "pose": sample.to_dict(),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "raw_pitch": round(raw[0], 2),
                "raw_yaw": round(raw[1], 2),
                "raw_roll": round(raw[2], 2),
                "yaw_threshold": yaw_threshold,
                "pitch_threshold": pitch_threshold,
                "pose_source": face.pose_source,
                "face_visible": True,
            },
        )

    def _classify(self, pitch: float, yaw: float) -> str:
        yaw_threshold = float(self.config.get("head_pose.yaw_threshold_deg", 15.0))
        pitch_threshold = float(self.config.get("head_pose.pitch_threshold_deg", 15.0))
        vertical = ""
        if pitch > pitch_threshold:
            vertical = "down"
        elif pitch < -pitch_threshold:
            vertical = "up"
        horizontal = ""
        if abs(yaw) > yaw_threshold:
            positive = str(self.config.get("head_pose.yaw_positive_label", "left")).lower()
            negative = "right" if positive == "left" else "left"
            horizontal = positive if yaw > 0 else negative
        if vertical and horizontal:
            return f"{vertical}_{horizontal}"
        if vertical:
            return vertical
        if horizontal:
            return horizontal
        return "forward"


class DistractionDetector(BaseFeature):
    feature = "distraction"

    def __init__(self, config: Any):
        super().__init__("distraction", config)
        self.events = EventManager(float(config.get("distraction.event_cooldown_ms", 3000)))
        self._reset_state()

    def _reset_state(self) -> None:
        self._away_since: float | None = None
        self._away_reason: str | None = None
        self._distracted_emitted = False
        self._last_face_t: float | None = None
        self._state = "unknown"
        self._max_away_ms = 0.0

    def load(self) -> None:
        self._reset_state()
        self._mark_ready("rule-based head-pose distraction heuristic (not a classifier)")

    def reset(self) -> None:
        self.events.reset()
        self._reset_state()

    def _update(self, pose_result: FeatureResult | None, timestamp: float) -> FeatureResult:
        timestamp = float(timestamp)
        if pose_result is None or not pose_result.valid:
            lost_ms = 0.0
            if self._last_face_t is not None:
                lost_ms = (timestamp - self._last_face_t) * 1000.0
            if lost_ms >= float(self.config.get("distraction.face_loss_reset_ms", 500)):
                self._away_since = None
                self._away_reason = None
                self._distracted_emitted = False
            self._state = "unknown"
            return FeatureResult(
                feature=self.feature,
                state="unknown",
                timestamp=timestamp,
                valid=False,
                measurements={
                    "away": False,
                    "face_visible": False,
                    "lost_ms": round(lost_ms, 1),
                    "note": "head pose unavailable; no distraction decision",
                },
                events=self.events.take(),
            )
        self._last_face_t = timestamp
        measurements = pose_result.measurements
        yaw = measurements.get("yaw")
        pitch = measurements.get("pitch")
        valid_pose = yaw is not None and pitch is not None
        if not valid_pose:
            return FeatureResult(
                feature=self.feature,
                state="unknown",
                timestamp=timestamp,
                valid=False,
                measurements={"away": False, "note": "pose values missing"},
                events=self.events.take(),
            )

        yaw = float(yaw)
        pitch = float(pitch)
        yaw_threshold = float(self.config.get("distraction.yaw_threshold_deg", 25.0))
        down_threshold = float(self.config.get("distraction.pitch_down_threshold_deg", 30.0))
        up_threshold = float(self.config.get("distraction.pitch_up_threshold_deg", 30.0))
        reason = None
        lateral = False
        if abs(yaw) > yaw_threshold:
            positive = str(self.config.get("head_pose.yaw_positive_label", "left")).lower()
            direction = positive if yaw > 0 else ("right" if positive == "left" else "left")
            reason = f"lateral_{direction}"
            lateral = True
        if pitch > down_threshold:
            reason = "downward"
        elif pitch < -up_threshold:
            reason = "upward"

        if reason is None:
            recovered = self._state == "distracted"
            self._away_since = None
            self._away_reason = None
            self._distracted_emitted = False
            self._state = "attentive"
            event = None
            if recovered:
                event = self.events.emit(
                    timestamp=timestamp,
                    feature=self.feature,
                    kind="attention_recovered",
                    severity="info",
                    message="Head returned to the forward region after a sustained deviation",
                )
            return FeatureResult(
                feature=self.feature,
                state="attentive",
                timestamp=timestamp,
                valid=True,
                measurements={
                    "away": False,
                    "away_ms": 0.0,
                    "max_away_ms": round(self._max_away_ms, 1),
                    "reason": None,
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "rule_score": 0.0,
                },
                events=self.events.take(),
            )

        if self._away_since is None or self._away_reason != reason:
            self._away_since = timestamp
            self._away_reason = reason
            self._distracted_emitted = False
        away_ms = max(0.0, (timestamp - self._away_since) * 1000.0)
        self._max_away_ms = max(self._max_away_ms, away_ms)
        glance_ms = float(self.config.get("distraction.glance_max_ms", 700))
        mirror_ms = float(self.config.get("distraction.mirror_glance_max_ms", 1200))
        hold_ms = float(self.config.get("distraction.distraction_hold_ms", 1500))
        if away_ms < glance_ms:
            state = "glance"
        elif lateral and away_ms < mirror_ms:
            state = "possible_mirror_check"
        elif away_ms < hold_ms:
            state = "extended_glance"
        else:
            state = "distracted"
            if not self._distracted_emitted:
                self._distracted_emitted = True
                self.events.emit(
                    timestamp=timestamp,
                    feature=self.feature,
                    kind="distracted",
                    severity="warning",
                    message=(
                        f"Sustained {reason.replace('_', ' ')} for {away_ms:.0f} ms "
                        "(rule-based head-pose heuristic; not a validated distraction classifier)"
                    ),
                    data={
                        "away_ms": round(away_ms, 1),
                        "reason": reason,
                        "yaw": round(yaw, 2),
                        "pitch": round(pitch, 2),
                    },
                )
        self._state = state
        rule_score = min(1.0, away_ms / hold_ms) if hold_ms > 0 else 0.0
        return FeatureResult(
            feature=self.feature,
            state=state,
            timestamp=timestamp,
            valid=True,
            measurements={
                "away": True,
                "away_ms": round(away_ms, 1),
                "max_away_ms": round(self._max_away_ms, 1),
                "reason": reason,
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "glance_max_ms": glance_ms,
                "mirror_glance_max_ms": mirror_ms,
                "distraction_hold_ms": hold_ms,
                "rule_score": round(rule_score, 4),
            },
            events=self.events.take(),
            message="Possible mirror check is tolerated for a bounded lateral glance; it is not a mirror classifier.",
        )
