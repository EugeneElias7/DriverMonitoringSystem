"""Behaviour state machines: drowsiness, smoking, distraction.

Each monitor consumes one frame's observation and returns its full set of
`ConditionSignal`s every frame. Deciding when a signal is worth alerting on is
the `AlertManager`'s job, not the monitor's.
"""

from __future__ import annotations

import math
import statistics

from .alerts import ConditionSignal, Severity
from .config import Thresholds
from .face import FaceObservation
from .objects import PHONE_LABELS, Detection
from .temporal import Sustained, TimeWindow


class DrowsinessMonitor:
    """PERCLOS, microsleep, yawning and driver-absent detection.

    The spec's "30-frame window" becomes a time window: at 15 fps, 30 frames is
    2 s, and a time window degrades gracefully when the Pi drops frames.
    """

    def __init__(self, thresholds: Thresholds):
        self.th = thresholds
        self._perclos = TimeWindow(thresholds.perclos_window_s)
        self._microsleep = Sustained(thresholds.microsleep_s)
        self._yawn = Sustained(thresholds.yawn_s)
        self._absent = Sustained(thresholds.face_missing_s)
        self._closed_prev = False
        self._closed_run = 0
        self._blink_start: float | None = None
        self._nod_start: float | None = None
        self._neutral_pitch: float | None = None
        self._pitch_samples: list[float] = []
        self.blink_durations = TimeWindow(300.0, maxlen=128)

    def update(self, now: float, face: FaceObservation) -> list[ConditionSignal]:
        th = self.th
        present = face.present and math.isfinite(face.ear)
        # Absence is about the face detector, not a bad landmark pass.
        absent = self._absent.update(now, not face.present)

        # Learn neutral pitch from usable frames.
        reliable = present and self._plausible(face)
        if reliable and math.isfinite(face.pitch):
            if len(self._pitch_samples) < 30:
                self._pitch_samples.append(face.pitch)
            elif self._neutral_pitch is None:
                self._neutral_pitch = sum(self._pitch_samples) / len(self._pitch_samples)

        if not reliable:
            # A face the landmark model lost (hand over the mouth, profile
            # turn) is not evidence of sleep. Skip the frame instead of
            # feeding a bogus EAR into the microsleep and PERCLOS timers.
            self._microsleep.update(now, False)
            self._yawn.update(now, False)
            self._closed_prev = False
            self._closed_run = 0
            self._blink_start = None
            self._perclos.prune(now)
            if not present:
                return self._signals(now, absent, 0.0, False, False, False)
            return self._signals(now, False, 0.0, False, False, False)

        # Use the averaged EAR, which is already robust against a single bad
        # eyelid: to get the average below the closed bar, both eyes must be
        # at least partly shut. Requiring both per-eye values below the same
        # bar breaks at a steep camera angle (this Pi sees the driver at
        # ~50 deg yaw), where the far eye's landmarks read high even when the
        # eyelid is shut. The per-eye values remain in the status line for
        # diagnosis.
        closed = face.ear < th.ear_closed
        self._closed_run = self._closed_run + 1 if closed else 0
        self._perclos.push(now, face.ear)
        self._track_blink(now, closed)

        # Nodding detection: head pitch drops significantly below neutral.
        nodding = False
        if self._neutral_pitch is not None and math.isfinite(face.pitch):
            pitch_drop = self._neutral_pitch - face.pitch
            if pitch_drop > 15.0:
                if self._nod_start is None:
                    self._nod_start = now
                if now - self._nod_start >= 1.5:
                    nodding = True
            else:
                self._nod_start = None

        microsleep = self._microsleep.update(now, closed)
        # Report only after a run of closed samples; the Sustained timer still
        # starts on the first closed frame, so the alert is not delayed twice.
        microsleep = microsleep and self._closed_run >= max(
            1, int(th.microsleep_min_frames)
        )
        yawning = self._yawn.update(now, face.mar > th.yawn_mar)
        # Nodding counts as drowsiness (microsleep).
        if nodding:
            microsleep = self._microsleep.update(now, True) or microsleep
        perclos = self._perclos.fraction(lambda v: v < th.ear_closed)
        covered = (
            self._perclos.span() >= 0.6 * th.perclos_window_s
            and len(self._perclos) >= max(1, int(th.perclos_min_samples))
        )
        perclos_active = covered and perclos >= th.perclos_alert and not microsleep
        return self._signals(now, absent, perclos, perclos_active, yawning, microsleep)

    def _plausible(self, face: FaceObservation) -> bool:
        """Reject landmark output that cannot be a normal open/closed face."""
        th = self.th
        if not (math.isfinite(face.ear) and math.isfinite(face.mar)):
            return False
        if face.ear > th.drowsy_valid_ear_max or face.mar > th.drowsy_valid_mar_max:
            return False
        if math.isfinite(face.yaw) and abs(face.yaw) > th.drowsy_max_yaw_deg:
            return False
        if math.isfinite(face.pitch) and abs(face.pitch) > th.drowsy_max_pitch_deg:
            return False
        return True

    def _track_blink(self, now: float, closed: bool) -> None:
        if closed and not self._closed_prev:
            self._blink_start = now
        elif not closed and self._closed_prev and self._blink_start is not None:
            duration = now - self._blink_start
            if 0.03 <= duration <= 1.5:
                self.blink_durations.push(now, duration)
            self._blink_start = None
        self._closed_prev = closed

    def _signals(self, now, absent, perclos, perclos_active, yawning, microsleep):
        return [
            ConditionSignal(
                "driver_absent",
                Severity.MAJOR,
                f"No face detected for {self._absent.elapsed(now):.0f}s",
                absent,
            ),
            ConditionSignal(
                "drowsy_microsleep",
                Severity.CRITICAL,
                f"Eyes closed for {self._microsleep.elapsed(now):.1f}s",
                microsleep,
            ),
            ConditionSignal(
                "drowsy_perclos",
                Severity.MAJOR,
                f"PERCLOS {perclos:.0%} over {self.th.perclos_window_s:.0f}s",
                perclos_active,
            ),
            ConditionSignal(
                "drowsy_yawning",
                Severity.MINOR,
                f"Yawning for {self._yawn.elapsed(now):.1f}s",
                yawning,
            ),
        ]

    @property
    def eyes_closed(self) -> bool:
        """True while the current run of closed-eye frames is unbroken.

        The app uses this as a face-first scheduling signal: while it is true
        the detector workers stop starting new passes, so the next face frame
        (which confirms or clears the microsleep) is produced as soon as
        possible.
        """
        return self._closed_run > 0

    @property
    def blink_median(self) -> float:
        return self.blink_durations.median()

    def blink_rate(self, now: float) -> float:
        """Blinks per minute over the last 60 s, extrapolated early on.

        NaN until at least one blink has been recorded.
        """
        count = self.blink_durations.count_in(now, 60.0)
        if count == 0:
            return math.nan
        span = min(60.0, self.blink_durations.span())
        if span < 1.0:
            return math.nan
        return count * 60.0 / span


class SmokingMonitor:
    """Cigarette only when a small box sits at the mouth.

    False smoking alerts are much worse than a missed cigarette, so the gate
    has three independent parts:

    * geometry: a small box within a tight radius of the mouth point;
    * confidence: a weak box needs a long continuous run, a strong box a
      shorter one, so a single flickering detector pass cannot speak;
    * time: the run spans at least two detector passes on the PC.

    The model republishes a result until the next pass, so the time bar is
    deliberately longer than one detector period. A real cigarette stays at
    the lips for several seconds; one puff still alerts. Once active, the
    alert is held through a brief miss so repeats are not cut short.
    """

    def __init__(self, thresholds: Thresholds, detector_has_cigarette: bool = True):
        self.th = thresholds
        self.detector_has_cigarette = detector_has_cigarette
        self._strong = Sustained(
            thresholds.cigarette_s, hold_s=thresholds.cigarette_hold_s
        )
        self._weak = Sustained(
            thresholds.cigarette_low_s, hold_s=thresholds.cigarette_hold_s
        )

    def candidate(self, detection: Detection, face: FaceObservation) -> bool:
        """A box plausible enough to be drawn in the preview as a cigarette."""
        return (
            detection.label == "cigarette"
            and detection.confidence >= self.th.cigarette_low_conf
            and self._near_face(detection, face)
        )

    def update(
        self, now: float, face: FaceObservation, detections: list[Detection]
    ) -> list[ConditionSignal]:
        if not self.detector_has_cigarette:
            return []
        cigs = [
            d for d in detections if d.label == "cigarette" and self._near_face(d, face)
        ]
        # The two bars are independent: a strong box alerts after
        # `cigarette_s`, a weaker box after `cigarette_low_s`. Both hold
        # through a short miss, and the longer elapsed time is reported.
        strong = self._strong.update(
            now, any(d.confidence >= self.th.cigarette_high_conf for d in cigs)
        )
        weak = self._weak.update(
            now, any(d.confidence >= self.th.cigarette_low_conf for d in cigs)
        )
        elapsed = max(self._strong.elapsed(now), self._weak.elapsed(now))
        return [
            ConditionSignal(
                "smoking",
                Severity.MAJOR,
                f"Cigarette at face for {elapsed:.1f}s",
                strong or weak,
            )
        ]

    @staticmethod
    def _mouth_anchor(face: FaceObservation) -> tuple[float, float, float] | None:
        """Mouth point plus face span; falls back to a box-derived anchor
        when landmark points are unavailable (scripted sources, tests)."""
        if face.bbox is None:
            return None
        x1, y1, x2, y2 = face.bbox
        span = y2 - y1
        mouth = face.mouth_point
        if mouth is None:
            return (x1 + x2) / 2, y2 - 0.25 * span, span
        return mouth[0], mouth[1], span

    def _near_face(self, detection: Detection, face: FaceObservation) -> bool:
        # Anchored to the mouth, not the face box: a hand or phone at the ear
        # sits inside a generous face box but far from the mouth. Labelled
        # positives land ~0.1-0.13 face-heights from the mouth point; the gate
        # is wider so a cigarette held at the lips or just below the chin is
        # not lost when the detector box includes part of the hand.
        if face.bbox is None:
            return False
        span = face.bbox[3] - face.bbox[1]
        if span <= 0:
            return False
        width = detection.box[2] - detection.box[0]
        height = detection.box[3] - detection.box[1]
        if max(width, height) > self.th.cigarette_max_size_span * span:
            return False
        if width <= 0 or height <= 0:
            return False
        aspect = max(width, height) / min(width, height)
        if aspect < self.th.cigarette_min_aspect:
            return False
        cx, cy = detection.center
        mouth = face.mouth_point
        if mouth is None:
            x1, y1, x2, y2 = face.bbox
            return (
                x1 - 0.75 * span <= cx <= x2 + 0.75 * span
                and y1 - 0.5 * span <= cy <= y2 + 0.75 * span
            )
        mx, my = mouth
        gate = self.th.cigarette_gate_span
        # Keep this tight around the lips. A phone at the ear is ~0.45
        # face-heights out, and hands at the chin land around 0.5; the default
        # 0.30 still covers a cigarette at the lips or just below them.
        if abs(cx - mx) > gate * span or abs(cy - my) > gate * span:
            return False
        bx1, by1, bx2, by2 = detection.box
        dx = max(bx1 - mx, 0.0, mx - bx2)
        dy = max(by1 - my, 0.0, my - by2)
        return math.hypot(dx, dy) <= self.th.cigarette_mouth_pad_span * span


class DrinkingMonitor:
    """The driver model's whole-gesture `drinking` class.

    The gesture is encoded in the class, so no mouth gate: a bottle on the
    passenger seat is not `drinking`, the model only fires for the
    hand-to-mouth action.
    """

    def __init__(self, thresholds: Thresholds):
        self.th = thresholds
        self._sustained = Sustained(thresholds.drinking_s)

    def update(
        self, now: float, face: FaceObservation, detections: list[Detection]
    ) -> list[ConditionSignal]:
        evidence = any(d.label == "drinking" for d in detections)
        active = self._sustained.update(now, evidence)
        return [
            ConditionSignal(
                "drinking",
                Severity.MAJOR,
                f"Drinking for {self._sustained.elapsed(now):.1f}s",
                active,
            )
        ]


class EatingMonitor:
    """Whole-gesture `eating` class from the driver-trained model.

    Like drinking, the gesture is encoded in the class itself, so no mouth
    gate: a burger on the passenger seat is not `eating`, the model only
    fires for the hand-to-mouth action.
    """

    def __init__(self, thresholds: Thresholds):
        self.th = thresholds
        self._sustained = Sustained(thresholds.eating_s)

    def update(
        self, now: float, face: FaceObservation, detections: list[Detection]
    ) -> list[ConditionSignal]:
        evidence = any(d.label == "eating" for d in detections)
        active = self._sustained.update(now, evidence)
        return [
            ConditionSignal(
                "eating",
                Severity.MINOR,
                f"Eating for {self._sustained.elapsed(now):.1f}s",
                active,
            )
        ]


class SeatbeltMonitor:
    """Absence rule: a driver in view with no seatbelt detected.

    The driver model detects the belt itself (0.65/0.90 on the demo
    images), so the violation is its sustained absence while a face is
    present. The delay absorbs the seconds it takes to buckle up after
    starting, and detector flicker.
    """

    def __init__(self, thresholds: Thresholds, detector_has_seatbelt: bool = True):
        self.th = thresholds
        self.detector_has_seatbelt = detector_has_seatbelt
        self._missing = Sustained(thresholds.seatbelt_missing_s)

    def update(
        self, now: float, face: FaceObservation, detections: list[Detection]
    ) -> list[ConditionSignal]:
        if not self.detector_has_seatbelt or not self.th.seatbelt_check:
            return []
        belt = any(d.label == "seatbelt" for d in detections)
        # Absence only accumulates while a driver is in view: an empty cabin
        # must not bank seconds that a later face flicker could cash in.
        missing = self._missing.update(now, not belt and face.present)
        active = missing and face.present
        return [
            ConditionSignal(
                "seatbelt_not_worn",
                Severity.MAJOR,
                f"No seatbelt detected for {self._missing.elapsed(now):.1f}s",
                active,
            )
        ]


class DistractionMonitor:
    """Head pose away from the road, escalated by phone use.

    A phone lying on the seat is not a distraction. The alert ladder is:

    * a phone near the driver's face/upper body is "in hand" and alerts after
      `phone_in_hand_s`;
    * a phone at the ear is a call and alerts after `phone_call_s`;
    * either of the above while the head is away is critical;
    * a phone that is merely visible stays silent.

    "In hand" is inferred from the phone box sitting in a generous gate around
    the face and upper chest. That gate rejects phones on the passenger
    seat/dashboard, which is what caused the old "phone visible" false alarms.
    """

    def __init__(self, thresholds: Thresholds):
        self.th = thresholds
        self._away = Sustained(thresholds.looking_away_s)
        # Hold the phone alert through a missed detector pass; it must keep
        # repeating until the phone is actually put down.
        self._call = Sustained(
            thresholds.phone_call_s, hold_s=thresholds.phone_hold_s
        )
        self._in_hand = Sustained(
            thresholds.phone_in_hand_s, hold_s=thresholds.phone_hold_s
        )
        self._neutral_yaw = 0.0
        self._neutral_pitch = 0.0
        self._calib_start: float | None = None
        self._calib_yaw: list[float] = []
        self._calib_pitch: list[float] = []
        self._calibrated = False

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def neutral_pose(self) -> tuple[float, float]:
        return self._neutral_yaw, self._neutral_pitch

    def _calibrate(self, now: float, face: FaceObservation) -> tuple[float, float] | None:
        """The driver's own forward pose, learned over the first seconds.

        The camera is rarely mounted on the driver's axis, and the pose
        estimate carries a constant bias for a given face and mount, so
        absolute yaw/pitch reads as a permanent look-away. The median of the
        first `gaze_warmup_s` seconds with a face becomes the forward
        reference. Returns None (and the caller suppresses look-away) until
        that reference exists.
        """
        if self._calibrated:
            return self._neutral_yaw, self._neutral_pitch
        if not (face.present and math.isfinite(face.yaw) and math.isfinite(face.pitch)):
            return None
        if self._calib_start is None:
            self._calib_start = now
        if len(self._calib_yaw) < 300:
            self._calib_yaw.append(face.yaw)
            self._calib_pitch.append(face.pitch)
        # A driver can start the session outside the frame; keep collecting
        # until the window elapsed and the median has enough samples to mean
        # something.
        if now - self._calib_start >= self.th.gaze_warmup_s and len(self._calib_yaw) >= 10:
            self._neutral_yaw = statistics.median(self._calib_yaw)
            self._neutral_pitch = statistics.median(self._calib_pitch)
            self._calib_yaw = []
            self._calib_pitch = []
            self._calibrated = True
            return self._neutral_yaw, self._neutral_pitch
        return None

    def update(
        self, now: float, face: FaceObservation, detections: list[Detection]
    ) -> list[ConditionSignal]:
        th = self.th
        neutral = self._calibrate(now, face)
        looking_away = False
        if neutral is not None and face.present and math.isfinite(face.yaw):
            yaw_off = abs(face.yaw - neutral[0])
            pitch_off = abs(face.pitch - neutral[1])
            looking_away = yaw_off > th.yaw_away_deg or pitch_off > th.pitch_away_deg
        phones = [
            d
            for d in detections
            if d.label in PHONE_LABELS and d.confidence >= th.phone_low_conf
        ]
        # The ear anchor is a much tighter place than "somewhere near the
        # face", so a weak phone may count there. Everywhere else a phone must
        # clear `phone_hand_conf`: one weak box is republished for a whole
        # detector cycle on the Pi, which is long enough to fake "in hand".
        held_phones = [d for d in phones if d.confidence >= th.phone_hand_conf]
        # A phone is "in hand" when its box sits in the face/upper-chest gate.
        # A phone on the seat stays out.
        phone_in_hand = any(self._near_face(d, face) for d in held_phones)
        phone_at_ear = any(self._at_ear(d, face) for d in phones)

        away = self._away.update(now, looking_away)
        call_active = self._call.update(now, phone_at_ear)
        in_hand_active = self._in_hand.update(now, phone_in_hand)

        if (call_active or in_hand_active) and away:
            phone_signal = ConditionSignal(
                "phone_use",
                Severity.CRITICAL,
                "Phone in use while looking away from the road",
                True,
            )
        elif call_active:
            phone_signal = ConditionSignal(
                "phone_use",
                Severity.MAJOR,
                f"Phone call (phone at ear) for {self._call.elapsed(now):.1f}s",
                True,
            )
        elif in_hand_active:
            phone_signal = ConditionSignal(
                "phone_use",
                Severity.MAJOR,
                f"Phone in hand for {self._in_hand.elapsed(now):.1f}s",
                True,
            )
        else:
            # Presence alone is not a distraction: a phone on the seat must
            # not speak. The box still shows in the status `det=` field.
            phone_signal = ConditionSignal(
                "phone_use",
                Severity.MINOR,
                "Phone visible (not in use)",
                False,
            )

        return [
            ConditionSignal(
                "gaze_off_road",
                Severity.MAJOR,
                f"Gaze off road for {self._away.elapsed(now):.1f}s",
                away,
            ),
            phone_signal,
        ]

    def _near_face(self, detection: Detection, face: FaceObservation) -> bool:
        """Phone box in the face/upper-chest region, i.e. being held."""
        if face.bbox is None:
            return False
        x1, y1, x2, y2 = face.bbox
        span = y2 - y1
        if span <= 0:
            return False
        cx, cy = detection.center
        width = detection.box[2] - detection.box[0]
        height = detection.box[3] - detection.box[1]
        # A full-frame box is a model failure, not a phone in the hand.
        if max(width, height) > self.th.phone_max_size_span * span:
            return False
        return (
            x1 - 0.40 * span <= cx <= x2 + 0.40 * span
            and y1 - 0.25 * span <= cy <= y2 + 1.0 * span
        )

    def _at_ear(self, detection: Detection, face: FaceObservation) -> bool:
        anchor = SmokingMonitor._mouth_anchor(face)
        if anchor is None:
            return False
        mx, my, span = anchor
        if span <= 0:
            return False
        width = detection.box[2] - detection.box[0]
        height = detection.box[3] - detection.box[1]
        if max(width, height) > self.th.phone_max_size_span * span:
            return False
        cx, cy = detection.center
        for ear_x in (mx - 0.45 * span, mx + 0.45 * span):
            if abs(cx - ear_x) <= 0.35 * span and abs(cy - (my - 0.25 * span)) <= 0.35 * span:
                return True
        return False
