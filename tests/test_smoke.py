"""Smoke checks: configs load, assets exist, models run on a blank frame."""

from __future__ import annotations

import math
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from dms.alerts import AlertManager, ConditionSignal, Severity
from dms.audio import PriorityPlaylist
from dms.config import AppConfig, Thresholds
from dms.face import FaceObservation, OpenCvFaceAnalyzer
from dms.behaviors import (
    DistractionMonitor,
    DrowsinessMonitor,
    SmokingMonitor,
    resolve_cigarette_phone_conflicts,
)
from dms.objects import (
    BackgroundDetector,
    Detection,
    DetectorGate,
    YoloDetector,
)
from dms.temporal import Sustained

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_profiles_load_and_assets_exist(self) -> None:
        for name in ("config_pc.json", "config_pi.json"):
            path = ROOT / name
            if not path.exists():
                # The Pi upload intentionally excludes the PC profile.
                continue
            with self.subTest(config=name):
                cfg = AppConfig.load(path)
                self.assertTrue(cfg.detectors)
                self.assertEqual(cfg.check_paths(ROOT), [])

    def test_configs_have_no_duplicate_json_keys(self) -> None:
        # json.load silently keeps the last duplicate, so a stale key can
        # override a tuned value without any error. This caught a duplicate
        # phone_max_size_span that kept the Pi at the old 1.0 limit.
        import json

        def reject_duplicates(pairs):
            keys = [key for key, _ in pairs]
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            if duplicates:
                raise ValueError(f"duplicate keys: {duplicates}")
            return dict(pairs)

        for name in ("config_pc.json", "config_pi.json"):
            path = ROOT / name
            if not path.exists():
                continue
            with self.subTest(config=name):
                json.loads(
                    path.read_text(encoding="utf-8"),
                    object_pairs_hook=reject_duplicates,
                )

    def test_credits_flag_prints_notices(self) -> None:
        import io
        from contextlib import redirect_stdout

        from dms.app import main

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--credits"]), 0)
        text = output.getvalue()
        self.assertIn("YuNet", text)
        self.assertIn("MediaPipe", text)
        self.assertIn("InsightFace", text)
        self.assertIn("Ultralytics", text)

    def test_pi_runs_detectors_in_parallel(self) -> None:
        pi = AppConfig.load(ROOT / "config_pi.json")
        pc_path = ROOT / "config_pc.json"
        pc = AppConfig.load(pc_path) if pc_path.exists() else None
        # Both YOLO models run on their own worker thread with one OpenCV
        # thread each. Live Pi timing: ~6.4 s and ~6.8 s concurrently, versus
        # ~12.5 s when serialized, while the face loop stays near 1 fps.
        self.assertFalse(pi.serialize_detectors)
        self.assertTrue(pi.face_first)
        if pc is not None:
            self.assertFalse(pc.serialize_detectors)
            self.assertFalse(pc.face_first)
            self.assertTrue(pc.self_test)
            self.assertTrue(pc.audio.announce_startup)
        # A single OpenCV worker keeps the YOLO passes on their own threads
        # instead of queueing behind the face path in a shared pool; with more
        # threads the status loop froze for the whole detector pass.
        self.assertEqual(pi.opencv_threads, 2)
        # The Pi skips the ~5 s model self-test, but keeps the short startup
        # phrase so the speaker path is proven before an alert is needed.
        self.assertFalse(pi.self_test)
        self.assertTrue(pi.audio.announce_startup)
        # Both detectors are enabled and run in parallel; list order only
        # matters if serialization is re-enabled for debugging.
        self.assertEqual(pi.detectors[0].name, "cigarette")
        self.assertEqual(pi.detectors[1].name, "phone")
        self.assertGreaterEqual(pi.detectors[0].target_hz, 0.5)
        self.assertGreaterEqual(pi.detectors[1].target_hz, 0.5)
        # Sleep must confirm only on a real closure: three consecutive fully
        # closed samples and a sustained timer. The old sub-second bar plus
        # two frames alarmed on a downward glance at a phone.
        self.assertGreaterEqual(pi.thresholds.microsleep_min_frames, 2)
        self.assertGreaterEqual(pi.thresholds.microsleep_s, 0.5)
        self.assertLessEqual(pi.thresholds.ear_closed, 0.20)
        self.assertLessEqual(pi.thresholds.perclos_min_samples, 6)
        self.assertGreaterEqual(pi.thresholds.perclos_alert, 0.40)
        # A yawn needs a high MAR plus a real mouth gap relative to the face,
        # held over several frames. MAR alone fired on a closed mouth.
        self.assertGreaterEqual(pi.thresholds.yawn_mar, 0.50)
        self.assertGreaterEqual(pi.thresholds.yawn_min_frames, 3)
        self.assertGreaterEqual(pi.thresholds.yawn_gap_span, 0.08)
        # This camera sees the driver at a steep angle; leave headroom above
        # the measured ~53 deg yaw so a valid closed-eye frame is not rejected.
        self.assertGreaterEqual(pi.thresholds.drowsy_max_yaw_deg, 80.0)
        # Phone alerts wait for a real handset-sized, elongated box in two
        # separate detector passes; one republished false box cannot alert.
        self.assertGreaterEqual(pi.thresholds.phone_call_s, 1.0)
        self.assertGreaterEqual(pi.thresholds.phone_in_hand_s, 2.0)
        self.assertLessEqual(pi.thresholds.phone_low_conf, 0.3)
        self.assertLessEqual(pi.thresholds.phone_hand_conf, 0.35)
        self.assertGreaterEqual(pi.thresholds.phone_min_size_span, 0.4)
        self.assertGreaterEqual(pi.thresholds.phone_max_size_span, 1.1)
        self.assertGreater(pi.thresholds.phone_min_aspect, 1.0)
        self.assertGreaterEqual(pi.thresholds.phone_face_margin_span, 0.7)
        self.assertGreaterEqual(pi.thresholds.phone_min_passes, 1)
        # A cigarette must hold at the mouth for a real time span and appear
        # in two passes. The mouth-open rejection is above the yawn bar so a
        # slightly parted mouth does not hide a real cigarette.
        self.assertGreaterEqual(pi.thresholds.cigarette_s, 1.0)
        self.assertGreaterEqual(
            pi.thresholds.cigarette_low_s, pi.thresholds.cigarette_s
        )
        self.assertLessEqual(pi.detectors[0].conf_threshold, 0.3)
        self.assertGreaterEqual(pi.thresholds.cigarette_low_conf, 0.20)
        self.assertLessEqual(pi.thresholds.cigarette_low_conf, 0.35)
        self.assertGreaterEqual(pi.thresholds.cigarette_high_conf, 0.45)
        self.assertLessEqual(pi.thresholds.cigarette_high_conf, 0.65)
        self.assertGreater(
            pi.thresholds.cigarette_high_conf, pi.thresholds.cigarette_low_conf
        )
        self.assertGreaterEqual(pi.thresholds.cigarette_min_passes, 2)
        self.assertGreater(
            pi.thresholds.cigarette_max_mouth_mar, pi.thresholds.yawn_mar
        )
        self.assertLessEqual(pi.thresholds.cigarette_max_size_span, 0.60)
        self.assertLessEqual(pi.thresholds.cigarette_max_thickness_span, 0.25)
        self.assertLessEqual(pi.thresholds.cigarette_mouth_pad_span, 0.25)
        self.assertLessEqual(pi.thresholds.cigarette_min_aspect, 1.25)
        self.assertGreaterEqual(pi.thresholds.cigarette_bridge_s, 0.5)


class TemporalTests(unittest.TestCase):
    def test_hold_bridges_a_short_gap(self) -> None:
        condition = Sustained(0.0, hold_s=2.0)
        self.assertTrue(condition.update(0.0, True))
        self.assertTrue(condition.update(1.0, True))
        self.assertTrue(condition.update(1.5, False))
        self.assertTrue(condition.update(2.9, False))   # 1.9 s gap, still held
        self.assertFalse(condition.update(3.1, False))  # 2.1 s gap, released

    def test_bridge_keeps_onset_timer_across_a_one_frame_miss(self) -> None:
        # A not-yet-active condition that drops out for one short frame must
        # not lose the evidence already banked. Without the bridge the second
        # True restarts the timer and the alert arrives 1 s later.
        condition = Sustained(1.0, bridge_s=1.0)
        self.assertFalse(condition.update(0.0, True))
        self.assertFalse(condition.update(0.6, False))  # one missed frame
        self.assertTrue(condition.update(1.1, True))    # 1.1 s since first true


class DrowsinessTests(unittest.TestCase):
    @staticmethod
    def _face(
        ear: float = 0.28,
        mar: float = 0.05,
        ear_left: float | None = None,
        ear_right: float | None = None,
        mouth_gap: float = math.nan,
        face_height: float = math.nan,
    ) -> FaceObservation:
        return FaceObservation(
            present=True,
            bbox=(100.0, 100.0, 200.0, 220.0),
            mouth=(150.0, 180.0),
            ear=ear,
            ear_left=ear if ear_left is None else ear_left,
            ear_right=ear if ear_right is None else ear_right,
            mar=mar,
            mouth_gap=mouth_gap,
            face_height=face_height,
            yaw=5.0,
            pitch=5.0,
        )

    @staticmethod
    def _active(signals, key: str) -> bool:
        return next(s.active for s in signals if s.key == key)

    def test_impossible_landmark_values_are_ignored(self) -> None:
        monitor = DrowsinessMonitor(Thresholds(microsleep_s=0.5))
        # Hand over the face: the landmark model reports an EAR no open or
        # closed eye can produce. It must not count as a closed eyelid.
        for t in (0.0, 1.0, 2.0, 3.0):
            signals = monitor.update(t, self._face(ear=0.80))
            self.assertFalse(self._active(signals, "drowsy_microsleep"))
            self.assertFalse(self._active(signals, "driver_absent"))

    def test_microsleep_needs_a_run_of_closed_samples(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(microsleep_s=0.1, microsleep_min_frames=3)
        )
        self.assertFalse(
            self._active(monitor.update(0.0, self._face(ear=0.08)), "drowsy_microsleep")
        )
        self.assertFalse(
            self._active(monitor.update(1.0, self._face(ear=0.08)), "drowsy_microsleep")
        )
        self.assertTrue(
            self._active(monitor.update(2.0, self._face(ear=0.08)), "drowsy_microsleep")
        )

    def test_perclos_is_suppressed_when_samples_are_too_sparse(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(
                microsleep_s=100.0,
                perclos_window_s=4.0,
                perclos_alert=0.35,
                perclos_min_samples=10,
            )
        )
        for t in (0.0, 1.0, 2.0, 3.0):
            signals = monitor.update(t, self._face(ear=0.10))
            self.assertFalse(self._active(signals, "drowsy_perclos"))

    def test_three_closed_frames_at_one_fps_alert_in_two_seconds(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(
                ear_closed=0.10,
                microsleep_s=1.2,
                microsleep_min_frames=3,
            )
        )
        self.assertFalse(
            self._active(monitor.update(0.0, self._face(ear=0.05)), "drowsy_microsleep")
        )
        self.assertFalse(
            self._active(monitor.update(1.0, self._face(ear=0.05)), "drowsy_microsleep")
        )
        self.assertTrue(
            self._active(monitor.update(2.0, self._face(ear=0.05)), "drowsy_microsleep")
        )

    def test_one_eye_artifact_does_not_count_as_closed(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(microsleep_s=0.1, microsleep_min_frames=2)
        )
        for t in (0.0, 1.0, 2.0, 3.0):
            signals = monitor.update(
                t, self._face(ear=0.19, ear_left=0.08, ear_right=0.30)
            )
            self.assertFalse(self._active(signals, "drowsy_microsleep"))

    def test_far_eye_reading_high_does_not_block_real_closure(self) -> None:
        # At a steep camera yaw the far eyelid can read ~0.17 while the
        # visible eye is fully shut. The averaged EAR is clearly closed, so
        # sleep must still alert; the old both-eyes-below-bar guard missed it.
        monitor = DrowsinessMonitor(
            Thresholds(
                ear_closed=0.12, microsleep_s=0.1, microsleep_min_frames=2
            )
        )
        for t in (0.0, 1.0, 2.0):
            signals = monitor.update(
                t, self._face(ear=0.11, ear_left=0.05, ear_right=0.17)
            )
        self.assertTrue(self._active(signals, "drowsy_microsleep"))

    def test_short_blink_does_not_reach_the_faster_bar(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(microsleep_s=1.2, microsleep_min_frames=3)
        )
        # A 0.4 s blink spans two samples at 2.5 fps but not the 1.2 s bar.
        self.assertFalse(
            self._active(monitor.update(0.0, self._face(ear=0.05)), "drowsy_microsleep")
        )
        self.assertFalse(
            self._active(monitor.update(0.4, self._face(ear=0.05)), "drowsy_microsleep")
        )
        self.assertFalse(
            self._active(monitor.update(0.8, self._face(ear=0.30)), "drowsy_microsleep")
        )

    def test_closed_mouth_with_high_mar_is_not_a_yawn(self) -> None:
        # A narrow, closed mouth can make MAR look high. The absolute lip gap
        # is only 3 px on a 120 px face, so this is not a yawn.
        monitor = DrowsinessMonitor(
            Thresholds(yawn_mar=0.35, yawn_s=0.0, yawn_min_frames=1)
        )
        for t in (0.0, 1.0, 2.0, 3.0):
            signals = monitor.update(
                t,
                self._face(mar=0.70, mouth_gap=3.0, face_height=120.0),
            )
            self.assertFalse(self._active(signals, "drowsy_yawning"))

    def test_real_yawn_needs_mouth_gap_and_frames(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(yawn_mar=0.55, yawn_s=0.0, yawn_min_frames=3)
        )
        face = self._face(mar=0.70, mouth_gap=20.0, face_height=120.0)
        self.assertFalse(self._active(monitor.update(0.0, face), "drowsy_yawning"))
        self.assertFalse(self._active(monitor.update(1.0, face), "drowsy_yawning"))
        self.assertTrue(self._active(monitor.update(2.0, face), "drowsy_yawning"))

    def test_phone_in_use_resets_sleep_evidence(self) -> None:
        monitor = DrowsinessMonitor(
            Thresholds(microsleep_s=0.1, microsleep_min_frames=2)
        )
        closed = self._face(ear=0.05)
        for t in (0.0, 1.0, 2.0):
            signals = monitor.update(t, closed, phone_in_use=True)
            self.assertFalse(self._active(signals, "drowsy_microsleep"))
        # The banked closure is discarded; a new run needs its own frames.
        self.assertFalse(
            self._active(monitor.update(3.0, closed), "drowsy_microsleep")
        )
        self.assertTrue(
            self._active(monitor.update(4.0, closed), "drowsy_microsleep")
        )


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.face = FaceObservation(
            present=True,
            bbox=(100.0, 100.0, 200.0, 220.0),
            mouth=(150.0, 180.0),
        )

    def test_mouth_gate_accepts_a_cigarette_and_rejects_a_full_frame_box(self):
        gate = SmokingMonitor(Thresholds())
        cigarette = Detection("cigarette", 0.8, (145, 170, 165, 195))
        full_frame = Detection("cigarette", 0.9, (0, 0, 300, 360))
        self.assertTrue(gate._near_face(cigarette, self.face))
        # Centred on the mouth, but far too large to be a cigarette.
        self.assertFalse(gate._near_face(full_frame, self.face))

    def test_phone_gate_rejects_a_full_frame_box(self):
        gate = DistractionMonitor(Thresholds())
        full_frame = Detection("phone", 0.9, (0, 0, 300, 360))
        self.assertFalse(gate._near_face(full_frame, self.face))

    def test_phone_gate_rejects_cigarette_sized_box_but_keeps_real_handset(self):
        # Live Pi measurement: a real phone at the ear was 48x116 in a
        # 159-pixel face (0.73 face-heights, aspect 2.4). Phone boxes the
        # model invents on a cigarette/hand are 0.23-0.31 face-heights.
        gate = DistractionMonitor(Thresholds(phone_min_size_span=0.35))
        real_phone = Detection("cell phone", 0.54, (85, 130, 145, 215))
        false_box_at_ear = Detection("cell phone", 0.46, (80, 135, 110, 165))
        false_box_at_mouth = Detection("cell phone", 0.46, (140, 160, 170, 190))
        self.assertTrue(gate._at_ear(real_phone, self.face))
        self.assertTrue(gate._near_face(real_phone, self.face))
        # A hand/cheek box at the ear and a cigarette box at the mouth both
        # pass the location checks but are far too small to be a handset.
        self.assertFalse(gate._at_ear(false_box_at_ear, self.face))
        self.assertFalse(gate._near_face(false_box_at_ear, self.face))
        self.assertFalse(gate._near_face(false_box_at_mouth, self.face))
        # The PC profile leaves the gate off, so its behaviour is unchanged.
        pc_gate = DistractionMonitor(Thresholds())
        self.assertTrue(pc_gate._at_ear(false_box_at_ear, self.face))
        self.assertTrue(pc_gate._near_face(false_box_at_mouth, self.face))

    def test_large_phone_beside_face_passes_relaxed_pi_gates(self):
        # Live Pi failure: a real handset held at chest level measured
        # 142x187 against a 179-pixel face (1.04 face-heights), just over
        # the old max size of 1.0, and sat 0.55 face-heights left of the
        # face. The Pi now accepts up to 1.5 and a 0.75 horizontal margin.
        gate = DistractionMonitor(
            Thresholds(
                phone_min_size_span=0.35,
                phone_max_size_span=1.5,
                phone_face_margin_span=0.75,
            )
        )
        live_like = Detection("cell phone", 0.62, (10, 150, 105, 275))
        self.assertTrue(gate._near_face(live_like, self.face))
        old_pi = DistractionMonitor(
            Thresholds(phone_min_size_span=0.35, phone_max_size_span=1.0)
        )
        self.assertFalse(old_pi._near_face(live_like, self.face))

    def test_mouth_gate_rejects_phone_sized_box_and_hand_below_chin(self):
        gate = SmokingMonitor(Thresholds(cigarette_mouth_pad_span=0.15))
        # Centred near the mouth but as large as the face: this is the shape
        # of the phone-in-hand false positive seen on the phone clips.
        phone_sized = Detection("cigarette", 0.9, (130, 150, 190, 230))
        below_chin = Detection("cigarette", 0.9, (145, 220, 165, 240))
        self.assertFalse(gate._near_face(phone_sized, self.face))
        self.assertFalse(gate._near_face(below_chin, self.face))

    def test_end_on_cigarette_passes_while_round_phone_box_does_not(self):
        # A real Pi box measured 43x37 (aspect 1.16) on the evidence frame.
        # A round phone false box measured 40x45 (aspect 1.12). The Pi uses
        # 1.15 so the real cigarette passes and the round box does not.
        gate = SmokingMonitor(Thresholds(cigarette_min_aspect=1.15))
        live_pi_box = Detection("cigarette", 0.69, (140, 175, 183, 212))
        end_on = Detection("cigarette", 0.62, (140, 175, 173, 215))
        round_box = Detection("cigarette", 0.8, (130, 160, 170, 205))
        self.assertTrue(gate._near_face(live_pi_box, self.face))
        self.assertTrue(gate._near_face(end_on, self.face))
        self.assertFalse(gate._near_face(round_box, self.face))

    def test_hand_sized_cigarette_box_at_mouth_is_rejected(self):
        # A hand at the lips is as thick as it is long. A real cigarette box
        # is thin on its short side (live Pi: 37 px in a 159-px face).
        gate = SmokingMonitor(
            Thresholds(
                cigarette_max_thickness_span=0.35,
                cigarette_mouth_pad_span=0.20,
            )
        )
        hand = Detection("cigarette", 0.8, (120, 155, 175, 200))
        thin = Detection("cigarette", 0.8, (140, 175, 165, 190))
        self.assertFalse(gate._near_face(hand, self.face))
        self.assertTrue(gate._near_face(thin, self.face))

    def test_cigarette_wins_over_weak_phone_at_ear(self):
        cigarette = Detection("cigarette", 0.58, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.46, (172, 120, 212, 180))
        kept = resolve_cigarette_phone_conflicts(
            [cigarette, phone], self.face, Thresholds()
        )
        self.assertEqual([d.label for d in kept], ["cigarette"])

    def test_cigarette_box_on_an_ear_phone_is_dropped(self):
        # The dedicated cigarette model can fire on the handset/hand at the
        # ear. The phone label wins on that object, so it can never become a
        # smoking alert.
        th = Thresholds(
            phone_min_size_span=0.50,
            phone_min_aspect=1.20,
            phone_max_aspect=3.5,
            phone_strong_size_span=0.55,
            cigarette_mouth_pad_span=0.20,
        )
        phone = Detection("cell phone", 0.62, (60, 130, 120, 215))
        on_phone = Detection("cigarette", 0.80, (105, 165, 140, 205))
        kept = resolve_cigarette_phone_conflicts([on_phone, phone], self.face, th)
        self.assertEqual([d.label for d in kept], ["cell phone"])
        # A real cigarette at the lips does not overlap the phone and is kept.
        at_lips = Detection("cigarette", 0.80, (140, 175, 165, 195))
        kept = resolve_cigarette_phone_conflicts([at_lips, phone], self.face, th)
        self.assertEqual([d.label for d in kept], ["cigarette", "cell phone"])

    def test_large_real_handset_at_ear_survives_cigarette_arbitration(self):
        # Live Pi real phone: 48x116 in a 159-pixel face (0.73 face-heights,
        # confidence 0.536). It must not be hidden by a cigarette at the
        # mouth just because its confidence is below phone_strong_conf.
        cigarette = Detection("cigarette", 0.58, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.54, (70, 130, 130, 215))
        th = Thresholds(
            phone_strong_size_span=0.55, cigarette_mouth_pad_span=0.28
        )
        kept = resolve_cigarette_phone_conflicts([cigarette, phone], self.face, th)
        self.assertEqual([d.label for d in kept], ["cigarette", "cell phone"])
        # A medium box at the ear (0.50 face-heights) is still treated as
        # the cigarette/hand and loses to the cigarette.
        medium = Detection("cell phone", 0.46, (160, 120, 220, 180))
        kept = resolve_cigarette_phone_conflicts([cigarette, medium], self.face, th)
        self.assertEqual([d.label for d in kept], ["cigarette"])

    def test_high_confidence_cigarette_sized_phone_still_loses(self):
        # A false phone box on a cigarette can score 0.8. Size must veto
        # confidence: a cigarette-sized box is not a handset.
        cigarette = Detection("cigarette", 0.58, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.80, (140, 168, 170, 198))
        th = Thresholds(
            phone_strong_size_span=0.55, cigarette_mouth_pad_span=0.28
        )
        kept = resolve_cigarette_phone_conflicts([cigarette, phone], self.face, th)
        self.assertEqual([d.label for d in kept], ["cigarette"])

    def test_low_confidence_cigarette_suppresses_low_confidence_phone(self):
        # The exact Pi failure: 0.34 cigarette and 0.46 phone on one object.
        # The cigarette is below the smoking alert bar but high enough to
        # arbitrate the label conflict.
        cigarette = Detection("cigarette", 0.34, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.46, (172, 120, 212, 180))
        kept = resolve_cigarette_phone_conflicts(
            [cigarette, phone], self.face, Thresholds()
        )
        self.assertEqual([d.label for d in kept], ["cigarette"])

    def test_strong_phone_wins_over_weak_cigarette(self):
        cigarette = Detection("cigarette", 0.34, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.80, (172, 120, 212, 180))
        kept = resolve_cigarette_phone_conflicts(
            [cigarette, phone], self.face, Thresholds()
        )
        self.assertEqual([d.label for d in kept], ["cell phone"])

    def test_phone_away_from_the_mouth_is_not_suppressed(self):
        cigarette = Detection("cigarette", 0.58, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.50, (230, 300, 270, 340))
        kept = resolve_cigarette_phone_conflicts(
            [cigarette, phone], self.face, Thresholds()
        )
        self.assertEqual(
            [d.label for d in kept], ["cigarette", "cell phone"]
        )

    def test_weak_cigarette_cannot_suppress_a_phone(self):
        cigarette = Detection("cigarette", 0.20, (140, 175, 183, 212))
        phone = Detection("cell phone", 0.46, (172, 120, 212, 180))
        kept = resolve_cigarette_phone_conflicts(
            [cigarette, phone], self.face, Thresholds()
        )
        self.assertEqual(
            [d.label for d in kept], ["cigarette", "cell phone"]
        )

    def test_wide_yawn_does_not_count_as_a_cigarette(self):
        gate = SmokingMonitor(Thresholds())
        yawning = FaceObservation(
            present=True,
            bbox=(100.0, 100.0, 200.0, 220.0),
            mouth=(150.0, 180.0),
            mar=0.8,
        )
        cigarette = Detection("cigarette", 0.9, (145, 170, 165, 195))
        self.assertFalse(gate._near_face(cigarette, yawning))

    def test_pi_gate_rejects_round_watch_and_far_cheek_box(self):
        gate = SmokingMonitor(
            Thresholds(
                cigarette_gate_span=0.24,
                cigarette_max_size_span=0.5,
                cigarette_min_aspect=1.15,
                cigarette_mouth_pad_span=0.12,
            )
        )
        round_watch = Detection("cigarette", 0.9, (145, 170, 165, 190))
        cheek = Detection("cigarette", 0.8, (140, 140, 160, 150))
        at_lips = Detection("cigarette", 0.8, (135, 170, 165, 180))
        self.assertFalse(gate._near_face(round_watch, self.face))
        self.assertFalse(gate._near_face(cheek, self.face))
        self.assertTrue(gate._near_face(at_lips, self.face))

    def test_smoking_does_not_alert_on_one_short_flicker(self):
        gate = SmokingMonitor(Thresholds())
        cigarette = Detection("cigarette", 0.9, (145, 170, 165, 195))
        self.assertFalse(gate.update(10.0, self.face, [cigarette])[0].active)
        self.assertFalse(gate.update(10.5, self.face, [cigarette])[0].active)
        # A gap longer than the hold clears the evidence.
        self.assertFalse(gate.update(12.0, self.face, [])[0].active)
        self.assertFalse(gate.update(13.0, self.face, [cigarette])[0].active)
        self.assertTrue(gate.update(14.0, self.face, [cigarette])[0].active)

    def test_low_confidence_cigarette_alerts_on_the_weak_path(self):
        # Live Pi boxes dipped to 0.27 while the driver was smoking. The low
        # path now starts at 0.25 and confirms in 0.6 s; geometry still
        # decides whether the box is at the mouth.
        gate = SmokingMonitor(
            Thresholds(
                cigarette_s=0.3,
                cigarette_low_s=0.6,
                cigarette_high_conf=0.50,
                cigarette_low_conf=0.25,
                cigarette_mouth_pad_span=0.15,
            )
        )
        cigarette = Detection("cigarette", 0.27, (145, 170, 165, 195))
        self.assertFalse(gate.update(0.0, self.face, [cigarette])[0].active)
        self.assertTrue(gate.update(0.7, self.face, [cigarette])[0].active)

    def test_phone_in_hand_alerts_after_half_a_second(self):
        gate = DistractionMonitor(
            Thresholds(
                phone_in_hand_s=0.5,
                phone_hold_s=0.0,
                phone_hand_conf=0.3,
                phone_low_conf=0.25,
            )
        )
        phone = Detection("cell phone", 0.62, (10, 150, 105, 275))
        first = gate.update(0.0, self.face, [phone])
        self.assertFalse(next(s.active for s in first if s.key == "phone_use"))
        second = gate.update(0.6, self.face, [phone])
        self.assertTrue(next(s.active for s in second if s.key == "phone_use"))

    def test_phone_gate_rejects_a_box_centred_on_the_face(self):
        gate = DistractionMonitor(
            Thresholds(
                phone_min_size_span=0.50,
                phone_min_aspect=1.20,
                phone_center_exclusion_span=0.30,
            )
        )
        # Large and elongated enough, but centred on the face: the hand/cheek
        # shape that caused the false phone alert while the driver was drowsy.
        on_face = Detection("cell phone", 0.70, (115, 150, 185, 200))
        self.assertFalse(gate._near_face(on_face, self.face))
        # A handset at the ear is offset from the face centre and still passes.
        at_ear = Detection("cell phone", 0.54, (60, 130, 120, 215))
        self.assertTrue(gate._near_face(at_ear, self.face))

    def test_round_phone_box_is_rejected(self):
        gate = DistractionMonitor(
            Thresholds(phone_min_size_span=0.50, phone_min_aspect=1.20)
        )
        round_box = Detection("cell phone", 0.9, (110, 140, 190, 220))
        self.assertFalse(gate._near_face(round_box, self.face))

    def test_phone_needs_two_distinct_detector_passes(self):
        gate = DistractionMonitor(
            Thresholds(
                phone_in_hand_s=0.5,
                phone_hold_s=0.0,
                phone_hand_conf=0.3,
                phone_low_conf=0.25,
                phone_min_size_span=0.50,
                phone_min_aspect=1.20,
                phone_max_aspect=3.5,
                phone_center_exclusion_span=0.30,
                phone_min_passes=2,
            )
        )
        phone_pass_1 = Detection("cell phone", 0.62, (10, 150, 105, 275), pass_id=1)
        for t in (0.0, 1.0, 2.0):
            signals = gate.update(t, self.face, [phone_pass_1])
            self.assertFalse(next(s.active for s in signals if s.key == "phone_use"))
        phone_pass_2 = Detection("cell phone", 0.62, (10, 150, 105, 275), pass_id=2)
        signals = gate.update(3.0, self.face, [phone_pass_2])
        self.assertTrue(next(s.active for s in signals if s.key == "phone_use"))

    def test_phone_in_use_flag_is_instant_geometry(self):
        gate = DistractionMonitor(
            Thresholds(
                phone_in_hand_s=100.0,
                phone_hand_conf=0.3,
                phone_low_conf=0.25,
                phone_min_size_span=0.50,
                phone_min_aspect=1.20,
                phone_max_aspect=3.5,
                phone_center_exclusion_span=0.30,
            )
        )
        phone = Detection("cell phone", 0.62, (10, 150, 105, 275))
        gate.update(0.0, self.face, [phone])
        self.assertTrue(gate.phone_in_use)
        gate.update(0.5, self.face, [])
        self.assertFalse(gate.phone_in_use)

    def test_cigarette_needs_two_distinct_detector_passes(self):
        gate = SmokingMonitor(
            Thresholds(
                cigarette_s=0.1,
                cigarette_low_s=0.1,
                cigarette_high_conf=0.6,
                cigarette_low_conf=0.25,
                cigarette_min_passes=2,
            )
        )
        cigarette = Detection("cigarette", 0.8, (145, 170, 165, 195), pass_id=1)
        for t in (0.0, 1.0, 2.0):
            self.assertFalse(gate.update(t, self.face, [cigarette])[0].active)
        next_pass = Detection("cigarette", 0.8, (145, 170, 165, 195), pass_id=2)
        self.assertTrue(gate.update(3.0, self.face, [next_pass])[0].active)

    def test_phone_alert_survives_a_missed_pass_after_onset(self):
        gate = DistractionMonitor(
            Thresholds(
                phone_in_hand_s=0.5,
                phone_hold_s=3.0,
                phone_hand_conf=0.3,
                phone_low_conf=0.25,
                phone_min_size_span=0.55,
                phone_min_aspect=1.20,
                phone_max_aspect=3.5,
                phone_center_exclusion_span=0.30,
                phone_min_passes=2,
            )
        )
        phone_pass_1 = Detection("cell phone", 0.62, (10, 150, 105, 275), pass_id=1)
        phone_pass_2 = Detection("cell phone", 0.62, (10, 150, 105, 275), pass_id=2)
        gate.update(0.0, self.face, [phone_pass_1])
        gate.update(1.0, self.face, [phone_pass_2])
        signals = gate.update(2.0, self.face, [])
        self.assertTrue(next(s.active for s in signals if s.key == "phone_use"))

    def test_cigarette_alert_survives_a_short_miss_after_onset(self):
        gate = SmokingMonitor(
            Thresholds(
                cigarette_s=0.1,
                cigarette_low_s=0.1,
                cigarette_high_conf=0.6,
                cigarette_low_conf=0.25,
                cigarette_hold_s=2.0,
                cigarette_min_passes=2,
            )
        )
        pass_1 = Detection("cigarette", 0.8, (145, 170, 165, 195), pass_id=1)
        pass_2 = Detection("cigarette", 0.8, (145, 170, 165, 195), pass_id=2)
        gate.update(0.0, self.face, [pass_1])
        self.assertTrue(gate.update(1.0, self.face, [pass_2])[0].active)
        self.assertTrue(gate.update(2.0, self.face, [])[0].active)

    def test_cigarette_gate_allows_a_slightly_parted_mouth(self):
        # The old mouth-open rejection reused the 0.35 yawn bar and hid real
        # cigarettes. 0.60 is open but below cigarette_max_mouth_mar.
        gate = SmokingMonitor(Thresholds())
        face = FaceObservation(
            present=True,
            bbox=(100.0, 100.0, 200.0, 220.0),
            mouth=(150.0, 180.0),
            mar=0.60,
        )
        cigarette = Detection("cigarette", 0.9, (145, 170, 165, 195))
        self.assertTrue(gate._near_face(cigarette, face))


class DetectorSchedulingTests(unittest.TestCase):
    @staticmethod
    def _wait_for(predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        raise AssertionError("condition not reached before timeout")

    def test_gate_serves_lower_priority_number_first(self) -> None:
        gate = DetectorGate()
        self.assertTrue(gate.acquire(0))
        order: list[str] = []

        def take(name: str, priority: int) -> None:
            if gate.acquire(priority):
                order.append(name)
                gate.release()

        phone = threading.Thread(target=take, args=("phone", 1))
        cigarette = threading.Thread(target=take, args=("cigarette", 0))
        phone.start()
        self._wait_for(lambda: gate.waiting == 1)
        cigarette.start()
        self._wait_for(lambda: gate.waiting == 2)
        gate.release()
        phone.join(2.0)
        cigarette.join(2.0)
        self.assertFalse(phone.is_alive())
        self.assertFalse(cigarette.is_alive())
        self.assertEqual(order, ["cigarette", "phone"])

    def test_serialized_workers_never_run_a_pass_concurrently(self) -> None:
        state = {"lock": threading.Lock(), "current": 0, "peak": 0}

        class SlowDetector:
            enabled = True
            name = "slow"

            def detect(self, frame):
                with state["lock"]:
                    state["current"] += 1
                    state["peak"] = max(state["peak"], state["current"])
                time.sleep(0.04)
                with state["lock"]:
                    state["current"] -= 1
                return []

        gate = DetectorGate()
        workers = [
            BackgroundDetector(SlowDetector(), target_hz=20.0, gate=gate, priority=i)
            for i in range(2)
        ]
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        for worker in workers:
            worker.start()
        for _ in range(10):
            for worker in workers:
                worker.submit(frame)
            time.sleep(0.02)
        for worker in workers:
            worker.request_stop()
        gate.close()
        for worker in workers:
            worker.stop()
        self.assertEqual(state["peak"], 1)

    def test_paused_worker_starts_no_new_pass(self) -> None:
        state = {"lock": threading.Lock(), "count": 0}

        class CountingDetector:
            enabled = True
            name = "counting"

            def detect(self, frame):
                with state["lock"]:
                    state["count"] += 1
                return []

        worker = BackgroundDetector(CountingDetector(), target_hz=20.0)
        worker.set_paused(True)
        worker.start()
        worker.submit(np.zeros((48, 64, 3), dtype=np.uint8))
        time.sleep(0.2)
        with state["lock"]:
            self.assertEqual(state["count"], 0)
        worker.set_paused(False)
        self._wait_for(lambda: state["count"] > 0)
        worker.stop()


class AlertPriorityTests(unittest.TestCase):
    def test_sleep_speaks_before_phone_before_smoking(self) -> None:
        seen: list[str] = []
        manager = AlertManager(sinks=[lambda alert: seen.append(alert.key)])
        alerts = manager.update(
            1.0,
            [
                ConditionSignal("smoking", Severity.MAJOR, "smoking", True),
                ConditionSignal("phone_use", Severity.MAJOR, "phone", True),
                ConditionSignal(
                    "drowsy_microsleep", Severity.CRITICAL, "sleep", True
                ),
            ],
        )
        expected = ["drowsy_microsleep", "phone_use", "smoking"]
        self.assertEqual([alert.key for alert in alerts], expected)
        self.assertEqual(seen, expected)

    def test_playlist_orders_by_priority_and_drops_least_urgent(self) -> None:
        playlist = PriorityPlaylist(maxsize=2)
        playlist.put(Path("stop_smoking.wav"), priority=30)
        playlist.put(Path("please_take_a_break.wav"), priority=0)
        playlist.put(Path("do_not_use_phone.wav"), priority=20)
        self.assertEqual(playlist.get_nowait().name, "please_take_a_break.wav")
        self.assertEqual(playlist.get_nowait().name, "do_not_use_phone.wav")
        self.assertIsNone(playlist.get_nowait())


class ModelTests(unittest.TestCase):
    def test_models_run_on_a_blank_frame(self) -> None:
        cfg = AppConfig.load(ROOT / "config_pc.json")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        face = OpenCvFaceAnalyzer(cfg.face)
        try:
            self.assertFalse(face.process(frame, 0).present)
        finally:
            face.close()

        for detector_cfg in cfg.detectors:
            if not detector_cfg.enabled:
                continue
            with self.subTest(detector=detector_cfg.name):
                detector = YoloDetector(detector_cfg)
                try:
                    self.assertEqual(detector.detect(frame), [])
                finally:
                    detector.close()


if __name__ == "__main__":
    unittest.main()
