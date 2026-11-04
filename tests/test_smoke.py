"""Smoke checks: configs load, assets exist, models run on a blank frame."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from dms.config import AppConfig, Thresholds
from dms.face import FaceObservation, OpenCvFaceAnalyzer
from dms.behaviors import DistractionMonitor, DrowsinessMonitor, SmokingMonitor
from dms.objects import Detection, YoloDetector
from dms.temporal import Sustained

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_profiles_load_and_assets_exist(self) -> None:
        for name in ("config_pc.json", "config_pi.json"):
            with self.subTest(config=name):
                cfg = AppConfig.load(ROOT / name)
                self.assertTrue(cfg.detectors)
                self.assertEqual(cfg.check_paths(ROOT), [])


class TemporalTests(unittest.TestCase):
    def test_hold_bridges_a_short_gap(self) -> None:
        condition = Sustained(0.0, hold_s=2.0)
        self.assertTrue(condition.update(0.0, True))
        self.assertTrue(condition.update(1.0, True))
        self.assertTrue(condition.update(1.5, False))
        self.assertTrue(condition.update(2.9, False))   # 1.9 s gap, still held
        self.assertFalse(condition.update(3.1, False))  # 2.1 s gap, released


class DrowsinessTests(unittest.TestCase):
    @staticmethod
    def _face(ear: float = 0.28, mar: float = 0.05) -> FaceObservation:
        return FaceObservation(
            present=True,
            bbox=(100.0, 100.0, 200.0, 220.0),
            mouth=(150.0, 180.0),
            ear=ear,
            ear_left=ear,
            ear_right=ear,
            mar=mar,
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

    def test_mouth_gate_rejects_phone_sized_box_and_hand_below_chin(self):
        gate = SmokingMonitor(Thresholds())
        # Centred near the mouth but as large as the face: this is the shape
        # of the phone-in-hand false positive seen on the phone clips.
        phone_sized = Detection("cigarette", 0.9, (130, 150, 190, 230))
        below_chin = Detection("cigarette", 0.9, (145, 220, 165, 240))
        self.assertFalse(gate._near_face(phone_sized, self.face))
        self.assertFalse(gate._near_face(below_chin, self.face))

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
