"""Smoke checks: configs load, assets exist, models run on a blank frame."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

import numpy as np

from dms.alerts import AlertManager, ConditionSignal, Severity
from dms.audio import PriorityPlaylist
from dms.config import AppConfig, Thresholds
from dms.face import FaceObservation, OpenCvFaceAnalyzer
from dms.behaviors import DistractionMonitor, DrowsinessMonitor, SmokingMonitor
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
            with self.subTest(config=name):
                cfg = AppConfig.load(ROOT / name)
                self.assertTrue(cfg.detectors)
                self.assertEqual(cfg.check_paths(ROOT), [])

    def test_only_the_pi_profile_serializes_detectors(self) -> None:
        pi = AppConfig.load(ROOT / "config_pi.json")
        pc = AppConfig.load(ROOT / "config_pc.json")
        self.assertTrue(pi.serialize_detectors)
        self.assertFalse(pc.serialize_detectors)
        self.assertTrue(pi.face_first)
        self.assertFalse(pc.face_first)
        self.assertEqual(pi.opencv_threads, 2)
        # Detector order is the priority order; cigarette must be first.
        self.assertEqual(pi.detectors[0].name, "cigarette")
        # Sleep must confirm in about 1.5-2 s, not after 4-5 s, while still
        # requiring a run of three fully closed frames.
        self.assertLessEqual(pi.thresholds.microsleep_min_frames, 3)
        self.assertLessEqual(pi.thresholds.microsleep_s, 1.2)
        self.assertLess(pi.thresholds.ear_closed, 0.15)


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
    def _face(
        ear: float = 0.28,
        mar: float = 0.05,
        ear_left: float | None = None,
        ear_right: float | None = None,
    ) -> FaceObservation:
        return FaceObservation(
            present=True,
            bbox=(100.0, 100.0, 200.0, 220.0),
            mouth=(150.0, 180.0),
            ear=ear,
            ear_left=ear if ear_left is None else ear_left,
            ear_right=ear if ear_right is None else ear_right,
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
