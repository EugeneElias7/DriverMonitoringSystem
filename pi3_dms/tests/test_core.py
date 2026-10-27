"""Deterministic tests for the face/eye/mouth/head core.

These tests do not open a camera.  They use synthetic face observations with
controlled timestamps so closure, yawning and distraction windows are exact.
"""

from __future__ import annotations

import unittest

from dms.core import MeasurementEngine, driver_state
from tests.helpers import make_face, update


class CoreStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MeasurementEngine(yawn_min_ms=800.0, distracted_ms=1500.0)

    def test_open_eyes_do_not_become_drowsy(self) -> None:
        state = None
        for index in range(100):
            state = update(self.engine, make_face(ear=0.30, mar=0.18), 1.0 + index * 0.1)
        self.assertEqual(state["eye_state"], "open")
        self.assertEqual(state["drowsiness_state"], "alert")
        self.assertEqual(state["state"], "alert")
        self.assertEqual(state["microsleeps"], 0)
        self.assertNotEqual(driver_state(state)[0], "DROWSY")

    def test_blink_is_not_drowsiness(self) -> None:
        update(self.engine, make_face(ear=0.30), 1.0)
        update(self.engine, make_face(ear=0.10), 2.0)
        state = update(self.engine, make_face(ear=0.30), 2.2)
        self.assertGreaterEqual(state["blinks"], 1)
        self.assertEqual(state["state"], "alert")
        self.assertEqual(state["closure_ms"], 0.0)
        self.assertEqual(state["microsleeps"], 0)

    def test_sustained_closure_and_microsleep(self) -> None:
        update(self.engine, make_face(ear=0.30), 1.0)
        update(self.engine, make_face(ear=0.10), 2.0)
        state = update(self.engine, make_face(ear=0.10), 3.1)
        self.assertEqual(state["state"], "prolonged_closure")
        self.assertEqual(driver_state(state)[0], "DROWSY")
        state = update(self.engine, make_face(ear=0.10), 3.7)
        self.assertEqual(state["state"], "microsleep")
        self.assertEqual(state["microsleeps"], 1)
        self.assertEqual(state["drowsiness_state"], "drowsy")
        self.assertEqual(driver_state(state)[0], "DROWSY")

    def test_no_face_is_uncertain_not_drowsy(self) -> None:
        update(self.engine, make_face(ear=0.30), 1.0)
        state = self.engine.update(
            None,
            2.0,
            source_frame_width=480,
            source_frame_height=270,
        )
        self.assertEqual(state["state"], "no_face")
        self.assertEqual(state["drowsiness_state"], "unknown")
        self.assertEqual(driver_state(state)[0], "TRACKING UNCERTAIN")

    def test_bad_quality_is_uncertain(self) -> None:
        state = self.engine.update(
            make_face(ear=0.30, quality="too_small", quality_detail="face too small"),
            1.0,
            source_frame_width=480,
            source_frame_height=270,
        )
        self.assertEqual(state["state"], "uncertain")
        self.assertEqual(state["tracking_confidence"], 0.0)
        self.assertEqual(driver_state(state)[0], "TRACKING UNCERTAIN")

    def test_stale_observation_does_not_advance_closure(self) -> None:
        update(self.engine, make_face(ear=0.30), 2.0)
        update(self.engine, make_face(ear=0.10), 2.1)
        state = update(self.engine, make_face(ear=0.10), 2.1)
        self.assertEqual(state["state"], "uncertain")
        self.assertEqual(state["quality"], "stale")
        # A fresh timestamp still works after the rejected duplicate.
        state = update(self.engine, make_face(ear=0.10), 2.2)
        self.assertEqual(state["state"], "eyes_closed")

    def test_yawning_is_sustained_and_requires_hold(self) -> None:
        update(self.engine, make_face(ear=0.30, mar=0.20), 1.0)
        # A short mouth opening is only mouth_open, not a yawn.
        for index, mar in enumerate((0.70, 0.72, 0.70, 0.25)):
            state = update(self.engine, make_face(ear=0.30, mar=mar), 2.0 + index * 0.2)
        self.assertNotEqual(state["mouth_state"], "yawning")
        update(self.engine, make_face(ear=0.30, mar=0.20), 4.0)
        # A sustained opening crosses the yawn threshold.
        for index in range(8):
            state = update(self.engine, make_face(ear=0.30, mar=0.75), 5.0 + index * 0.15)
        self.assertEqual(state["mouth_state"], "yawning")
        self.assertGreaterEqual(state["yawns"], 1)
        self.assertEqual(state["state"], "yawning")

    def test_yawn_and_closure_are_independent(self) -> None:
        update(self.engine, make_face(ear=0.30, mar=0.20), 1.0)
        for index in range(10):
            state = update(
                self.engine,
                make_face(ear=0.10, mar=0.75),
                2.0 + index * 0.15,
            )
        self.assertIn(state["state"], {"prolonged_closure", "microsleep"})
        self.assertEqual(state["mouth_state"], "yawning")
        self.assertEqual(state["drowsiness_state"], "drowsy")
        self.assertTrue(state["yawns"] >= 1)

    def test_glance_then_distracted_without_confusing_drowsiness(self) -> None:
        update(self.engine, make_face(ear=0.30, yaw=0.0), 1.0)
        state = update(self.engine, make_face(ear=0.30, yaw=40.0), 2.0)
        self.assertEqual(state["attention_state"], "glance")
        self.assertEqual(state["drowsiness_state"], "alert")
        state = update(self.engine, make_face(ear=0.30, yaw=40.0), 3.6)
        self.assertEqual(state["attention_state"], "distracted")
        self.assertEqual(driver_state(state)[0], "DISTRACTED")
        self.assertEqual(state["eye_state"], "open")

    def test_small_head_turn_stays_forward(self) -> None:
        update(self.engine, make_face(ear=0.30, yaw=0.0), 1.0)
        state = update(self.engine, make_face(ear=0.30, yaw=20.0), 2.0)
        self.assertEqual(state["head_state"], "forward")
        self.assertEqual(state["attention_state"], "forward")

    def test_missing_head_pose_does_not_claim_attentive(self) -> None:
        state = update(self.engine, make_face(ear=0.30, yaw=None, pitch=None), 1.0)
        self.assertEqual(state["head_state"], "unknown")
        self.assertEqual(state["attention_state"], "unknown")
        self.assertEqual(driver_state(state)[0], "TRACKING UNCERTAIN")

    def test_tracking_confidence_is_a_conservative_heuristic(self) -> None:
        state = update(self.engine, make_face(ear=0.30, face_width=30.0), 1.0)
        self.assertEqual(state["tracking_confidence"], 0.0)  # too_small gate
        self.engine = MeasurementEngine()
        state = update(self.engine, make_face(ear=0.30, face_width=120.0), 1.0)
        self.assertGreater(float(state["tracking_confidence"]), 0.0)
        self.assertLessEqual(float(state["tracking_confidence"]), 1.0)
        self.assertIn("heuristic", state["tracking_confidence_basis"])


if __name__ == "__main__":
    unittest.main()
