"""Simulated temporal-fusion tests: no camera and no object model are required."""

from __future__ import annotations

import unittest

from dms.fusion import BehaviourConfig, BehaviourFusion, FusionEngine
from dms.objects import ObjectDetection, ObjectSnapshot


NOW = 100.0
FACE = {
    "quality": "good",
    "quality_detail": "face and eye geometry available",
    "eye_state": "open",
    "state": "alert",
    "tracking_confidence": 0.82,
    "tracking_confidence_basis": "heuristic: size, pose, continuity",
    "source_frame_width": 480,
    "source_frame_height": 270,
    "face_bbox": [100.0, 50.0, 300.0, 250.0],
    "mouth_center": [200.0, 200.0],
    "drowsiness_state": "alert",
    "attention_state": "forward",
    "mouth_state": "clear",
    "distraction_state": "forward",
    "head_state": "forward",
    "closure_ms": 0.0,
    "ear_smoothed": 0.30,
    "mar_smoothed": 0.20,
}


def detection(label: str, bbox=(0.40, 0.70, 0.48, 0.80), confidence=0.80, timestamp=NOW):
    return ObjectDetection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        timestamp=timestamp,
        source="test",
        raw_label=label,
    )


def snapshot(*items, timestamp=NOW, status="running"):
    return ObjectSnapshot(
        status=status,
        enabled=True,
        detections=tuple(items),
        timestamp=timestamp,
        latency_ms=20.0,
        model="test-model",
        classes=("phone", "bottle", "cigarette"),
        stale_ms=1500.0,
    )


def config(**overrides):
    base = dict(
        stale_ms=1500.0,
        min_confidence=0.35,
        cigarette_min_confidence=0.45,
        phone_min_ms=900.0,
        drinking_min_ms=900.0,
        smoking_min_ms=900.0,
        min_observations=2,
        face_radius_scale=0.40,
        mouth_radius_scale=0.55,
        gap_ms=400.0,
    )
    base.update(overrides)
    return BehaviourConfig(**base)


class BehaviourFusionTests(unittest.TestCase):
    def test_bottle_near_mouth_is_drinking_not_smoking(self) -> None:
        fusion = BehaviourFusion(config())
        bottle = detection("bottle")
        states = None
        for index in range(3):
            states = fusion.update(
                FACE,
                snapshot(bottle, timestamp=NOW + index * 0.5),
                NOW + index * 0.5,
            )
        self.assertEqual(states["drinking"]["state"], "suspected")
        self.assertTrue(states["drinking"]["suspected"])
        self.assertEqual(states["smoking"]["state"], "clear")
        self.assertFalse(states["smoking"]["suspected"])
        self.assertEqual(states["phone"]["state"], "clear")

    def test_cigarette_near_mouth_is_smoking(self) -> None:
        fusion = BehaviourFusion(config())
        cigarette = detection("cigarette", confidence=0.80)
        states = None
        for index in range(3):
            states = fusion.update(
                FACE,
                snapshot(cigarette, timestamp=NOW + index * 0.5),
                NOW + index * 0.5,
            )
        self.assertEqual(states["smoking"]["state"], "suspected")
        self.assertEqual(states["drinking"]["state"], "clear")

    def test_phone_near_face_is_phone_use_but_phone_elsewhere_is_not(self) -> None:
        fusion = BehaviourFusion(config())
        near = detection("phone", bbox=(0.42, 0.25, 0.55, 0.45))
        states = None
        for index in range(3):
            states = fusion.update(
                FACE,
                snapshot(near, timestamp=NOW + index * 0.5),
                NOW + index * 0.5,
            )
        self.assertEqual(states["phone"]["state"], "suspected")

        fusion = BehaviourFusion(config())
        far = detection("phone", bbox=(0.85, 0.80, 0.95, 0.95))
        for index in range(3):
            states = fusion.update(
                FACE,
                snapshot(far, timestamp=NOW + index * 0.5),
                NOW + index * 0.5,
            )
        self.assertEqual(states["phone"]["state"], "clear")
        self.assertFalse(states["phone"]["suspected"])

    def test_short_detection_does_not_become_behaviour(self) -> None:
        fusion = BehaviourFusion(config(phone_min_ms=1200.0, min_observations=3))
        states = fusion.update(
            FACE, snapshot(detection("phone"), timestamp=NOW), NOW
        )
        self.assertEqual(states["phone"]["state"], "possible")
        # One second later, but with no new detector snapshot: duration alone
        # is not enough; observations are still below the configured minimum.
        states = fusion.update(
            FACE, snapshot(detection("phone"), timestamp=NOW), NOW + 1.0
        )
        self.assertEqual(states["phone"]["state"], "possible")

    def test_stale_detections_are_rejected(self) -> None:
        fusion = BehaviourFusion(config())
        states = fusion.update(
            FACE,
            snapshot(detection("bottle"), timestamp=NOW - 3.0),
            NOW,
        )
        self.assertEqual(states["drinking"]["state"], "stale")
        self.assertFalse(states["drinking"]["suspected"])

    def test_face_loss_makes_behaviour_uncertain(self) -> None:
        fusion = BehaviourFusion(config())
        uncertain_face = dict(FACE)
        uncertain_face.update({"quality": "no_face", "state": "no_face", "eye_state": "unknown"})
        states = fusion.update(
            uncertain_face,
            snapshot(detection("cigarette"), timestamp=NOW),
            NOW,
        )
        self.assertEqual(states["smoking"]["state"], "uncertain")
        self.assertFalse(states["smoking"]["suspected"])

    def test_disabled_and_unavailable_modules_are_explicit(self) -> None:
        fusion = BehaviourFusion(config())
        disabled = ObjectSnapshot.disabled()
        states = fusion.update(FACE, disabled, NOW)
        for value in states.values():
            self.assertEqual(value["state"], "disabled")
            self.assertFalse(value["suspected"])

        unavailable = ObjectSnapshot.unavailable("model missing")
        states = fusion.update(FACE, unavailable, NOW)
        for value in states.values():
            self.assertEqual(value["state"], "unavailable")
            self.assertFalse(value["suspected"])

    def test_multiple_simultaneous_behaviours(self) -> None:
        fusion = BehaviourFusion(
            config(smoking_min_ms=500.0, drinking_min_ms=500.0, phone_min_ms=500.0)
        )
        phone = detection("phone", bbox=(0.42, 0.25, 0.55, 0.45))
        bottle = detection("bottle")
        cigarette = detection("cigarette")
        states = None
        for index in range(4):
            states = fusion.update(
                FACE,
                snapshot(phone, bottle, cigarette, timestamp=NOW + index * 0.25),
                NOW + index * 0.25,
            )
        self.assertTrue(states["phone"]["suspected"])
        self.assertTrue(states["drinking"]["suspected"])
        self.assertTrue(states["smoking"]["suspected"])

    def test_fusion_never_marks_uncertain_tracking_attentive(self) -> None:
        fusion = FusionEngine(config())
        uncertain = dict(FACE)
        uncertain.update({"quality": "no_face", "state": "no_face", "eye_state": "unknown"})
        result = fusion.update(uncertain, ObjectSnapshot.disabled(), NOW)
        self.assertEqual(result["main_state"], "TRACKING UNCERTAIN")
        self.assertEqual(result["tracking"]["state"], "uncertain")

    def test_object_status_is_reported_with_detections(self) -> None:
        fusion = FusionEngine(config())
        result = fusion.update(
            FACE, snapshot(detection("phone"), timestamp=NOW), NOW
        )
        self.assertEqual(result["object"]["status"], "running")
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["detections"][0]["label"], "phone")
        self.assertIn("confidence", result["detections"][0])


if __name__ == "__main__":
    unittest.main()
