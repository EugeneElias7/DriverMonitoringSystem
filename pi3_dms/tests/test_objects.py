"""Object plumbing tests; no camera is opened."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dms.objects import (
    CANONICAL_LABELS,
    HttpObjectDetector,
    MediaPipeObjectDetector,
    ObjectDetection,
    ObjectSnapshot,
    canonical_label,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "object" / "efficientdet_lite0.tflite"


class ObjectLabelTests(unittest.TestCase):
    def test_only_explicit_aliases_are_mapped(self) -> None:
        self.assertEqual(canonical_label("cell phone"), "phone")
        self.assertEqual(canonical_label("bottle"), "bottle")
        self.assertEqual(canonical_label("cigarette"), "cigarette")
        self.assertIsNone(canonical_label("smoke"))
        self.assertIsNone(canonical_label("cup"))
        self.assertEqual(CANONICAL_LABELS, {"phone", "bottle", "cigarette"})

    def test_stale_snapshot_suppresses_detections(self) -> None:
        detection = ObjectDetection(
            label="phone",
            confidence=0.9,
            bbox=(0.1, 0.2, 0.3, 0.4),
            timestamp=1.0,
            source="test",
        )
        snapshot = ObjectSnapshot(
            status="running",
            enabled=True,
            detections=(detection,),
            timestamp=1.0,
            stale_ms=1000.0,
        )
        fresh = snapshot.to_dict(1.5)
        self.assertEqual(fresh["status"], "running")
        self.assertEqual(len(fresh["detections"]), 1)
        stale = snapshot.to_dict(1.0 + 2.0)
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["detections"], [])

    def test_disabled_snapshot_is_not_running(self) -> None:
        snapshot = ObjectSnapshot.disabled()
        payload = snapshot.to_dict(100.0)
        self.assertEqual(payload["status"], "disabled")
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["detections"], [])


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._payload


class HttpDetectorTests(unittest.TestCase):
    def test_http_backend_parses_normalized_detections(self) -> None:
        payload = {
            "detections": [
                {
                    "label": "bottle",
                    "confidence": 0.72,
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                },
                {
                    "label": "cigarette",
                    "confidence": 0.66,
                    "bbox": [0.2, 0.2, 0.25, 0.25],
                },
                {
                    "label": "smoke",
                    "confidence": 0.99,
                    "bbox": [0.0, 0.0, 0.1, 0.1],
                },
            ]
        }
        with patch("dms.objects.urllib.request.urlopen", return_value=_Response(payload)):
            detector = HttpObjectDetector("http://127.0.0.1:8010/detect")
            detections = detector.detect(
                np.zeros((270, 480, 3), dtype=np.uint8),
                timestamp=12.0,
                frame_id=7,
            )
        self.assertEqual([d.label for d in detections], ["bottle", "cigarette"])
        self.assertEqual(detections[0].timestamp, 12.0)
        self.assertEqual(detections[0].frame_id, 7)
        self.assertEqual(detections[0].source, "http")

    def test_http_backend_filters_unconfigured_classes(self) -> None:
        payload = {
            "detections": [
                {"label": "bottle", "confidence": 0.72, "bbox": [0.1, 0.2, 0.3, 0.4]},
                {"label": "cigarette", "confidence": 0.66, "bbox": [0.2, 0.2, 0.25, 0.25]},
            ]
        }
        with patch("dms.objects.urllib.request.urlopen", return_value=_Response(payload)):
            detector = HttpObjectDetector(
                "http://127.0.0.1:8010/detect", allowed_classes={"phone"}
            )
            detections = detector.detect(
                np.zeros((270, 480, 3), dtype=np.uint8),
                timestamp=12.0,
                frame_id=7,
            )
        self.assertEqual(detections, [])

    def test_http_backend_rejects_invalid_boxes(self) -> None:
        payload = {
            "detections": [
                {"label": "bottle", "confidence": 0.7, "bbox": [0.5, 0.5, 0.5, 0.6]},
                {"label": "bottle", "confidence": 1.7, "bbox": [0.1, 0.1, 0.2, 0.2]},
            ]
        }
        with patch("dms.objects.urllib.request.urlopen", return_value=_Response(payload)):
            detector = HttpObjectDetector("http://127.0.0.1:8010/detect")
            detections = detector.detect(
                np.zeros((270, 480, 3), dtype=np.uint8), timestamp=12.0
            )
        self.assertEqual(detections, [])


@unittest.skipUnless(MODEL.exists(), "shipped optional COCO model is not present")
class MediaPipeObjectDetectorTests(unittest.TestCase):
    def test_model_loads_and_invents_no_labels_on_blank_frame(self) -> None:
        detector = MediaPipeObjectDetector(
            MODEL,
            allowed_classes={"phone", "bottle"},
            min_confidence=0.5,
        )
        try:
            detections = detector.detect(
                np.zeros((270, 480, 3), dtype=np.uint8), timestamp=1.0
            )
        finally:
            detector.close()
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
