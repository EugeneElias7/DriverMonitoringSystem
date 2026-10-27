"""Camera geometry and colour-preprocessing tests without a camera."""

from __future__ import annotations

import unittest

import numpy as np

from dms.camera import FrameStore, letterbox, night_gray, preprocess_frame, soft_gray_world


class CameraHelperTests(unittest.TestCase):
    def test_letterbox_preserves_aspect_and_center(self) -> None:
        source = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        out = letterbox(source, (480, 270))
        self.assertEqual(out.shape, (270, 480, 3))
        # A 16:9 source in a 16:9 target has no padding.
        self.assertTrue(np.all(out == 128))

    def test_letterbox_pads_portrait_source(self) -> None:
        source = np.full((400, 200, 3), 255, dtype=np.uint8)
        out = letterbox(source, (480, 270))
        self.assertEqual(out.shape, (270, 480, 3))
        self.assertTrue(np.all(out[:, :120] == 0))
        self.assertTrue(np.all(out[:, 360:] == 0))

    def test_soft_gray_world_reduces_simple_channel_cast(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # blue
        frame[:, :, 1] = 50
        frame[:, :, 2] = 50
        corrected = soft_gray_world(frame)
        before = frame.reshape(-1, 3).mean(axis=0)
        after = corrected.reshape(-1, 3).mean(axis=0)
        self.assertLess(after[0], before[0])
        self.assertGreater(after[2], before[2])

    def test_night_mode_is_grayscale_and_keeps_three_channels(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :, 1] = 120
        out = night_gray(frame)
        self.assertEqual(out.shape, (20, 20, 3))
        self.assertTrue(np.all(out[:, :, 0] == out[:, :, 1]))
        self.assertTrue(np.all(out[:, :, 1] == out[:, :, 2]))

    def test_unknown_colour_mode_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_frame(np.zeros((4, 4, 3), dtype=np.uint8), "magic")

    def test_frame_store_keeps_raw_frame_separate(self) -> None:
        from dms.camera import FramePacket

        store = FrameStore()
        stream = np.full((54, 96, 3), 20, dtype=np.uint8)
        process = np.full((27, 48, 3), 10, dtype=np.uint8)
        raw = np.full((108, 192, 3), 30, dtype=np.uint8)
        store.put(FramePacket(stream, process, 1.0, 1, raw), {"backend": "test"})
        got_raw, info = store.latest_raw()
        self.assertIs(got_raw, raw)
        self.assertEqual(info["backend"], "test")


if __name__ == "__main__":
    unittest.main()
