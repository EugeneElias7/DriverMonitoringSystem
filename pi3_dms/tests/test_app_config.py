"""Configuration tests for the optional module and preprocessing flags."""

from __future__ import annotations

import unittest

import app


class ArgumentTests(unittest.TestCase):
    def test_objects_disabled_by_default(self) -> None:
        args = app.parse_args([])
        self.assertFalse(args.objects)
        self.assertEqual(args.object_backend, "none")
        self.assertEqual(args.object_classes, ["phone", "bottle"])

    def test_objects_enable_mediapipe_and_do_not_require_http(self) -> None:
        args = app.parse_args(["--objects"])
        self.assertTrue(args.objects)
        self.assertEqual(args.object_backend, "mediapipe")

    def test_no_tint_fix_selects_raw_mode(self) -> None:
        args = app.parse_args(["--no-tint-fix"])
        self.assertEqual(args.tint_mode, "off")
        self.assertEqual(args.tint_label, "off (raw)")

    def test_night_mode_is_explicit(self) -> None:
        args = app.parse_args(["--night"])
        self.assertEqual(args.tint_mode, "night")

    def test_manual_awb_requires_both_gains(self) -> None:
        with self.assertRaises(SystemExit):
            app.parse_args(["--awb", "manual"])
        args = app.parse_args(["--awb", "manual", "--red-gain", "1.4", "--blue-gain", "1.1"])
        self.assertEqual(args.awb, "manual")
        self.assertEqual(args.red_gain, 1.4)

    def test_unknown_object_class_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            app.parse_args(["--objects", "--object-classes", "phone,bottle,cup"])

    def test_disabled_service_has_explicit_object_state(self) -> None:
        args = app.parse_args([])
        service = app.DMSService(args)
        status = service.status()
        self.assertEqual(status["object"]["status"], "disabled")
        self.assertEqual(status["alcohol"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
