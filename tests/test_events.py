"""Tests for evidence capture, JSONL logging, and Telegram notifications."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from dms.events.config import EventConfig
from dms.events.event_logger import EventLogger
from dms.events.event_manager import EventManager, format_caption
from dms.events.evidence_capture import EvidenceCapture
from dms.events.telegram_env import load_telegram_env
from dms.events.telegram_notifier import NotificationResult, TelegramNotifier


class FakeNotifier:
    configured = True

    def __init__(self, sent: bool = True, error: str | None = None):
        self.sent = sent
        self.error = error
        self.calls: list[tuple[Path, str]] = []

    def send_photo(self, image_path, caption):
        self.calls.append((Path(image_path), caption))
        return NotificationResult(self.sent, self.error)


class FailingCapture:
    def save(self, frame, folder, event_id):
        raise OSError("disk full")


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class EventManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame[:, :, 1] = 80
        self.fake = FakeNotifier()
        self.manager = EventManager(
            vehicle_number="KA03MN1234",
            evidence_dir=self.root / "evidence",
            log_path=self.root / "logs" / "events.jsonl",
            notifier=self.fake,
            rearm_grace_s=1.0,
        )
        self.timestamp = datetime(2026, 11, 11, 17, 30, 20)

    def tearDown(self) -> None:
        self.manager.close()
        self._tmp.cleanup()

    def records(self) -> list[dict]:
        path = self.root / "logs" / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_one_event_saves_original_frame_and_record(self) -> None:
        created = self.manager.process(
            {"drowsy_microsleep"},
            self.frame,
            now=0.0,
            timestamp=self.timestamp,
        )
        self.assertEqual(created, ["EVT-20261111-000001"])
        self.assertTrue(self.manager.wait_idle(5.0))

        records = self.records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["event_id"], "EVT-20261111-000001")
        self.assertEqual(record["vehicle_number"], "KA03MN1234")
        self.assertEqual(record["alert_type"], "DROWSINESS")
        self.assertEqual(record["timestamp"], "2026-11-11T17:30:20")
        self.assertEqual(record["notification_status"], "SENT")
        self.assertNotIn("notification_error", record)

        image_path = Path(record["image_path"])
        self.assertTrue(image_path.is_file())
        self.assertEqual(image_path.name, "EVT-20261111-000001.jpg")
        self.assertEqual(image_path.parent.name, "drowsiness")
        image = cv2.imread(str(image_path))
        self.assertEqual(image.shape[:2], (480, 640))

        self.assertEqual(len(self.fake.calls), 1)
        caption = self.fake.calls[0][1]
        self.assertIn("<b>DMS ALERT</b>", caption)
        self.assertIn("Vehicle Number: KA03MN1234", caption)
        self.assertIn("Alert Type: 😴 DROWSINESS", caption)
        self.assertIn("Time: 11 Nov 2026, 05:30 PM", caption)
        self.assertIn("Event ID: EVT-20261111-000001", caption)

    def test_continuous_event_creates_one_record(self) -> None:
        self.manager.process(
            {"drowsy_microsleep"}, self.frame, now=0.0, timestamp=self.timestamp
        )
        self.manager.wait_idle(5.0)

        # The same condition, a second pass, and the microsleep -> PERCLOS
        # transition all belong to one continuous drowsiness event.
        self.manager.process(
            {"drowsy_microsleep"}, self.frame, now=1.0, timestamp=self.timestamp
        )
        self.manager.process(
            {"drowsy_perclos"}, self.frame, now=1.5, timestamp=self.timestamp
        )
        self.manager.process(
            {"drowsy_microsleep", "drowsy_perclos"},
            self.frame,
            now=2.0,
            timestamp=self.timestamp,
        )
        self.manager.wait_idle(5.0)

        self.assertEqual(len(self.records()), 1)
        self.assertEqual(len(self.fake.calls), 1)

    def test_event_rearms_after_grace(self) -> None:
        self.manager.process(
            {"smoking"}, self.frame, now=0.0, timestamp=self.timestamp
        )
        self.manager.wait_idle(5.0)
        self.manager.process(set(), self.frame, now=1.0)
        self.manager.process(
            {"smoking"}, self.frame, now=2.0, timestamp=self.timestamp
        )
        self.manager.wait_idle(5.0)

        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(
            [record["event_id"] for record in records],
            ["EVT-20261111-000001", "EVT-20261111-000002"],
        )
        self.assertEqual(len(self.fake.calls), 2)

    def test_all_event_types_use_expected_folder(self) -> None:
        cases = [
            ("drowsy_microsleep", "DROWSINESS", "drowsiness"),
            ("drowsy_perclos", "DROWSINESS", "drowsiness"),
            ("drowsy_yawning", "YAWNING", "yawning"),
            ("phone_use", "PHONE_USAGE", "phone"),
            ("smoking", "SMOKING", "smoking"),
        ]
        for index, (key, alert_type, folder) in enumerate(cases):
            with self.subTest(key=key):
                manager = EventManager(
                    vehicle_number="KA03MN1234",
                    evidence_dir=self.root / f"case{index}" / "evidence",
                    log_path=self.root / f"case{index}" / "events.jsonl",
                    notifier=FakeNotifier(),
                )
                try:
                    manager.process(
                        {key}, self.frame, now=0.0, timestamp=self.timestamp
                    )
                    self.assertTrue(manager.wait_idle(5.0))
                finally:
                    manager.close()
                record = json.loads(
                    (self.root / f"case{index}" / "events.jsonl").read_text(
                        encoding="utf-8"
                    ).strip()
                )
                self.assertEqual(record["alert_type"], alert_type)
                self.assertEqual(Path(record["image_path"]).parent.name, folder)

    def test_notification_failure_is_logged_without_crashing(self) -> None:
        manager = EventManager(
            vehicle_number="KA03MN1234",
            evidence_dir=self.root / "failed_evidence",
            log_path=self.root / "failed_events.jsonl",
            notifier=FakeNotifier(sent=False, error="network down"),
        )
        try:
            manager.process(
                {"phone_use"}, self.frame, now=0.0, timestamp=self.timestamp
            )
            self.assertTrue(manager.wait_idle(5.0))
        finally:
            manager.close()

        record = json.loads(
            (self.root / "failed_events.jsonl").read_text(encoding="utf-8").strip()
        )
        self.assertEqual(record["notification_status"], "FAILED")
        self.assertEqual(record["notification_error"], "network down")
        self.assertTrue(Path(record["image_path"]).is_file())

    def test_image_failure_is_logged_without_calling_telegram(self) -> None:
        fake = FakeNotifier()
        manager = EventManager(
            vehicle_number="KA03MN1234",
            evidence_dir=self.root / "unused",
            log_path=self.root / "image_fail_events.jsonl",
            notifier=fake,
            capture=FailingCapture(),
        )
        try:
            manager.process(
                {"smoking"}, self.frame, now=0.0, timestamp=self.timestamp
            )
            self.assertTrue(manager.wait_idle(5.0))
        finally:
            manager.close()

        record = json.loads(
            (self.root / "image_fail_events.jsonl").read_text(encoding="utf-8").strip()
        )
        self.assertEqual(record["notification_status"], "FAILED")
        self.assertEqual(record["image_path"], "")
        self.assertIn("image save failed", record["notification_error"])
        self.assertEqual(fake.calls, [])


class EventLoggerTests(unittest.TestCase):
    def test_ids_continue_from_existing_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text(
                '{"event_id": "EVT-20261111-000007"}\n'
                'not json\n'
                '{"event_id": "EVT-20261110-000099"}\n',
                encoding="utf-8",
            )
            logger = EventLogger(path)
            at = datetime(2026, 11, 11, 17, 30, 20)
            self.assertEqual(logger.next_event_id(at), "EVT-20261111-000008")
            self.assertEqual(logger.next_event_id(at), "EVT-20261111-000009")

    def test_new_day_starts_at_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = EventLogger(Path(tmp) / "events.jsonl")
            self.assertEqual(
                logger.next_event_id(datetime(2026, 11, 11)), "EVT-20261111-000001"
            )
            self.assertEqual(
                logger.next_event_id(datetime(2026, 11, 12)), "EVT-20261112-000001"
            )


class EventConfigTests(unittest.TestCase):
    def test_load_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config_events.json"
            path.write_text('{"vehicle_number": "KA03MN1234"}', encoding="utf-8")
            self.assertEqual(EventConfig.load(path).vehicle_number, "KA03MN1234")

            missing = Path(tmp) / "missing.json"
            self.assertEqual(EventConfig.load(missing).vehicle_number, "UNKNOWN")

            path.write_text('{"vehicle_number": "  "}', encoding="utf-8")
            with self.assertRaises(ValueError):
                EventConfig.load(path)

            path.write_text('{"vehicle_number": "KA03MN1234", "x": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                EventConfig.load(path)


class EvidenceCaptureTests(unittest.TestCase):
    def test_saves_frame_without_resizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:, :, 2] = 120
            path = EvidenceCapture(Path(tmp) / "evidence").save(
                frame, "drowsiness", "EVT-20261111-000001"
            )
            image = cv2.imread(str(path))
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (720, 1280))
            self.assertGreater(path.stat().st_size, 0)

    def test_rejects_empty_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                EvidenceCapture(Path(tmp) / "evidence").save(
                    np.zeros((0, 0, 3), dtype=np.uint8),
                    "drowsiness",
                    "EVT-20261111-000001",
                )


class CaptionTests(unittest.TestCase):
    def test_caption_format(self) -> None:
        caption = format_caption(
            "KA03MN1234",
            "DROWSINESS",
            datetime(2026, 11, 11, 17, 30, 20),
            "EVT-20261111-000001",
        )
        self.assertEqual(
            caption,
            "🚨 <b>DMS ALERT</b>\n\n"
            "Vehicle Number: KA03MN1234\n"
            "Alert Type: 😴 DROWSINESS\n"
            "Time: 11 Nov 2026, 05:30 PM\n"
            "Event ID: EVT-20261111-000001",
        )


class TelegramEnvTests(unittest.TestCase):
    @staticmethod
    def _write(root: Path, text: str) -> Path:
        path = root / ".env"
        path.write_text(text, encoding="utf-8")
        return path

    def test_loads_allowed_keys_and_ignores_unknown_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            path = self._write(
                Path(tmp),
                "# local credentials\n"
                "export TELEGRAM_BOT_TOKEN='123:abc'\n"
                'TELEGRAM_CHAT_ID="456"\n'
                "OTHER_KEY=ignored\n"
                "malformed line\n"
                "\n",
            )
            loaded = load_telegram_env(path)
            self.assertEqual(loaded, path)
            self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "123:abc")
            self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "456")
            self.assertNotIn("OTHER_KEY", os.environ)

    def test_environment_variables_take_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "from-shell", "TELEGRAM_CHAT_ID": "shell-chat"},
            clear=True,
        ):
            path = self._write(
                Path(tmp),
                "TELEGRAM_BOT_TOKEN=from-file\n"
                "TELEGRAM_CHAT_ID=file-chat\n",
            )
            load_telegram_env(path)
            self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "from-shell")
            self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "shell-chat")

    def test_override_replaces_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"TELEGRAM_CHAT_ID": "shell-chat"}, clear=True
        ):
            path = self._write(Path(tmp), "TELEGRAM_CHAT_ID=file-chat\n")
            load_telegram_env(path, override=True)
            self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "file-chat")

    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertIsNone(load_telegram_env(Path(tmp) / "missing.env"))

    def test_dms_env_file_selects_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            path = self._write(Path(tmp), "TELEGRAM_CHAT_ID=777\n")
            os.environ["DMS_ENV_FILE"] = str(path)
            self.assertEqual(load_telegram_env(), path)
            self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "777")

    def test_utf8_bom_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            path = Path(tmp) / ".env"
            path.write_text(
                "TELEGRAM_CHAT_ID=888\n", encoding="utf-8-sig"
            )
            load_telegram_env(path)
            self.assertEqual(os.environ["TELEGRAM_CHAT_ID"], "888")


class TelegramNotifierTests(unittest.TestCase):
    def _image(self, root: Path) -> Path:
        path = root / "EVT-20261111-000001.jpg"
        cv2.imwrite(
            str(path),
            np.zeros((48, 64, 3), dtype=np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        return path

    def test_send_photo_uses_multipart_and_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = self._image(Path(tmp))
            with mock.patch(
                "dms.events.telegram_notifier.urllib.request.urlopen",
                return_value=FakeResponse({"ok": True}),
            ) as urlopen:
                notifier = TelegramNotifier(
                    token="SECRET_TOKEN", chat_id="999", retries=0
                )
                result = notifier.send_photo(image, "<b>DMS ALERT</b>")

            self.assertTrue(result.sent)
            self.assertEqual(urlopen.call_count, 1)
            request = urlopen.call_args.args[0]
            self.assertIn("sendPhoto", request.full_url)
            self.assertIn(b'name="chat_id"', request.data)
            self.assertIn(b"999", request.data)
            self.assertIn(b'name="photo"; filename="EVT-20261111-000001.jpg"', request.data)
            self.assertIn(b"<b>DMS ALERT</b>", request.data)
            self.assertIn(b"\xff\xd8", request.data)

    def test_transient_network_errors_are_retried(self) -> None:
        import urllib.error

        with tempfile.TemporaryDirectory() as tmp:
            image = self._image(Path(tmp))
            side_effect = [
                urllib.error.URLError("offline"),
                urllib.error.URLError("still offline"),
                FakeResponse({"ok": True}),
            ]
            with mock.patch(
                "dms.events.telegram_notifier.urllib.request.urlopen",
                side_effect=side_effect,
            ) as urlopen:
                notifier = TelegramNotifier(
                    token="SECRET_TOKEN",
                    chat_id="999",
                    retries=2,
                    retry_delay_s=0.0,
                )
                result = notifier.send_photo(image, "caption")

            self.assertTrue(result.sent)
            self.assertEqual(urlopen.call_count, 3)

    def test_http_error_does_not_leak_token(self) -> None:
        import urllib.error

        with tempfile.TemporaryDirectory() as tmp:
            image = self._image(Path(tmp))
            error = urllib.error.HTTPError(
                "https://api.telegram.org/botSECRET_TOKEN/sendPhoto",
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"ok":false,"description":"Unauthorized"}'),
            )
            with mock.patch(
                "dms.events.telegram_notifier.urllib.request.urlopen",
                side_effect=error,
            ):
                notifier = TelegramNotifier(
                    token="SECRET_TOKEN", chat_id="999", retries=3
                )
                result = notifier.send_photo(image, "caption")

            self.assertFalse(result.sent)
            self.assertIn("HTTP 401", result.error)
            self.assertNotIn("SECRET_TOKEN", result.error)

    def test_http_error_body_cannot_leak_token(self) -> None:
        import urllib.error

        with tempfile.TemporaryDirectory() as tmp:
            image = self._image(Path(tmp))
            error = urllib.error.HTTPError(
                "https://api.telegram.org/botSECRET_TOKEN/sendPhoto",
                400,
                "Bad Request",
                {},
                io.BytesIO(b'{"ok":false,"description":"bad SECRET_TOKEN"}'),
            )
            with mock.patch(
                "dms.events.telegram_notifier.urllib.request.urlopen",
                side_effect=error,
            ):
                notifier = TelegramNotifier(
                    token="SECRET_TOKEN", chat_id="999", retries=3
                )
                result = notifier.send_photo(image, "caption")

            self.assertFalse(result.sent)
            self.assertIn("HTTP 400", result.error)
            self.assertNotIn("SECRET_TOKEN", result.error)
            self.assertIn("<redacted>", result.error)

    def test_missing_credentials_is_a_clean_failure(self) -> None:
        notifier = TelegramNotifier(token="", chat_id="")
        result = notifier.send_photo(Path("does-not-matter.jpg"), "caption")
        self.assertFalse(result.sent)
        self.assertIn("credentials", result.error)

    def test_missing_image_is_a_clean_failure(self) -> None:
        notifier = TelegramNotifier(token="token", chat_id="999")
        result = notifier.send_photo(Path("does-not-exist.jpg"), "caption")
        self.assertFalse(result.sent)
        self.assertIn("image not found", result.error)
