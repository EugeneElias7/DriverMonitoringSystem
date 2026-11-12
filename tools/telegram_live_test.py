"""Manual Telegram smoke test; never runs as part of the automated suite.

Usage (from the project root):

    # Put TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env once, then:
    python tools/telegram_live_test.py --camera
    python tools/telegram_live_test.py --image evidence/smoking/EVT-....jpg

Shell environment variables override .env values. With no arguments, a small
synthetic JPEG is generated. --camera captures one live frame from the laptop
webcam, and --image sends an existing JPEG. Exit code 0 means Telegram accepted
the photo; 1 means the request failed; 2 means configuration, the camera, or
the input image is missing. The bot token is never printed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dms.events.event_manager import format_caption  # noqa: E402
from dms.events.telegram_env import load_telegram_env  # noqa: E402
from dms.events.telegram_notifier import TelegramNotifier  # noqa: E402


def _synthetic_image(path: Path) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[:, :, 1] = 80
    cv2.putText(
        image,
        "DMS Telegram test",
        (120, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
        raise OSError(f"could not write test image: {path}")


def _camera_image(path: Path, index: int) -> None:
    """Capture one live frame from a local camera and save it as JPEG."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise OSError(f"could not open camera index {index}")
    try:
        frame = None
        for _ in range(10):
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
                break
            time.sleep(0.05)
        if frame is None:
            raise OSError(f"camera {index} did not deliver a frame")
        if not cv2.imwrite(
            str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ):
            raise OSError(f"could not write camera image: {path}")
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send one real Telegram photo")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="existing JPEG evidence image")
    source.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        metavar="INDEX",
        help="capture one live frame from this camera index (default: 0)",
    )
    parser.add_argument("--caption", help="override the default caption")
    args = parser.parse_args(argv)

    load_telegram_env()
    notifier = TelegramNotifier()
    if not notifier.configured:
        print(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the "
            "environment or in .env",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        if args.camera is not None:
            image = Path(tmp) / "dms_camera_test.jpg"
            try:
                _camera_image(image, args.camera)
            except OSError as exc:
                print(f"camera error: {exc}", file=sys.stderr)
                return 2
        elif args.image:
            image = Path(args.image)
            if not image.is_file():
                print(f"image not found: {image}", file=sys.stderr)
                return 2
        else:
            image = Path(tmp) / "dms_telegram_test.jpg"
            _synthetic_image(image)

        caption = args.caption or format_caption(
            vehicle_number="TEST",
            alert_type="MANUAL TEST",
            timestamp=datetime.now(),
            event_id="EVT-TELEGRAM-TEST",
        )
        result = notifier.send_photo(image, caption)

    if result.sent:
        print("SENT")
        return 0
    print(f"FAILED: {result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
