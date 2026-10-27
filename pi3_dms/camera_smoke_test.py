"""Manual camera smoke test, separate from the simulated unit tests.

This starts the same single-camera path used by the dashboard, waits for frames,
and prints the processor/object status.  It does not assert that a face is
present: a camera can be working while the driver is not in view.

Examples:
    python camera_smoke_test.py --camera 0 --seconds 8
    python camera_smoke_test.py --image test.jpg --seconds 2
"""

from __future__ import annotations

import argparse
import json
import time

from app import DMSService, parse_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "picamera2", "opencv"), default="auto")
    parser.add_argument("--image", default=None, help="use a still image instead of a camera")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--objects", action="store_true", help="also exercise the object module")
    args = parser.parse_args()

    app_args = parse_args(
        [
            *(["--image", args.image] if args.image else []),
            "--camera",
            str(args.camera),
            "--camera-backend",
            args.camera_backend,
            "--camera-fps",
            "10",
            "--process-width",
            "640",
            "--process-height",
            "360",
            "--tint-mode",
            "soft",
            *(["--objects"] if args.objects else []),
        ]
    )
    service = DMSService(app_args)
    service.start()
    try:
        time.sleep(max(1.0, float(args.seconds)))
        status = service.status()
        print(
            json.dumps(
                {
                    "mode": status.get("mode"),
                    "camera": status.get("camera"),
                    "performance": status.get("performance"),
                    "main_state": status.get("main_state"),
                    "main_reason": status.get("main_reason"),
                    "tracking": status.get("tracking"),
                    "object": status.get("object"),
                    "resources": status.get("resources"),
                },
                indent=2,
            )
        )
        camera = status.get("camera") or {}
        if camera.get("error"):
            print(f"camera smoke test failed: {camera['error']}")
            return 1
        if not (status.get("performance") or {}).get("process_fps"):
            print("camera smoke test failed: no processed frames")
            return 1
        print("camera smoke test: pipeline produced processed frames")
        return 0
    finally:
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
