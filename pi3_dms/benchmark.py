"""Desktop benchmark for the face core and optional object detector.

This script measures inference only. It does not open a camera. Run it with a
real face image and, optionally, the shipped object model. Pi 3 numbers must be
measured separately on the target board; desktop numbers do not transfer.

Example:
    python benchmark.py --image C:\\path\\face.jpg --objects
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dms.camera import letterbox
from dms.core import FaceModel
from dms.objects import MediaPipeObjectDetector

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FACE = BASE_DIR / "models" / "face_landmarker.task"
DEFAULT_OBJECT = BASE_DIR / "models" / "object" / "efficientdet_lite0.tflite"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(label: str, iterations: int, callable_fn) -> dict[str, float]:
    # One warm-up is deliberately excluded from the result.
    callable_fn(0)
    process = psutil.Process(os.getpid()) if psutil is not None else None
    if process is not None:
        process.cpu_percent(interval=None)
        start_process_time = process.cpu_times()
    else:
        start_process_time = None
    wall_start = time.perf_counter()
    times: list[float] = []
    for index in range(iterations):
        started = time.perf_counter()
        callable_fn(index)
        times.append((time.perf_counter() - started) * 1000.0)
    wall = time.perf_counter() - wall_start
    payload = {
        "iterations": float(iterations),
        "latency_mean_ms": round(statistics.mean(times), 2),
        "latency_median_ms": round(statistics.median(times), 2),
        "latency_p95_ms": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 2),
        "throughput_fps": round(iterations / wall, 1) if wall > 0 else 0.0,
    }
    if psutil is not None and start_process_time is not None and process is not None:
        end_process_time = process.cpu_times()
        cpu_seconds = (
            (end_process_time.user - start_process_time.user)
            + (end_process_time.system - start_process_time.system)
        )
        memory = process.memory_info()
        payload.update(
            {
                "process_cpu_percent": round(cpu_seconds / wall * 100.0, 1) if wall > 0 else 0.0,
                "process_rss_mb": round(memory.rss / (1024 * 1024), 1),
                "system_cpu_percent_after": round(psutil.cpu_percent(interval=None), 1),
                "system_ram_percent": round(psutil.virtual_memory().percent, 1),
                "system_ram_total_mb": round(
                    psutil.virtual_memory().total / (1024 * 1024), 1
                ),
            }
        )
    print(
        f"{label:20s} mean={payload['latency_mean_ms']:7.2f} ms  "
        f"median={payload['latency_median_ms']:7.2f} ms  "
        f"fps={payload['throughput_fps']:5.1f}",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="test image with a face")
    parser.add_argument("--face-model", default=str(DEFAULT_FACE))
    parser.add_argument("--object-model", default=str(DEFAULT_OBJECT))
    parser.add_argument("--objects", action="store_true", help="also measure the object model")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args(argv)

    image = cv2.imread(str(Path(args.image)))
    if image is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 2
    frame = letterbox(image, (args.width, args.height))
    iterations = max(1, int(args.iterations))
    face_model = FaceModel(Path(args.face_model))
    object_model = None
    try:
        results: dict[str, object] = {
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "cpu": platform.processor() or "unknown",
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "ram_total_mb": (
                    round(psutil.virtual_memory().total / (1024 * 1024), 1)
                    if psutil is not None
                    else None
                ),
            },
            "input": {
                "image": str(Path(args.image).resolve()),
                "inference_resolution": f"{args.width}x{args.height}",
                "face_model": str(Path(args.face_model).resolve()),
                "face_sha256": _sha256(Path(args.face_model)),
            },
            "notes": [
                "Desktop measurement only; not a Raspberry Pi 3 result.",
                "Sequential combined timing runs face and object inference back-to-back.",
                "The application runs object inference periodically and asynchronously, "
                "so the sequential number is a conservative upper bound on per-frame work.",
                "No accuracy or safety claim is made.",
            ],
        }

        face_result = _measure(
            "face core",
            iterations,
            lambda index: face_model.detect(frame, 1.0 + index * 0.1),
        )
        results["face_core"] = face_result

        if args.objects and Path(args.object_model).exists():
            object_model = MediaPipeObjectDetector(
                Path(args.object_model),
                allowed_classes={"phone", "bottle"},
                min_confidence=0.35,
            )
            object_result = _measure(
                "object model",
                iterations,
                lambda index: object_model.detect(frame, 1.0 + index * 0.1, index),
            )
            results["object_model"] = object_result
            combined_result = _measure(
                "combined sequential",
                iterations,
                lambda index: (
                    face_model.detect(frame, 100.0 + index * 0.1),
                    object_model.detect(frame, 100.0 + index * 0.1, index),
                ),
            )
            results["combined_sequential"] = combined_result
            results["input"]["object_model"] = str(Path(args.object_model).resolve())
            results["input"]["object_sha256"] = _sha256(Path(args.object_model))
        elif args.objects:
            results["object_model"] = {"error": f"model not found: {args.object_model}"}

        print(json.dumps(results, indent=2), flush=True)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        if args.markdown_out:
            lines = [
                "# DMS benchmark",
                "",
                f"Measured: {results['measured_at']}",
                "",
                f"Environment: {results['environment']['platform']}; "
                f"Python {results['environment']['python']}; "
                f"OpenCV {results['environment']['opencv']}",
                "",
                f"Input: {args.width}x{args.height}",
                "",
                "| Path | Mean latency | Median | p95 | Throughput | Process CPU | RSS | System CPU after | System RAM |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for key, label in (
                ("face_core", "Face core"),
                ("object_model", "Object model"),
                ("combined_sequential", "Combined sequential"),
            ):
                data = results.get(key)
                if not isinstance(data, dict):
                    continue
                lines.append(
                    f"| {label} | {data.get('latency_mean_ms', '-')} ms | "
                    f"{data.get('latency_median_ms', '-')} ms | "
                    f"{data.get('latency_p95_ms', '-')} ms | "
                    f"{data.get('throughput_fps', '-')} fps | "
                    f"{data.get('process_cpu_percent', '-')}% | "
                    f"{data.get('process_rss_mb', '-')} MB | "
                    f"{data.get('system_cpu_percent_after', '-')}% | "
                    f"{data.get('system_ram_percent', '-')}% |"
                )
            lines.extend(
                [
                    "",
                    "Process CPU can exceed 100% because TFLite/XNNPACK uses more than one CPU thread. "
                    "System CPU is sampled after the loop; system RAM is the whole machine, not the process.",
                    "Desktop only. Pi 3 FPS, latency, CPU, RAM and thermals remain unverified.",
                    "",
                ]
            )
            Path(args.markdown_out).write_text("\n".join(lines), encoding="utf-8")
        return 0
    finally:
        face_model.close()
        if object_model is not None:
            object_model.close()


if __name__ == "__main__":
    raise SystemExit(main())
