"""Command-line entry point for image, folder, video and webcam runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .config import Config, ConfigError, load_config
from .drawing import draw_result
from .pipeline import DMSPipeline
from .utils.io import append_jsonl, imread, imwrite, write_json
from .utils.perf import LatencyAccumulator

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "image":
            return run_image(args, config)
        if args.command == "folder":
            return run_folder(args, config)
        if args.command == "video":
            return run_video(args, config)
        if args.command == "webcam":
            return run_webcam(args, config)
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dms",
        description="Offline driver-monitoring research prototype (not a safety device).",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="YAML profile (default: %(default)s)")
    common.add_argument("--smoke-backend", choices=["yolo_onnx", "tflite_ssd", "none"], help="override smoking backend")
    common.add_argument("--no-smoking", action="store_true", help="disable smoking inference")
    common.add_argument("--smoke-model", help="override smoking ONNX model path")
    common.add_argument("--smoke-class-names", help="comma-separated classes matching the checkpoint")
    common.add_argument("--face-model", help="override face landmarker path")
    common.add_argument("--face-mode", choices=["image", "video"], help="override face running mode")
    common.add_argument("--resize-width", type=int, help="resize frames once before processing (0 = original)")
    common.add_argument("--events", help="JSONL event output path")
    common.add_argument("--quiet", action="store_true", help="suppress per-frame progress output")
    common.add_argument("--no-json", action="store_true", help="do not write per-frame JSON")
    common.add_argument("--no-annotated", action="store_true", help="do not write annotated media")

    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("image", parents=[common], help="process one image")
    image.add_argument("--source", required=True, help="input image")
    image.add_argument("--output", help="annotated image path")
    image.add_argument("--roi", help="face ROI hint in pixels: x1,y1,x2,y2")
    image.add_argument("--display", action="store_true", help="show the annotated image")

    folder = sub.add_parser("folder", parents=[common], help="process a folder of images")
    folder.add_argument("--source", required=True, help="input image directory")
    folder.add_argument("--output-dir", help="annotated image output directory")
    folder.add_argument("--limit", type=int, default=0, help="maximum images (0 = all)")
    folder.add_argument("--roi", help="fixed face ROI hint: x1,y1,x2,y2")

    video = sub.add_parser("video", parents=[common], help="process a video file")
    video.add_argument("--source", required=True, help="input video")
    video.add_argument("--output", help="annotated video output path")
    video.add_argument("--max-frames", type=int, default=0, help="process at most N frames")
    video.add_argument("--display", action="store_true", help="show frames while processing")
    video.add_argument("--roi", help="fixed face ROI hint: x1,y1,x2,y2")

    webcam = sub.add_parser("webcam", parents=[common], help="process a live camera")
    webcam.add_argument("--camera", type=int, default=0, help="camera index")
    webcam.add_argument("--output", help="optional annotated video output")
    webcam.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    webcam.add_argument("--display", action="store_true", default=True, help="show frames (default: true)")
    webcam.add_argument("--no-display", dest="display", action="store_false")
    webcam.add_argument("--width", type=int, default=0, help="requested capture width")
    webcam.add_argument("--height", type=int, default=0, help="requested capture height")
    webcam.add_argument("--fps", type=float, default=15.0, help="output writer FPS")
    webcam.add_argument("--roi", help="fixed face ROI hint: x1,y1,x2,y2")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if args.no_smoking and args.smoke_backend and args.smoke_backend != "none":
        raise ConfigError("--no-smoking conflicts with --smoke-backend")
    if args.no_smoking:
        config.set("smoking.backend", "none")
    elif args.smoke_backend:
        config.set("smoking.backend", args.smoke_backend)
    if args.smoke_model:
        config.set("smoking.model_path", args.smoke_model)
    if args.smoke_class_names:
        names = [part.strip() for part in args.smoke_class_names.split(",") if part.strip()]
        config.set("smoking.class_names", names)
    if args.face_model:
        config.set("face.model_path", args.face_model)
    if args.face_mode:
        config.set("face.mode", args.face_mode)
    elif args.command in {"image", "folder"}:
        # Still-images should not depend on video tracking state.
        config.set("face.mode", "image")
    if args.resize_width is not None:
        config.set("pipeline.resize_width", int(args.resize_width))
    config.validate()
    return config


def run_image(args: argparse.Namespace, config: Config) -> int:
    source = Path(args.source)
    original = imread(source)
    if original is None:
        raise ValueError(f"could not read image: {source}")
    pipeline = DMSPipeline(config, mode="image")
    pipeline.load()
    roi = parse_roi(args.roi) if args.roi else None
    frame = pipeline.resize(original)
    hint = scale_roi(roi, original, frame) if roi is not None else None
    result = pipeline.process(frame, timestamp=time.time(), roi_hint=hint)
    output_path = Path(args.output) if args.output else _default_image_output(config, source)
    events_path = _events_path(args, config, source.stem, is_image=True)
    if not args.no_annotated:
        annotated = draw_result(frame, result, face=pipeline.face.last_observation)
        if not imwrite(output_path, annotated):
            raise OSError(f"could not write annotated image: {output_path}")
    if not args.no_json:
        write_json(output_path.with_suffix(".json"), result)
    _write_events(events_path, result)
    if args.display:
        cv2.imshow("DMS image (press any key)", draw_result(frame, result, pipeline.face.last_observation))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    pipeline.close()
    summary = {
        "mode": "image",
        "source": str(source),
        "output": None if args.no_annotated else str(output_path),
        "json": None if args.no_json else str(output_path.with_suffix(".json")),
        "events": None if events_path is None else str(events_path),
        "event_count": len(result["events"]),
        "face": bool(result["face"].get("present")),
        "states": {name: (value or {}).get("state") for name, value in result["features"].items()},
        "timing_ms": result["timing_ms"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_folder(args: argparse.Namespace, config: Config) -> int:
    source = Path(args.source)
    if not source.is_dir():
        raise ValueError(f"not a directory: {source}")
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"no images in {source}")
    output_dir = Path(args.output_dir) if args.output_dir else _default_folder_output(config, source)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = _events_path(args, config, source.name, is_image=True)
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")
    pipeline = DMSPipeline(config, mode="image")
    pipeline.load()
    roi = parse_roi(args.roi) if args.roi else None
    latencies = LatencyAccumulator()
    event_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    failures = 0
    for index, path in enumerate(files):
        frame = imread(path)
        if frame is None:
            failures += 1
            continue
        pipeline.reset()
        processed = pipeline.resize(frame)
        hint = scale_roi(roi, frame, processed) if roi else None
        started = time.perf_counter()
        result = pipeline.process(processed, timestamp=time.time() + index / 5.0, frame_index=index, roi_hint=hint)
        latencies.add((time.perf_counter() - started) * 1000.0)
        for event in result["events"]:
            event_counts[event["kind"]] += 1
        if events_path is not None:
            with events_path.open("a", encoding="utf-8") as handle:
                for event in result["events"]:
                    append_jsonl(handle, event)
        out_path = output_dir / f"{path.stem}_annotated.jpg"
        if not args.no_annotated:
            annotated = draw_result(processed, result, face=pipeline.face.last_observation)
            imwrite(out_path, annotated)
        if not args.no_json:
            write_json(output_dir / f"{path.stem}.json", result)
        records.append(
            {
                "source": str(path),
                "face": bool(result["face"].get("present")),
                "states": {name: (value or {}).get("state") for name, value in result["features"].items()},
                "events": [event["kind"] for event in result["events"]],
            }
        )
        if not args.quiet and (index + 1) % 20 == 0:
            print(f"processed {index + 1}/{len(files)}", flush=True)
    pipeline.close()
    summary = {
        "mode": "folder",
        "source": str(source),
        "images": len(files),
        "failed_reads": failures,
        "output_dir": str(output_dir),
        "events": None if events_path is None else str(events_path),
        "event_counts": dict(event_counts),
        "latency": latencies.summary(),
    }
    write_json(output_dir / "summary.json", {**summary, "records": records})
    print(json.dumps(summary, indent=2))
    return 0


def run_video(args: argparse.Namespace, config: Config) -> int:
    source = Path(args.source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (0.1 <= fps <= 240.0):
        fps = float(config.get("video.fps_fallback", 30.0))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("video reported invalid dimensions")
    pipeline = DMSPipeline(config, mode="video")
    pipeline.load()
    roi = parse_roi(args.roi) if args.roi else None
    writer = None
    output_path = Path(args.output) if args.output else None
    events_path = _events_path(args, config, source.stem, is_image=False)
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")
    latencies = LatencyAccumulator()
    event_counts: Counter[str] = Counter()
    wall_start = time.time()
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.max_frames and index >= args.max_frames:
            break
        processed = pipeline.resize(frame)
        hint = scale_roi(roi, frame, processed) if roi else None
        timestamp = wall_start + index / max(fps, 0.1)
        started = time.perf_counter()
        result = pipeline.process(processed, timestamp=timestamp, frame_index=index, roi_hint=hint)
        latencies.add((time.perf_counter() - started) * 1000.0)
        if not args.no_annotated:
            annotated = draw_result(processed, result, face=pipeline.face.last_observation)
            if output_path is not None:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                    if not writer.isOpened():
                        raise OSError(f"could not open video writer: {output_path}")
                writer.write(annotated)
            if args.display:
                cv2.imshow("DMS video (q to quit)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
        if events_path is not None and result["events"]:
            with events_path.open("a", encoding="utf-8") as handle:
                for event in result["events"]:
                    append_jsonl(handle, event)
        for event in result["events"]:
            event_counts[event["kind"]] += 1
        index += 1
        if not args.quiet and index % 30 == 0:
            print(f"frame {index} states={ {k: (v or {}).get('state') for k, v in result['features'].items()} }", flush=True)
    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    pipeline.close()
    summary = {
        "mode": "video",
        "source": str(source),
        "frames": index,
        "fps_source": round(fps, 3),
        "output": None if output_path is None or args.no_annotated else str(output_path),
        "events": None if events_path is None else str(events_path),
        "event_counts": dict(event_counts),
        "latency": latencies.summary(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_webcam(args: argparse.Namespace, config: Config) -> int:
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(args.camera)
    else:
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise ValueError(f"could not open camera {args.camera}")
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    pipeline = DMSPipeline(config, mode="webcam")
    pipeline.load()
    roi = parse_roi(args.roi) if args.roi else None
    writer = None
    output_path = Path(args.output) if args.output else None
    events_path = _events_path(args, config, "webcam", is_image=False)
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text("", encoding="utf-8")
    latencies = LatencyAccumulator()
    event_counts: Counter[str] = Counter()
    index = 0
    wall_start = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.max_frames and index >= args.max_frames:
            break
        processed = pipeline.resize(frame)
        hint = scale_roi(roi, frame, processed) if roi else None
        result = pipeline.process(
            processed,
            timestamp=wall_start + (time.time() - wall_start),
            frame_index=index,
            roi_hint=hint,
        )
        latencies.add(float(result["timing_ms"]["total"]))
        if not args.no_annotated:
            annotated = draw_result(processed, result, face=pipeline.face.last_observation)
            if output_path is not None:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(args.fps),
                        (annotated.shape[1], annotated.shape[0]),
                    )
                    if not writer.isOpened():
                        raise OSError(f"could not open video writer: {output_path}")
                writer.write(annotated)
            if args.display:
                cv2.imshow("DMS webcam (q to quit)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
        if events_path is not None and result["events"]:
            with events_path.open("a", encoding="utf-8") as handle:
                for event in result["events"]:
                    append_jsonl(handle, event)
        for event in result["events"]:
            event_counts[event["kind"]] += 1
        index += 1
        if not args.quiet and index % 30 == 0:
            print(f"frame {index} latency {result['timing_ms']['total']:.1f} ms", flush=True)
    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    pipeline.close()
    summary = {
        "mode": "webcam",
        "frames": index,
        "output": None if output_path is None or args.no_annotated else str(output_path),
        "events": None if events_path is None else str(events_path),
        "event_counts": dict(event_counts),
        "latency": latencies.summary(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def parse_roi(text: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2")
    roi = tuple(int(float(v)) for v in parts)
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("ROI has non-positive width/height")
    return roi  # type: ignore[return-value]


def scale_roi(
    roi: tuple[int, int, int, int],
    original: Any,
    processed: Any,
) -> tuple[int, int, int, int]:
    oh, ow = original.shape[:2]
    ph, pw = processed.shape[:2]
    if (ow, oh) == (pw, ph):
        return roi
    sx = pw / max(ow, 1)
    sy = ph / max(oh, 1)
    x1, y1, x2, y2 = roi
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _default_image_output(config: Config, source: Path) -> Path:
    base = config.resolve_path(config.get("pipeline.output_dir", "results")) / "annotated"
    return base / f"{source.stem}_annotated.jpg"


def _default_folder_output(config: Config, source: Path) -> Path:
    return config.resolve_path(config.get("pipeline.output_dir", "results")) / "annotated" / source.name


def _events_path(args: argparse.Namespace, config: Config, stem: str, is_image: bool) -> Path | None:
    if args.events:
        return Path(args.events)
    if not bool(config.get("pipeline.save_events_jsonl", True)):
        return None
    base = config.resolve_path(config.get("pipeline.output_dir", "results"))
    return base / ("events" if not is_image else "events_images") / f"{stem}.jsonl"


def _write_events(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        for event in result.get("events", []):
            append_jsonl(handle, event)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
