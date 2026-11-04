"""Command-line application: capture loop, overlay and alert logging."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from .alerts import CsvAlertLog, console_sink
from .audio import AudioAlerter
from .config import AppConfig
from .pipeline import Analyzer, FrameObservation, Perception
from .state import summarize

_SEVERITY_COLOURS = {
    "INFO": (205, 205, 205),
    "MINOR": (120, 210, 255),
    "MAJOR": (60, 170, 255),
    "CRITICAL": (70, 70, 255),
}

# Detector boxes older than this are drawn grey and labelled with their age:
# on the Pi a pass takes 12-16 s, and a green box drawn on a live frame reads
# as "the model sees the phone over there right now" when it does not.
_STALE_DETECTION_S = 1.5


class PicameraCapture:
    """CSI-camera frames through libcamera/picamera2.

    Raspberry Pi OS drives the ribbon camera (e.g. the OV5647 night-vision
    module) through libcamera; a plain V4L2 open succeeds but the sensor
    never streams. This wrapper serves the same read()/get() interface as
    cv2.VideoCapture, in BGR to match the rest of the pipeline.
    """

    def __init__(
        self, width: int, height: int, fps: float, controls: dict | None = None
    ):
        from picamera2 import Picamera2  # lazy: only the Pi uses this source

        self._cam = Picamera2()
        # RGB888 is the ISP's native format; asking libcamera for BGR888
        # makes it run a converter that some Pi 3 pipelines do not expose.
        config = self._cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.set_controls({"FrameRate": float(fps)})
        self._cam.start()
        if controls:
            # libcamera wants tuples for array controls; JSON only has lists.
            self._cam.set_controls(
                {
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in controls.items()
                }
            )
        self.fps = float(fps)

    def read(self):
        try:
            # A camera/CSI fault can leave capture_array blocked forever, which
            # also defeats --max-seconds and keeps the camera locked after the
            # run. Bounded wait turns that into a normal "camera stopped"
            # exit so the process releases the device.
            frame = self._cam.capture_array("main", wait=2.0)
        except (RuntimeError, TimeoutError):
            return False, None
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def get(self, prop: int) -> float:
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop: int, value: float) -> bool:
        return False

    def release(self) -> None:
        self._cam.stop()
        self._cam.close()


def _open_capture(spec: str, cfg: AppConfig) -> cv2.VideoCapture | PicameraCapture:
    if spec == "picamera":
        return PicameraCapture(
            cfg.capture_width, cfg.capture_height, cfg.target_fps, cfg.camera.controls
        )
    if spec.isdigit():
        index = int(spec)
        # The default Windows backend (MSMF) frequently opens but never
        # delivers a frame; DirectShow is the reliable path there.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            raise SystemExit(f"could not open camera index {spec}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
        cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if not Path(spec).exists():
        raise SystemExit(f"video source not found: {spec}")
    cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {spec}")
    return cap


def _read_frame(cap: cv2.VideoCapture, timeout_s: float = 0.6):
    """Grab a frame, tolerating the warm-up failures a webcam often returns.

    USB cameras (especially on the Pi) routinely take a second or two after
    opening before the first frame arrives, so the startup call passes a
    wider window than the per-frame watchdog.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        ok, frame = cap.read()
        if ok:
            return True, frame
        if time.monotonic() >= deadline:
            return False, None
        time.sleep(0.03)


class AnnotatedRecorder:
    """Write the preview frame (boxes, state, warnings) to an MP4 file."""

    def __init__(self, path: str, fps: float):
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 1.0 else 25.0
        self._writer: cv2.VideoWriter | None = None
        self._failed = False

    def write(self, frame) -> None:
        if self._failed:
            return
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.path),
                fourcc,
                self.fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not self._writer.isOpened():
                print(
                    f"warning: could not open video writer for {self.path}",
                    flush=True,
                )
                self._writer = None
                self._failed = True
                return
            print(f"recording: {self.path}", flush=True)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _read_rss_mb() -> float | None:
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


def _status_line(
    observation: FrameObservation,
    analyzer: Analyzer,
    fps: float,
    perception: Perception | None,
) -> str:
    """One line of every live value, for runs without a window (the Pi)."""
    face = observation.face
    gaze = analyzer.distraction
    ref = (
        f"ref {gaze.neutral_pose[0]:6.1f}/{gaze.neutral_pose[1]:6.1f}"
        if gaze.calibrated
        else "ref calibrating"
    )
    found = (
        " ".join(f"{d.label}:{d.confidence:.2f}" for d in observation.detections)
        or "none"
    )
    state_text, _ = summarize(face, analyzer.active_signals)
    blink_rate = analyzer.drowsiness.blink_rate(observation.timestamp)
    parts = [
        f"[status] t={observation.timestamp:6.1f}s fps={fps:4.1f}",
        f"state={state_text.replace(' - ', ': ')}",
        f"face={'yes' if face.present else 'no':>3}",
        f"ear={face.ear:5.3f} mar={face.mar:5.3f}",
        f"yaw={face.yaw:6.1f} pitch={face.pitch:6.1f} roll={face.roll:6.1f}",
        ref,
        f"det={found}",
    ]
    if math.isfinite(blink_rate):
        parts.insert(4, f"blinks={blink_rate:3.0f}/min")
    if perception is not None and perception.workers:
        stats = " ".join(
            f"{worker.name}={worker.last_latency_ms:.0f}ms"
            + (
                f"/wait{worker.last_wait_ms:.0f}ms"
                if perception.cfg.serialize_detectors
                else ""
            )
            for worker in perception.workers
        )
        parts.append(f"detectors={stats}")
        if perception.detectors_paused:
            parts.append("detectors_paused=yes")
        if observation.det_age is not None:
            stale = " STALE" if observation.det_age > _STALE_DETECTION_S else ""
            parts.append(f"det_age={observation.det_age:.1f}s{stale}")
        errors = [
            f"{worker.name}: {worker.last_error}"
            for worker in perception.workers
            if worker.last_error
        ]
        if errors:
            parts.append("DETECTOR_ERROR=" + " | ".join(errors))
    return " ".join(parts)


def _display_works() -> bool:
    """Check highgui in a subprocess first.

    On the Pi, OpenCV's Qt backend can block forever inside `imshow` when
    DISPLAY points at an XWayland socket the SSH session cannot use. A probe
    with a hard timeout turns that hang into a clean "run headless" fallback.
    """
    script = (
        "import cv2, numpy as np; "
        "cv2.imshow('probe', np.zeros((64, 64, 3), np.uint8)); "
        "cv2.waitKey(50); cv2.destroyAllWindows()"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            # The first `import cv2` alone can take several seconds on a
            # Pi 3; the timeout is for a genuine highgui hang, not startup.
            timeout=12.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _power_note() -> str | None:
    """A Pi on a weak supply throttles, which doubles every model's latency."""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out or "throttled=0x0" in out:
        return None
    return (
        f"the Pi reports under-voltage/throttling ({out}); inference runs at a "
        "reduced clock — use a 5 V 2.5 A supply and a short cable, then check "
        "that `vcgencmd get_throttled` reads 0x0"
    )


_FONT_STATE = cv2.FONT_HERSHEY_DUPLEX
_FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def _corner_box(frame, box, colour, length: int = 18, thickness: int = 2) -> None:
    """Draw four corner brackets instead of a full face rectangle."""
    x1, y1, x2, y2 = (int(v) for v in box)
    for x, y, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(
            frame, (x, y), (x + dx * length, y), colour, thickness, cv2.LINE_AA
        )
        cv2.line(
            frame, (x, y), (x, y + dy * length), colour, thickness, cv2.LINE_AA
        )


def _tag(frame, text: str, x: int, y: int, colour) -> None:
    """Compact filled label with its bottom-left at (x, y)."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT_SMALL, 0.45, 1)
    top = max(0, y - th - baseline - 6)
    right = min(frame.shape[1] - 1, x + tw + 8)
    cv2.rectangle(frame, (x, top), (right, y), colour, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        _FONT_SMALL,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _state_pill(frame, text: str, colour) -> None:
    scale = 0.8
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, _FONT_STATE, scale, thickness)
    x, y = 12, 12
    pad_x, pad_y = 14, 9
    right = x + pad_x * 2 + tw + 10
    bottom = y + pad_y * 2 + th + baseline
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (right, bottom), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (x, y), (x + 6, bottom), colour, -1)
    cv2.putText(
        frame,
        text,
        (x + pad_x + 6, y + pad_y + th),
        _FONT_STATE,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _detection_colour(label: str) -> tuple[int, int, int]:
    if label == "cigarette":
        return (60, 60, 255)
    if label in ("phone", "cell phone"):
        return (0, 170, 255)
    return (80, 220, 80)


def _draw(
    frame,
    observation: FrameObservation,
    analyzer: Analyzer,
    fps: float,
    label: str,
) -> None:
    height, width = frame.shape[:2]
    face = observation.face

    # Skip stale detector boxes instead of painting an old pass on a live
    # frame. On a Pi one pass can be several seconds old.
    stale = observation.det_age is not None and observation.det_age > _STALE_DETECTION_S
    if not stale:
        for detection in observation.detections:
            # Only draw cigarette boxes that could actually contribute to an
            # alert. A weak box on a chair or a hand is not a cigarette and
            # must not paint a red label on a clean preview.
            if detection.label == "cigarette" and not analyzer.smoking.candidate(
                detection, face
            ):
                continue
            x1, y1, x2, y2 = detection.box
            colour = _detection_colour(detection.label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
            _tag(
                frame,
                f"{detection.label} {detection.confidence:.2f}",
                x1,
                max(14, y1 - 2),
                colour,
            )

    # Corner brackets keep the face itself unobstructed.
    if face.bbox is not None:
        _corner_box(frame, face.bbox, (255, 190, 0))

    state_text, state_colour = summarize(face, analyzer.active_signals)
    _state_pill(frame, state_text, state_colour)

    # Live eye/mouth measurements. These are the values drowsiness uses, so
    # they stay on screen instead of being buried in the headless status line.
    if face.present and math.isfinite(face.ear):
        metrics = f"EAR {face.ear:.3f}   MAR {face.mar:.2f}"
        cv2.putText(
            frame,
            metrics,
            (14, 86),
            _FONT_SMALL,
            0.45,
            (25, 25, 25),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            metrics,
            (14, 86),
            _FONT_SMALL,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    meta = f"{label}  {fps:4.1f} fps"
    (tw, th), _ = cv2.getTextSize(meta, _FONT_SMALL, 0.45, 1)
    cv2.putText(
        frame,
        meta,
        (width - tw - 12, 24),
        _FONT_SMALL,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Active warnings along the bottom, newest closest to the edge.
    messages = analyzer.active_messages()[-3:]
    y = height - 10
    for signal in reversed(messages):
        text = f"{signal.severity.name}: {signal.message}"
        (tw, th), baseline = cv2.getTextSize(text, _FONT_SMALL, 0.5, 1)
        top = y - th - baseline - 8
        cv2.rectangle(frame, (8, top), (16 + tw, y + 2), (18, 18, 18), -1)
        cv2.putText(
            frame,
            text,
            (12, y - 3),
            _FONT_SMALL,
            0.5,
            _SEVERITY_COLOURS.get(signal.severity.name, (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )
        y = top - 6


def run(
    cfg: AppConfig,
    source: str,
    max_seconds: float | None = None,
    realtime: bool = False,
    loop: bool = False,
    record: str | None = None,
) -> int:
    sinks = [console_sink]
    log = CsvAlertLog(cfg.log_path) if cfg.log_path else None
    if log is not None:
        sinks.append(log)
    audio: AudioAlerter | None = None
    if cfg.audio.enabled:
        audio = AudioAlerter(cfg.audio.wav_dir, device=cfg.audio.device)
        if audio.available:
            sinks.append(audio)
            if cfg.audio.announce_startup:
                audio.announce_startup()
        else:
            print("note: voice alerts unavailable on this machine", flush=True)

    perception: Perception | None = None
    analyzer: Analyzer | None = None
    cap: cv2.VideoCapture | PicameraCapture | None = None
    recorder: AnnotatedRecorder | None = None
    show = cfg.display
    if show and sys.platform != "win32" and not _display_works():
        print(
            "note: no usable display from this session; running headless "
            "(launch from the Pi desktop for the preview window)",
            flush=True,
        )
        show = False
    emitted: Counter[str] = Counter()
    frame_count = 0
    fps = 0.0

    try:
        perception = Perception(cfg)
        if cfg.self_test:
            for line in perception.self_test():
                print(f"self-test: {line}", flush=True)
        perception.start()
        cap = _open_capture(source, cfg)
        labels = perception.labels
        label = f"source {source}"

        analyzer = Analyzer(cfg, detector_labels=labels, sinks=sinks)
        for note in analyzer.notes:
            print(f"note: {note}", flush=True)
        power = _power_note()
        if power:
            print(f"note: {power}", flush=True)

        fps_time = time.monotonic()
        fps_frames = 0
        status_time = time.monotonic()
        wall_start = time.monotonic()
        source_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0.0
        frame_period = 1.0 / source_fps if realtime and source_fps > 1.0 else 0.0
        live_source = source.isdigit() or source == "picamera"
        if record:
            recorder = AnnotatedRecorder(
                record, source_fps if source_fps > 1.0 else cfg.target_fps
            )

        while True:
            loop_started = time.monotonic()
            ok, frame = _read_frame(
                cap,
                timeout_s=(
                    (5.0 if frame_count == 0 else 2.0) if live_source else 0.0
                ),
            )
            if not ok:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    wall_start = time.monotonic()
                    continue
                print("camera stopped delivering frames", flush=True)
                if source.isdigit():
                    print(
                        "hint: for the Pi ribbon (CSI) camera use "
                        "--source picamera; for a USB camera check the "
                        "cable/power and ls /dev/video*",
                        flush=True,
                    )
                break
            if frame.shape[1] > cfg.capture_width:
                # Cameras often ignore the requested size; running the
                # detectors at 720p costs ~3x for no accuracy gain.
                scale = cfg.capture_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (cfg.capture_width, int(round(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            observation = perception.analyze(frame, time.monotonic() - wall_start)

            alerts = analyzer.update(observation)
            emitted.update(alert.key for alert in alerts)
            if cfg.face_first and perception is not None:
                # Eyes closed or no face: stop starting YOLO passes and give
                # the CPU to the next face frame. A pass already in flight
                # finishes, but the next one waits until the face is back.
                perception.set_detectors_paused(
                    not observation.face.present
                    or analyzer.drowsiness.eyes_closed
                )
            frame_count += 1

            if audio is not None and not any(
                s.active for s in analyzer.active_signals
            ):
                # Conditions cleared: drop queued phrases so stale warnings
                # do not keep playing after the object is gone.
                audio.flush()

            fps_frames += 1
            elapsed = time.monotonic() - fps_time
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_time = time.monotonic()
                fps_frames = 0

            if not show and time.monotonic() - status_time >= 1.0:
                status_time = time.monotonic()
                print(
                    _status_line(observation, analyzer, fps, perception),
                    flush=True,
                )

            if max_seconds is not None:
                if time.monotonic() - wall_start >= max_seconds:
                    break

            if show or recorder is not None:
                _draw(frame, observation, analyzer, fps, label)
            if recorder is not None:
                recorder.write(frame)
            if show:
                try:
                    cv2.imshow("drivermon", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                    # Closing the window must end the run as well, otherwise
                    # the process keeps the camera open with no window and
                    # blocks every other app from using it.
                    if cv2.getWindowProperty("drivermon", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error as exc:
                    print(f"display disabled: {exc}", flush=True)
                    show = False

            if frame_period:
                # Play video files at their native rate instead of as fast as
                # the detectors allow, so a preview looks like footage.
                slack = frame_period - (time.monotonic() - loop_started)
                if slack > 0:
                    time.sleep(slack)

        rss = _read_rss_mb()
        print("\n--- summary ---")
        print(f"frames processed: {frame_count}")
        if rss is not None:
            print(f"resident memory: {rss:.1f} MB")
        if emitted:
            for key, count in emitted.most_common():
                print(f"  {key}: {count}")
        else:
            print("  no alerts fired")
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        return 130
    finally:
        if show:
            cv2.destroyAllWindows()
        if recorder is not None:
            recorder.close()
        if analyzer is not None:
            analyzer.close()
        if perception is not None:
            perception.close()
        if cap is not None:
            cap.release()
        if audio is not None:
            audio.close()
        if log is not None:
            log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drivermon", description="On-device driver monitoring prototype"
    )
    parser.add_argument(
        "--source",
        default="0",
        help='camera index ("0"), "picamera" (Pi CSI camera), or a video file path',
    )
    parser.add_argument("--config", help="path to a JSON config overriding defaults")
    parser.add_argument("--log", help="write alerts to this CSV path")
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="write the annotated preview to an MP4 file",
    )
    parser.add_argument("--headless", action="store_true", help="do not open a window")
    parser.add_argument("--realtime", action="store_true", help="play video files at their native frame rate")
    parser.add_argument("--loop", action="store_true", help="restart a video file when it ends")
    parser.add_argument("--max-seconds", type=float, help="stop after N seconds")
    parser.add_argument(
        "--no-seatbelt",
        action="store_true",
        help="disable the seatbelt rule (desk tests have no belt in view)",
    )
    parser.add_argument("--dump-config", action="store_true", help="print the effective config and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = AppConfig.load(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: could not load config {args.config}: {exc}", flush=True)
        return 2
    if args.log:
        cfg.log_path = args.log
    if args.headless:
        cfg.display = False
    if args.no_seatbelt:
        cfg.thresholds.seatbelt_check = False
        print("note: seatbelt rule disabled (--no-seatbelt)", flush=True)

    if args.dump_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    problems = cfg.check_paths()
    if problems:
        for problem in problems:
            print(f"error: {problem}", flush=True)
        return 2

    return run(
        cfg,
        args.source,
        args.max_seconds,
        realtime=args.realtime,
        loop=args.loop,
        record=args.record,
    )
