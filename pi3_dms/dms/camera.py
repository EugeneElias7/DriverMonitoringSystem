"""Single camera capture path and conservative colour preprocessing.

The camera thread is the only code that opens the camera.  It publishes one
newest frame packet for both the face processor and the optional object worker,
so enabling object detection never opens a second camera.

Colour correction is deliberately conservative and configurable.  A magenta or
purple image is usually a camera white-balance/IR-filter problem, not something
that should be repaired by permanently amplifying colour channels.  The raw
frame is retained in every packet for debugging and for the raw snapshot
endpoint.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class FramePacket:
    stream_bgr: np.ndarray
    process_bgr: np.ndarray
    timestamp: float
    sequence: int
    raw_bgr: np.ndarray | None = None


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize without changing geometry; pad with black if needed."""
    target_w, target_h = int(size[0]), int(size[1])
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return frame
    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if new_w == target_w and new_h == target_h:
        return resized
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def soft_gray_world(
    frame: np.ndarray,
    *,
    strength: float = 0.55,
    clamp: tuple[float, float] = (0.88, 1.12),
) -> np.ndarray:
    """Apply a partial gray-world correction.

    The full gray-world transform is intentionally not used: in a scene with a
    large coloured object (a blue cab interior, a red shirt, an IR illuminator)
    it can make the face colour worse.  The gain is blended toward unity and
    clipped.  This is a display/inference aid, never a colourimetric claim.
    """
    image = frame.astype(np.float32)
    means = image.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    raw_gains = gray / np.maximum(means, 1.0)
    clipped = np.clip(raw_gains, clamp[0], clamp[1])
    gains = 1.0 + (clipped - 1.0) * float(np.clip(strength, 0.0, 1.0))
    return np.clip(image * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def night_gray(frame: np.ndarray, *, clip_limit: float = 2.0) -> np.ndarray:
    """Convert an IR/night frame to grayscale with restrained local contrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(
        clipLimit=max(0.5, float(clip_limit)), tileGridSize=(8, 8)
    ).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if not mode or mode in {"off", "none", "raw"}:
        return frame
    if mode == "night":
        return night_gray(frame)
    if mode == "soft":
        return soft_gray_world(frame)
    if mode == "grayworld":
        return soft_gray_world(frame, strength=1.0, clamp=(0.72, 1.38))
    raise ValueError(f"unknown colour mode: {mode}")


class FrameStore:
    """Bounded newest-frame store shared by all consumers.

    There is no unbounded frame queue.  The processor and object worker both
    read the latest packet; an old frame can therefore never accumulate into a
    latency backlog.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._packet: FramePacket | None = None
        self._camera_info: dict[str, Any] = {}
        self._error: str | None = None
        self._capture_times: deque[float] = deque(maxlen=30)
        self._sequence = 0

    def put(self, packet: FramePacket, camera_info: dict[str, Any] | None = None) -> None:
        with self._changed:
            self._packet = packet
            self._sequence = packet.sequence
            if camera_info is not None:
                self._camera_info = camera_info
            self._error = None
            self._capture_times.append(time.monotonic())
            self._changed.notify_all()

    def set_error(self, message: str) -> None:
        with self._changed:
            self._error = message

    def _fps_locked(self) -> float:
        if len(self._capture_times) < 2:
            return 0.0
        span = self._capture_times[-1] - self._capture_times[0]
        return (len(self._capture_times) - 1) / span if span > 0 else 0.0

    def latest(self) -> tuple[FramePacket | None, dict[str, Any], str | None, float]:
        with self._lock:
            return (
                self._packet,
                dict(self._camera_info),
                self._error,
                self._fps_locked(),
            )

    def latest_raw(self) -> tuple[np.ndarray | None, dict[str, Any]]:
        with self._lock:
            packet = self._packet
            info = dict(self._camera_info)
        if packet is None:
            return None, info
        return packet.raw_bgr if packet.raw_bgr is not None else packet.stream_bgr, info

    def wait_next(
        self, last_sequence: int, timeout: float = 0.5
    ) -> tuple[FramePacket | None, dict[str, Any], str | None, float]:
        with self._changed:
            self._changed.wait_for(
                lambda: self._packet is not None and self._sequence != last_sequence,
                timeout=timeout,
            )
            return (
                self._packet,
                dict(self._camera_info),
                self._error,
                self._fps_locked(),
            )


class CameraWorker(threading.Thread):
    """Open the camera once and publish newest frames to the FrameStore."""

    def __init__(self, args: argparse.Namespace, store: FrameStore):
        super().__init__(name="camera", daemon=True)
        self.args = args
        self.store = store
        self.stop_event = threading.Event()
        self.sequence = 0
        self._info: dict[str, Any] = {
            "backend": "starting",
            "capture_resolution": "-",
            "capture_fps": None,
            "stream_resolution": f"{args.stream_width}x{args.stream_height}",
            "process_resolution": f"{args.process_width}x{args.process_height}",
            "error": None,
            "color_mode": getattr(args, "tint_mode", "off"),
            "awb": getattr(args, "awb", "auto"),
            "camera_controls": "default",
        }

    def stop(self) -> None:
        self.stop_event.set()

    def _publish(self, frame_bgr: np.ndarray) -> None:
        # Keep the unmirrored, uncorrected capture for debugging.  The mirror
        # and resizes used by the UI/inference are separate derived views.
        raw = frame_bgr
        display = cv2.flip(frame_bgr, 1) if self.args.mirror else frame_bgr
        stream_size = (int(self.args.stream_width), int(self.args.stream_height))
        process_size = (int(self.args.process_width), int(self.args.process_height))
        stream = letterbox(display, stream_size)
        process = letterbox(display, process_size)
        self.sequence += 1
        self._info["error"] = None
        self.store.put(
            FramePacket(stream, process, time.monotonic(), self.sequence, raw),
            dict(self._info),
        )

    def run(self) -> None:
        if self.args.image:
            self._run_image()
            return
        if self.args.camera_backend in {"auto", "picamera2"}:
            try:
                self._run_picamera2()
                return
            except Exception as exc:
                if self.args.camera_backend == "picamera2":
                    self._error(f"Picamera2 failed: {type(exc).__name__}: {exc}")
                    return
                self._info["backend"] = f"opencv (Picamera2 unavailable: {type(exc).__name__})"
        self._run_opencv()

    def _error(self, message: str) -> None:
        print(f"[camera] {message}", flush=True)
        self._info["error"] = message
        self.store.set_error(message)
        self.sequence += 1
        empty_stream = np.zeros(
            (self.args.stream_height, self.args.stream_width, 3), dtype=np.uint8
        )
        empty_process = np.zeros(
            (self.args.process_height, self.args.process_width, 3), dtype=np.uint8
        )
        self.store.put(
            FramePacket(
                empty_stream,
                empty_process,
                time.monotonic(),
                self.sequence,
                empty_stream,
            ),
            dict(self._info),
        )

    def _run_image(self) -> None:
        path = Path(self.args.image)
        image = cv2.imread(str(path))
        if image is None:
            self._error(f"cannot read image: {path}")
            return
        self._info.update(
            {
                "backend": "image",
                "capture_resolution": f"{image.shape[1]}x{image.shape[0]}",
                "capture_fps": self.args.camera_fps,
            }
        )
        interval = 1.0 / max(1.0, float(self.args.camera_fps))
        while not self.stop_event.is_set():
            self._publish(image.copy())
            self.stop_event.wait(interval)

    def _run_opencv(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(self.args.camera), backend)
        if not cap.isOpened():
            cap.release()
            self._error(f"camera {self.args.camera} could not be opened")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.args.capture_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.args.capture_height))
        cap.set(cv2.CAP_PROP_FPS, int(self.args.camera_fps))
        if sys.platform.startswith("win"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._info.update(
            {
                "backend": "opencv",
                "capture_resolution": f"{actual_w}x{actual_h}",
                "capture_fps": self.args.camera_fps,
            }
        )
        interval = 1.0 / max(1.0, float(self.args.camera_fps))
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._error("camera returned an empty frame")
                    self.stop_event.wait(2.0)
                    continue
                self._publish(frame)
                elapsed = time.monotonic() - started
                if elapsed < interval:
                    self.stop_event.wait(interval - elapsed)
        finally:
            cap.release()

    def _picamera_controls(self) -> dict[str, Any]:
        controls: dict[str, Any] = {"FrameRate": float(self.args.camera_fps)}
        awb = getattr(self.args, "awb", "auto")
        red = getattr(self.args, "red_gain", None)
        blue = getattr(self.args, "blue_gain", None)
        if awb == "manual" or (red is not None and blue is not None):
            controls["AwbEnable"] = False
            if red is not None and blue is not None:
                controls["ColourGains"] = (float(red), float(blue))
        exposure_us = getattr(self.args, "exposure_us", None)
        if exposure_us is not None and float(exposure_us) > 0:
            controls["AeEnable"] = False
            controls["ExposureTime"] = int(exposure_us)
        analogue_gain = getattr(self.args, "analogue_gain", None)
        if analogue_gain is not None and float(analogue_gain) > 0:
            controls["AnalogueGain"] = float(analogue_gain)
        return controls

    def _run_picamera2(self) -> None:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        capture_size = (int(self.args.capture_width), int(self.args.capture_height))
        stream_size = (int(self.args.stream_width), int(self.args.stream_height))
        process_size = (int(self.args.process_width), int(self.args.process_height))
        controls = self._picamera_controls()

        # Ask the ISP for a full camera resolution main stream and a tiny
        # inference stream.  If the lores combination is unsupported, fall back
        # to one full-resolution stream and resize in software.
        has_lores = True
        try:
            config = picam2.create_video_configuration(
                main={"size": capture_size, "format": "RGB888"},
                lores={"size": process_size, "format": "RGB888"},
                controls=controls,
                buffer_count=3,
            )
            picam2.configure(config)
        except Exception:
            picam2.close()
            picam2 = Picamera2()
            has_lores = False
            config = picam2.create_video_configuration(
                main={"size": capture_size, "format": "RGB888"},
                controls={"FrameRate": float(self.args.camera_fps)},
                buffer_count=3,
            )
            picam2.configure(config)
            self._info["camera_controls"] = "fallback: requested colour controls unavailable"
        try:
            from libcamera import controls as lib_controls

            picam2.set_controls({"AfMode": lib_controls.AfModeEnum.Continuous})
        except Exception:
            pass
        try:
            picam2.set_controls(controls)
        except Exception as exc:
            self._info["camera_controls"] = f"controls warning: {type(exc).__name__}: {exc}"
        picam2.start()
        self._info.update(
            {
                "backend": "picamera2",
                "capture_resolution": f"{capture_size[0]}x{capture_size[1]}",
                "capture_fps": self.args.camera_fps,
            }
        )
        try:
            while not self.stop_event.is_set():
                request = picam2.capture_request()
                try:
                    main = request.make_array("main")
                    lores = request.make_array("lores") if has_lores else None
                finally:
                    request.release()
                if main.ndim == 2:
                    raw = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
                else:
                    raw = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
                display = cv2.flip(raw, 1) if self.args.mirror else raw
                stream = letterbox(display, stream_size)
                if has_lores and lores is not None:
                    process = (
                        cv2.cvtColor(lores, cv2.COLOR_RGB2BGR)
                        if lores.ndim == 3
                        else cv2.cvtColor(lores, cv2.COLOR_GRAY2BGR)
                    )
                    if self.args.mirror:
                        process = cv2.flip(process, 1)
                else:
                    process = letterbox(display, process_size)
                self._info["error"] = None
                self.sequence += 1
                self.store.put(
                    FramePacket(stream, process, time.monotonic(), self.sequence, raw),
                    dict(self._info),
                )
        finally:
            picam2.stop()
            picam2.close()
