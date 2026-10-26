"""Raspberry Pi 3 DMS core: face, eyes, mouth and head orientation.

This file is intentionally self-contained. It needs exactly one model:
``models/face_landmarker.task``.

It runs on a laptop webcam for development and on a Pi 3 with a CSI camera
through Picamera2. The camera may be a 1080p module: the sensor stream is
downscaled to a small inference frame while the browser stream stays large
enough for crisp overlay text.

This is a research/demo application, not a safety or medical device.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import statistics
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "face_landmarker.task"

# MediaPipe Face Mesh indices.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
FACE_LEFT = 33
FACE_RIGHT = 263
NOSE_TIP = 1

# Colours are BGR.
GREEN = (90, 220, 105)
YELLOW = (40, 210, 240)
ORANGE = (35, 150, 245)
RED = (60, 60, 235)
CYAN = (220, 210, 60)
WHITE = (245, 245, 245)
DARK = (13, 18, 23)
MUTED = (165, 180, 195)

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi 3 DMS - Core</title>
<style>
  :root { color-scheme:dark; --bg:#0b1117; --panel:#141e28; --line:#2a3a49;
          --text:#edf4f9; --muted:#98a9b8; --good:#58d68d; --warn:#ffc857;
          --bad:#ff6b6b; --accent:#59c3ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 "Segoe UI", system-ui, sans-serif; }
  header { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
           padding:12px 16px; border-bottom:1px solid var(--line); }
  header h1 { font-size:19px; margin:0; letter-spacing:.2px; }
  .pill { border:1px solid var(--line); border-radius:999px; padding:3px 9px;
          font-size:12px; color:var(--muted); }
  .pill.good { color:var(--good); border-color:#2b6b46; }
  .pill.warn { color:var(--warn); border-color:#725d2b; }
  .pill.bad { color:var(--bad); border-color:#7a3535; }
  main { max-width:1600px; margin:auto; padding:14px; display:grid;
         grid-template-columns:minmax(0,1.7fr) minmax(330px,1fr); gap:14px; }
  .video { background:#000; border:1px solid var(--line); border-radius:12px;
           overflow:hidden; display:flex; align-items:center; }
  .video img { width:100%; height:auto; display:block; }
  .side { display:flex; flex-direction:column; gap:9px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:11px; padding:11px 13px; }
  .card h2 { margin:0 0 6px; color:var(--muted); font-size:11px;
             letter-spacing:.10em; text-transform:uppercase; }
  .big { font-size:25px; line-height:1.2; font-weight:700; }
  .reason { color:var(--muted); font-size:13px; min-height:19px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
  .metrics { display:grid; grid-template-columns:1fr auto; gap:2px 12px;
             font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .metrics b { color:var(--text); font-weight:600; }
  .row { display:flex; justify-content:space-between; gap:10px;
         font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .row b { color:var(--text); font-weight:600; }
  .good { color:var(--good); } .warn { color:var(--warn); }
  .bad { color:var(--bad); }
  .note { font-size:12px; color:var(--muted); }
  .event { border-left:3px solid var(--warn); padding-left:8px; }
  footer { max-width:1600px; margin:auto; padding:2px 16px 22px;
           color:var(--muted); font-size:11px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Pi 3 DMS - Core</h1>
  <span class="pill" id="mode">starting...</span>
  <span class="pill" id="camera">camera</span>
  <span class="note">face + eyes + mouth + head orientation; one model</span>
</header>
<main>
  <section class="video"><img src="/video_feed" alt="annotated camera"></section>
  <section class="side">
    <div class="card">
      <h2>Driver state</h2>
      <div class="big" id="main_state">Starting...</div>
      <div class="reason" id="main_reason">Loading the face model.</div>
      <div style="margin-top:7px"><span class="pill" id="tracking">tracking uncertain</span></div>
    </div>
    <div class="grid2">
      <div class="card">
        <h2>Eyes</h2>
        <div class="big" id="eyes">-</div>
        <div class="metrics">
          <span>EAR raw</span><b id="ear_raw">-</b>
          <span>EAR smoothed</span><b id="ear_smoothed">-</b>
          <span>Closed threshold</span><b id="threshold">-</b>
          <span>Closure</span><b id="closure">-</b>
          <span>Blinks</span><b id="blinks">-</b>
          <span>Microsleeps</span><b id="microsleeps">-</b>
        </div>
      </div>
      <div class="card">
        <h2>Mouth / head</h2>
        <div class="big" id="mouth">-</div>
        <div class="metrics">
          <span>MAR raw / smooth</span><b id="mar">-</b>
          <span>Mouth open</span><b id="yawn_ms">-</b>
          <span>Yawns</span><b id="yawns">-</b>
          <span>Head</span><b id="head">-</b>
          <span>Yaw / pitch / roll</span><b id="pose">-</b>
          <span>Head away</span><b id="away">-</b>
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Fatigue index (measurement only)</h2>
      <div class="row"><span>PERCLOS, 60 s</span><b id="perclos">-</b></div>
      <div class="row"><span>Window</span><b id="perclos_window">-</b></div>
      <div class="note">Shown separately. DROWSY is only raised for a current sustained closure.</div>
    </div>
    <div class="card">
      <h2>Performance</h2>
      <div class="row"><span>Processed FPS</span><b id="fps">-</b></div>
      <div class="row"><span>Process latency</span><b id="latency">-</b></div>
      <div class="row"><span>Capture-to-process age</span><b id="age">-</b></div>
      <div class="row"><span>Camera</span><b id="camera_detail">-</b></div>
      <div class="row"><span>Stream / inference</span><b id="resolution">-</b></div>
      <div class="row"><span>Purple / night</span><b id="tint">-</b></div>
    </div>
    <div class="card">
      <h2>Latest event</h2>
      <div class="event reason" id="event">No event yet.</div>
    </div>
    <div class="card">
      <h2>Pi 3 scope</h2>
      <div class="note">
        This build runs the face/eye/mouth/head core only. Smoking, phone and
        drinking models are not run on the Pi 3 because they are too heavy for
        real-time use and are not validated DMS classes.
      </div>
    </div>
  </section>
</main>
<footer>
  Research prototype. Not a safety device. No accuracy, precision, recall or
  fatigue-diagnosis claim is made from unlabelled camera video. Keep faces and
  recordings local and obtain consent.
</footer>
<script>
const $ = (id) => document.getElementById(id);
const n = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : Number(v);
const f = (v, d=1, suffix="") => (v === null || v === undefined || Number.isNaN(Number(v))) ? "-" : Number(v).toFixed(d) + suffix;
function setState(el, value) {
  el.textContent = value;
  const bad = ["DROWSY", "DISTRACTED"];
  const warn = ["EYES CLOSED", "YAWNING", "GLANCE", "TRACKING UNCERTAIN"];
  el.className = "big " + (bad.includes(value) ? "bad" : (warn.includes(value) ? "warn" : "good"));
}
async function refresh() {
  try {
    const s = await (await fetch("/status", {cache:"no-store"})).json();
    $("mode").textContent = s.mode || "unknown";
    $("mode").className = "pill " + (s.mode === "running" ? "good" : "warn");
    setState($("main_state"), s.main_state || "UNKNOWN");
    $("main_reason").textContent = s.main_reason || "";
    const tr = s.tracking || {};
    $("tracking").textContent = "tracking: " + (tr.state || "uncertain") + " - " + (tr.reason || "");
    $("tracking").className = "pill " + (tr.state === "good" ? "good" : "warn");
    const e = s.eyes || {};
    $("eyes").textContent = e.state || "-";
    $("eyes").className = "big " + (e.state === "closed" ? "warn" : "good");
    $("ear_raw").textContent = f(e.ear_raw, 3);
    $("ear_smoothed").textContent = f(e.ear_smoothed, 3);
    $("threshold").textContent = f(e.threshold, 3);
    $("closure").textContent = f(e.closure_ms, 0, " ms");
    $("blinks").textContent = n(s.blinks);
    $("microsleeps").textContent = n(s.microsleeps);
    const m = s.mouth || {};
    $("mouth").textContent = m.state || "-";
    $("mouth").className = "big " + (m.state === "yawning" ? "warn" : "good");
    $("mar").textContent = f(m.mar_raw, 2) + " / " + f(m.mar_smoothed, 2);
    $("yawn_ms").textContent = f(m.open_ms, 0, " ms");
    $("yawns").textContent = n(s.yawns);
    const h = s.head || {};
    $("head").textContent = h.state || "-";
    $("pose").textContent = f(h.yaw, 1) + " / " + f(h.pitch, 1) + " / " + f(h.roll, 1);
    $("away").textContent = f(h.away_ms, 0, " ms");
    const p = s.fatigue || {};
    $("perclos").textContent = p.perclos === null || p.perclos === undefined ? "-" : f(p.perclos * 100, 1, "%");
    $("perclos_window").textContent = f(p.window_s, 0, " s");
    const perf = s.performance || {};
    $("fps").textContent = f(perf.process_fps, 1);
    $("latency").textContent = f(perf.process_ms, 1, " ms");
    $("age").textContent = f(perf.frame_age_ms, 1, " ms");
    const cam = s.camera || {};
    $("camera").textContent = cam.backend || "camera";
    $("camera_detail").textContent = (cam.capture_resolution || "-") + " @ " + f(cam.capture_fps, 1) + " fps";
    $("resolution").textContent = (cam.stream_resolution || "-") + " / " + (cam.process_resolution || "-");
    $("tint").textContent = s.tint || "-";
    $("event").textContent = s.last_event || "No event yet.";
    if (s.camera && s.camera.error) {
      $("camera").className = "pill bad";
      $("camera").textContent = "camera error";
    }
  } catch (err) {
    $("mode").textContent = "server unavailable";
    $("mode").className = "pill bad";
  }
}
refresh();
setInterval(refresh, 300);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Face landmarks and measurements
# ---------------------------------------------------------------------------


@dataclass
class FaceObservation:
    points: np.ndarray
    bbox: tuple[float, float, float, float]
    face_width: float
    ear_left: float
    ear_right: float
    ear: float
    mar: float
    mouth_center: tuple[float, float]
    yaw: float | None
    pitch: float | None
    roll: float | None
    quality: str
    quality_detail: str


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _ear(points: np.ndarray, indices: list[int]) -> float:
    p = np.asarray(points, dtype=np.float64)[indices]
    denominator = 2.0 * _dist(p[0], p[3])
    if denominator <= 1e-6:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / denominator


def _mar(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    width = _dist(p[MOUTH_LEFT], p[MOUTH_RIGHT])
    if width <= 1e-6:
        return 0.0
    return _dist(p[MOUTH_TOP], p[MOUTH_BOTTOM]) / width


def _rotation_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


class FaceModel:
    """MediaPipe Face Landmarker in VIDEO mode."""

    def __init__(
        self,
        model_path: Path,
        *,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"face model not found: {model_path}")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = None
        self._last_timestamp_ms = -1
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=int(num_faces),
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=0.5,
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        *,
        min_face_width_px: float = 40.0,
    ) -> FaceObservation | None:
        if self._mp is None:
            import mediapipe as mp

            self._mp = mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        stamp = max(int(timestamp_s * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = stamp
        result = self._landmarker.detect_for_video(image, stamp)
        if not result.face_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        landmarks = result.face_landmarks[0]
        points = np.asarray(
            [(float(lm.x) * width, float(lm.y) * height) for lm in landmarks],
            dtype=np.float32,
        )
        if points.shape[0] <= max(RIGHT_EYE):
            return None

        face_w = _dist(points[FACE_LEFT], points[FACE_RIGHT])
        x1, y1 = points[:, 0].min(), points[:, 1].min()
        x2, y2 = points[:, 0].max(), points[:, 1].max()
        pad_x = max(2.0, (x2 - x1) * 0.03)
        pad_y = max(2.0, (y2 - y1) * 0.03)
        bbox = (
            float(max(0.0, x1 - pad_x)),
            float(max(0.0, y1 - pad_y)),
            float(min(float(width), x2 + pad_x)),
            float(min(float(height), y2 + pad_y)),
        )

        yaw = pitch = roll = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
            if matrix.shape == (4, 4):
                pitch, yaw, roll = _rotation_to_euler(matrix[:3, :3])

        ear_left = _ear(points, LEFT_EYE)
        ear_right = _ear(points, RIGHT_EYE)
        ear = max(ear_left, ear_right)
        mar = _mar(points)
        quality = "good"
        detail = "face and eye geometry available"
        if face_w < min_face_width_px:
            quality = "too_small"
            detail = "face too small for a precise eye measurement"
        elif not (0.01 <= ear <= 0.9):
            quality = "invalid_ear"
            detail = "eye geometry is not reliable"
        elif yaw is not None and abs(yaw) > 55.0:
            quality = "low_yaw"
            detail = "head turned too far for eye measurement"
        elif pitch is not None and abs(pitch) > 40.0:
            quality = "low_pitch"
            detail = "head angle too steep for eye measurement"

        center = (points[MOUTH_TOP] + points[MOUTH_BOTTOM]) / 2.0
        return FaceObservation(
            points=points,
            bbox=bbox,
            face_width=face_w,
            ear_left=ear_left,
            ear_right=ear_right,
            ear=ear,
            mar=mar,
            mouth_center=(float(center[0]), float(center[1])),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            quality=quality,
            quality_detail=detail,
        )


# ---------------------------------------------------------------------------
# Drowsiness and head state
# ---------------------------------------------------------------------------


class MeasurementEngine:
    """Smooth, threshold and time-limit the face measurements."""

    def __init__(
        self,
        *,
        ear_smoothing_frames: int = 3,
        mar_smoothing_frames: int = 5,
        perclos_window_s: float = 60.0,
        perclos_min_window_s: float = 10.0,
        prolonged_ms: float = 900.0,
        microsleep_ms: float = 1500.0,
        blink_max_ms: float = 500.0,
        yawn_mar: float = 0.60,
        yawn_min_ms: float = 800.0,
        distracted_ms: float = 1500.0,
    ):
        self.ear_history: deque[float] = deque(maxlen=max(1, ear_smoothing_frames))
        self.mar_history: deque[float] = deque(maxlen=max(1, mar_smoothing_frames))
        self.baseline = 0.26
        self.samples = 0
        self.eye_state = "unknown"
        self.closed_since: float | None = None
        self.closure_candidate: float | None = None
        self.closure_ms = 0.0
        self.prolonged_emitted = False
        self.microsleep_emitted = False
        self.yawn_since: float | None = None
        self.yawn_ms = 0.0
        self.yawn_emitted = False
        self.yawn_state = "clear"
        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.perclos: deque[tuple[float, bool]] = deque()
        self.perclos_window_s = perclos_window_s
        self.perclos_min_window_s = perclos_min_window_s
        self.prolonged_ms = prolonged_ms
        self.microsleep_ms = microsleep_ms
        self.blink_max_ms = blink_max_ms
        self.yawn_mar = yawn_mar
        self.yawn_min_ms = yawn_min_ms
        self.distracted_ms = distracted_ms
        self.away_since: float | None = None
        self.away_ms = 0.0
        self.last_state = "unknown"

    def reset_tracking(self) -> None:
        self.ear_history.clear()
        self.mar_history.clear()
        self.eye_state = "unknown"
        self.closed_since = None
        self.closure_candidate = None
        self.closure_ms = 0.0
        self.prolonged_emitted = False
        self.microsleep_emitted = False
        self.yawn_since = None
        self.yawn_ms = 0.0
        self.yawn_emitted = False
        self.yawn_state = "clear"

    def _threshold(self, pitch: float | None) -> float:
        ratio = 0.55 if pitch is not None and pitch > 25.0 else 0.65
        # Do not let a low personal baseline drive the threshold so low that a
        # real closure is missed. 0.15 is a floor for this landmark model; the
        # 0.20 cap prevents a high baseline from calling an open eye closed.
        return min(0.20, max(0.15, self.baseline * ratio))

    def _adapt_baseline(self, ear: float, obs: FaceObservation) -> None:
        if obs.yaw is not None and abs(obs.yaw) > 30.0:
            return
        if obs.pitch is not None and abs(obs.pitch) > 20.0:
            return
        alpha = 0.05 if self.samples < 30 else 0.01
        self.baseline = (1.0 - alpha) * self.baseline + alpha * ear
        self.baseline = min(0.40, max(0.16, self.baseline))
        self.samples += 1

    def _append_perclos(self, timestamp: float, closed: bool) -> None:
        self.perclos.append((timestamp, closed))
        cutoff = timestamp - self.perclos_window_s
        while self.perclos and self.perclos[0][0] < cutoff:
            self.perclos.popleft()

    def _perclos_value(self) -> tuple[float | None, float]:
        if not self.perclos:
            return None, 0.0
        window = self.perclos[-1][0] - self.perclos[0][0]
        if window < self.perclos_min_window_s:
            return None, window
        closed = sum(1 for _, value in self.perclos if value)
        return closed / len(self.perclos), window

    def update(
        self,
        obs: FaceObservation | None,
        timestamp: float,
        *,
        source_frame_width: int,
        source_frame_height: int,
    ) -> dict[str, Any]:
        if obs is None:
            self.reset_tracking()
            self.away_since = None
            self.away_ms = 0.0
            self.last_state = "no_face"
            return self._result(
                state="no_face",
                eye_state="unknown",
                quality="no_face",
                quality_detail="face not detected",
                ear_raw=None,
                ear_smoothed=None,
                threshold=None,
                mar_raw=None,
                mar_smoothed=None,
                head_state="unknown",
                obs=None,
                distraction_state="unknown",
            )

        if obs.quality != "good":
            self.reset_tracking()
            self.away_since = None
            self.away_ms = 0.0
            self.last_state = "uncertain"
            return self._result(
                state="uncertain",
                eye_state="unknown",
                quality=obs.quality,
                quality_detail=obs.quality_detail,
                ear_raw=obs.ear,
                ear_smoothed=None,
                threshold=None,
                mar_raw=obs.mar,
                mar_smoothed=None,
                head_state=_head_state(obs.yaw, obs.pitch),
                obs=obs,
                distraction_state="unknown",
            )

        threshold = self._threshold(obs.pitch)
        raw_ear = float(obs.ear)
        if self.eye_state == "closed" and raw_ear >= threshold:
            # Reopening is never delayed by the closing median.
            self.ear_history.clear()
        self.ear_history.append(raw_ear)
        ear_smoothed = float(statistics.median(self.ear_history))
        if raw_ear <= threshold:
            if self.closure_candidate is None:
                self.closure_candidate = timestamp
        else:
            # Clear the start time as soon as the raw eye is open. Keeping it
            # through a marginal open frame can later report a long closure
            # that never happened.
            self.closure_candidate = None

        self.mar_history.append(float(obs.mar))
        mar_smoothed = float(statistics.median(self.mar_history))
        if self.eye_state == "closed" and raw_ear >= threshold:
            eye_state = "open"
        else:
            eye_state = "closed" if ear_smoothed <= threshold else "open"
        self.eye_state = eye_state

        if eye_state == "closed":
            if self.closed_since is None:
                self.closed_since = self.closure_candidate or timestamp
                self.prolonged_emitted = False
                self.microsleep_emitted = False
            self.closure_ms = max(0.0, (timestamp - self.closed_since) * 1000.0)
            self.yawn_since = None
            self.yawn_ms = 0.0
            self.yawn_emitted = False
            self.yawn_state = "clear"
            self._append_perclos(timestamp, True)
            if self.closure_ms >= self.microsleep_ms:
                if not self.microsleep_emitted:
                    self.microsleeps += 1
                    self.microsleep_emitted = True
                    self.prolonged_emitted = True
                state = "microsleep"
            elif self.closure_ms >= self.prolonged_ms:
                self.prolonged_emitted = True
                state = "prolonged_closure"
            else:
                state = "eyes_closed"
        else:
            if self.closed_since is not None:
                closed_ms = max(0.0, (timestamp - self.closed_since) * 1000.0)
                if closed_ms < self.blink_max_ms:
                    self.blinks += 1
                self.closed_since = None
                self.closure_ms = 0.0
                self.prolonged_emitted = False
                self.microsleep_emitted = False
            self._adapt_baseline(ear_smoothed, obs)
            self._append_perclos(timestamp, False)

            if mar_smoothed >= self.yawn_mar:
                if self.yawn_since is None:
                    self.yawn_since = timestamp
                    self.yawn_emitted = False
                self.yawn_ms = max(0.0, (timestamp - self.yawn_since) * 1000.0)
                if self.yawn_ms >= self.yawn_min_ms:
                    self.yawn_state = "yawning"
                    if not self.yawn_emitted:
                        self.yawns += 1
                        self.yawn_emitted = True
                else:
                    self.yawn_state = "mouth_open"
            else:
                self.yawn_since = None
                self.yawn_ms = 0.0
                self.yawn_emitted = False
                self.yawn_state = "clear"

            state = "yawning" if self.yawn_state == "yawning" else "alert"

        head_state = _head_state(obs.yaw, obs.pitch)
        if head_state == "unknown":
            self.away_since = None
            self.away_ms = 0.0
            distraction_state = "unknown"
        elif head_state == "forward":
            self.away_since = None
            self.away_ms = 0.0
            distraction_state = "forward"
        else:
            if self.away_since is None:
                self.away_since = timestamp
            self.away_ms = max(0.0, (timestamp - self.away_since) * 1000.0)
            distraction_state = "distracted" if self.away_ms >= self.distracted_ms else "glance"

        self.last_state = state
        return self._result(
            state=state,
            eye_state=eye_state,
            quality=obs.quality,
            quality_detail=obs.quality_detail,
            ear_raw=obs.ear,
            ear_smoothed=ear_smoothed,
            threshold=threshold,
            mar_raw=obs.mar,
            mar_smoothed=mar_smoothed,
            head_state=head_state,
            obs=obs,
            distraction_state=distraction_state,
        )

    def _result(
        self,
        *,
        state: str,
        eye_state: str,
        quality: str,
        quality_detail: str,
        ear_raw: float | None,
        ear_smoothed: float | None,
        threshold: float | None,
        mar_raw: float | None,
        mar_smoothed: float | None,
        head_state: str,
        obs: FaceObservation | None,
        distraction_state: str,
    ) -> dict[str, Any]:
        perclos, window = self._perclos_value()
        return {
            "state": state,
            "eye_state": eye_state,
            "quality": quality,
            "quality_detail": quality_detail,
            "ear_raw": ear_raw,
            "ear_smoothed": ear_smoothed,
            "threshold": threshold,
            "closure_ms": self.closure_ms,
            "mar_raw": mar_raw,
            "mar_smoothed": mar_smoothed,
            "mouth_state": self.yawn_state,
            "mouth_open_ms": self.yawn_ms,
            "yawns": self.yawns,
            "blinks": self.blinks,
            "microsleeps": self.microsleeps,
            "head_state": head_state,
            "yaw": None if obs is None else obs.yaw,
            "pitch": None if obs is None else obs.pitch,
            "roll": None if obs is None else obs.roll,
            "away_ms": self.away_ms,
            "distraction_state": distraction_state,
            "perclos": perclos,
            "perclos_window_s": window,
            "face_width": None if obs is None else obs.face_width,
        }


def _head_state(yaw: float | None, pitch: float | None) -> str:
    if yaw is None or pitch is None:
        return "unknown"
    # A small dead zone prevents a camera mounted slightly off-centre from
    # reporting a permanent left/right turn. Normal mirror glances stay inside
    # this band; a clear cabin-facing turn does not.
    if yaw > 30.0:
        return "left"
    if yaw < -30.0:
        return "right"
    if pitch > 25.0:
        return "down"
    if pitch < -20.0:
        return "up"
    return "forward"


def _driver_state(measurements: dict[str, Any]) -> tuple[str, str]:
    tracking = measurements["quality"] == "good" and measurements["eye_state"] in {"open", "closed"}
    if measurements["state"] in {"no_face", "uncertain"} or not tracking:
        reason = measurements["quality_detail"]
        return "TRACKING UNCERTAIN", reason
    if measurements["state"] in {"prolonged_closure", "microsleep"}:
        return (
            "DROWSY",
            f"eyes closed {measurements['closure_ms']:.0f} ms; "
            f"smoothed EAR {measurements['ear_smoothed']:.3f}",
        )
    if measurements["distraction_state"] == "distracted":
        return (
            "DISTRACTED",
            f"head {measurements['head_state']} for {measurements['away_ms']:.0f} ms",
        )
    if measurements["state"] == "yawning":
        return (
            "YAWNING",
            f"mouth open {measurements['mouth_open_ms']:.0f} ms; "
            f"smoothed MAR {measurements['mar_smoothed']:.2f}",
        )
    if measurements["eye_state"] == "closed":
        if measurements["closure_ms"] < 500.0:
            return (
                "BLINK",
                f"brief blink; closure {measurements['closure_ms']:.0f} ms",
            )
        return (
            "EYES CLOSED",
            f"closure {measurements['closure_ms']:.0f} ms; sustained alert not reached",
        )
    if measurements["distraction_state"] == "glance":
        return (
            "GLANCE",
            f"head {measurements['head_state']} for {measurements['away_ms']:.0f} ms",
        )
    return "ATTENTIVE", "face tracked; eyes open; head forward"


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def _text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _panel(frame: np.ndarray, x: int, y: int, w: int, h: int, colour=(9, 14, 19)) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)


def _state_colour(main_state: str) -> tuple[int, int, int]:
    if main_state in {"DROWSY", "DISTRACTED"}:
        return RED
    if main_state in {"EYES CLOSED", "YAWNING", "GLANCE", "TRACKING UNCERTAIN"}:
        return YELLOW
    return GREEN


def annotate(
    frame: np.ndarray,
    *,
    obs: FaceObservation | None,
    measurements: dict[str, Any],
    main_state: str,
    main_reason: str,
    process_size: tuple[int, int],
    process_fps: float,
    process_ms: float,
    frame_age_ms: float,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    proc_w, proc_h = process_size
    sx = width / float(proc_w)
    sy = height / float(proc_h)
    colour = _state_colour(main_state)

    state_scale = 1.02
    reason_scale = 0.59
    state_w, _ = cv2.getTextSize(
        main_state, cv2.FONT_HERSHEY_DUPLEX, state_scale, 2
    )[0]
    reason_w, _ = cv2.getTextSize(
        main_reason, cv2.FONT_HERSHEY_DUPLEX, reason_scale, 1
    )[0]
    panel_max = max(330, width - 300)
    max_reason_w = max(220, panel_max - 52)
    if reason_w > max_reason_w:
        reason_scale *= max_reason_w / float(reason_w)
        reason_w, _ = cv2.getTextSize(
            main_reason, cv2.FONT_HERSHEY_DUPLEX, reason_scale, 1
        )[0]
    panel_w = min(panel_max, max(330, max(state_w, reason_w) + 52))
    _panel(canvas, 18, 18, panel_w, 116)
    cv2.rectangle(canvas, (18, 18), (24, 134), colour, -1)
    _text(canvas, main_state, (38, 61), scale=state_scale, colour=colour, thickness=2)
    _text(canvas, main_reason, (38, 93), scale=reason_scale, colour=WHITE, thickness=1)

    if obs is not None:
        pts = obs.points.copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        p = np.asarray(pts, dtype=np.float64)
        for indices in (LEFT_EYE, RIGHT_EYE):
            curve = np.asarray([p[i] for i in indices], dtype=np.int32)
            cv2.polylines(canvas, [curve.reshape(-1, 1, 2)], True, CYAN, 2, cv2.LINE_AA)
        mouth = np.asarray(
            [p[MOUTH_LEFT], p[MOUTH_TOP], p[MOUTH_RIGHT], p[MOUTH_BOTTOM]],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [mouth.reshape(-1, 1, 2)], True, CYAN, 2, cv2.LINE_AA)
        center = (int(p[MOUTH_TOP][0] + p[MOUTH_BOTTOM][0]) // 2,
                  int(p[MOUTH_TOP][1] + p[MOUTH_BOTTOM][1]) // 2)
        cv2.circle(canvas, center, 3, CYAN, -1, cv2.LINE_AA)

        x1, y1, x2, y2 = [int(round(v)) for v in (
            obs.bbox[0] * sx,
            obs.bbox[1] * sy,
            obs.bbox[2] * sx,
            obs.bbox[3] * sy,
        )]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        if obs.yaw is not None and obs.pitch is not None:
            nose = (int(p[NOSE_TIP][0]), int(p[NOSE_TIP][1]))
            angle = math.radians(obs.yaw)
            tilt = math.radians(-obs.pitch)
            end = (
                int(nose[0] + math.sin(angle) * 70.0),
                int(nose[1] + math.sin(tilt) * 70.0),
            )
            cv2.arrowedLine(canvas, nose, end, colour, 2, cv2.LINE_AA, tipLength=0.18)

    # Small, readable metric line; detailed numbers stay in the HTML panel.
    eye = measurements["eye_state"]
    ear = measurements["ear_smoothed"]
    ear_text = "-" if ear is None else f"{ear:.3f}"
    closure = measurements["closure_ms"]
    metric_text = (
        f"EAR {ear_text}  eye {eye}  closure {closure:.0f} ms  "
        f"head {measurements['head_state']}"
    )
    metric_w, _ = cv2.getTextSize(metric_text, cv2.FONT_HERSHEY_DUPLEX, 0.63, 1)[0]
    _panel(canvas, 18, height - 48, min(width - 36, metric_w + 26), 34)
    _text(
        canvas,
        metric_text,
        (30, height - 24),
        scale=0.63,
        colour=WHITE,
        thickness=1,
    )
    perf_text = f"{process_fps:.1f} fps  {process_ms:.0f} ms  age {frame_age_ms:.0f}"
    perf_w, _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)[0]
    perf_x = width - perf_w - 38
    _panel(canvas, perf_x, 18, perf_w + 24, 42)
    _text(
        canvas,
        perf_text,
        (perf_x + 12, 45),
        scale=0.48,
        colour=WHITE,
        thickness=1,
    )
    return canvas


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------


@dataclass
class FramePacket:
    stream_bgr: np.ndarray
    process_bgr: np.ndarray
    timestamp: float
    sequence: int


def _letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize without changing face geometry; pad with black if needed."""
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


class FrameStore:
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

    def latest(self) -> tuple[FramePacket | None, dict[str, Any], str | None, float]:
        with self._lock:
            packet = self._packet
            info = dict(self._camera_info)
            error = self._error
            fps = 0.0
            if len(self._capture_times) >= 2:
                span = self._capture_times[-1] - self._capture_times[0]
                if span > 0:
                    fps = (len(self._capture_times) - 1) / span
            return packet, info, error, fps

    def wait_next(self, last_sequence: int, timeout: float = 0.5) -> tuple[FramePacket | None, dict[str, Any], str | None, float]:
        with self._changed:
            self._changed.wait_for(
                lambda: self._packet is not None and self._sequence != last_sequence,
                timeout=timeout,
            )
            packet = self._packet
            info = dict(self._camera_info)
            error = self._error
            fps = 0.0
            if len(self._capture_times) >= 2:
                span = self._capture_times[-1] - self._capture_times[0]
                if span > 0:
                    fps = (len(self._capture_times) - 1) / span
            return packet, info, error, fps


class CameraWorker(threading.Thread):
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
        }

    def stop(self) -> None:
        self.stop_event.set()

    def _publish(self, frame_bgr: np.ndarray) -> None:
        if self.args.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        self._info["error"] = None
        stream_size = (int(self.args.stream_width), int(self.args.stream_height))
        process_size = (int(self.args.process_width), int(self.args.process_height))
        stream = _letterbox(frame_bgr, stream_size)
        process = _letterbox(frame_bgr, process_size)
        self.sequence += 1
        self.store.put(
            FramePacket(stream, process, time.monotonic(), self.sequence),
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
        self.store.put(
            FramePacket(
                np.zeros((self.args.stream_height, self.args.stream_width, 3), dtype=np.uint8),
                np.zeros((self.args.process_height, self.args.process_width, 3), dtype=np.uint8),
                time.monotonic(),
                self.sequence,
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

    def _run_picamera2(self) -> None:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        capture_size = (int(self.args.capture_width), int(self.args.capture_height))
        stream_size = (int(self.args.stream_width), int(self.args.stream_height))
        process_size = (int(self.args.process_width), int(self.args.process_height))

        # Ask the ISP for a full camera resolution main stream and a tiny
        # inference stream. If the lores combination is unsupported, fall back
        # to one full-resolution stream and resize in software.
        has_lores = True
        try:
            config = picam2.create_video_configuration(
                main={"size": capture_size, "format": "RGB888"},
                lores={"size": process_size, "format": "RGB888"},
                controls={"FrameRate": float(self.args.camera_fps)},
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
        try:
            from libcamera import controls

            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception:
            pass
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
                    stream = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
                else:
                    stream = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
                if has_lores and lores is not None:
                    process = cv2.cvtColor(lores, cv2.COLOR_RGB2BGR) if lores.ndim == 3 else cv2.cvtColor(lores, cv2.COLOR_GRAY2BGR)
                else:
                    process = _letterbox(stream, process_size)
                if self.args.mirror:
                    stream = cv2.flip(stream, 1)
                    process = cv2.flip(process, 1)
                if (stream.shape[1], stream.shape[0]) != stream_size:
                    stream = _letterbox(stream, stream_size)
                self._info["error"] = None
                self.sequence += 1
                self.store.put(
                    FramePacket(stream, process, time.monotonic(), self.sequence),
                    dict(self._info),
                )
        finally:
            picam2.stop()
            picam2.close()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DMSService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_event = threading.Event()
        self.store = FrameStore()
        self.camera = CameraWorker(args, self.store)
        self.model: FaceModel | None = None
        self.engine = MeasurementEngine(
            ear_smoothing_frames=args.ear_smoothing,
            mar_smoothing_frames=args.mar_smoothing,
            prolonged_ms=args.drowsy_ms,
            microsleep_ms=args.microsleep_ms,
            yawn_mar=args.yawn_mar,
            yawn_min_ms=args.yawn_ms,
            distracted_ms=args.distracted_ms,
        )
        self._lock = threading.Lock()
        self._status: dict[str, Any] = self._initial_status()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._thread: threading.Thread | None = None
        self._process_times: deque[float] = deque(maxlen=40)
        self._last_state: str = "STARTING"
        self._last_event: str | None = None
        self._last_measurements: dict[str, Any] = {}

    def _initial_status(self) -> dict[str, Any]:
        return {
            "mode": "starting",
            "main_state": "STARTING",
            "main_reason": "Loading the face model.",
            "tracking": {"state": "uncertain", "reason": "starting"},
            "eyes": {
                "state": "-",
                "ear_raw": None,
                "ear_smoothed": None,
                "threshold": None,
                "closure_ms": None,
            },
            "mouth": {"state": "-", "mar_raw": None, "mar_smoothed": None, "open_ms": None},
            "head": {"state": "-", "yaw": None, "pitch": None, "roll": None, "away_ms": None},
            "fatigue": {"perclos": None, "window_s": 0.0},
            "blinks": 0,
            "microsleeps": 0,
            "yawns": 0,
            "performance": {"process_fps": None, "process_ms": None, "frame_age_ms": None},
            "camera": {
                "backend": "-",
                "capture_resolution": "-",
                "capture_fps": None,
                "stream_resolution": f"{self.args.stream_width}x{self.args.stream_height}",
                "process_resolution": f"{self.args.process_width}x{self.args.process_height}",
                "error": None,
            },
            "tint": self.args.tint_label,
            "last_event": None,
        }

    def start(self) -> None:
        self.model = FaceModel(
            Path(self.args.model),
            num_faces=1,
            min_detection_confidence=self.args.face_detection_confidence,
            min_tracking_confidence=self.args.face_tracking_confidence,
        )
        self.camera.start()
        self._thread = threading.Thread(target=self._run, name="processor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.camera.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.model is not None:
            self.model.close()

    def _run(self) -> None:
        assert self.model is not None
        last_sequence = -1
        while not self.stop_event.is_set():
            packet, camera_info, camera_error, capture_fps = self.store.wait_next(last_sequence, timeout=0.5)
            if packet is None or packet.sequence == last_sequence:
                continue
            last_sequence = packet.sequence
            started = time.perf_counter()
            try:
                stream_bgr, process_bgr = _preprocess_frames(
                    [packet.stream_bgr, packet.process_bgr],
                    self.args.night,
                    self.args.tint_fix,
                )
                obs = self.model.detect(
                    process_bgr,
                    packet.timestamp,
                    min_face_width_px=self.args.min_face_width,
                )
                now = time.monotonic()
                measurements = self.engine.update(
                    obs,
                    now,
                    source_frame_width=self.args.process_width,
                    source_frame_height=self.args.process_height,
                )
                main_state, main_reason = _driver_state(measurements)
                process_ms = (time.perf_counter() - started) * 1000.0
                frame_age_ms = (now - packet.timestamp) * 1000.0
                self._process_times.append(now)
                fps = 0.0
                if len(self._process_times) >= 2:
                    span = self._process_times[-1] - self._process_times[0]
                    if span > 0:
                        fps = (len(self._process_times) - 1) / span

                if main_state != self._last_state and main_state in {
                    "DROWSY",
                    "EYES CLOSED",
                    "YAWNING",
                    "DISTRACTED",
                    "TRACKING UNCERTAIN",
                }:
                    stamp = datetime.now().strftime("%H:%M:%S")
                    self._last_event = f"{stamp}  {main_state}: {main_reason}"
                self._last_state = main_state

                annotated = annotate(
                    stream_bgr,
                    obs=obs,
                    measurements=measurements,
                    main_state=main_state,
                    main_reason=main_reason,
                    process_size=(self.args.process_width, self.args.process_height),
                    process_fps=fps,
                    process_ms=process_ms,
                    frame_age_ms=frame_age_ms,
                )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, int(self.args.jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                self._last_measurements = measurements
                self._update_status(
                    measurements=measurements,
                    main_state=main_state,
                    main_reason=main_reason,
                    camera_info=camera_info,
                    camera_error=camera_error,
                    capture_fps=capture_fps,
                    process_fps=fps,
                    process_ms=process_ms,
                    frame_age_ms=frame_age_ms,
                )
                with self._lock:
                    self._jpeg = encoded.tobytes()
                    self._jpeg_seq += 1
            except Exception as exc:
                with self._lock:
                    self._status.update(
                        {
                            "mode": "error",
                            "main_state": "ERROR",
                            "main_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                print(f"[processor] {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.5)

    def _update_status(
        self,
        *,
        measurements: dict[str, Any],
        main_state: str,
        main_reason: str,
        camera_info: dict[str, Any],
        camera_error: str | None,
        capture_fps: float,
        process_fps: float,
        process_ms: float,
        frame_age_ms: float,
    ) -> None:
        camera = dict(camera_info)
        camera["capture_fps"] = round(capture_fps, 1) if capture_fps else camera.get("capture_fps")
        if camera_error:
            camera["error"] = camera_error
        tracking_state = (
            "good"
            if (
                measurements["quality"] == "good"
                and measurements["eye_state"] in {"open", "closed"}
                and measurements["state"] not in {"no_face", "uncertain"}
            )
            else "uncertain"
        )
        tracking_reason = measurements["quality_detail"]
        with self._lock:
            self._status.update(
                {
                    "mode": "running",
                    "main_state": main_state,
                    "main_reason": main_reason,
                    "tracking": {"state": tracking_state, "reason": tracking_reason},
                    "eyes": {
                        "state": measurements["eye_state"],
                        "ear_raw": _round(measurements["ear_raw"], 4),
                        "ear_smoothed": _round(measurements["ear_smoothed"], 4),
                        "threshold": _round(measurements["threshold"], 4),
                        "closure_ms": _round(measurements["closure_ms"], 1),
                    },
                    "mouth": {
                        "state": measurements["mouth_state"],
                        "mar_raw": _round(measurements["mar_raw"], 4),
                        "mar_smoothed": _round(measurements["mar_smoothed"], 4),
                        "open_ms": _round(measurements["mouth_open_ms"], 1),
                    },
                    "head": {
                        "state": measurements["head_state"],
                        "yaw": _round(measurements["yaw"], 1),
                        "pitch": _round(measurements["pitch"], 1),
                        "roll": _round(measurements["roll"], 1),
                        "away_ms": _round(measurements["away_ms"], 1),
                    },
                    "fatigue": {
                        "perclos": _round(measurements["perclos"], 4),
                        "window_s": _round(measurements["perclos_window_s"], 1),
                    },
                    "blinks": measurements["blinks"],
                    "microsleeps": measurements["microsleeps"],
                    "yawns": measurements["yawns"],
                    "performance": {
                        "process_fps": round(process_fps, 1) if process_fps else None,
                        "process_ms": round(process_ms, 1),
                        "frame_age_ms": round(frame_age_ms, 1),
                    },
                    "camera": camera,
                    "last_event": self._last_event,
                }
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))

    def latest_jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._jpeg_seq


def _round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Web application
# ---------------------------------------------------------------------------


def create_app(service: DMSService) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(HTML, mimetype="text/html")

    @app.get("/status")
    def status():
        response = jsonify(service.status())
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": service.status().get("mode") == "running"})

    @app.get("/snapshot.jpg")
    def snapshot():
        jpeg, _ = service.latest_jpeg()
        if jpeg is None:
            return Response("not ready", status=503)
        return Response(jpeg, mimetype="image/jpeg")

    @app.get("/video_feed")
    def video_feed():
        def frames():
            last_seq = -1
            while True:
                jpeg, seq = service.latest_jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                if seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.after_request
    def no_cache(response):
        if request.path in {"/status", "/snapshot.jpg"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 3 DMS core (face, eyes, mouth, head orientation)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the browser on this machine")
    parser.add_argument("--image", help="use a still image instead of a camera (for UI testing)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="auto tries Picamera2 on Linux, then OpenCV",
    )
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--stream-width", type=int, default=960)
    parser.add_argument("--stream-height", type=int, default=540)
    parser.add_argument("--process-width", type=int, default=480)
    parser.add_argument("--process-height", type=int, default=270)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--mirror", action="store_true", help="mirror the camera image")
    parser.add_argument("--night", action="store_true", help="grayscale + CLAHE for an IR/night camera")
    parser.add_argument("--tint-fix", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--no-tint-fix", action="store_false", dest="tint_fix")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--face-detection-confidence", type=float, default=0.5)
    parser.add_argument("--face-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-face-width", type=float, default=40.0,
                        help="minimum face width in inference pixels")
    parser.add_argument("--ear-smoothing", type=int, default=3)
    parser.add_argument("--mar-smoothing", type=int, default=5)
    parser.add_argument("--drowsy-ms", type=float, default=900.0)
    parser.add_argument("--microsleep-ms", type=float, default=1500.0)
    parser.add_argument("--yawn-mar", type=float, default=0.60)
    parser.add_argument("--yawn-ms", type=float, default=800.0)
    parser.add_argument("--distracted-ms", type=float, default=1500.0)
    args = parser.parse_args(argv)

    args.process_width = max(160, min(int(args.process_width), int(args.stream_width)))
    args.process_height = max(90, min(int(args.process_height), int(args.stream_height)))
    args.stream_width = max(320, int(args.stream_width))
    args.stream_height = max(180, int(args.stream_height))
    args.jpeg_quality = max(50, min(95, int(args.jpeg_quality)))
    args.camera_fps = max(1.0, min(60.0, float(args.camera_fps)))
    args.tint_label = "night (IR)" if args.night else ("gray-world" if args.tint_fix else "off")
    return args


def _fix_purple(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(np.float32)
    means = image.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    gains = np.clip(gray / np.maximum(means, 1.0), 0.72, 1.38)
    return np.clip(image * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def _fix_night(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _preprocess_frames(frames: list[np.ndarray], night: bool, tint_fix: bool) -> list[np.ndarray]:
    out = []
    for frame in frames:
        if night:
            out.append(_fix_night(frame))
        elif tint_fix:
            out.append(_fix_purple(frame))
        else:
            out.append(frame)
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.night and args.tint_fix:
        args.tint_fix = False
        args.tint_label = "night (IR)"
    service = DMSService(args)
    print(f"Loading face model: {args.model}", flush=True)
    try:
        service.start()
    except Exception as exc:
        print(f"failed to start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    app = create_app(service)
    atexit.register(service.stop)
    print("", flush=True)
    print("Pi 3 DMS core", flush=True)
    print(f"  model     {args.model}", flush=True)
    print(
        f"  camera    {args.camera_backend} capture={args.capture_width}x{args.capture_height} "
        f"@{args.camera_fps:g} fps",
        flush=True,
    )
    print(
        f"  view      stream={args.stream_width}x{args.stream_height} "
        f"process={args.process_width}x{args.process_height}",
        flush=True,
    )
    print(f"  open      http://{args.host}:{args.port}", flush=True)
    print("  stop      Ctrl+C", flush=True)
    if args.open:
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")
        ).start()
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
