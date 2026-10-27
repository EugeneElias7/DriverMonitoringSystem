"""Raspberry Pi 3 DMS application.

The face core lives in :mod:`dms.core` and is always independent.  The optional
object/behaviour module lives in :mod:`dms.objects` and :mod:`dms.fusion`; it is
disabled by default and cannot prevent the face core from starting.

The web dashboard is a research/demo interface, not a safety or medical device.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from dms.camera import CameraWorker, FrameStore, preprocess_frame
from dms.core import (
    LEFT_EYE,
    MOUTH_BOTTOM,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    MOUTH_TOP,
    NOSE_TIP,
    RIGHT_EYE,
    FaceObservation,
    FaceModel,
    MeasurementEngine,
)
from dms.fusion import BehaviourConfig, FusionEngine
from dms.objects import (
    CANONICAL_LABELS,
    ObjectSnapshot,
    ObjectWorker,
    build_object_detector,
)
from dms.resources import ResourceMonitor

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "face_landmarker.task"
DEFAULT_OBJECT_MODEL = BASE_DIR / "models" / "object" / "efficientdet_lite0.tflite"

# Colours are BGR.
GREEN = (90, 220, 105)
YELLOW = (40, 210, 240)
ORANGE = (35, 150, 245)
RED = (60, 60, 235)
CYAN = (220, 210, 60)
WHITE = (245, 245, 245)
MUTED = (165, 180, 195)

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi 3 DMS</title>
<style>
  :root { color-scheme:dark; --bg:#0b1117; --panel:#141e28; --line:#2a3a49;
          --text:#edf4f9; --muted:#98a9b8; --good:#58d68d; --warn:#ffc857;
          --bad:#ff6b6b; --accent:#59c3ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 "Segoe UI", system-ui, sans-serif; }
  header { display:flex; gap:9px; align-items:center; flex-wrap:wrap;
           padding:12px 16px; border-bottom:1px solid var(--line); }
  header h1 { font-size:19px; margin:0; letter-spacing:.2px; }
  .pill { border:1px solid var(--line); border-radius:999px; padding:3px 9px;
          font-size:12px; color:var(--muted); white-space:nowrap; }
  .pill.good { color:var(--good); border-color:#2b6b46; }
  .pill.warn { color:var(--warn); border-color:#725d2b; }
  .pill.bad { color:var(--bad); border-color:#7a3535; }
  main { max-width:1600px; margin:auto; padding:14px; display:grid;
         grid-template-columns:minmax(0,1.7fr) minmax(340px,1fr); gap:14px; }
  .video { background:#000; border:1px solid var(--line); border-radius:12px;
           overflow:hidden; display:flex; align-items:center; }
  .video img { width:100%; height:auto; display:block; }
  .side { display:flex; flex-direction:column; gap:9px; min-width:0; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:11px; padding:11px 13px; min-width:0; }
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
  .bad { color:var(--bad); } .muted { color:var(--muted); }
  .note { font-size:12px; color:var(--muted); }
  .event { border-left:3px solid var(--warn); padding-left:8px; }
  .event-item { font-size:12px; color:var(--muted); padding:2px 0; }
  .event-item b { color:var(--text); font-weight:600; }
  .event-item.behaviour { border-left:2px solid var(--accent); padding-left:6px; }
  .detection-list { display:flex; flex-direction:column; gap:5px; }
  .detection { display:flex; justify-content:space-between; gap:8px;
               border-left:3px solid var(--accent); padding-left:7px;
               font-size:12px; color:var(--muted); }
  .detection b { color:var(--text); }
  .behaviour { border:1px solid var(--line); border-radius:8px; padding:7px 8px;
               margin-top:5px; }
  .behaviour-head { display:flex; justify-content:space-between; gap:8px;
                    align-items:center; font-size:13px; }
  .behaviour-head b { font-weight:700; }
  .behaviour .note { margin-top:3px; }
  .raw-link { color:var(--accent); text-decoration:none; font-size:12px; }
  .raw-link:hover { text-decoration:underline; }
  footer { max-width:1600px; margin:auto; padding:2px 16px 22px;
           color:var(--muted); font-size:11px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Pi 3 DMS</h1>
  <span class="pill" id="mode">starting...</span>
  <span class="pill" id="camera">camera</span>
  <span class="pill" id="object_pill">objects disabled</span>
  <span class="note">face core always on; object/behaviour module optional</span>
</header>
<main>
  <section class="video"><img src="/video_feed" alt="annotated camera"></section>
  <section class="side">
    <div class="card">
      <h2>Driver state (display priority)</h2>
      <div class="big" id="main_state">Starting...</div>
      <div class="reason" id="main_reason">Loading the face model.</div>
      <div style="margin-top:7px"><span class="pill" id="tracking">tracking uncertain</span></div>
      <div class="metrics" style="margin-top:7px">
        <span>Tracking confidence</span><b id="tracking_confidence">-</b>
        <span>Confidence source</span><b id="tracking_basis">heuristic, not a model probability</b>
      </div>
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
      <h2>Independent core states</h2>
      <div class="row"><span>Drowsiness</span><b id="drowsiness">-</b></div>
      <div class="row"><span>Attention</span><b id="attention">-</b></div>
      <div class="row"><span>Yawning</span><b id="yawning">-</b></div>
      <div class="row"><span>PERCLOS, 60 s</span><b id="perclos">-</b></div>
      <div class="row"><span>Window</span><b id="perclos_window">-</b></div>
      <div class="note" style="margin-top:5px">DROWSY is raised only by a current sustained closure. PERCLOS is shown separately.</div>
    </div>
    <div class="card">
      <h2>Object detections <span class="muted">(detector output, not behaviour)</span></h2>
      <div class="row"><span>Module</span><b id="object_status">-</b></div>
      <div class="row"><span>Model</span><b id="object_model">-</b></div>
      <div class="row"><span>Allowed classes</span><b id="object_classes">-</b></div>
      <div class="row"><span>Inference latency</span><b id="object_latency">-</b></div>
      <div class="row"><span>Detections age</span><b id="object_age">-</b></div>
      <div class="detection-list" id="detections" style="margin-top:7px">
        <div class="note">No active detections.</div>
      </div>
      <div class="note" id="object_error" style="margin-top:6px"></div>
      <div style="margin-top:6px"><a class="raw-link" id="raw_link" href="/snapshot_raw.jpg" target="_blank" rel="noreferrer">Open unannotated raw frame for debugging</a></div>
      <div class="note" style="margin-top:5px">Detector confidence is not evidence of behaviour. A bottle is never relabelled as a cigarette.</div>
    </div>
    <div class="card">
      <h2>Behaviour evidence <span class="muted">(suspected, not proof)</span></h2>
      <div class="behaviour" id="beh_phone">
        <div class="behaviour-head"><span>Phone use</span><b class="muted">-</b></div>
        <div class="note"></div>
      </div>
      <div class="behaviour" id="beh_drinking">
        <div class="behaviour-head"><span>Drinking</span><b class="muted">-</b></div>
        <div class="note"></div>
      </div>
      <div class="behaviour" id="beh_smoking">
        <div class="behaviour-head"><span>Smoking</span><b class="muted">-</b></div>
        <div class="note"></div>
      </div>
      <div class="note" style="margin-top:5px">A suspected behaviour requires a matching detector label, face/mouth proximity, persistence, and enough observations. Face loss or stale detections clear the evidence.</div>
    </div>
    <div class="card">
      <h2>Performance / resources</h2>
      <div class="row"><span>Processed FPS</span><b id="fps">-</b></div>
      <div class="row"><span>Process latency</span><b id="latency">-</b></div>
      <div class="row"><span>Capture-to-process age</span><b id="age">-</b></div>
      <div class="row"><span>Camera</span><b id="camera_detail">-</b></div>
      <div class="row"><span>Face model</span><b id="face_model">-</b></div>
      <div class="row"><span>Stream / inference</span><b id="resolution">-</b></div>
      <div class="row"><span>Colour preprocessing</span><b id="tint">-</b></div>
      <div class="row"><span>Camera AWB / controls</span><b id="awb">-</b></div>
      <div class="row"><span>CPU</span><b id="cpu">-</b></div>
      <div class="row"><span>Process CPU</span><b id="process_cpu">-</b></div>
      <div class="row"><span>RAM</span><b id="ram">-</b></div>
      <div class="row"><span>Temperature</span><b id="temperature">-</b></div>
    </div>
    <div class="card">
      <h2>Recent events</h2>
      <div id="events" class="note">No event yet.</div>
    </div>
    <div class="card">
      <h2>Scope / unavailable</h2>
      <div class="note" id="scope">
        Alcohol is unavailable: video cannot measure breath alcohol. A physical sensor is required.
        Object detection is optional and experimental; the shipped COCO control model has no cigarette class.
      </div>
    </div>
  </section>
</main>
<footer>
  Research prototype. Not a safety device. No accuracy, precision, recall or fatigue-diagnosis claim
  is made from unlabelled camera video. Keep faces and recordings local and obtain consent.
</footer>
<script>
const $ = (id) => document.getElementById(id);
const isNum = (v) => v !== null && v !== undefined && !Number.isNaN(Number(v));
const n = (v) => isNum(v) ? Number(v) : "-";
const f = (v, d=1, suffix="") => isNum(v) ? Number(v).toFixed(d) + suffix : "-";
const pct = (v) => isNum(v) ? f(Number(v) * 100, 1, "%") : "-";
const pillClass = (value) => {
  if (["suspected", "DROWSY", "DISTRACTED", "error"].includes(value)) return "bad";
  if (["possible", "EYES CLOSED", "YAWNING", "GLANCE", "TRACKING UNCERTAIN",
       "uncertain", "stale", "unavailable", "starting"].includes(value)) return "warn";
  if (["clear", "running", "ATTENTIVE", "disabled"].includes(value)) return "good";
  return "";
};
const setState = (el, value) => {
  el.textContent = value || "-";
  el.className = "big " + pillClass(value);
};
const setRow = (id, value) => { const el = $(id); if (el) el.textContent = value; };
const pretty = (value) => (value || "-").replaceAll("_", " ");
function renderDetections(items) {
  const box = $("detections");
  box.replaceChildren();
  if (!items || !items.length) {
    const d = document.createElement("div"); d.className = "note";
    d.textContent = "No active detections."; box.appendChild(d); return;
  }
  for (const item of items) {
    const row = document.createElement("div"); row.className = "detection";
    const left = document.createElement("span");
    left.textContent = (item.label || "?") + "  " + f(item.confidence, 2);
    const right = document.createElement("b");
    const b = item.bbox || [];
    right.textContent = b.length === 4 ? "box " + b.map(v => f(v, 2)).join(",") : "";
    right.title = "normalized bounding box and detector confidence";
    row.appendChild(left); row.appendChild(right); box.appendChild(row);
  }
}
function renderBehaviour(id, data) {
  const box = $(id); if (!box) return;
  const head = box.querySelector(".behaviour-head b");
  const note = box.querySelector(".note");
  const state = data && data.state ? data.state : "unavailable";
  head.textContent = pretty(state); head.className = pillClass(state);
  note.textContent = (data && data.evidence) ? data.evidence : "No evidence.";
}
function renderEvents(events) {
  const box = $("events"); box.replaceChildren();
  if (!events || !events.length) {
    const d = document.createElement("div"); d.textContent = "No event yet."; box.appendChild(d); return;
  }
  for (const item of events) {
    const d = document.createElement("div");
    d.className = "event-item " + (item.kind === "behaviour" ? "behaviour" : "");
    const b = document.createElement("b"); b.textContent = item.time + "  ";
    d.appendChild(b); d.appendChild(document.createTextNode((item.label || "") + ": " + (item.detail || "")));
    box.appendChild(d);
  }
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
    setRow("tracking_confidence", isNum(tr.confidence) ? f(tr.confidence, 2) : "-");
    setRow("tracking_basis", tr.confidence_basis || "heuristic, not a model probability");

    const e = s.eyes || {};
    $("eyes").textContent = e.state || "-";
    $("eyes").className = "big " + (e.state === "closed" ? "warn" : "good");
    setRow("ear_raw", f(e.ear_raw, 3));
    setRow("ear_smoothed", f(e.ear_smoothed, 3));
    setRow("threshold", f(e.threshold, 3));
    setRow("closure", f(e.closure_ms, 0, " ms"));
    setRow("blinks", n(s.blinks));
    setRow("microsleeps", n(s.microsleeps));

    const m = s.mouth || {};
    $("mouth").textContent = m.state || "-";
    $("mouth").className = "big " + (m.state === "yawning" ? "warn" : "good");
    setRow("mar", f(m.mar_raw, 2) + " / " + f(m.mar_smoothed, 2));
    setRow("yawn_ms", f(m.open_ms, 0, " ms"));
    setRow("yawns", n(s.yawns));

    const h = s.head || {};
    setRow("head", h.state || "-");
    setRow("pose", f(h.yaw, 1) + " / " + f(h.pitch, 1) + " / " + f(h.roll, 1));
    setRow("away", f(h.away_ms, 0, " ms"));

    const ind = s.independent || {};
    setRow("drowsiness", pretty(ind.drowsiness));
    setRow("attention", pretty(ind.attention));
    setRow("yawning", ind.yawning ? "yes" : "no");
    const pf = s.fatigue || {};
    setRow("perclos", pf.perclos === null || pf.perclos === undefined ? "-" : pct(pf.perclos));
    setRow("perclos_window", f(pf.window_s, 0, " s"));

    const obj = s.object || {};
    const objStatus = obj.status || "disabled";
    $("object_pill").textContent = "objects: " + objStatus;
    $("object_pill").className = "pill " + pillClass(objStatus);
    setRow("object_status", pretty(objStatus));
    setRow("object_model", obj.model || "-");
    setRow("object_classes", (obj.classes || []).join(", ") || "-");
    setRow("object_latency", f(obj.latency_ms, 1, " ms"));
    setRow("object_age", f(obj.age_ms, 0, " ms"));
    $("object_error").textContent = obj.error || "";
    renderDetections(s.detections || []);
    renderBehaviour("beh_phone", (s.behaviours || {}).phone);
    renderBehaviour("beh_drinking", (s.behaviours || {}).drinking);
    renderBehaviour("beh_smoking", (s.behaviours || {}).smoking);

    const perf = s.performance || {};
    setRow("fps", f(perf.process_fps, 1));
    setRow("latency", f(perf.process_ms, 1, " ms"));
    setRow("age", f(perf.frame_age_ms, 1, " ms"));
    const cam = s.camera || {};
    $("camera").textContent = cam.backend || "camera";
    setRow("camera_detail", (cam.capture_resolution || "-") + " @ " + f(cam.capture_fps, 1) + " fps");
    setRow("face_model", ((s.face_model || {}).name || "-") + " (" + ((s.face_model || {}).status || "?") + ")");
    setRow("resolution", (cam.stream_resolution || "-") + " / " + (cam.process_resolution || "-"));
    const pre = s.preprocessing || {};
    setRow("tint", pre.label || "-");
    setRow("awb", (pre.awb || "-") + " / " + (pre.camera_controls || "-"));
    if (cam.error) {
      $("camera").className = "pill bad"; $("camera").textContent = "camera error";
    } else {
      $("camera").className = "pill good";
    }

    const res = s.resources || {};
    setRow("cpu", isNum(res.cpu_percent) ? f(res.cpu_percent, 1, "%") : "-");
    setRow("process_cpu", isNum(res.process_cpu_percent) ? f(res.process_cpu_percent, 1, "%") : "-");
    setRow("ram", isNum(res.ram_percent) ? f(res.ram_percent, 1, "%") : "-");
    setRow("temperature", isNum(res.temperature_c) ? f(res.temperature_c, 1, " C") : "-");

    renderEvents(s.recent_events || []);
  } catch (err) {
    $("mode").textContent = "server unavailable";
    $("mode").className = "pill bad";
  }
}
refresh();
setInterval(refresh, 400);
</script>
</body>
</html>"""


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


def _object_colour(label: str) -> tuple[int, int, int]:
    return {
        "phone": CYAN,
        "bottle": (230, 180, 80),
        "cigarette": ORANGE,
    }.get(label, WHITE)


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
    detections: list[dict[str, Any]] | None = None,
    behaviours: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    proc_w, proc_h = process_size
    sx = width / float(proc_w)
    sy = height / float(proc_h)
    colour = _state_colour(main_state)

    for detection in detections or []:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1 = int(max(0.0, min(1.0, float(bbox[0]))) * width)
        y1 = int(max(0.0, min(1.0, float(bbox[1]))) * height)
        x2 = int(max(0.0, min(1.0, float(bbox[2]))) * width)
        y2 = int(max(0.0, min(1.0, float(bbox[3]))) * height)
        object_colour = _object_colour(str(detection.get("label", "")))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), object_colour, 2, cv2.LINE_AA)
        label = f"{detection.get('label', '?')} {float(detection.get('confidence', 0.0)):.2f}"
        _text(canvas, label, (x1 + 3, max(16, y1 - 6)), scale=0.52, colour=object_colour, thickness=1)

    state_scale = 1.02
    reason_scale = 0.59
    state_w, _ = cv2.getTextSize(main_state, cv2.FONT_HERSHEY_DUPLEX, state_scale, 2)[0]
    reason_w, _ = cv2.getTextSize(main_reason, cv2.FONT_HERSHEY_DUPLEX, reason_scale, 1)[0]
    panel_max = max(330, width - 300)
    max_reason_w = max(220, panel_max - 52)
    if reason_w > max_reason_w:
        reason_scale *= max_reason_w / float(reason_w)
        reason_w, _ = cv2.getTextSize(main_reason, cv2.FONT_HERSHEY_DUPLEX, reason_scale, 1)[0]
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
        center = (
            int(p[MOUTH_TOP][0] + p[MOUTH_BOTTOM][0]) // 2,
            int(p[MOUTH_TOP][1] + p[MOUTH_BOTTOM][1]) // 2,
        )
        cv2.circle(canvas, center, 3, CYAN, -1, cv2.LINE_AA)

        x1, y1, x2, y2 = [
            int(round(v))
            for v in (
                obs.bbox[0] * sx,
                obs.bbox[1] * sy,
                obs.bbox[2] * sx,
                obs.bbox[3] * sy,
            )
        ]
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
    _text(canvas, metric_text, (30, height - 24), scale=0.63, colour=WHITE, thickness=1)
    perf_text = f"{process_fps:.1f} fps  {process_ms:.0f} ms  age {frame_age_ms:.0f}"
    perf_w, _ = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)[0]
    perf_x = width - perf_w - 38
    _panel(canvas, perf_x, 18, perf_w + 24, 42)
    _text(canvas, perf_text, (perf_x + 12, 45), scale=0.48, colour=WHITE, thickness=1)

    active = [
        key
        for key, value in (behaviours or {}).items()
        if value.get("suspected")
    ]
    if active:
        active_text = "suspected: " + ", ".join(active)
        active_w, _ = cv2.getTextSize(active_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
        _panel(canvas, 18, 142, active_w + 26, 32)
        _text(canvas, active_text, (30, 164), scale=0.55, colour=ORANGE, thickness=1)
    return canvas


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
        self.fusion = FusionEngine(
            BehaviourConfig(
                stale_ms=args.object_stale_ms,
                min_confidence=args.object_min_confidence,
                phone_min_ms=args.object_phone_ms,
                drinking_min_ms=args.object_drink_ms,
                smoking_min_ms=args.object_smoke_ms,
                min_observations=args.object_min_observations,
                face_radius_scale=args.object_face_radius,
                mouth_radius_scale=args.object_mouth_radius,
            )
        )
        self.resource_monitor = ResourceMonitor(min_interval_s=1.0)
        self.object_worker: ObjectWorker | None = None
        self._object_start_thread: threading.Thread | None = None
        self._object_lock = threading.Lock()
        self._object_snapshot = self._initial_object_snapshot()
        self._lock = threading.Lock()
        self._status: dict[str, Any] = self._initial_status()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._thread: threading.Thread | None = None
        self._process_times: deque[float] = deque(maxlen=40)
        self._last_state: str = "STARTING"
        self._last_behaviour_states: dict[str, str] = {}
        self._last_object_status: str | None = None
        self._last_event: str | None = None
        self._recent_events: deque[dict[str, str]] = deque(maxlen=10)

    def _initial_object_snapshot(self) -> ObjectSnapshot:
        if not self.args.objects:
            return ObjectSnapshot.disabled()
        return ObjectSnapshot(
            status="starting",
            enabled=True,
            model=Path(self.args.object_model).name if self.args.object_model else None,
            classes=tuple(self.args.object_classes),
            stale_ms=self.args.object_stale_ms,
        )

    def _initial_status(self) -> dict[str, Any]:
        return {
            "mode": "starting",
            "main_state": "STARTING",
            "main_reason": "Loading the face model.",
            "tracking": {
                "state": "uncertain",
                "reason": "starting",
                "confidence": None,
                "confidence_basis": "heuristic, not a model probability",
            },
            "eyes": {
                "state": "-",
                "ear_raw": None,
                "ear_smoothed": None,
                "threshold": None,
                "closure_ms": None,
            },
            "mouth": {
                "state": "-",
                "mar_raw": None,
                "mar_smoothed": None,
                "open_ms": None,
            },
            "head": {
                "state": "-",
                "yaw": None,
                "pitch": None,
                "roll": None,
                "away_ms": None,
            },
            "independent": {
                "drowsiness": "unknown",
                "attention": "unknown",
                "yawning": False,
            },
            "fatigue": {"perclos": None, "window_s": 0.0},
            "blinks": 0,
            "microsleeps": 0,
            "yawns": 0,
            "performance": {
                "process_fps": None,
                "process_ms": None,
                "frame_age_ms": None,
            },
            "camera": {
                "backend": "-",
                "capture_resolution": "-",
                "capture_fps": None,
                "stream_resolution": f"{self.args.stream_width}x{self.args.stream_height}",
                "process_resolution": f"{self.args.process_width}x{self.args.process_height}",
                "error": None,
            },
            "face_model": {
                "name": Path(self.args.model).name,
                "status": "starting",
                "source": "MediaPipe Face Landmarker",
            },
            "preprocessing": {
                "mode": self.args.tint_mode,
                "label": self.args.tint_label,
                "awb": self.args.awb,
                "camera_controls": "default",
            },
            "resources": self.resource_monitor.sample(force=True),
            "object": self._object_snapshot.to_dict(time.monotonic()),
            "detections": [],
            "behaviours": {},
            "active_behaviours": [],
            "tint": self.args.tint_label,
            "last_event": None,
            "recent_events": [],
            "alcohol": {
                "status": "unavailable",
                "reason": "No physical breath-alcohol sensor is connected; video cannot measure alcohol.",
            },
        }

    def start(self) -> None:
        self.model = FaceModel(
            Path(self.args.model),
            num_faces=1,
            min_detection_confidence=self.args.face_detection_confidence,
            min_tracking_confidence=self.args.face_tracking_confidence,
        )
        self.camera.start()
        # Start the face processor before attempting to load the optional object
        # model.  A missing, slow or incompatible object backend can therefore
        # never delay or block the face core.
        self._thread = threading.Thread(target=self._run, name="processor", daemon=True)
        self._thread.start()
        if self.args.objects:
            self._object_start_thread = threading.Thread(
                target=self._start_objects, name="object-start", daemon=True
            )
            self._object_start_thread.start()

    def _start_objects(self) -> None:
        detector = None
        try:
            detector = build_object_detector(self.args)
            with self._object_lock:
                if self.stop_event.is_set():
                    detector.close()
                    return
                worker = ObjectWorker(
                    self.store,
                    detector,
                    interval_ms=self.args.object_interval_ms,
                    stale_ms=self.args.object_stale_ms,
                    classes=tuple(self.args.object_classes),
                    color_mode=self.args.tint_mode,
                )
                self.object_worker = worker
                worker.start()
            print(
                f"Object module: {self.args.object_backend} "
                f"model={self.args.object_model} classes={','.join(self.args.object_classes)}",
                flush=True,
            )
        except Exception as exc:
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            reason = f"object module unavailable: {type(exc).__name__}: {exc}"
            self._object_snapshot = ObjectSnapshot.unavailable(
                reason, model=Path(self.args.object_model).name
            )
            print(f"[objects] {reason}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        self.camera.stop()
        if self._object_start_thread is not None:
            self._object_start_thread.join(timeout=3.0)
        with self._object_lock:
            worker = self.object_worker
            self.object_worker = None
        if worker is not None:
            worker.stop()
            worker.join(timeout=2.0)
            worker.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.model is not None:
            self.model.close()

    def _snapshot(self) -> ObjectSnapshot:
        if self.object_worker is not None:
            return self.object_worker.snapshot()
        return self._object_snapshot

    def _record_event(self, *, kind: str, label: str, detail: str) -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "kind": kind,
            "label": label,
            "detail": detail,
        }
        with self._lock:
            self._recent_events.appendleft(entry)
            self._last_event = f"{entry['time']}  {label}: {detail}"
        print(f"[event] {label}: {detail}", flush=True)

    def _run(self) -> None:
        assert self.model is not None
        last_sequence = -1
        while not self.stop_event.is_set():
            packet, camera_info, camera_error, capture_fps = self.store.wait_next(
                last_sequence, timeout=0.5
            )
            if packet is None or packet.sequence == last_sequence:
                continue
            last_sequence = packet.sequence
            started = time.perf_counter()
            try:
                process_bgr = preprocess_frame(packet.process_bgr, self.args.tint_mode)
                stream_bgr = preprocess_frame(packet.stream_bgr, self.args.tint_mode)
                obs = self.model.detect(
                    process_bgr,
                    packet.timestamp,
                    min_face_width_px=self.args.min_face_width,
                )
                now = time.monotonic()
                measurements = self.engine.update(
                    obs,
                    packet.timestamp,
                    source_frame_width=self.args.process_width,
                    source_frame_height=self.args.process_height,
                )
                object_snapshot = self._snapshot()
                fusion_result = self.fusion.update(measurements, object_snapshot, now)
                main_state = fusion_result["main_state"]
                main_reason = fusion_result["main_reason"]
                process_ms = (time.perf_counter() - started) * 1000.0
                frame_age_ms = (now - packet.timestamp) * 1000.0
                self._process_times.append(now)
                fps = 0.0
                if len(self._process_times) >= 2:
                    span = self._process_times[-1] - self._process_times[0]
                    if span > 0:
                        fps = (len(self._process_times) - 1) / span

                self._record_state_events(fusion_result, now)
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
                    detections=fusion_result["detections"],
                    behaviours=fusion_result["behaviours"],
                )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, int(self.args.jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                self._update_status(
                    measurements=measurements,
                    fusion_result=fusion_result,
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

    def _record_state_events(self, fusion_result: dict[str, Any], now: float) -> None:
        main_state = fusion_result["main_state"]
        if main_state != self._last_state and main_state in {
            "DROWSY",
            "EYES CLOSED",
            "YAWNING",
            "DISTRACTED",
            "TRACKING UNCERTAIN",
        }:
            self._record_event(
                kind="core",
                label=main_state,
                detail=fusion_result["main_reason"],
            )
        self._last_state = main_state

        for key, behaviour in fusion_result["behaviours"].items():
            state = str(behaviour.get("state", ""))
            previous = self._last_behaviour_states.get(key)
            if state == "suspected" and previous != "suspected":
                self._record_event(
                    kind="behaviour",
                    label=behaviour.get("name", key),
                    detail=behaviour.get("evidence", "suspected from object evidence"),
                )
            self._last_behaviour_states[key] = state

        object_status = str(fusion_result["object"].get("status", ""))
        if object_status in {"error", "unavailable", "stale"} and object_status != self._last_object_status:
            self._record_event(
                kind="object",
                label="objects",
                detail=f"module {object_status}: {fusion_result['object'].get('error') or 'no fresh detections'}",
            )
        self._last_object_status = object_status

    def _update_status(
        self,
        *,
        measurements: dict[str, Any],
        fusion_result: dict[str, Any],
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
        with self._lock:
            self._status.update(
                {
                    "mode": "running",
                    "main_state": fusion_result["main_state"],
                    "main_reason": fusion_result["main_reason"],
                    "tracking": fusion_result["tracking"],
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
                    "independent": fusion_result["independent"],
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
                    "face_model": {
                        "name": Path(self.args.model).name,
                        "status": "running",
                        "source": "MediaPipe Face Landmarker",
                    },
                    "preprocessing": {
                        "mode": self.args.tint_mode,
                        "label": self.args.tint_label,
                        "awb": self.args.awb,
                        "camera_controls": camera.get("camera_controls", "default"),
                    },
                    "resources": self.resource_monitor.sample(),
                    "object": fusion_result["object"],
                    "detections": fusion_result["detections"],
                    "behaviours": fusion_result["behaviours"],
                    "active_behaviours": fusion_result["active_behaviours"],
                    "tint": self.args.tint_label,
                    "last_event": self._last_event,
                    "recent_events": list(self._recent_events),
                    "alcohol": {
                        "status": "unavailable",
                        "reason": "No physical breath-alcohol sensor is connected; video cannot measure alcohol.",
                    },
                }
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status))

    def latest_jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._jpeg_seq

    def latest_raw_jpeg(self) -> bytes | None:
        raw, _info = self.store.latest_raw()
        if raw is None:
            return None
        ok, encoded = cv2.imencode(
            ".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, min(85, int(self.args.jpeg_quality))]
        )
        return encoded.tobytes() if ok else None


def _round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


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

    @app.get("/snapshot_raw.jpg")
    def snapshot_raw():
        jpeg = service.latest_raw_jpeg()
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
        if request.path in {"/status", "/snapshot.jpg", "/snapshot_raw.jpg"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 3 DMS: face core plus optional object/behaviour module"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="open the browser on this machine")
    parser.add_argument("--image", help="use a still image instead of a camera (UI testing)")
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
    parser.add_argument(
        "--tint-mode",
        choices=("off", "soft", "grayworld", "night"),
        default="soft",
        help="conservative colour preprocessing; default soft gray-world",
    )
    parser.add_argument("--tint-fix", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-tint-fix", action="store_false", dest="tint_fix", help="disable software colour correction")
    parser.add_argument("--awb", choices=("auto", "manual"), default="auto")
    parser.add_argument("--red-gain", type=float, default=None)
    parser.add_argument("--blue-gain", type=float, default=None)
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--analogue-gain", type=float, default=None)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--face-detection-confidence", type=float, default=0.5)
    parser.add_argument("--face-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--min-face-width", type=float, default=40.0)
    parser.add_argument("--ear-smoothing", type=int, default=3)
    parser.add_argument("--mar-smoothing", type=int, default=5)
    parser.add_argument("--drowsy-ms", type=float, default=900.0)
    parser.add_argument("--microsleep-ms", type=float, default=1500.0)
    parser.add_argument("--yawn-mar", type=float, default=0.60)
    parser.add_argument("--yawn-ms", type=float, default=800.0)
    parser.add_argument("--distracted-ms", type=float, default=1500.0)

    parser.add_argument("--objects", action="store_true", help="enable optional object detection")
    parser.add_argument(
        "--object-backend",
        choices=("none", "mediapipe", "http"),
        default="none",
        help="mediapipe local TFLite or a separate HTTP object service",
    )
    parser.add_argument("--object-model", default=str(DEFAULT_OBJECT_MODEL))
    parser.add_argument("--object-service-url", default="http://127.0.0.1:8010/detect")
    parser.add_argument("--object-timeout-s", type=float, default=2.0)
    parser.add_argument("--object-classes", default="phone,bottle",
                        help="comma-separated labels to accept: phone,bottle,cigarette")
    parser.add_argument("--object-min-confidence", type=float, default=0.35)
    parser.add_argument("--object-max-results", type=int, default=10)
    parser.add_argument("--object-interval-ms", type=float, default=500.0)
    parser.add_argument("--object-stale-ms", type=float, default=1500.0)
    parser.add_argument("--object-phone-ms", type=float, default=1200.0)
    parser.add_argument("--object-drink-ms", type=float, default=1000.0)
    parser.add_argument("--object-smoke-ms", type=float, default=1000.0)
    parser.add_argument("--object-min-observations", type=int, default=3)
    parser.add_argument("--object-face-radius", type=float, default=0.40)
    parser.add_argument("--object-mouth-radius", type=float, default=0.55)

    args = parser.parse_args(argv)

    args.process_width = max(160, min(int(args.process_width), int(args.stream_width)))
    args.process_height = max(90, min(int(args.process_height), int(args.stream_height)))
    args.stream_width = max(320, int(args.stream_width))
    args.stream_height = max(180, int(args.stream_height))
    args.jpeg_quality = max(50, min(95, int(args.jpeg_quality)))
    args.camera_fps = max(1.0, min(60.0, float(args.camera_fps)))

    if args.night:
        args.tint_mode = "night"
    if args.tint_fix is False:
        args.tint_mode = "off"
    elif args.tint_fix is True:
        args.tint_mode = "soft"
    args.tint_label = {
        "off": "off (raw)",
        "soft": "soft gray-world",
        "grayworld": "gray-world",
        "night": "night (grayscale + CLAHE)",
    }[args.tint_mode]

    if (args.red_gain is None) != (args.blue_gain is None):
        parser.error("--red-gain and --blue-gain must be supplied together")
    if args.awb == "manual" and (args.red_gain is None or args.blue_gain is None):
        parser.error("--awb manual requires --red-gain and --blue-gain")
    if args.objects and args.object_backend == "none":
        args.object_backend = "mediapipe"
    args.object_classes = [
        item.strip().lower()
        for item in str(args.object_classes).split(",")
        if item.strip()
    ]
    unknown = set(args.object_classes) - CANONICAL_LABELS
    if unknown:
        parser.error(f"unsupported --object-classes: {sorted(unknown)}")
    args.object_interval_ms = max(100.0, float(args.object_interval_ms))
    args.object_stale_ms = max(args.object_interval_ms * 1.5, float(args.object_stale_ms))
    args.object_min_observations = max(1, int(args.object_min_observations))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    print("Pi 3 DMS", flush=True)
    print(f"  face model {args.model}", flush=True)
    print(
        f"  camera     {args.camera_backend} capture={args.capture_width}x{args.capture_height} "
        f"@{args.camera_fps:g} fps",
        flush=True,
    )
    print(
        f"  view       stream={args.stream_width}x{args.stream_height} "
        f"process={args.process_width}x{args.process_height}",
        flush=True,
    )
    print(f"  colour     {args.tint_mode} ({args.tint_label}) awb={args.awb}", flush=True)
    if args.objects:
        print(
            f"  objects    {args.object_backend} model={args.object_model} "
            f"classes={','.join(args.object_classes)} interval={args.object_interval_ms:g} ms",
            flush=True,
        )
    else:
        print("  objects    disabled (face core continues)", flush=True)
    print(f"  open       http://{args.host}:{args.port}", flush=True)
    print("  stop       Ctrl+C", flush=True)
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
