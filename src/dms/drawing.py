"""Annotated-frame rendering for demos, reports and manual inspection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .core.types import FaceObservation

_GREEN = (80, 220, 100)
_YELLOW = (40, 210, 240)
_RED = (60, 60, 235)
_ORANGE = (40, 150, 245)
_GRAY = (150, 150, 150)
_WHITE = (245, 245, 245)
_DARK = (22, 22, 22)
_CYAN = (220, 210, 60)

# MediaPipe landmark indices used for compact eye/mouth overlay.
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_MOUTH = [61, 13, 14, 291, 78, 308]


def draw_result(
    frame: np.ndarray,
    result: dict[str, Any],
    face: FaceObservation | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Return an annotated copy of ``frame`` without modifying the input."""
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if scale is None:
        scale = max(0.38, min(1.0, width / 1200.0))
    thickness = max(1, int(round(scale * 2)))

    _header(canvas, result, scale)

    face_info = result.get("face") or {}
    if face_info.get("present"):
        bbox = _bbox(face_info.get("bbox"))
        if bbox:
            cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), _GREEN, thickness)
            _label(canvas, "face", (bbox[0], max(0, bbox[1] - 5)), _GREEN, scale, thickness)

    if face is not None:
        _draw_face_points(canvas, face, scale)
        _draw_pose_arrow(canvas, face, scale)

    for det in result.get("detections", []):
        label = str(det.get("label", ""))
        if label == "face":
            continue
        bbox = _bbox(det.get("bbox"))
        if not bbox:
            continue
        metadata = det.get("metadata") or {}
        near = bool(metadata.get("near_mouth"))
        if label.lower() == "cigarette":
            color = _RED if near else _ORANGE
        elif label.lower() == "smoke":
            color = _GRAY
        else:
            color = _GRAY
        cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        text = f"{label} {float(det.get('confidence', 0.0)):.2f}"
        if near:
            text += " near-mouth"
        _label(canvas, text, (bbox[0], max(0, bbox[1] - 5)), color, scale * 0.9, thickness)

    _state_panel(canvas, result, scale)
    _event_banner(canvas, result.get("events") or [], scale)
    return canvas


def _header(canvas: np.ndarray, result: dict[str, Any], scale: float) -> None:
    height, width = canvas.shape[:2]
    band_h = int(34 * scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, band_h), _DARK, -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    text = "DMS research prototype | not a safety device"
    cv2.putText(
        canvas,
        text,
        (int(10 * scale), int(23 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * scale,
        _WHITE,
        max(1, int(round(scale * 2))),
        cv2.LINE_AA,
    )
    frame_no = result.get("frame_index", "?")
    stamp = str(result.get("iso_time", ""))[11:23]
    cv2.putText(
        canvas,
        f"frame {frame_no}  {stamp}",
        (max(0, width - int(300 * scale)), int(23 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50 * scale,
        _WHITE,
        max(1, int(round(scale * 1.6))),
        cv2.LINE_AA,
    )


def _state_panel(canvas: np.ndarray, result: dict[str, Any], scale: float) -> None:
    features = result.get("features") or {}
    drowsy = features.get("drowsiness") or {}
    pose = features.get("head_pose") or {}
    distract = features.get("distraction") or {}
    smoke = features.get("smoking") or {}
    face = result.get("face") or {}
    dm = drowsy.get("measurements") or {}
    pm = pose.get("measurements") or {}
    xm = distract.get("measurements") or {}
    sm = smoke.get("measurements") or {}

    lines = [
        f"Face: {'yes' if face.get('present') else 'no'}  EARmax {_fmt(dm.get('ear_max'), 3)}  MAR {_fmt(dm.get('mar'), 2)}",
        f"Drowsy: {drowsy.get('state', 'n/a')}  PERCLOS {_fmt(dm.get('perclos'), 2)}  closure {_fmt(dm.get('closure_ms'), 0)} ms",
        f"Head: {pose.get('state', 'n/a')}  yaw {_fmt(pm.get('yaw'), 1)}  pitch {_fmt(pm.get('pitch'), 1)}",
        f"Distraction: {distract.get('state', 'n/a')}  away {_fmt(xm.get('away_ms'), 0)} ms",
        f"Smoking: {smoke.get('state', 'n/a')}  near {_fmt(sm.get('near_ms'), 0)} ms  conf {sm.get('confirmed')}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 * scale
    line_h = int(19 * scale)
    padding = int(8 * scale)
    max_width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, max(1, int(scale)))
        max_width = max(max_width, tw)
    panel_w = min(canvas.shape[1] - 2 * padding, max_width + 2 * padding)
    panel_h = line_h * len(lines) + padding * 2
    x0 = padding
    y0 = int(42 * scale)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), _DARK, -1)
    cv2.addWeighted(overlay, 0.66, canvas, 0.34, 0, canvas)
    for idx, line in enumerate(lines):
        y = y0 + padding + line_h * (idx + 1) - int(3 * scale)
        cv2.putText(
            canvas,
            line,
            (x0 + padding, y),
            font,
            font_scale,
            _WHITE,
            max(1, int(round(scale))),
            cv2.LINE_AA,
        )


def _event_banner(canvas: np.ndarray, events: list[dict[str, Any]], scale: float) -> None:
    if not events:
        return
    event = events[-1]
    height, width = canvas.shape[:2]
    band_h = int(30 * scale)
    y0 = height - band_h
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (width, height), _DARK, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    severity = str(event.get("severity", "info"))
    color = _RED if severity == "critical" else (_ORANGE if severity == "warning" else _CYAN)
    text = f"{event.get('feature', '')}: {event.get('kind', '')} - {event.get('message', '')}"
    max_chars = max(20, int(width / (8.5 * scale)))
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    cv2.putText(
        canvas,
        text,
        (int(10 * scale), height - int(9 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * scale,
        color,
        max(1, int(round(scale))),
        cv2.LINE_AA,
    )


def _draw_face_points(canvas: np.ndarray, face: FaceObservation, scale: float) -> None:
    points = np.asarray(face.landmarks)
    if points.ndim != 2 or points.shape[0] < 468:
        return
    radius = max(1, int(round(scale * 1.5)))
    for indices, color in ((_LEFT_EYE, _CYAN), (_RIGHT_EYE, _CYAN), (_MOUTH, _ORANGE)):
        poly = points[indices].astype(np.int32)
        cv2.polylines(canvas, [poly], True, color, max(1, int(round(scale))), cv2.LINE_AA)
        for x, y in poly:
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    if face.mouth_center is not None:
        center = (int(face.mouth_center[0]), int(face.mouth_center[1]))
        cv2.drawMarker(canvas, center, _RED, cv2.MARKER_CROSS, int(12 * scale), max(1, int(round(scale))))


def _draw_pose_arrow(canvas: np.ndarray, face: FaceObservation, scale: float) -> None:
    if face.yaw is None or face.face_width is None:
        return
    cx, cy = ((face.bbox[0] + face.bbox[2]) / 2.0, (face.bbox[1] + face.bbox[3]) / 2.0)
    length = max(18.0, face.face_width * 0.55)
    import math

    angle = math.radians(face.yaw)
    x2 = int(cx + math.sin(angle) * length)
    y2 = int(cy - math.cos(angle) * length * 0.35)
    cv2.arrowedLine(
        canvas,
        (int(cx), int(cy)),
        (x2, y2),
        _YELLOW,
        max(1, int(round(scale * 2))),
        cv2.LINE_AA,
        tipLength=0.25,
    )


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 * scale
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    y = max(th + 4, y)
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + baseline), _DARK, -1)
    cv2.putText(canvas, text, (x + 2, y - 2), font, font_scale, color, thickness, cv2.LINE_AA)


def _bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not value or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in value)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
