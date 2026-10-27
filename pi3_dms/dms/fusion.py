"""Central temporal fusion for face state and optional object detections.

The fusion engine has no camera, model or Flask dependency.  It is deterministic
when supplied with timestamps, which is what the simulated tests exercise.
Behaviours are kept independent: phone use, drinking and smoking can all be
active at once, and a bottle is never promoted to a smoking cue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .core import driver_state
from .objects import ObjectDetection, ObjectSnapshot

BEHAVIOUR_KEYS = ("phone", "drinking", "smoking")


@dataclass
class BehaviourConfig:
    stale_ms: float = 1500.0
    min_confidence: float = 0.35
    cigarette_min_confidence: float = 0.45
    phone_min_ms: float = 1200.0
    drinking_min_ms: float = 1000.0
    smoking_min_ms: float = 1000.0
    min_observations: int = 3
    face_radius_scale: float = 0.40
    mouth_radius_scale: float = 0.55
    gap_ms: float = 400.0
    min_mouth_radius: float = 0.035


@dataclass
class _Episode:
    since: float | None = None
    last_seen: float | None = None
    last_object_timestamp: float | None = None
    observations: int = 0
    duration_ms: float = 0.0
    detection: ObjectDetection | None = None

    def reset(self) -> None:
        self.since = None
        self.last_seen = None
        self.last_object_timestamp = None
        self.observations = 0
        self.duration_ms = 0.0
        self.detection = None


def _norm_face(measurements: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = measurements.get("face_bbox")
    width = measurements.get("source_frame_width")
    height = measurements.get("source_frame_height")
    if not bbox or len(bbox) != 4 or not width or not height:
        return None
    w = float(width)
    h = float(height)
    x1, y1, x2, y2 = (max(0.0, min(1.0, float(v) / dim)) for v, dim in zip(bbox, (w, h, w, h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _norm_mouth(measurements: dict[str, Any], face: tuple[float, float, float, float]) -> tuple[float, float]:
    mouth = measurements.get("mouth_center")
    width = measurements.get("source_frame_width")
    height = measurements.get("source_frame_height")
    if mouth and len(mouth) == 2 and width and height:
        return float(mouth[0]) / float(width), float(mouth[1]) / float(height)
    return _center(face)


def _inside_expanded(
    point: tuple[float, float],
    face: tuple[float, float, float, float],
    scale: float,
) -> bool:
    face_w = face[2] - face[0]
    face_h = face[3] - face[1]
    pad_x = face_w * float(scale)
    pad_y = face_h * float(scale)
    return (
        face[0] - pad_x <= point[0] <= face[2] + pad_x
        and face[1] - pad_y <= point[1] <= face[3] + pad_y
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _behaviour_state_name(key: str) -> str:
    return {"phone": "phone_use", "drinking": "drinking", "smoking": "smoking"}[key]


class BehaviourFusion:
    """Accumulate object evidence into independent, time-limited behaviours."""

    def __init__(self, config: BehaviourConfig | None = None):
        self.config = config or BehaviourConfig()
        self._episodes: dict[str, _Episode] = {key: _Episode() for key in BEHAVIOUR_KEYS}

    def reset(self) -> None:
        for episode in self._episodes.values():
            episode.reset()

    def _fresh_detections(
        self, snapshot: ObjectSnapshot, now: float
    ) -> list[ObjectDetection]:
        max_age_s = self.config.stale_ms / 1000.0
        fresh: list[ObjectDetection] = []
        for detection in snapshot.detections:
            age = max(0.0, now - float(detection.timestamp))
            if age <= max_age_s:
                fresh.append(detection)
        return fresh

    def _matching_detection(
        self,
        detections: list[ObjectDetection],
        *,
        key: str,
        face: tuple[float, float, float, float],
        mouth: tuple[float, float],
    ) -> tuple[ObjectDetection | None, str]:
        if key == "phone":
            candidates = [
                d
                for d in detections
                if d.label == "phone"
                and d.confidence >= self.config.min_confidence
                and _inside_expanded(_center(d.bbox), face, self.config.face_radius_scale)
            ]
            if not candidates:
                return None, "no phone near the face"
            best = max(candidates, key=lambda d: d.confidence)
            return best, "phone inside the face/upper-body region"

        if key == "drinking":
            face_w = face[2] - face[0]
            radius = max(self.config.min_mouth_radius, face_w * self.config.mouth_radius_scale)
            candidates = [
                d
                for d in detections
                if d.label == "bottle"
                and d.confidence >= self.config.min_confidence
                and _distance(_center(d.bbox), mouth) <= radius
            ]
            if not candidates:
                return None, "no bottle near the mouth"
            best = max(candidates, key=lambda d: d.confidence)
            return best, "bottle near the mouth"

        if key == "smoking":
            face_w = face[2] - face[0]
            radius = max(self.config.min_mouth_radius, face_w * self.config.mouth_radius_scale)
            candidates = [
                d
                for d in detections
                if d.label == "cigarette"
                and d.confidence >= self.config.cigarette_min_confidence
                and _distance(_center(d.bbox), mouth) <= radius
            ]
            if not candidates:
                return None, "no cigarette near the mouth"
            best = max(candidates, key=lambda d: d.confidence)
            return best, "cigarette near the mouth"

        return None, "unknown behaviour"

    def _update_one(
        self,
        key: str,
        *,
        detections: list[ObjectDetection],
        snapshot_status: str,
        face_ok: bool,
        face: tuple[float, float, float, float] | None,
        mouth: tuple[float, float] | None,
        now: float,
        snapshot_timestamp: float | None,
    ) -> dict[str, Any]:
        episode = self._episodes[key]
        name = _behaviour_state_name(key)

        if snapshot_status == "disabled":
            episode.reset()
            return {
                "name": name,
                "state": "disabled",
                "suspected": False,
                "evidence": "object module disabled",
                "duration_ms": None,
                "observations": 0,
                "last_seen_ms": None,
                "detection": None,
            }
        if snapshot_status in {"unavailable", "error", "starting"}:
            episode.reset()
            return {
                "name": name,
                "state": snapshot_status,
                "suspected": False,
                "evidence": f"object module {snapshot_status}",
                "duration_ms": None,
                "observations": 0,
                "last_seen_ms": None,
                "detection": None,
            }
        if snapshot_status == "stale":
            episode.reset()
            return {
                "name": name,
                "state": "stale",
                "suspected": False,
                "evidence": "object detections are stale; rejected",
                "duration_ms": None,
                "observations": 0,
                "last_seen_ms": None,
                "detection": None,
            }
        if not face_ok or face is None or mouth is None:
            episode.reset()
            return {
                "name": name,
                "state": "uncertain",
                "suspected": False,
                "evidence": "face tracking or mouth position is unreliable",
                "duration_ms": None,
                "observations": 0,
                "last_seen_ms": None,
                "detection": None,
            }

        best, proximity_reason = self._matching_detection(
            detections, key=key, face=face, mouth=mouth
        )
        if best is None:
            # Keep a very short gap to avoid erasing one episode between two
            # detector runs, but never keep a stale/old detection alive.
            if episode.last_seen is not None and (now - episode.last_seen) * 1000.0 <= self.config.gap_ms:
                pass
            else:
                episode.reset()
            last_seen_ms = (
                None
                if episode.last_seen is None
                else round(max(0.0, now - episode.last_seen) * 1000.0, 1)
            )
            return {
                "name": name,
                "state": "clear",
                "suspected": False,
                "evidence": proximity_reason,
                "duration_ms": None,
                "observations": episode.observations,
                "last_seen_ms": last_seen_ms,
                "detection": None,
            }

        if episode.since is None:
            episode.since = now
            episode.observations = 0
            episode.last_object_timestamp = None
        episode.last_seen = now
        if episode.last_object_timestamp != snapshot_timestamp:
            episode.observations += 1
            episode.last_object_timestamp = snapshot_timestamp
        episode.duration_ms = max(0.0, (now - episode.since) * 1000.0)
        episode.detection = best

        minimum_ms = {
            "phone": self.config.phone_min_ms,
            "drinking": self.config.drinking_min_ms,
            "smoking": self.config.smoking_min_ms,
        }[key]
        enough_time = episode.duration_ms >= minimum_ms
        enough_observations = episode.observations >= self.config.min_observations
        suspected = enough_time and enough_observations
        return {
            "name": name,
            "state": "suspected" if suspected else "possible",
            "suspected": suspected,
            "evidence": (
                f"{proximity_reason}; {episode.observations} observation(s), "
                f"{episode.duration_ms:.0f} ms; detector confidence "
                f"{best.confidence:.2f}"
            ),
            "duration_ms": round(episode.duration_ms, 1),
            "observations": episode.observations,
            "last_seen_ms": 0.0,
            "detection": best.to_dict(now),
        }

    def update(
        self,
        measurements: dict[str, Any],
        snapshot: ObjectSnapshot,
        now: float,
    ) -> dict[str, dict[str, Any]]:
        status = snapshot.effective_status(now)
        face = _norm_face(measurements)
        mouth = _norm_mouth(measurements, face) if face is not None else None
        face_ok = (
            measurements.get("quality") == "good"
            and measurements.get("eye_state") in {"open", "closed"}
            and measurements.get("state") not in {"no_face", "uncertain"}
            and face is not None
        )
        detections = self._fresh_detections(snapshot, now) if status == "running" else []
        # Filter individual stale detections even when the snapshot's frame is
        # fresh.  A sidecar must not be able to hold an old bottle on screen.
        return {
            key: self._update_one(
                key,
                detections=detections,
                snapshot_status=status,
                face_ok=face_ok,
                face=face,
                mouth=mouth,
                now=now,
                snapshot_timestamp=snapshot.timestamp,
            )
            for key in BEHAVIOUR_KEYS
        }


class FusionEngine:
    """Combine face measurements and object evidence into one status model."""

    def __init__(self, config: BehaviourConfig | None = None):
        self.behaviour = BehaviourFusion(config)

    def update(
        self,
        measurements: dict[str, Any],
        snapshot: ObjectSnapshot,
        now: float,
    ) -> dict[str, Any]:
        main_state, main_reason = driver_state(measurements)
        behaviours = self.behaviour.update(measurements, snapshot, now)
        active = [key for key, value in behaviours.items() if value["suspected"]]
        object_status = snapshot.to_dict(now)
        tracking_good = (
            measurements.get("quality") == "good"
            and measurements.get("eye_state") in {"open", "closed"}
            and measurements.get("state") not in {"no_face", "uncertain"}
        )
        tracking = {
            "state": "good" if tracking_good else "uncertain",
            "reason": measurements.get("quality_detail", "unknown"),
            "confidence": measurements.get("tracking_confidence"),
            "confidence_basis": measurements.get("tracking_confidence_basis", ""),
            "source": "face landmarks",
        }
        events = {
            "drowsy": measurements.get("drowsiness_state") == "drowsy",
            "eyes_closed": measurements.get("eye_state") == "closed",
            "yawning": measurements.get("mouth_state") == "yawning",
            "distracted": measurements.get("attention_state") == "distracted",
            "phone_use": "phone" in active,
            "drinking": "drinking" in active,
            "smoking": "smoking" in active,
        }
        return {
            "main_state": main_state,
            "main_reason": main_reason,
            "tracking": tracking,
            "independent": {
                "eye_state": measurements.get("eye_state"),
                "drowsiness": measurements.get("drowsiness_state"),
                "attention": measurements.get("attention_state"),
                "yawning": measurements.get("mouth_state") == "yawning",
            },
            "behaviours": behaviours,
            "active_behaviours": active,
            "events": events,
            "detections": object_status["detections"],
            "object": object_status,
        }
