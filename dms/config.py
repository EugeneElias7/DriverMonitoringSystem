"""Typed configuration for the driver-monitoring app.

The face path is OpenCV YuNet + Google's 478-point landmark ONNX. One or
more YOLOv8/YOLO11 ONNX detectors supply driver-object classes (cigarette,
phone, drinking, eating, seatbelt). Thresholds are tuned against the bundled
test clips and the PC webcam, not claimed as universal operating points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class FaceConfig:
    detector_model_path: str = "models/face_detection_yunet_2023mar.onnx"
    landmark_model_path: str = "models/2d106det.onnx"
    min_detection_confidence: float = 0.2


@dataclass
class DetectorConfig:
    # A readable name used in the preview/status line and in errors.
    name: str = "driver"
    model_path: str = "models/best.onnx"
    classes_path: str = "models/driver_classes.txt"
    # Optional keep-list: any class not listed is reported as an empty label
    # and dropped. Class ids stay aligned with the model output.
    classes_whitelist: list[str] | None = None
    # Per-class confidence floors, for classes that need a different bar than
    # the global threshold.
    class_conf_overrides: dict[str, float] = field(default_factory=dict)
    input_size: int = 640
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    # Inference is far too slow to run inline with the face loop on a Pi.
    target_hz: float = 2.0
    enabled: bool = True


@dataclass
class AudioConfig:
    # Spoken driver alerts through the machine's speakers: pre-recorded WAV
    # phrases with an OS text-to-speech fallback, no extra dependencies.
    # Repeats while a condition stays active are paced by alert_cooldown_s.
    enabled: bool = True
    wav_dir: str = "audio"
    # Optional ALSA device for aplay on Linux, e.g. "plughw:Headphones".
    # Empty keeps ALSA's "default" PCM, which some Pi OS setups cannot open.
    device: str = ""


@dataclass
class Thresholds:
    # --- drowsiness ---
    # 0.33 is this driver's open-eye EAR; 0.28 catches heavy lids and squints
    # before the eyes fully shut.
    ear_closed: float = 0.15
    perclos_window_s: float = 4.0
    perclos_alert: float = 0.35
    microsleep_s: float = 1.2
    # A real yawn opens the mouth well past speech; 0.55 with a 1.0 s hold
    # catches it without firing on normal talking or a drag on a cigarette.
    yawn_mar: float = 0.55
    yawn_s: float = 1.0
    face_missing_s: float = 2.0
    # A hand over the face, a profile view or a landmark glitch produces
    # impossible EAR/MAR values. Those frames are skipped instead of being
    # counted as "eyes closed": on a 1 fps Pi frame rate one bad sample would
    # otherwise be enough to trigger a false microsleep alarm.
    drowsy_valid_ear_max: float = 0.50
    drowsy_valid_mar_max: float = 1.20
    # Generous absolute pose bounds: a desk-mounted Pi camera often sees the
    # driver at 40-45 degrees of yaw even when they face the screen. The EAR
    # range above is the primary guard against lost landmarks.
    drowsy_max_yaw_deg: float = 60.0
    drowsy_max_pitch_deg: float = 50.0
    # Require a run of closed samples, so a single glitch cannot alarm even
    # when the frame rate is very low. The timer starts on the first closed
    # frame; these samples only gate the report.
    microsleep_min_frames: int = 3
    # PERCLOS is a ratio over the window; with only a handful of frames it is
    # noise. Below this sample count the ratio is not reported at all.
    perclos_min_samples: int = 10

    # --- smoking ---
    # A single false "Stop smoking" is unacceptable, so a box must first pass
    # the tight mouth/size gate below and then hold for a time span longer
    # than one detector period. The dedicated cigarette model publishes at
    # 2 Hz on the PC, so 1.0 s means at least two consecutive passes; a real
    # puff lasts longer than that and still alerts promptly.
    cigarette_s: float = 1.0
    # A cigarette box this confident uses `cigarette_s`.
    cigarette_high_conf: float = 0.60
    # A weaker box must persist for `cigarette_low_s`.
    cigarette_low_conf: float = 0.45
    cigarette_low_s: float = 1.5
    # How far the cigarette box centre may sit from the mouth, in face
    # heights. Real boxes measured on smoking clips land 0.01-0.27 away;
    # hands/phones at the chin land around 0.47-0.57, so 0.30 separates them.
    cigarette_gate_span: float = 0.30
    # A cigarette box larger than this many face heights is not an object at
    # the mouth. Tested real boxes reach 0.51; a phone held at the chin was
    # 0.61 and is rejected here.
    cigarette_max_size_span: float = 0.55
    # Cigarettes are elongated; a watch, ring or round pen cap is not. The
    # default 1.0 disables the check (PC profile); the Pi profile raises it.
    cigarette_min_aspect: float = 1.0
    # The mouth point must lie within this many face heights of the box
    # (zero when the mouth is inside it). A cigarette at the lips overlaps
    # the mouth; a watch on the cheek does not. The default is effectively
    # unlimited so only profiles that opt in are tightened.
    cigarette_mouth_pad_span: float = 1.0
    # Bridge one missed detector pass without re-announcing a finished puff.
    cigarette_hold_s: float = 1.0

    # --- drinking ---
    drinking_s: float = 1.5

    # --- eating ---
    eating_s: float = 1.5

    # --- seatbelt ---
    seatbelt_missing_s: float = 3.0
    # Desk tests have no belt in frame, so the absence rule fires forever and
    # drowns the log. Disabling keeps the monitor quiet without touching the
    # detection thresholds.
    seatbelt_check: bool = True

    # --- distraction ---
    yaw_away_deg: float = 28.0
    pitch_away_deg: float = 25.0
    # Seconds of face-present frames used to learn the driver's neutral head
    # pose (the camera is not on the driver's axis, and the pose estimate has
    # a constant bias per face). Look-away is suppressed until it is learned.
    gaze_warmup_s: float = 4.0
    looking_away_s: float = 1.2
    # A phone at the ear must persist long enough to outlast a cigarette or
    # hand that the general phone model briefly mistakes for one. A real call
    # lasts far longer, so 1.5 s still feels immediate.
    phone_call_s: float = 1.5
    phone_in_hand_s: float = 1.0
    # Weak phone detections at or above this confidence count only at the ear
    # anchor. That keeps a partially hidden phone in a call detectable while
    # not treating any weak box near the face as "in hand".
    phone_low_conf: float = 0.15
    # A phone counted as held near the face/chest must clear this confidence.
    # Real in-hand boxes on the COCO phone model score 0.45-0.9; the 0.2 floor
    # keeps those while dropping the 0.1-0.15 boxes the model invents on hands.
    phone_hand_conf: float = 0.20
    # Reject full-frame phone boxes: a held phone is never larger than the
    # driver's face by this factor.
    phone_max_size_span: float = 1.2
    # Keep the phone alert on through brief detector misses.
    phone_hold_s: float = 3.0


@dataclass
class CameraConfig:
    # Raw libcamera controls applied to the CSI camera after it starts, e.g.
    # {"AwbEnable": false, "ColourGains": [0.95, 1.4]}. Night-vision modules
    # (OV5647 without an IR-cut filter) render daylight magenta, which the
    # detectors were never trained on; fixed colour gains are the software
    # workaround. Keys are libcamera control names.
    controls: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    face: FaceConfig = field(default_factory=FaceConfig)
    detectors: list[DetectorConfig] = field(default_factory=lambda: [DetectorConfig()])
    thresholds: Thresholds = field(default_factory=Thresholds)
    audio: AudioConfig = field(default_factory=AudioConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    capture_width: int = 640
    capture_height: int = 480
    target_fps: int = 15
    # Repeat cadence for a condition that stays active: the warning (voice,
    # console, log) fires again every N seconds until the condition clears.
    alert_cooldown_s: float = 2.5
    log_path: str | None = None
    display: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        cfg = cls()
        if path is None:
            return cfg
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _apply_overrides(cfg, data)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def check_paths(self, root: str | Path = ".") -> list[str]:
        """Return human-readable problems with configured asset paths."""
        root = Path(root)
        problems: list[str] = []
        for label, rel in (
            ("face detector model", self.face.detector_model_path),
            ("face landmark model", self.face.landmark_model_path),
        ):
            if not (root / rel).exists():
                problems.append(f"{label} not found: {rel}")
        for detector in self.detectors:
            if not detector.enabled:
                continue
            for label, rel in (
                ("detector model", detector.model_path),
                ("classes file", detector.classes_path),
            ):
                if not (root / rel).exists():
                    problems.append(
                        f"{detector.name} {label} not found: {rel}"
                    )
        return problems


def _apply_overrides(target: Any, data: dict[str, Any]) -> None:
    known = {f.name for f in fields(target)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for name, value in data.items():
        current = getattr(target, name)
        if name == "detectors":
            if not isinstance(value, list):
                raise ValueError("detectors must be a list")
            setattr(
                target,
                name,
                [_build(DetectorConfig, item) for item in value],
            )
        elif is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(target, name, value)


def _build(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{cls.__name__} entries must be JSON objects")
    obj = cls()
    _apply_overrides(obj, data)
    return obj


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(item) for item in obj]
    return obj
