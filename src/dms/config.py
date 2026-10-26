"""Configuration loading and validation.

The default YAML is the single source of truth. Profiles are deep-merged on top
of it, then explicit CLI overrides are applied last.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """Raised for malformed or internally inconsistent configuration."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Nested dictionary with dotted access and path resolution."""

    def __init__(self, data: dict[str, Any], base_dir: Path | None = None):
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be a mapping")
        self.data = copy.deepcopy(data)
        self.base_dir = Path(base_dir or PROJECT_ROOT)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        node = self.data
        parts = path.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"configuration section {name!r} is missing or not a mapping")
        return value

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.base_dir / candidate).resolve()

    def copy(self) -> Config:
        return Config(copy.deepcopy(self.data), self.base_dir)

    def merged(self, override: dict[str, Any]) -> Config:
        return Config(_deep_merge(self.data, override), self.base_dir)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def validate(self) -> None:
        _require(self.get("face.num_faces", 1) >= 1, "face.num_faces must be >= 1")
        for key in (
            "face.min_detection_confidence",
            "face.min_presence_confidence",
            "face.min_tracking_confidence",
            "face.smoothing_alpha",
            "drowsiness.ear_adapt_alpha",
            "drowsiness.closed_ratio",
            "drowsiness.open_ratio",
            "drowsiness.perclos_drowsy",
            "smoking.confidence_threshold",
            "smoking.nms_iou",
        ):
            value = self.get(key)
            if value is not None:
                _require(0.0 <= float(value) <= 1.0, f"{key} must be in [0, 1]")

        _require(float(self.get("drowsiness.blink_max_ms")) > 0, "blink_max_ms must be > 0")
        _require(
            float(self.get("drowsiness.blink_max_ms"))
            < float(self.get("drowsiness.prolonged_ms"))
            < float(self.get("drowsiness.microsleep_ms")),
            "expected blink_max_ms < prolonged_ms < microsleep_ms",
        )
        _require(
            float(self.get("drowsiness.closed_ratio"))
            < float(self.get("drowsiness.open_ratio")),
            "closed_ratio must be lower than open_ratio",
        )
        _require(
            float(self.get("drowsiness.face_loss_grace_ms"))
            <= float(self.get("drowsiness.face_lost_event_ms")),
            "face_loss_grace_ms must be <= face_lost_event_ms",
        )
        _require(
            float(self.get("distraction.glance_max_ms"))
            <= float(self.get("distraction.mirror_glance_max_ms"))
            <= float(self.get("distraction.distraction_hold_ms")),
            "expected glance_max_ms <= mirror_glance_max_ms <= distraction_hold_ms",
        )
        for key in (
            "distraction.yaw_threshold_deg",
            "distraction.pitch_down_threshold_deg",
            "distraction.pitch_up_threshold_deg",
            "head_pose.yaw_threshold_deg",
            "head_pose.pitch_threshold_deg",
        ):
            _require(float(self.get(key)) > 0, f"{key} must be > 0")

        backend = self.get("smoking.backend")
        _require(backend in {"yolo_onnx", "tflite_ssd", "none"}, f"unsupported smoking.backend: {backend!r}")
        class_names = self.get("smoking.class_names")
        _require(isinstance(class_names, list) and class_names, "smoking.class_names must be a non-empty list")
        trigger = self.get("smoking.trigger_classes")
        _require(isinstance(trigger, list) and trigger, "smoking.trigger_classes must be a non-empty list")
        target = self.get("smoking.expected_target_class")
        _require(target in trigger, "expected_target_class must be listed in trigger_classes")
        _require(int(self.get("smoking.inference_interval", 1)) >= 1, "inference_interval must be >= 1")
        _require(int(self.get("pipeline.process_every_n_frames", 1)) >= 1, "process_every_n_frames must be >= 1")
        if int(self.get("smoking.inference_interval", 1)) > 1:
            # This is a warning, not a hard failure: the user may deliberately trade
            # temporal coverage for CPU. Keep a machine-readable note in the config.
            self.set("smoking._interval_warning", "inference_interval > 1 increases missed-event risk")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load default YAML, merge a profile, apply overrides, and validate."""
    if not DEFAULT_CONFIG_PATH.exists():
        raise ConfigError(f"default config not found: {DEFAULT_CONFIG_PATH}")
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if path:
        profile_path = Path(path)
        if not profile_path.is_absolute():
            profile_path = (PROJECT_ROOT / profile_path).resolve()
        with profile_path.open("r", encoding="utf-8") as handle:
            profile = yaml.safe_load(handle) or {}
        data = _deep_merge(data, profile)
    if overrides:
        data = _deep_merge(data, _parse_overrides(overrides))
    config = Config(data)
    config.validate()
    return config


def _parse_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{'smoking.backend': 'none'}`` into nested dictionaries."""
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        node = nested
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested
