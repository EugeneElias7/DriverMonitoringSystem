"""Small, separate configuration for evidence events.

The existing DMS profiles stay untouched. This file only supplies the vehicle
number used in event records and Telegram captions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventConfig:
    vehicle_number: str = "UNKNOWN"

    @classmethod
    def load(cls, path: str | Path = "config_events.json") -> "EventConfig":
        """Load the event config, returning defaults when the file is absent."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not read event config: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("event config must be a JSON object")

        unknown = set(data) - {"vehicle_number"}
        if unknown:
            raise ValueError(f"unknown event config keys: {sorted(unknown)}")

        vehicle = data.get("vehicle_number", cls().vehicle_number)
        if not isinstance(vehicle, str) or not vehicle.strip():
            raise ValueError("vehicle_number must be a non-empty string")
        return cls(vehicle_number=vehicle.strip())
