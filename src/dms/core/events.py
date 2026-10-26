"""Throttled structured event creation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from .types import Event


class EventManager:
    """Emit state transitions with a per-feature/kind cooldown.

    This is intentionally simple: it prevents event floods but is not a temporal
    debouncer. Modules decide when a transition is meaningful.
    """

    def __init__(self, cooldown_ms: float = 3000.0):
        self.cooldown_ms = float(cooldown_ms)
        self._last: dict[str, float] = {}
        self.events: list[Event] = []
        self._clock = time.monotonic

    def emit(
        self,
        *,
        timestamp: float,
        feature: str,
        kind: str,
        message: str,
        severity: str = "info",
        confidence: float | None = None,
        data: dict[str, Any] | None = None,
        cooldown_key: str | None = None,
    ) -> Event | None:
        key = cooldown_key or f"{feature}:{kind}"
        now_ms = float(timestamp) * 1000.0
        last_ms = self._last.get(key)
        if last_ms is not None and now_ms - last_ms < self.cooldown_ms:
            return None
        event = Event(
            id=uuid.uuid4().hex[:12],
            timestamp=float(timestamp),
            feature=feature,
            kind=kind,
            severity=severity,
            message=message,
            data=dict(data or {}),
            confidence=confidence,
        )
        self._last[key] = now_ms
        self.events.append(event)
        return event

    def reset(self) -> None:
        self._last.clear()
        self.events.clear()

    def take(self) -> list[Event]:
        out = self.events
        self.events = []
        return out
