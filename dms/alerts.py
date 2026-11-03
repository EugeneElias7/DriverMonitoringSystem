"""Condition signals, alert de-duplication, and alert sinks."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable, TextIO


class Severity(IntEnum):
    INFO = 1
    MINOR = 2
    MAJOR = 3
    CRITICAL = 4


@dataclass(frozen=True)
class ConditionSignal:
    """A monitor's current view of one condition.

    Monitors report state every frame; `AlertManager` decides when a state
    transition is worth surfacing to the driver.
    """

    key: str
    severity: Severity
    message: str
    active: bool


@dataclass(frozen=True)
class Alert:
    key: str
    severity: Severity
    message: str
    timestamp: float

    def format(self) -> str:
        return f"[{self.severity.name:<8}] {self.message}"


class AlertManager:
    """Emits on rising edge, re-emits every cooldown while the condition
    stays active, and re-emits immediately on escalation. A condition that
    never clears keeps alerting until it does."""

    def __init__(
        self,
        sinks: Iterable[Callable[[Alert], None]] | None = None,
        cooldown_s: float = 15.0,
        cooldown_overrides: dict[str, float] | None = None,
    ):
        self._sinks = list(sinks or [])
        self.cooldown_s = float(cooldown_s)
        self._cooldown_overrides = dict(cooldown_overrides or {})
        self._active: dict[str, bool] = {}
        self._severity: dict[str, Severity] = {}
        self._last_emit: dict[str, float] = {}

    def add_sink(self, sink: Callable[[Alert], None]) -> None:
        self._sinks.append(sink)

    def update(self, now: float, signals: Iterable[ConditionSignal]) -> list[Alert]:
        emitted: list[Alert] = []
        for signal in signals:
            previous_active = self._active.get(signal.key, False)
            previous_severity = self._severity.get(signal.key, Severity.INFO)
            self._active[signal.key] = signal.active
            self._severity[signal.key] = signal.severity

            if not signal.active:
                continue
            escalated = previous_active and signal.severity > previous_severity
            cooldown = self._cooldown_overrides.get(signal.key, self.cooldown_s)
            last = self._last_emit.get(signal.key)
            if not escalated and last is not None and (now - last) < cooldown:
                continue

            alert = Alert(signal.key, signal.severity, signal.message, now)
            self._last_emit[signal.key] = now
            emitted.append(alert)
            for sink in self._sinks:
                sink(alert)
        return emitted

    def active_keys(self) -> list[str]:
        return sorted(k for k, v in self._active.items() if v)


class CsvAlertLog:
    """Append-only alert log; safe to use as an `AlertManager` sink."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if self.path.stat().st_size == 0:
            self._writer.writerow(["wall_clock", "monotonic", "severity", "key", "message"])
            self._fh.flush()

    def __call__(self, alert: Alert) -> None:
        self._writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                f"{alert.timestamp:.3f}",
                alert.severity.name,
                alert.key,
                alert.message,
            ]
        )
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "CsvAlertLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def console_sink(alert: Alert) -> None:
    print(f"{time.strftime('%H:%M:%S')} {alert.format()}", flush=True)
