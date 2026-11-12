"""Append-only JSONL event log and event-id allocation."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_EVENT_ID_RE = re.compile(r"^EVT-(\d{8})-(\d+)$")


class EventLogger:
    """Allocate sequential IDs and append one JSON object per line.

    IDs are scanned from the existing log on startup, so restarting the app
    does not reuse an ID from earlier the same day.
    """

    def __init__(self, path: str | Path = "logs/events.jsonl"):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq_by_date: dict[str, int] = {}
        self._scan_all()

    def _scan_all(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    self._remember_line(line)
        except OSError:
            return

    def _remember_line(self, line: str) -> None:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            return
        if not isinstance(record, dict):
            return
        event_id = record.get("event_id")
        if not isinstance(event_id, str):
            return
        match = _EVENT_ID_RE.match(event_id)
        if match is None:
            return
        date, sequence = match.group(1), int(match.group(2))
        self._seq_by_date[date] = max(self._seq_by_date.get(date, 0), sequence)

    def next_event_id(self, at: datetime) -> str:
        date = at.strftime("%Y%m%d")
        with self._lock:
            sequence = self._seq_by_date.get(date, 0) + 1
            self._seq_by_date[date] = sequence
        return f"EVT-{date}-{sequence:06d}"

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
