"""Turn confirmed behaviour signals into evidence, JSONL records, and alerts.

The camera loop calls :meth:`EventManager.process` once per frame with the
currently active signal keys. Confirmed signals are mapped to four event
categories; each category fires once per continuous event, then re-arms only
after it has been clear for a short grace period. Image writing and Telegram
delivery happen on one background worker so the camera loop never waits for
disk or network I/O.
"""

from __future__ import annotations

import html
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .event_logger import EventLogger
from .evidence_capture import EvidenceCapture
from .telegram_notifier import NotificationResult, TelegramNotifier

# Behaviour signals from the analyzer, grouped into event categories.
_SIGNAL_CATEGORY = {
    "drowsy_microsleep": "drowsiness",
    "drowsy_perclos": "drowsiness",
    "drowsy_yawning": "yawning",
    "phone_use": "phone",
    "smoking": "smoking",
}

# category -> (alert_type written to JSONL, evidence subfolder)
_CATEGORY_INFO = {
    "drowsiness": ("DROWSINESS", "drowsiness"),
    "yawning": ("YAWNING", "yawning"),
    "phone": ("PHONE_USAGE", "phone"),
    "smoking": ("SMOKING", "smoking"),
}

_ALERT_EMOJI = {
    "DROWSINESS": "😴",
    "YAWNING": "🥱",
    "PHONE_USAGE": "📱",
    "SMOKING": "🚬",
    "MANUAL TEST": "🧪",
}
_FALLBACK_EMOJI = "⚠️"

_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


@dataclass(frozen=True)
class _EventJob:
    event_id: str
    alert_type: str
    folder: str
    timestamp: datetime
    frame: object


class EventManager:
    """Capture one evidence image and one notification per continuous event."""

    def __init__(
        self,
        vehicle_number: str = "UNKNOWN",
        evidence_dir: str | Path = "evidence",
        log_path: str | Path = "logs/events.jsonl",
        notifier: TelegramNotifier | None = None,
        capture: EvidenceCapture | None = None,
        rearm_grace_s: float = 2.0,
    ):
        self.vehicle_number = str(vehicle_number).strip() or "UNKNOWN"
        self.rearm_grace_s = max(0.0, float(rearm_grace_s))
        self._capture = capture if capture is not None else EvidenceCapture(evidence_dir)
        self._logger = EventLogger(log_path)
        self._notifier = notifier if notifier is not None else TelegramNotifier()

        self._state_lock = threading.Lock()
        self._fired: dict[str, bool] = {}
        self._last_active: dict[str, float] = {}

        self._queue: queue.Queue[_EventJob | None] = queue.Queue()
        self._condition = threading.Condition()
        self._pending = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="dms-events", daemon=True
        )
        self._thread.start()

    @property
    def telegram_configured(self) -> bool:
        return bool(getattr(self._notifier, "configured", False))

    def process(
        self,
        active_keys,
        frame,
        now: float | None = None,
        timestamp: datetime | None = None,
    ) -> list[str]:
        """Record any newly confirmed events and return their event IDs.

        ``active_keys`` is any iterable of behaviour signal keys (a live
        generator is fine). ``now`` is a monotonic timestamp from the caller;
        tests can provide a deterministic value.
        """
        keys = set(active_keys)
        at = time.monotonic() if now is None else float(now)
        when = timestamp if timestamp is not None else datetime.now()
        active = {
            _SIGNAL_CATEGORY[key]
            for key in keys
            if key in _SIGNAL_CATEGORY
        }

        created: list[str] = []
        with self._state_lock:
            for category, (alert_type, folder) in _CATEGORY_INFO.items():
                if category in active:
                    self._last_active[category] = at
                    if self._fired.get(category, False):
                        continue
                    self._fired[category] = True
                    event_id = self._logger.next_event_id(when)
                    created.append(event_id)
                    self._enqueue(
                        _EventJob(
                            event_id=event_id,
                            alert_type=alert_type,
                            folder=folder,
                            timestamp=when,
                            frame=_copy_frame(frame),
                        )
                    )
                else:
                    last = self._last_active.get(category)
                    if (
                        self._fired.get(category, False)
                        and last is not None
                        and at - last >= self.rearm_grace_s
                    ):
                        self._fired[category] = False
        return created

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Wait until all queued evidence/notification jobs have finished."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=10.0)

    def _enqueue(self, job: _EventJob) -> None:
        with self._condition:
            self._pending += 1
        self._queue.put(job)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                self._handle(job)
            except Exception as exc:  # never let evidence stop the worker
                print(
                    f"note: event record skipped ({type(exc).__name__})",
                    flush=True,
                )
            finally:
                self._queue.task_done()
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()

    def _handle(self, job: _EventJob) -> None:
        image_path = ""
        try:
            image_path = str(self._capture.save(job.frame, job.folder, job.event_id))
        except Exception as exc:
            self._write_record(
                job,
                image_path="",
                status="FAILED",
                error=f"image save failed ({type(exc).__name__})",
            )
            return

        caption = format_caption(
            self.vehicle_number,
            job.alert_type,
            job.timestamp,
            job.event_id,
        )
        try:
            result = self._notifier.send_photo(image_path, caption)
        except Exception as exc:  # notifier should return, but stay defensive
            result = NotificationResult(
                False, f"notification failed ({type(exc).__name__})"
            )

        self._write_record(
            job,
            image_path=image_path,
            status="SENT" if result.sent else "FAILED",
            error=result.error,
        )

    def _write_record(
        self,
        job: _EventJob,
        image_path: str,
        status: str,
        error: str | None,
    ) -> None:
        record = {
            "event_id": job.event_id,
            "vehicle_number": self.vehicle_number,
            "alert_type": job.alert_type,
            "timestamp": job.timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "image_path": image_path,
            "notification_status": status,
        }
        if error:
            record["notification_error"] = str(error)
        self._logger.append(record)


def _copy_frame(frame):
    """Snapshot the frame because the camera buffer may be reused."""
    copy = getattr(frame, "copy", None)
    return copy() if callable(copy) else frame


def format_caption(
    vehicle_number: str,
    alert_type: str,
    timestamp: datetime,
    event_id: str,
) -> str:
    """Build the HTML caption used for the Telegram photo."""
    emoji = _ALERT_EMOJI.get(alert_type, _FALLBACK_EMOJI)
    return (
        "🚨 <b>DMS ALERT</b>\n\n"
        f"Vehicle Number: {html.escape(str(vehicle_number))}\n"
        f"Alert Type: {emoji} {html.escape(str(alert_type))}\n"
        f"Time: {_format_time(timestamp)}\n"
        f"Event ID: {html.escape(str(event_id))}"
    )


def _format_time(timestamp: datetime) -> str:
    hour = timestamp.hour % 12 or 12
    suffix = "AM" if timestamp.hour < 12 else "PM"
    return (
        f"{timestamp.day:02d} {_MONTHS[timestamp.month - 1]} {timestamp.year}, "
        f"{hour:02d}:{timestamp.minute:02d} {suffix}"
    )
