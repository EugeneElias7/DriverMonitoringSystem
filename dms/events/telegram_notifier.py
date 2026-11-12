"""Telegram Bot API ``sendPhoto`` client with bounded retries.

The notifier is deliberately independent of the camera and behaviour layers.
It reads credentials from the environment, never logs the bot token, and turns
all network failures into a result instead of raising into the DMS loop.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NotificationResult:
    sent: bool
    error: str | None = None


class _Retryable(Exception):
    """Internal marker for a failure worth retrying."""


class TelegramNotifier:
    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        timeout_s: float = 15.0,
        retries: int = 3,
        retry_delay_s: float = 1.0,
    ):
        self._token = (
            token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ).strip()
        self._chat_id = (
            chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        ).strip()
        self.timeout_s = max(1.0, float(timeout_s))
        self.retries = max(0, int(retries))
        self.retry_delay_s = max(0.0, float(retry_delay_s))

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def send_photo(self, image_path: str | Path, caption: str) -> NotificationResult:
        if not self.configured:
            return NotificationResult(False, "telegram credentials not configured")

        path = Path(image_path)
        if not path.is_file():
            return NotificationResult(False, "evidence image not found")
        try:
            photo = path.read_bytes()
        except OSError as exc:
            return NotificationResult(
                False, f"could not read evidence image ({type(exc).__name__})"
            )

        last_error = "telegram request failed"
        attempts = self.retries + 1
        for attempt in range(attempts):
            try:
                self._post_photo(path.name, photo, caption)
                return NotificationResult(True)
            except _Retryable as exc:
                last_error = self._redact(str(exc) or last_error)
            except Exception as exc:  # keep the DMS alive on any API failure
                return NotificationResult(False, self._redact(self._safe_error(exc)))

            if attempt + 1 < attempts and self.retry_delay_s:
                time.sleep(self.retry_delay_s * (2**attempt))

        return NotificationResult(
            False, self._redact(f"{last_error} after {attempts} attempt(s)")
        )

    def _post_photo(self, filename: str, photo: bytes, caption: str) -> None:
        boundary = "----DMSBoundary" + uuid.uuid4().hex
        body = _multipart_body(
            boundary,
            fields={
                "chat_id": self._chat_id,
                "caption": caption,
                "parse_mode": "HTML",
            },
            files={
                "photo": (filename, "image/jpeg", photo),
            },
        )
        url = f"https://api.telegram.org/bot{self._token}/sendPhoto"
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = self._http_detail(exc)
            if exc.code == 429 or exc.code >= 500:
                raise _Retryable(detail) from None
            raise RuntimeError(detail) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _Retryable(self._network_error(exc)) from None

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RuntimeError("invalid Telegram response") from None
        if not isinstance(payload, dict) or not payload.get("ok"):
            description = payload.get("description") if isinstance(payload, dict) else None
            if description:
                raise RuntimeError(f"Telegram rejected the photo: {str(description)[:200]}")
            raise RuntimeError("Telegram rejected the photo")

    @staticmethod
    def _http_detail(exc: urllib.error.HTTPError) -> str:
        description = None
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            if isinstance(payload, dict):
                description = payload.get("description")
        except (OSError, UnicodeDecodeError, ValueError):
            pass
        if description:
            return f"HTTP {exc.code}: {str(description)[:200]}"
        return f"HTTP {exc.code}"

    @staticmethod
    def _network_error(exc: BaseException) -> str:
        reason = getattr(exc, "reason", None)
        name = type(reason if reason is not None else exc).__name__
        return f"network error ({name})"

    @staticmethod
    def _safe_error(exc: BaseException) -> str:
        if isinstance(exc, RuntimeError):
            return str(exc)
        return f"telegram request failed ({type(exc).__name__})"

    def _redact(self, text: str) -> str:
        """Make sure a server or proxy response cannot leak the bot token."""
        if self._token:
            return text.replace(self._token, "<redacted>")
        return text


def _multipart_body(
    boundary: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, str, bytes]],
) -> bytes:
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for name, (filename, content_type, payload) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body)
