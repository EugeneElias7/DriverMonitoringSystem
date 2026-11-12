"""Load Telegram credentials from a local ``.env`` file.

The file is optional and exists for local convenience. Environment variables
already set by the shell or service take priority, so a Pi systemd unit can
still provide the credentials without touching the project directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def load_telegram_env(
    path: str | Path | None = None,
    override: bool = False,
) -> Path | None:
    """Load ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from a dotenv file.

    Returns the path that was read, or ``None`` when no file is available.
    Only the two Telegram keys are imported; other dotenv entries are ignored.
    """
    env_path = Path(path) if path is not None else _find_env_file()
    if env_path is None or not env_path.is_file():
        return None

    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return None

    for line in text.splitlines():
        key, value = _parse_line(line)
        if key is None:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def _find_env_file() -> Path | None:
    custom = os.environ.get("DMS_ENV_FILE", "").strip()
    if custom:
        return Path(custom)

    project_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / ".env", project_root / ".env"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_line(line: str) -> tuple[str | None, str]:
    text = line.strip()
    if not text or text.startswith("#"):
        return None, ""
    if text.startswith("export "):
        text = text[7:].lstrip()
    if "=" not in text:
        return None, ""

    key, value = text.split("=", 1)
    key = key.strip()
    if key not in _ALLOWED_KEYS:
        return None, ""

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value
