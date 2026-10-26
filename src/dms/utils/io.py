"""Filesystem helpers that tolerate Windows Unicode paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray | None:
    """Read an image without relying on OpenCV's ANSI filename handling."""
    file_path = Path(path)
    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(file_path))


def imwrite(path: str | Path, image: np.ndarray) -> bool:
    """Write an image, creating parent directories and using an extension-aware codec."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower() or ".jpg"
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    try:
        ok, encoded = cv2.imencode(suffix, image, params)
        if not ok:
            return False
        encoded.tofile(str(file_path))
        return True
    except Exception:
        return bool(cv2.imwrite(str(file_path), image))


def write_json(path: str | Path, payload: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return file_path


def append_jsonl(handle: Any, payload: Any) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
