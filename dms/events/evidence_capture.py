"""Write one JPEG evidence frame per confirmed event."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class EvidenceCapture:
    """Save camera frames without resizing or drawing on them."""

    def __init__(self, root: str | Path = "evidence", jpeg_quality: int = 85):
        self.root = Path(root)
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))

    def save(self, frame: np.ndarray, folder: str, event_id: str) -> Path:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("empty evidence frame")

        path = self.root / folder / f"{event_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        wrote = cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not wrote or not path.exists() or path.stat().st_size == 0:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise OSError(f"could not write evidence image: {path}")
        return path
