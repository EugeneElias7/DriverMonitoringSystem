"""Common model/module lifecycle and health tracking."""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np

from .types import DetectorHealth, Detection


def validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be HxWx3 BGR, got shape {getattr(frame, 'shape', None)}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("frame is too small")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must be uint8 BGR, got dtype {frame.dtype}")
    if frame.size == 0:
        raise ValueError("frame is empty")


class HealthMixin:
    """Lifecycle plus best-effort error isolation for a single module."""

    def __init__(self, name: str):
        self.name = name
        self.health = DetectorHealth(name=name)
        self._latencies: deque[float] = deque(maxlen=100)
        self._loaded = False

    def _begin(self) -> float:
        return time.perf_counter()

    def _finish(self, started: float, error: Exception | None = None) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.health.last_latency_ms = elapsed_ms
        self._latencies.append(elapsed_ms)
        self.health.mean_latency_ms = sum(self._latencies) / len(self._latencies)
        self.health.frames_processed += 1
        if error is None:
            self.health.failures = self.health.failures
            if self.health.state == "error":
                self.health.state = "ok"
                self.health.message = ""
            return
        self.health.failures += 1
        self.health.state = "error"
        self.health.last_error = f"{type(error).__name__}: {error}"
        self.health.message = self.health.last_error
        # Keep the traceback available for debugging tools, not JSON output.
        self.health.last_traceback = traceback.format_exc()  # type: ignore[attr-defined]

    def _mark_ready(self, message: str = "") -> None:
        self._loaded = True
        self.health.ready = True
        self.health.state = "ok"
        self.health.message = message

    def _mark_disabled(self, message: str) -> None:
        self._loaded = False
        self.health.ready = False
        self.health.state = "disabled"
        self.health.message = message

    def _mark_error(self, message: str) -> None:
        self._loaded = False
        self.health.ready = False
        self.health.state = "error"
        self.health.message = message
        self.health.last_error = message

    def get_status(self) -> DetectorHealth:
        return self.health

    def close(self) -> None:
        self._loaded = False
        self.health.ready = False
        self.health.state = "closed"


class BaseDetector(HealthMixin, ABC):
    """Common frame detector interface required by the project brief."""

    def __init__(self, name: str, config: Any = None):
        super().__init__(name)
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load weights and validate tensor/class metadata. Must be idempotent."""

    @abstractmethod
    def _detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        """Run inference. May raise; the public ``detect`` isolates failures."""

    def detect(self, frame: np.ndarray, timestamp: float) -> list[Detection]:
        if not self._loaded:
            return []
        started = self._begin()
        error: Exception | None = None
        detections: list[Detection] = []
        try:
            validate_frame(frame)
            detections = self._detect(frame, timestamp)
        except Exception as exc:  # isolate a module failure from the pipeline
            error = exc
            detections = []
        finally:
            self._finish(started, error)
        return detections

    def reset(self) -> None:
        """Clear sequence state, if any. Weights remain loaded."""


class BaseFeature(HealthMixin, ABC):
    """Interface for landmark-derived feature modules (no extra frame model)."""

    feature = "feature"

    def __init__(self, name: str, config: Any = None):
        super().__init__(name)
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Prepare the module (usually no weights). Must be idempotent."""

    @abstractmethod
    def _update(self, face: Any, timestamp: float) -> Any:
        """Update the module. May raise; public ``update`` isolates failures."""

    def update(self, face: Any, timestamp: float) -> Any:
        from .types import FeatureResult

        if not self._loaded:
            return FeatureResult(
                feature=self.feature,
                state="unavailable",
                timestamp=timestamp,
                valid=False,
                message=self.health.message,
            )
        started = self._begin()
        error: Exception | None = None
        try:
            result = self._update(face, timestamp)
        except Exception as exc:  # keep other modules alive
            error = exc
            result = FeatureResult(
                feature=self.feature,
                state="error",
                timestamp=timestamp,
                valid=False,
                message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._finish(started, error)
        return result
