"""Small performance helpers used by the CLI, benchmarks and health output."""

from __future__ import annotations

import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared dependency
    psutil = None  # type: ignore[assignment]


class Stopwatch:
    def __init__(self) -> None:
        self.elapsed_ms: float | None = None
        self._start = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


@contextmanager
def timed() -> Iterator[Stopwatch]:
    watch = Stopwatch()
    with watch:
        yield watch


def rss_mb() -> float:
    if psutil is None:
        return 0.0
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


class LatencyAccumulator:
    def __init__(self, maxlen: int = 500):
        self.samples: deque[float] = deque(maxlen=maxlen)

    def add(self, milliseconds: float) -> None:
        self.samples.append(float(milliseconds))

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "fps": None}
        values = sorted(self.samples)
        mean = statistics.fmean(values)
        p50 = _percentile(values, 0.50)
        p95 = _percentile(values, 0.95)
        return {
            "count": len(values),
            "mean_ms": round(mean, 3),
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "fps": round(1000.0 / mean, 3) if mean > 0 else None,
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
