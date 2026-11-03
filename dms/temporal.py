"""Time-based primitives shared by the behaviour monitors."""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Callable, Iterable


class TimeWindow:
    """Samples on a rolling time span, pruned by timestamp.

    Time-based rather than frame-based so behaviour stays comparable when the
    frame rate collapses on a loaded Pi.
    """

    def __init__(self, window_s: float, maxlen: int = 1024):
        self.window_s = float(window_s)
        self.maxlen = maxlen
        self._t: deque[float] = deque()
        self._v: deque[float] = deque()

    def push(self, t: float, value: float) -> None:
        self._t.append(t)
        self._v.append(value)
        self.prune(t)

    def prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._t and (self._t[0] < cutoff or len(self._t) > self.maxlen):
            self._t.popleft()
            self._v.popleft()

    def __len__(self) -> int:
        return len(self._v)

    @property
    def values(self) -> list[float]:
        return list(self._v)

    def span(self) -> float:
        """Seconds covered by the retained samples (0 for fewer than two)."""
        if len(self._t) < 2:
            return 0.0
        return self._t[-1] - self._t[0]

    def fraction(self, predicate: Callable[[float], bool]) -> float:
        if not self._v:
            return 0.0
        return sum(1 for v in self._v if predicate(v)) / len(self._v)

    def median(self) -> float:
        return statistics.median(self._v) if self._v else math.nan

    def mean(self) -> float:
        return statistics.fmean(self._v) if self._v else math.nan

    def count_in(self, now: float, lookback_s: float) -> int:
        """Samples whose timestamp falls within `lookback_s` of `now`."""
        cutoff = now - lookback_s
        return sum(1 for t in self._t if t >= cutoff)

    def clear(self) -> None:
        self._t.clear()
        self._v.clear()


class Sustained:
    """True once a condition has held continuously for `duration_s`.

    With `hold_s=0` (the default) it drops back to False the moment the
    condition breaks, so short gaps reset the timer. A non-zero `hold_s` keeps
    an already-active condition on through gaps shorter than that, which is
    what stops one missed detector pass from silencing a smoking or phone
    warning. The onset still uses `duration_s`; the hold only delays release.
    """

    def __init__(self, duration_s: float, hold_s: float = 0.0):
        self.duration_s = float(duration_s)
        self.hold_s = float(hold_s)
        self._since: float | None = None
        self._last_true: float | None = None
        self._active = False

    def update(self, now: float, condition: bool) -> bool:
        if condition:
            self._last_true = now
            if self._since is None:
                self._since = now
            if not self._active and (now - self._since) >= self.duration_s:
                self._active = True
            return self._active
        if self._active and self._last_true is not None:
            if (now - self._last_true) < self.hold_s:
                return True
        self._since = None
        self._last_true = None
        self._active = False
        return False

    def elapsed(self, now: float) -> float:
        return 0.0 if self._since is None else now - self._since

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._since = None
        self._last_true = None
        self._active = False


class RecentSamples:
    """The last few event values, evicted by both count and age.

    Blinks and head settles arrive at irregular, driver-dependent intervals, so
    a time window either holds too much history to see a change or goes empty
    during quiet periods. Capping by count bounds how much recent history
    dilutes the median; capping by age means a signal goes quiet once the
    behaviour stops happening, instead of reporting stale evidence forever.
    """

    def __init__(self, maxlen: int = 5, max_age_s: float = 15.0):
        self._items: deque[tuple[float, float]] = deque(maxlen=maxlen)
        self.max_age_s = float(max_age_s)

    def push(self, t: float, value: float) -> None:
        self._items.append((t, value))

    def values(self, now: float) -> list[float]:
        cutoff = now - self.max_age_s
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()
        return [value for _, value in self._items]

    def median(self, now: float) -> float:
        values = self.values(now)
        return statistics.median(values) if values else math.nan


class Baseline:
    """Per-driver reference value taken from an early warm-up period."""

    def __init__(self, min_samples: int = 8, cap: int = 512):
        self.min_samples = min_samples
        self._samples: deque[float] = deque(maxlen=cap)

    def add(self, value: float) -> None:
        if value is not None and math.isfinite(value):
            self._samples.append(value)

    @property
    def ready(self) -> bool:
        return len(self._samples) >= self.min_samples

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def value(self) -> float:
        return statistics.median(self._samples) if self._samples else math.nan

    def ratio(self, value: float) -> float:
        """value / baseline. NaN when the baseline is unusable."""
        base = self.value
        if not math.isfinite(base) or abs(base) < 1e-9 or not math.isfinite(value):
            return math.nan
        return value / base


def mean_finite(values: Iterable[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(finite) if finite else math.nan
