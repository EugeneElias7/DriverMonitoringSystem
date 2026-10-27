"""Small, optional system-resource sampler.

Values are reported only when a real source is available.  If ``psutil`` is not
installed, the sampler falls back to Linux ``/proc``; on other systems it
returns ``None`` rather than inventing numbers.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:  # optional, but listed in requirements.txt
    import psutil  # type: ignore
except Exception:  # pragma: no cover - depends on deployment
    psutil = None


def _read_linux_meminfo() -> dict[str, float] | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    values: dict[str, float] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if parts:
                values[key] = float(parts[0]) / 1024.0  # MiB
    except Exception:
        return None
    return values


def _read_linux_cpu() -> tuple[float, float] | None:
    path = Path("/proc/stat")
    if not path.exists():
        return None
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].split()
        if not first or first[0] != "cpu":
            return None
        fields = [float(v) for v in first[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
        total = sum(fields)
        return idle, total
    except Exception:
        return None


def _read_temperature() -> float | None:
    for path in (
        Path("/sys/class/thermal/thermal_zone0/temp"),
        Path("/sys/class/hwmon/hwmon0/temp1_input"),
    ):
        try:
            if path.exists():
                raw = float(path.read_text(encoding="utf-8").strip())
                return round(raw / 1000.0 if raw > 1000 else raw, 1)
        except Exception:
            continue
    if psutil is not None:
        try:
            sensors = psutil.sensors_temperatures()
            for entries in sensors.values():
                for entry in entries:
                    if entry.current is not None:
                        return round(float(entry.current), 1)
        except Exception:
            pass
    return None


class ResourceMonitor:
    def __init__(self, *, min_interval_s: float = 1.0):
        self.min_interval_s = float(min_interval_s)
        self._last_sample = 0.0
        self._last_payload: dict[str, Any] = {
            "available": False,
            "cpu_percent": None,
            "process_cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "temperature_c": None,
        }
        self._prev_linux_cpu: tuple[float, float] | None = None
        self._prev_process_cpu: float | None = None
        self._prev_wall: float | None = None
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def sample(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force and now - self._last_sample < self.min_interval_s:
            return dict(self._last_payload)
        self._last_sample = now

        payload: dict[str, Any] = {
            "available": False,
            "cpu_percent": None,
            "process_cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "temperature_c": _read_temperature(),
        }

        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                process = psutil.Process(os.getpid())
                payload.update(
                    {
                        "available": True,
                        "cpu_percent": round(float(cpu), 1),
                        "ram_percent": round(float(memory.percent), 1),
                        "ram_used_mb": round(float(memory.used) / (1024 * 1024), 1),
                        "ram_total_mb": round(float(memory.total) / (1024 * 1024), 1),
                        "process_rss_mb": round(
                            float(process.memory_info().rss) / (1024 * 1024), 1
                        ),
                        "process_cpu_percent": round(
                            float(process.cpu_percent(interval=None)), 1
                        ),
                    }
                )
            except Exception:
                pass

        if not payload["available"]:
            meminfo = _read_linux_meminfo()
            if meminfo:
                total = meminfo.get("MemTotal")
                available = meminfo.get("MemAvailable")
                if total and available is not None:
                    used = max(0.0, total - available)
                    payload.update(
                        {
                            "available": True,
                            "ram_total_mb": round(total, 1),
                            "ram_used_mb": round(used, 1),
                            "ram_percent": round(used / total * 100.0, 1),
                        }
                    )
            cpu_now = _read_linux_cpu()
            if cpu_now is not None:
                payload["available"] = True
                if self._prev_linux_cpu is not None:
                    idle0, total0 = self._prev_linux_cpu
                    idle1, total1 = cpu_now
                    total_delta = total1 - total0
                    idle_delta = idle1 - idle0
                    if total_delta > 0:
                        payload["cpu_percent"] = round(
                            max(0.0, min(100.0, (1.0 - idle_delta / total_delta) * 100.0)),
                            1,
                        )
                self._prev_linux_cpu = cpu_now

        # A process-level fallback that also works on Windows without psutil.
        if payload["process_cpu_percent"] is None:
            wall = time.monotonic()
            process_cpu = time.process_time()
            if self._prev_wall is not None and self._prev_process_cpu is not None:
                elapsed = wall - self._prev_wall
                if elapsed > 0:
                    payload["process_cpu_percent"] = round(
                        max(0.0, (process_cpu - self._prev_process_cpu) / elapsed * 100.0),
                        1,
                    )
            self._prev_wall = wall
            self._prev_process_cpu = process_cpu

        self._last_payload = payload
        return dict(payload)
