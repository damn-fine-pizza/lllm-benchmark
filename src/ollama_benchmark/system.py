"""Lightweight CPU / RAM sampling from ``/proc`` and ``/sys`` (no dependencies)."""

from __future__ import annotations

import glob
import os


def cpu_count() -> int:
    return os.cpu_count() or 1


def read_cpu_snapshot() -> tuple[int, int] | None:
    """Return ``(total_jiffies, idle_jiffies)`` from the aggregate ``/proc/stat``.

    CPU utilization is the change in ``(total - idle)`` over ``total`` between
    two snapshots, so callers diff consecutive readings.
    """
    try:
        with open("/proc/stat", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("cpu "):
                    parts = [int(x) for x in line.split()[1:]]
                    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle+iowait
                    return sum(parts), idle
    except (OSError, ValueError):
        pass
    return None


def cpu_percent(prev: tuple[int, int] | None,
                cur: tuple[int, int] | None) -> float | None:
    if not prev or not cur:
        return None
    dtotal = cur[0] - prev[0]
    didle = cur[1] - prev[1]
    if dtotal <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (dtotal - didle) / dtotal))


def read_loadavg() -> tuple[float, float, float] | None:
    """1/5/15-minute load averages from ``/proc/loadavg``."""
    try:
        with open("/proc/loadavg", encoding="ascii") as fh:
            p = fh.read().split()
        return float(p[0]), float(p[1]), float(p[2])
    except (OSError, ValueError, IndexError):
        return None


_FREQ_FILES: list[str] | None = None


def read_cpu_freq_mhz() -> float | None:
    """Average current core frequency (MHz) from cpufreq sysfs."""
    global _FREQ_FILES
    if _FREQ_FILES is None:
        _FREQ_FILES = glob.glob(
            "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")
    if not _FREQ_FILES:
        return None
    vals = []
    for path in _FREQ_FILES:
        try:
            with open(path, encoding="ascii") as fh:
                vals.append(int(fh.read().strip()))
        except (OSError, ValueError):
            pass
    return (sum(vals) / len(vals)) / 1000.0 if vals else None  # kHz -> MHz


_TEMP_PATH: str | None | bool = False  # False=unsearched, None=none, str=path


def read_cpu_temp_c() -> float | None:
    """CPU package temperature (°C) from a k10temp/coretemp/zenpower hwmon."""
    global _TEMP_PATH
    if _TEMP_PATH is False:
        _TEMP_PATH = _find_cpu_temp_path()
    if not _TEMP_PATH:
        return None
    try:
        with open(_TEMP_PATH, encoding="ascii") as fh:
            return int(fh.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _find_cpu_temp_path() -> str | None:
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(f"{hw}/name", encoding="ascii") as fh:
                name = fh.read().strip()
        except OSError:
            continue
        if name in ("k10temp", "coretemp", "zenpower"):
            inputs = sorted(glob.glob(f"{hw}/temp*_input"))
            if inputs:
                return inputs[0]
    return None


def read_mem() -> tuple[int, int] | None:
    """Return ``(used_bytes, total_bytes)`` (used = total - available)."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, val = line.partition(":")
                info[key.strip()] = int(val.split()[0]) * 1024  # kB -> bytes
        total = info["MemTotal"]
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        return max(0, total - avail), total
    except (OSError, KeyError, ValueError, IndexError):
        return None
