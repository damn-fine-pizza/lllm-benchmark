"""AMD GPU sampling via ``rocm-smi`` (plus CPU/RAM from ``/proc``).

Samples VRAM usage, power, temperature and utilization — and system CPU load
and RAM — in a background thread so we can capture peak/average values while a
model is generating.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from . import system


ROCM_SMI = shutil.which("rocm-smi")
NVIDIA_SMI = shutil.which("nvidia-smi")
AMD_SMI = shutil.which("amd-smi")
XRT_SMI = shutil.which("xrt-smi")
_MIB = 1024 * 1024
_XRT_POWER_RE = re.compile(r"Estimated Power\s*:\s*([\d.]+)\s*Watts", re.IGNORECASE)

# Known monitoring tools, in preference order, with what each targets. Detection
# is extensible: add a tool + sampler and it participates automatically.
KNOWN_TOOLS = [
    ("rocm-smi", ROCM_SMI, "AMD GPU"),
    ("nvidia-smi", NVIDIA_SMI, "NVIDIA GPU"),
    ("amd-smi", AMD_SMI, "AMD GPU/APU (newer)"),
    ("xrt-smi", XRT_SMI, "AMD XDNA NPU"),
]


def available_tools() -> list[tuple[str, str, str]]:
    """Return ``[(name, path, target)]`` for monitoring tools found on PATH."""
    return [(n, p, t) for (n, p, t) in KNOWN_TOOLS if p]


def gpu_vendor() -> str | None:
    """Detect the GPU sampling backend: ``'amd'``, ``'nvidia'`` or ``None``.

    Prefers AMD when both tools are present; override with the ``GPU_VENDOR``
    environment variable (``amd`` / ``nvidia``).
    """
    forced = os.environ.get("GPU_VENDOR", "").strip().lower()
    if forced in ("amd", "rocm") and ROCM_SMI:
        return "amd"
    if forced in ("nvidia", "cuda") and NVIDIA_SMI:
        return "nvidia"
    if ROCM_SMI:
        return "amd"
    if NVIDIA_SMI:
        return "nvidia"
    return None


def gpu_available() -> bool:
    return gpu_vendor() is not None


# Backwards-compatible alias (some call sites still ask specifically for ROCm).
def rocm_available() -> bool:
    return gpu_vendor() == "amd"


def _to_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def sample_gpu() -> dict:
    """Return a single point-in-time reading per card, or ``{}`` on failure.

    Works with AMD (``rocm-smi``) or NVIDIA (``nvidia-smi``). Keys are normalized
    to: ``vram_used_b``, ``vram_total_b``, ``power_w``, ``temp_c``,
    ``gpu_use_pct``. Cards are keyed ``card0``, ``card1``, ... for both vendors.
    """
    vendor = gpu_vendor()
    if vendor == "amd":
        return _sample_rocm()
    if vendor == "nvidia":
        return _sample_nvidia()
    return {}


def _sample_rocm() -> dict:
    try:
        out = subprocess.run(
            [ROCM_SMI, "--showmeminfo", "vram", "--showpower",
             "--showtemp", "--showuse", "--json"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    try:
        raw = json.loads(out.stdout)
    except json.JSONDecodeError:
        return {}

    cards: dict = {}
    for card, fields in raw.items():
        if not card.startswith("card") or not isinstance(fields, dict):
            continue
        reading = {}
        for key, val in fields.items():
            if "VRAM Total Used Memory" in key:
                reading["vram_used_b"] = _to_float(val)
            elif "VRAM Total Memory" in key:
                reading["vram_total_b"] = _to_float(val)
            elif "Graphics Package Power" in key or key.startswith("Average Graphics Package Power"):
                reading["power_w"] = _to_float(val)
            elif "Temperature (Sensor edge)" in key:
                reading["temp_c"] = _to_float(val)
            elif "GPU use (%)" in key:
                reading["gpu_use_pct"] = _to_float(val)
        cards[card] = reading
    return cards


def _sample_nvidia() -> dict:
    """Query ``nvidia-smi`` (CSV) and normalize; memory is MiB -> bytes."""
    try:
        out = subprocess.run(
            [NVIDIA_SMI,
             "--query-gpu=index,memory.used,memory.total,power.draw,"
             "temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    if out.returncode != 0 or not out.stdout.strip():
        return {}

    cards: dict = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx = _to_float(parts[0])
        idx = int(idx) if idx is not None else len(cards)
        used = _to_float(parts[1])
        total = _to_float(parts[2])
        cards[f"card{idx}"] = {
            "vram_used_b": used * _MIB if used is not None else None,
            "vram_total_b": total * _MIB if total is not None else None,
            "power_w": _to_float(parts[3]),
            "temp_c": _to_float(parts[4]),
            "gpu_use_pct": _to_float(parts[5]),
        }
    return cards


def npu_available() -> bool:
    return XRT_SMI is not None


def sample_npu() -> dict:
    """Return a single point-in-time NPU reading (power only), or ``{}``.

    Uses ``xrt-smi`` (AMD XDNA / Ryzen AI). Unlike ``rocm-smi``/``nvidia-smi``,
    xrt-smi exposes no VRAM, temperature or utilization for the NPU — only an
    ``Estimated Power`` figure (Watts), and only on Strix (STX) NPUs and newer.
    Keyed ``npu0`` (distinct from the GPU samplers' ``card0``) so an NPU
    reading never collides with — or overwrites — a real GPU card's panel.
    """
    if not XRT_SMI:
        return {}
    try:
        out = subprocess.run(
            [XRT_SMI, "examine", "-r", "platform"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    match = _XRT_POWER_RE.search(out.stdout)
    if not match:
        return {}
    return {"npu0": {
        "vram_used_b": None,
        "vram_total_b": None,
        "power_w": _to_float(match.group(1)),
        "temp_c": None,
        "gpu_use_pct": None,
    }}


def wait_vram_settle(card: str = "card0", timeout: float = 10.0,
                     poll: float = 0.3) -> None:
    """Block until VRAM usage stops falling, so the next baseline is clean.

    ``ollama stop`` returns before the driver has actually released the model's
    VRAM. Sampling a baseline immediately would still include the previous
    model's footprint and distort the next model's ``vram_delta``. We poll
    until two consecutive readings stop decreasing (unload finished) or the
    timeout elapses.
    """
    if not gpu_available():
        return
    deadline = time.monotonic() + timeout
    prev = None
    stable = 0
    while time.monotonic() < deadline:
        cards = sample_gpu()
        reading = cards.get(card) or (next(iter(cards.values()), {}) if cards else {})
        used = reading.get("vram_used_b")
        if used is None:
            return
        if prev is not None and used >= prev - 5_000_000:  # <5 MB drop = settled
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        prev = used
        time.sleep(poll)


@dataclass
class GpuStats:
    """Aggregated GPU + CPU/RAM statistics over a sampling window."""

    samples: int = 0
    vram_baseline_b: float | None = None
    vram_peak_b: float | None = None
    vram_delta_b: float | None = None
    vram_total_b: float | None = None
    power_avg_w: float | None = None
    power_peak_w: float | None = None
    temp_peak_c: float | None = None
    gpu_use_avg_pct: float | None = None
    gpu_use_peak_pct: float | None = None
    # CPU / RAM
    cpu_avg_pct: float | None = None
    cpu_peak_pct: float | None = None
    cpu_freq_avg_mhz: float | None = None
    cpu_freq_peak_mhz: float | None = None
    cpu_temp_peak_c: float | None = None
    ram_baseline_b: float | None = None
    ram_peak_b: float | None = None
    ram_delta_b: float | None = None
    ram_total_b: float | None = None

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "vram_baseline_b": self.vram_baseline_b,
            "vram_peak_b": self.vram_peak_b,
            "vram_delta_b": self.vram_delta_b,
            "vram_delta_mb": round(self.vram_delta_b / 1e6, 1) if self.vram_delta_b is not None else None,
            "vram_peak_mb": round(self.vram_peak_b / 1e6, 1) if self.vram_peak_b is not None else None,
            "vram_total_b": self.vram_total_b,
            "power_avg_w": round(self.power_avg_w, 2) if self.power_avg_w is not None else None,
            "power_peak_w": round(self.power_peak_w, 2) if self.power_peak_w is not None else None,
            "temp_peak_c": self.temp_peak_c,
            "gpu_use_avg_pct": round(self.gpu_use_avg_pct, 1) if self.gpu_use_avg_pct is not None else None,
            "gpu_use_peak_pct": self.gpu_use_peak_pct,
        }

    def cpu_dict(self) -> dict:
        return {
            "cpu_avg_pct": round(self.cpu_avg_pct, 1) if self.cpu_avg_pct is not None else None,
            "cpu_peak_pct": round(self.cpu_peak_pct, 1) if self.cpu_peak_pct is not None else None,
            "cpu_freq_avg_mhz": round(self.cpu_freq_avg_mhz) if self.cpu_freq_avg_mhz is not None else None,
            "cpu_freq_peak_mhz": round(self.cpu_freq_peak_mhz) if self.cpu_freq_peak_mhz is not None else None,
            "cpu_temp_peak_c": round(self.cpu_temp_peak_c, 1) if self.cpu_temp_peak_c is not None else None,
            "ram_baseline_b": self.ram_baseline_b,
            "ram_peak_b": self.ram_peak_b,
            "ram_delta_b": self.ram_delta_b,
            "ram_delta_mb": round(self.ram_delta_b / 1e6, 1) if self.ram_delta_b is not None else None,
            "ram_peak_mb": round(self.ram_peak_b / 1e6, 1) if self.ram_peak_b is not None else None,
            "ram_total_b": self.ram_total_b,
        }


class GpuSampler:
    """Polls ``rocm-smi`` in a background thread until stopped.

    Usage::

        sampler = GpuSampler(card="card0")
        sampler.start()          # records a baseline immediately
        ...run the workload...
        stats = sampler.stop()   # returns GpuStats
    """

    def __init__(self, card: str = "card0", interval: float = 0.25,
                 on_sample=None, sample_fn=None):
        self.card = card
        self.interval = interval
        # Optional callback invoked (from the sampler thread) with each
        # normalized card reading — used to drive a live UI.
        self.on_sample = on_sample
        # Which hardware to poll each tick — defaults to the GPU vendor
        # sampler; backends can inject a different one (e.g. sample_npu).
        self.sample_fn = sample_fn or sample_gpu
        self._readings: list[dict] = []
        self._baseline: dict | None = None
        self._ram_baseline: float | None = None
        self._prev_cpu: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _pick_card(self, cards: dict) -> dict:
        if self.card in cards:
            return cards[self.card]
        # Fall back to the first card if the requested one is absent.
        return next(iter(cards.values()), {})

    def _sample_all(self):
        """Sample GPUs + system CPU%/RAM/freq/temp/load. Returns (cards, cpu)."""
        cards = self.sample_fn()
        cur_cpu = system.read_cpu_snapshot()
        cpu = {
            "cpu_pct": system.cpu_percent(self._prev_cpu, cur_cpu),
            "cpu_freq_mhz": system.read_cpu_freq_mhz(),
            "cpu_temp_c": system.read_cpu_temp_c(),
            "load": system.read_loadavg(),
        }
        self._prev_cpu = cur_cpu
        mem = system.read_mem()
        if mem:
            cpu["ram_used_b"], cpu["ram_total_b"] = mem
        return cards, cpu

    def _merged(self, cards, cpu) -> dict:
        """Merged reading for the picked card (used for record aggregation)."""
        reading = dict(self._pick_card(cards)) if cards else {}
        for key in ("cpu_pct", "cpu_freq_mhz", "cpu_temp_c",
                    "ram_used_b", "ram_total_b"):
            if cpu.get(key) is not None:
                reading[key] = cpu[key]
        return reading

    def _emit(self, cards, cpu) -> None:
        """Push a full live payload (all cards + cpu/ram) to the UI callback."""
        if self.on_sample is None:
            return
        payload = {"cards": cards, **cpu}
        try:
            self.on_sample(payload)
        except Exception:  # noqa: BLE001 - UI must never break sampling
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            cards, cpu = self._sample_all()
            merged = self._merged(cards, cpu)
            if merged:
                self._readings.append(merged)
            self._emit(cards, cpu)
            self._stop.wait(self.interval)

    def start(self) -> None:
        cards = self.sample_fn()
        self._baseline = self._pick_card(cards) if cards else {}
        self._prev_cpu = system.read_cpu_snapshot()
        mem = system.read_mem()
        self._ram_baseline = mem[0] if mem else None
        self._readings = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> GpuStats:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

        # Guarantee at least one in-window reading: fast workloads (small
        # models, embeddings) can finish before the polling thread — with its
        # per-call rocm-smi latency — records anything, which would leave VRAM
        # metrics empty. This final synchronous sample is taken while the model
        # is still loaded, so it reflects peak footprint.
        cards, cpu = self._sample_all()
        final = self._merged(cards, cpu)
        if final:
            self._readings.append(final)

        stats = GpuStats(samples=len(self._readings))
        if self._baseline:
            stats.vram_baseline_b = self._baseline.get("vram_used_b")
            stats.vram_total_b = self._baseline.get("vram_total_b")
        stats.ram_baseline_b = self._ram_baseline

        def collect(key: str) -> list[float]:
            return [r[key] for r in self._readings if r.get(key) is not None]

        vram = collect("vram_used_b")
        if vram:
            stats.vram_peak_b = max(vram)
            if stats.vram_total_b is None:
                totals = collect("vram_total_b")
                stats.vram_total_b = totals[0] if totals else None
            if stats.vram_baseline_b is not None:
                stats.vram_delta_b = max(0.0, stats.vram_peak_b - stats.vram_baseline_b)

        power = collect("power_w")
        if power:
            stats.power_avg_w = sum(power) / len(power)
            stats.power_peak_w = max(power)

        temp = collect("temp_c")
        if temp:
            stats.temp_peak_c = max(temp)

        use = collect("gpu_use_pct")
        if use:
            stats.gpu_use_avg_pct = sum(use) / len(use)
            stats.gpu_use_peak_pct = max(use)

        cpu = collect("cpu_pct")
        if cpu:
            stats.cpu_avg_pct = sum(cpu) / len(cpu)
            stats.cpu_peak_pct = max(cpu)

        freq = collect("cpu_freq_mhz")
        if freq:
            stats.cpu_freq_avg_mhz = sum(freq) / len(freq)
            stats.cpu_freq_peak_mhz = max(freq)

        ctemp = collect("cpu_temp_c")
        if ctemp:
            stats.cpu_temp_peak_c = max(ctemp)

        ram = collect("ram_used_b")
        if ram:
            stats.ram_peak_b = max(ram)
            totals = collect("ram_total_b")
            stats.ram_total_b = totals[0] if totals else None
            if stats.ram_baseline_b is not None:
                stats.ram_delta_b = max(0.0, stats.ram_peak_b - stats.ram_baseline_b)

        return stats
