"""Backend abstraction: a uniform interface over local and remote LLM runtimes.

Two measurement modes fall out of one property, ``supports_local_metrics``:

* **local / onboard** (e.g. Ollama): the native client gives rich, exact
  metrics (nanosecond eval timings → precise tokens/s) and the benchmark also
  samples GPU/CPU/RAM directly.
* **remote** (OpenAI-style API): only what the API exposes is available, so
  tokens/s is *computed* (streamed tokens ÷ wall-time) and hardware panels read
  "n/d" because we cannot see the remote machine.

Streaming contract for :meth:`Backend.stream`: yield ``{"text": str}`` for each
content delta, then exactly one ``{"final": <raw provider payload>}`` at the
end. :meth:`Backend.performance` turns that raw payload (plus the measured
wall-time and streamed token count) into the normalized performance dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class Backend(ABC):
    #: short id, e.g. "ollama" / "openai" / "fastflowllm"
    name: str = "backend"
    #: True → sample GPU/CPU/RAM locally and expect native perf metrics.
    supports_local_metrics: bool = False
    #: Hardware sampling function to use when supports_local_metrics is True.
    #: None → the runner's default (auto GPU vendor via rocm-smi/nvidia-smi).
    sample_fn = None

    # --- discovery / lifecycle ------------------------------------------------
    @abstractmethod
    def list_models(self) -> list[dict]:
        """Return ``[{name, size, capabilities, details}]`` (missing → None/{})."""

    def prepare(self) -> None:
        """Optional one-time setup before the run (e.g. free VRAM)."""

    @abstractmethod
    def warmup(self, model: str, num_ctx: int | None) -> None:
        """Load the model so the measured run reflects steady-state throughput."""

    def loaded_info(self, model: str) -> dict | None:
        """Runtime placement/context info, or None if the backend can't report it."""
        return None

    def unload(self, model: str) -> None:
        """Free the model (frees VRAM locally); no-op for remote backends."""

    # --- workloads ------------------------------------------------------------
    @abstractmethod
    def stream(self, model: str, prompt: str, num_predict: int,
               num_ctx: int | None) -> Iterator[dict]:
        """Yield ``{"text": ...}`` deltas then one ``{"final": raw}``."""

    @abstractmethod
    def performance(self, raw: dict, wall_s: float, streamed: int) -> dict:
        """Normalize provider timings into the standard performance dict."""

    def embed(self, model: str, text: str, num_ctx: int | None) -> tuple:
        """Return ``(embedding_dim, extra_metrics_dict)``; optional."""
        raise NotImplementedError(f"{self.name} backend does not support embeddings")


def blank_performance(wall_s: float) -> dict:
    return {
        "tokens_per_s": None,
        "prompt_tokens_per_s": None,
        "eval_count": 0,
        "prompt_eval_count": 0,
        "eval_duration_s": None,
        "prompt_eval_duration_s": None,
        "load_duration_s": None,
        "total_duration_s": None,
        "wall_time_s": round(wall_s, 4),
        "tokens_per_s_method": None,
    }
