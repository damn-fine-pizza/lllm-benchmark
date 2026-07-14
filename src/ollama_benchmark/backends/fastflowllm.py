"""FastFlowLM (FTL) backend.

FastFlowLM exposes an OpenAI-compatible server, so this reuses the OpenAI
backend with a dedicated default base URL. FTL runs on the AMD Ryzen AI NPU,
which ``rocm-smi``/``nvidia-smi`` do not measure. When ``xrt-smi`` is on
``PATH`` we sample it instead for NPU power (Watts) — VRAM/temp/utilization
stay "n/d" since xrt-smi doesn't expose them for the NPU. Without xrt-smi,
hardware panels read "n/d" entirely (``supports_local_metrics`` stays False).
"""

from __future__ import annotations

from ..gpu import XRT_SMI, sample_npu
from .openai_backend import OpenAIBackend

# FTL's default local server; override with --base-url if yours differs.
DEFAULT_BASE_URL = "http://localhost:11434/v1"


class FastFlowLLMBackend(OpenAIBackend):
    name = "fastflowllm"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 900.0):
        super().__init__(base_url or DEFAULT_BASE_URL, api_key=api_key,
                         mode="remote", timeout=timeout)
        # NPU workload — invisible to GPU samplers; use xrt-smi if present.
        self.supports_local_metrics = XRT_SMI is not None
        self.sample_fn = sample_npu
