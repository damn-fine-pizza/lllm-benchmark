"""FastFlowLM (FTL) backend.

FastFlowLM exposes an OpenAI-compatible server, so this reuses the OpenAI
backend with a dedicated default base URL. FTL runs on the AMD Ryzen AI NPU,
which ``rocm-smi``/``nvidia-smi`` do not measure. When ``xrt-smi`` is on
``PATH`` we sample it instead for NPU power (Watts) — VRAM/temp/utilization
stay "n/d" since xrt-smi doesn't expose them for the NPU. Without xrt-smi,
hardware panels read "n/d" entirely (``supports_local_metrics`` stays False).
"""

from __future__ import annotations

import json
import os
import time

import requests

from ..gpu import XRT_SMI, sample_npu
from .openai_backend import OpenAIBackend

# FTL's default local server; override with --base-url if yours differs.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

# FLM has crashed (XRT "run destructed while command is still in progress")
# when the next model's warmup request lands right as the previous model's
# NPU run is being torn down. A short pause between models avoids the race.
NPU_SETTLE_S = 3.0


class FastFlowLLMBackend(OpenAIBackend):
    name = "fastflowllm"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 900.0):
        super().__init__(base_url or DEFAULT_BASE_URL, api_key=api_key,
                         mode="remote", timeout=timeout)
        # NPU workload — invisible to GPU samplers; use xrt-smi if present.
        self.supports_local_metrics = XRT_SMI is not None
        self.sample_fn = sample_npu
        self._warmup_load_s: float | None = None

    def unload(self, model: str) -> None:
        time.sleep(NPU_SETTLE_S)

    # The cold-load time only shows up in *this* request's usage — by the
    # time the benchmarked generation call runs, the model is already warm
    # and reports a near-zero load_duration of its own.
    def warmup(self, model: str, num_ctx: int | None) -> None:
        payload = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 1, "stream": False}
        self._warmup_load_s = None
        try:
            resp = requests.post(f"{self.base_url}/chat/completions",
                                 headers=self._headers(), json=payload,
                                 timeout=self.timeout)
            usage = (resp.json() or {}).get("usage") or {}
            self._warmup_load_s = usage.get("load_duration")
        except (requests.RequestException, ValueError):
            pass  # warmup is best-effort

    # /v1/models only returns {id}, so params/quant/ctx/disk size stay "-" in
    # the report. FLM's own model_list.json (same file it loads models from,
    # pointed to by FLM_CONFIG_PATH) has all of that — enrich with it when
    # available rather than leaving the table empty.
    def list_models(self) -> list[dict]:
        models = super().list_models()
        spec = self._load_model_spec()
        if not spec:
            return models
        for m in models:
            family, _, size_key = (m.get("name") or "").partition(":")
            info = spec.get(family, {}).get(size_key)
            if not info:
                continue
            det = info.get("details") or {}
            m["size"] = info.get("size")
            m["details"] = {
                "family": det.get("family"),
                "parameter_size": det.get("parameter_size"),
                "quantization_level": det.get("quantization_level"),
                "context_length": info.get("default_context_length"),
                "format": det.get("format"),
            }
        return models

    @staticmethod
    def _load_model_spec() -> dict | None:
        path = os.environ.get("FLM_CONFIG_PATH")
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return (json.load(fh) or {}).get("models") or {}
        except (OSError, json.JSONDecodeError):
            return None

    # Everything FLM serves runs on the NPU — no per-model placement query
    # exists over the API, so this is a fixed label rather than a lookup.
    def loaded_info(self, model: str) -> dict | None:
        return {"processor": "100% NPU"}

    # FLM's usage payload carries native decode/prefill throughput and load
    # time (not just token counts), which beats the wall-clock estimate the
    # generic OpenAI backend falls back to — use it when present.
    def performance(self, raw: dict, wall_s: float, streamed: int) -> dict:
        perf = super().performance(raw, wall_s, streamed)
        usage = raw.get("usage") or {}
        decode_tps = usage.get("decoding_speed_tps")
        prefill_tps = usage.get("prefill_speed_tps")
        load_s = self._warmup_load_s if self._warmup_load_s is not None \
            else usage.get("load_duration")
        if decode_tps:
            perf["tokens_per_s"] = round(decode_tps, 2)
            perf["tokens_per_s_method"] = "native"
        if prefill_tps:
            perf["prompt_tokens_per_s"] = round(prefill_tps, 2)
        if load_s is not None:
            perf["load_duration_s"] = round(load_s, 4)
        return perf
