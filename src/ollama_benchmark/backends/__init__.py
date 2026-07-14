"""Backend factory."""

from __future__ import annotations

from ..client import DEFAULT_HOST, OllamaClient
from .base import Backend
from .fastflowllm import FastFlowLLMBackend
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAIBackend

PLATFORMS = ("ollama", "openai", "fastflowllm")


def make_backend(platform: str, *, host: str = DEFAULT_HOST,
                 base_url: str | None = None, api_key: str | None = None,
                 mode: str = "auto", card: str = "card0",
                 timeout: float = 900.0) -> Backend:
    """Build a Backend for the given platform.

    * ``ollama``      — local native client (``host`` is the Ollama URL).
    * ``openai``      — OpenAI-compatible API at ``base_url`` (LM Studio, vLLM,
                        llama.cpp server, or a remote API). ``mode`` =
                        ``auto``/``local``/``remote`` controls local sampling.
    * ``fastflowllm`` — FTL's OpenAI-style server (NPU; no local metrics).
    """
    if platform == "ollama":
        return OllamaBackend(OllamaClient(host=host, timeout=timeout), card=card)
    if platform == "openai":
        if not base_url:
            raise ValueError("--base-url is required for the openai platform")
        return OpenAIBackend(base_url, api_key=api_key, mode=mode, timeout=timeout)
    if platform == "fastflowllm":
        return FastFlowLLMBackend(base_url=base_url, api_key=api_key, timeout=timeout)
    raise ValueError(f"unknown platform: {platform!r} (choose from {PLATFORMS})")
