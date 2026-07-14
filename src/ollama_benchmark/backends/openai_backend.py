"""OpenAI-compatible backend — remote APIs, LM Studio, vLLM, llama.cpp server.

No native eval timings, so tokens/s is computed as streamed tokens ÷ wall-time
(measured from the first to the last streamed token). Hardware panels read
"n/d" unless the server is local and the user opts into local sampling.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from urllib.parse import urlparse

import requests

from .base import Backend

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, base_url: str, api_key: str | None = None,
                 mode: str = "auto", timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        host = (urlparse(self.base_url).hostname or "").lower()
        is_local = host in LOCAL_HOSTS
        if mode == "local":
            self.supports_local_metrics = True
        elif mode == "remote":
            self.supports_local_metrics = False
        else:  # auto: only a localhost server sits on hardware we can sample
            self.supports_local_metrics = is_local

    # --- http helpers ---------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # --- discovery ------------------------------------------------------------
    def list_models(self) -> list[dict]:
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(),
                                timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"cannot reach OpenAI API at {self.base_url}: {exc}") from exc
        data = resp.json().get("data", [])
        out = []
        for m in data:
            out.append({
                "name": m.get("id"),
                "size": None,           # not exposed by the API
                "digest": None,
                "capabilities": ["completion"],  # assume chat/completion
                "details": {},          # no params/quant/ctx over the API
            })
        return out

    def warmup(self, model: str, num_ctx: int | None) -> None:
        # A tiny non-streamed request just to load / spin up the model.
        payload = {"model": model, "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 1, "stream": False}
        try:
            requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                          json=payload, timeout=self.timeout)
        except requests.RequestException:
            pass  # warmup is best-effort

    # --- streaming ------------------------------------------------------------
    def stream(self, model: str, prompt: str, num_predict: int,
               num_ctx: int | None) -> Iterator[dict]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": num_predict,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        usage: dict = {}
        with requests.post(f"{self.base_url}/chat/completions",
                           headers=self._headers(), json=payload,
                           stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                text = line.decode("utf-8", "ignore").strip()
                if not text.startswith("data:"):
                    continue
                data = text[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield {"text": piece}
        yield {"final": {"usage": usage}}

    def performance(self, raw: dict, wall_s: float, streamed: int) -> dict:
        usage = raw.get("usage") or {}
        completion = usage.get("completion_tokens") or streamed
        prompt_tokens = usage.get("prompt_tokens")
        tok_s = (completion / wall_s) if (completion and wall_s) else None
        return {
            "tokens_per_s": round(tok_s, 2) if tok_s else None,
            "prompt_tokens_per_s": None,
            "eval_count": completion,
            "prompt_eval_count": prompt_tokens,
            "eval_duration_s": None,
            "prompt_eval_duration_s": None,
            "load_duration_s": None,
            "total_duration_s": None,
            "wall_time_s": round(wall_s, 4),
            "tokens_per_s_method": "wall",  # computed, not native
        }

    def embed(self, model: str, text: str, num_ctx: int | None) -> tuple:
        payload = {"model": model, "input": text}
        resp = requests.post(f"{self.base_url}/embeddings", headers=self._headers(),
                             json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        dim = len(data[0]["embedding"]) if data and data[0].get("embedding") else None
        return dim, {}
