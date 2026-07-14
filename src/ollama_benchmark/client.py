"""Thin wrapper around the local Ollama CLI and HTTP API.

Discovery (``ollama ls`` / ``ollama show`` / ``ollama ps`` / ``ollama stop``)
goes through the CLI, matching how a user drives Ollama. The actual timed
generation uses the local HTTP API because it returns exact token counts and
nanosecond-accurate timings (the same numbers ``ollama run --verbose`` prints),
which makes tokens/s measurement reliable.
"""

from __future__ import annotations

import json
import subprocess

import requests

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 900.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # --- CLI-backed discovery -------------------------------------------------

    def cli_version(self) -> str:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return out.stdout.strip() or out.stderr.strip()

    def list_models(self) -> list[dict]:
        """Return model metadata from the API's ``/api/tags`` (mirrors ``ollama ls``).

        Using the API here gives us the ``capabilities`` array and structured
        ``details`` (family, parameter size, quantization, context length) in
        one call instead of scraping ``ollama show`` per model.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        return resp.json().get("models", [])

    def show(self, model: str) -> dict:
        """Full model info incl. capabilities, via ``/api/show``."""
        resp = requests.post(f"{self.host}/api/show", json={"model": model},
                             timeout=60)
        resp.raise_for_status()
        return resp.json()

    def ps(self) -> str:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        return out.stdout.strip()

    def ps_for(self, model: str) -> dict | None:
        """Parse ``ollama ps`` for one model: SIZE / PROCESSOR / CONTEXT.

        PROCESSOR reveals the CPU/GPU placement (e.g. ``100% GPU`` or
        ``40%/60% CPU/GPU``) — key info for a GPU benchmark.
        """
        text = self.ps()
        lines = text.splitlines()
        if len(lines) < 2:
            return None
        header = lines[0]
        # Column starts are stable in `ollama ps` output; use them to slice
        # so multi-word PROCESSOR values aren't split.
        cols = ["NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"]
        idx = {c: header.find(c) for c in cols if header.find(c) != -1}
        order = [c for c in cols if c in idx]
        for line in lines[1:]:
            name = line[idx["NAME"]:idx.get("ID", len(line))].strip()
            if name != model:
                continue
            row = {}
            for i, c in enumerate(order):
                start = idx[c]
                end = idx[order[i + 1]] if i + 1 < len(order) else len(line)
                row[c.lower()] = line[start:end].strip()
            return {
                "size": row.get("size"),
                "processor": row.get("processor"),
                "context": row.get("context"),
            }
        return None

    def stop(self, model: str) -> None:
        """Unload a model from memory (frees VRAM) via ``ollama stop``."""
        subprocess.run(["ollama", "stop", model], capture_output=True, text=True)

    def loaded_models(self) -> list[str]:
        """Names of models currently resident in memory (from ``ollama ps``)."""
        text = self.ps()
        lines = text.splitlines()
        if len(lines) < 2:
            return []
        id_col = lines[0].find("ID")
        names = []
        for line in lines[1:]:
            name = (line[:id_col].strip() if id_col > 0 else line.split()[0]).strip()
            if name:
                names.append(name)
        return names

    def stop_all(self) -> list[str]:
        """Unload every currently-loaded model. Returns the names stopped."""
        loaded = self.loaded_models()
        for name in loaded:
            self.stop(name)
        return loaded

    # --- API-backed workloads -------------------------------------------------

    @staticmethod
    def _options(num_predict: int, num_ctx: int | None) -> dict:
        opts = {"num_predict": num_predict, "temperature": 0.0}
        if num_ctx:
            opts["num_ctx"] = num_ctx
        return opts

    def generate(self, model: str, prompt: str, num_predict: int = 128,
                 keep_alive: str | int = "5m", num_ctx: int | None = None) -> dict:
        """Non-streamed completion. Returns the raw API response with timings."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": self._options(num_predict, num_ctx),
        }
        resp = requests.post(f"{self.host}/api/generate", json=payload,
                             timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def generate_stream(self, model: str, prompt: str, num_predict: int = 128,
                        keep_alive: str | int = "5m", num_ctx: int | None = None):
        """Stream a completion, yielding each API chunk as a dict.

        Non-final chunks carry incremental ``response`` text; the final chunk
        (``done: true``) carries the timing/count fields used for tokens/s.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": keep_alive,
            "options": self._options(num_predict, num_ctx),
        }
        with requests.post(f"{self.host}/api/generate", json=payload,
                          stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def embed(self, model: str, text: str, num_ctx: int | None = None) -> dict:
        """Embedding request. Returns raw API response (incl. embeddings)."""
        payload = {"model": model, "input": text, "keep_alive": "5m"}
        if num_ctx:
            payload["options"] = {"num_ctx": num_ctx}
        resp = requests.post(f"{self.host}/api/embed", json=payload,
                             timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
