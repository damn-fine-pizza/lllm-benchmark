"""Local Ollama backend — the native client, with full metrics."""

from __future__ import annotations

from collections.abc import Iterator

from ..client import OllamaClient
from ..gpu import wait_vram_settle
from .base import Backend


class OllamaBackend(Backend):
    name = "ollama"
    supports_local_metrics = True

    def __init__(self, client: OllamaClient, card: str = "card0"):
        self.client = client
        self.card = card

    def list_models(self) -> list[dict]:
        out = []
        for m in self.client.list_models():
            out.append({
                "name": m.get("name") or m.get("model"),
                "size": m.get("size"),
                "digest": m.get("digest"),
                "capabilities": m.get("capabilities") or [],
                "details": m.get("details") or {},
            })
        return out

    def prepare(self) -> None:
        already = self.client.stop_all()
        if already:
            print(f"[info] unloaded {len(already)} pre-loaded model(s) before starting")
        wait_vram_settle(card=self.card, timeout=10.0)

    def warmup(self, model: str, num_ctx: int | None) -> None:
        self.client.generate(model, "hi", num_predict=1, num_ctx=num_ctx)

    def loaded_info(self, model: str) -> dict | None:
        return self.client.ps_for(model)

    def unload(self, model: str) -> None:
        self.client.stop(model)
        wait_vram_settle(card=self.card)

    def stream(self, model: str, prompt: str, num_predict: int,
               num_ctx: int | None) -> Iterator[dict]:
        for chunk in self.client.generate_stream(model, prompt,
                                                 num_predict=num_predict,
                                                 num_ctx=num_ctx):
            piece = chunk.get("response", "")
            if piece:
                yield {"text": piece}
            if chunk.get("done"):
                yield {"final": chunk}

    def performance(self, raw: dict, wall_s: float, streamed: int) -> dict:
        eval_count = raw.get("eval_count") or 0
        eval_dur_ns = raw.get("eval_duration") or 0
        prompt_count = raw.get("prompt_eval_count") or 0
        prompt_dur_ns = raw.get("prompt_eval_duration") or 0

        if eval_dur_ns:
            tok_s = eval_count / (eval_dur_ns / 1e9)
            method = "native"
        else:  # degenerate fallback
            tok_s = (streamed / wall_s) if wall_s else None
            method = "wall"
        prompt_tok_s = (prompt_count / (prompt_dur_ns / 1e9)) if prompt_dur_ns else None

        return {
            "tokens_per_s": round(tok_s, 2) if tok_s else None,
            "prompt_tokens_per_s": round(prompt_tok_s, 2) if prompt_tok_s else None,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_count,
            "eval_duration_s": round(eval_dur_ns / 1e9, 4) if eval_dur_ns else None,
            "prompt_eval_duration_s": round(prompt_dur_ns / 1e9, 4) if prompt_dur_ns else None,
            "load_duration_s": round(raw.get("load_duration", 0) / 1e9, 4),
            "total_duration_s": round(raw.get("total_duration", 0) / 1e9, 4),
            "wall_time_s": round(wall_s, 4),
            "tokens_per_s_method": method,
        }

    def embed(self, model: str, text: str, num_ctx: int | None) -> tuple:
        resp = self.client.embed(model, text, num_ctx=num_ctx)
        embeddings = resp.get("embeddings") or []
        dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else None
        extra = {
            "load_duration_s": round(resp.get("load_duration", 0) / 1e9, 4)
            if resp.get("load_duration") else None,
            "total_duration_s": round(resp.get("total_duration", 0) / 1e9, 4)
            if resp.get("total_duration") else None,
        }
        return dim, extra
