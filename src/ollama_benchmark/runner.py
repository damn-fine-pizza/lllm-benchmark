"""Benchmark orchestration: run each model and emit one JSONL record per model."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone

from .backends.base import Backend, blank_performance
from .gpu import GpuSampler, available_tools, gpu_available, gpu_vendor, npu_available
from .ui import Reporter

# A prompt that reliably produces a long, steady stream of tokens so tokens/s
# is measured over a meaningful window rather than a two-word reply.
GEN_PROMPT = (
    "Write a detailed technical explanation of how a CPU executes instructions, "
    "covering fetch, decode, execute and write-back stages. Be thorough."
)
EMBED_TEXT = "The quick brown fox jumps over the lazy dog. " * 8


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _details(model: dict) -> dict:
    d = model.get("details", {}) or {}
    return {
        "family": d.get("family"),
        "families": d.get("families"),
        "parameter_size": d.get("parameter_size"),
        "quantization_level": d.get("quantization_level"),
        "context_length": d.get("context_length"),
        "embedding_length": d.get("embedding_length"),
        "format": d.get("format"),
    }


def benchmark_model(backend: Backend, model: dict, reporter: Reporter, *,
                    num_predict: int, card: str, sample_interval: float,
                    num_ctx: int | None = None, prompt: str | None = None) -> dict:
    """Benchmark one model and return a JSON-serializable result record."""
    name = model.get("name") or model.get("model")
    caps = model.get("capabilities") or []
    is_embedding = "embedding" in caps and "completion" not in caps
    local = backend.supports_local_metrics

    record: dict = {
        "model": name,
        "platform": backend.name,
        "timestamp": _now_iso(),
        "size_bytes": model.get("size"),
        "size_mb": round(model["size"] / 1e6, 1) if model.get("size") else None,
        "digest": (model.get("digest") or "")[:12] or None,
        "capabilities": caps,
        "details": _details(model),
        "workload": "embedding" if is_embedding else "completion",
        "num_ctx_override": num_ctx,
        "local_metrics": local,
        "placement": None,
        "gpu": None,
        "cpu": None,
        "kv_cache_mb": None,  # vram_delta - disk ≈ KV cache footprint
        "performance": {},
        "ok": False,
        "error": None,
    }

    reporter.model_start(record)
    # Only sample hardware when the backend runs on measurable local hardware.
    sampler = GpuSampler(card=card, interval=sample_interval,
                         on_sample=reporter.gpu,
                         sample_fn=backend.sample_fn) if local else None
    try:
        if is_embedding:
            _run_embedding(backend, name, record, sampler, reporter, num_ctx)
        else:
            _run_completion(backend, name, record, sampler, reporter,
                            num_predict, num_ctx, prompt)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - one bad model shouldn't abort the run
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=3)
    finally:
        try:
            backend.unload(name)
        except Exception:  # noqa: BLE001
            pass

    # Make the KV-cache footprint explicit: VRAM delta beyond the weights on disk.
    vram = (record.get("gpu") or {}).get("vram_delta_mb")
    disk = record.get("size_mb")
    if vram is not None and disk:
        record["kv_cache_mb"] = round(vram - disk, 1)

    reporter.model_done(record)
    return record


def _run_completion(backend, name, record, sampler, reporter, num_predict,
                    num_ctx=None, prompt=None) -> None:
    prompt = prompt or GEN_PROMPT
    # Baseline sampled with the model unloaded, so vram_delta captures the full
    # model VRAM footprint (not just per-token overhead).
    if sampler:
        sampler.start()
    # Warm-up so the measured run reflects steady decode throughput.
    backend.warmup(name, num_ctx)
    record["placement"] = backend.loaded_info(name)
    reporter.model_loaded(record["placement"])

    t0 = time.perf_counter()
    raw: dict = {}
    streamed = 0
    for ev in backend.stream(name, prompt, num_predict, num_ctx):
        piece = ev.get("text")
        if piece:
            reporter.token(piece)
            streamed += 1
        if "final" in ev:
            raw = ev["final"] or {}
        if reporter.should_abort():
            break
    wall = time.perf_counter() - t0

    if sampler:
        stats = sampler.stop()
        record["gpu"] = stats.as_dict()
        record["cpu"] = stats.cpu_dict()
    record["performance"] = backend.performance(raw, wall, streamed)


def _run_embedding(backend, name, record, sampler, reporter, num_ctx=None) -> None:
    if sampler:
        sampler.start()
    backend.embed(name, "warmup", num_ctx)  # warm-up load
    record["placement"] = backend.loaded_info(name)
    reporter.model_loaded(record["placement"])

    t0 = time.perf_counter()
    dim, extra = backend.embed(name, EMBED_TEXT, num_ctx)
    wall = time.perf_counter() - t0

    if sampler:
        stats = sampler.stop()
        record["gpu"] = stats.as_dict()
        record["cpu"] = stats.cpu_dict()
    perf = blank_performance(wall)
    perf["embedding_dim"] = dim
    perf.update({k: v for k, v in extra.items() if v is not None})
    record["performance"] = perf


def _is_embedding(model: dict) -> bool:
    caps = model.get("capabilities") or []
    return "embedding" in caps and "completion" not in caps


def run(backend: Backend, out_path: str, reporter: Reporter, *,
        num_predict: int = 128, card: str = "card0", sample_interval: float = 0.25,
        only: list[str] | None = None, include_embeddings: bool = False,
        num_ctx: int | None = None, prompt: str | None = None) -> list[dict]:
    """Benchmark every available model, writing JSONL incrementally to *out_path*."""
    models = backend.list_models()
    if only:
        wanted = set(only)
        models = [m for m in models if (m.get("name") or m.get("model")) in wanted]
    if not include_embeddings:
        skipped = [m.get("name") or m.get("model") for m in models if _is_embedding(m)]
        models = [m for m in models if not _is_embedding(m)]
        if skipped:
            print(f"[info] skipping {len(skipped)} embedding model(s) "
                  f"(use --include-embeddings to benchmark them)")

    print(f"[info] platform: {backend.name}  "
          f"(local metrics: {'on' if backend.supports_local_metrics else 'off — remote'})")
    if backend.supports_local_metrics:
        tools = available_tools()
        if tools:
            print("[info] monitoring tools detected: "
                  + ", ".join(f"{n} ({t})" for n, _, t in tools))
        if backend.name == "fastflowllm":
            if not npu_available():
                print("[warn] no NPU tool (xrt-smi) found; NPU metrics will be empty.")
            else:
                print("[info] NPU sampling backend: xrt-smi "
                      "(power only — VRAM/temp/utilization not exposed)")
        elif not gpu_available():
            print("[warn] no GPU tool (rocm-smi/nvidia-smi) found; "
                  "GPU metrics will be empty.")
        else:
            print(f"[info] GPU sampling backend: {gpu_vendor()}")

    reporter.set_local_metrics(backend.supports_local_metrics)
    reporter.run_start(len(models), card)
    backend.prepare()

    results: list[dict] = []
    aborted = False
    with open(out_path, "w", encoding="utf-8") as fh:
        for model in models:
            if reporter.should_abort():
                aborted = True
                break
            rec = benchmark_model(backend, model, reporter, num_predict=num_predict,
                                  card=card, sample_interval=sample_interval,
                                  num_ctx=num_ctx, prompt=prompt)
            # If the user quit mid-model, don't record the partial result.
            if reporter.should_abort():
                aborted = True
                break
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(rec)
    reporter.run_done(results)
    if aborted:
        print(f"[info] aborted by user after {len(results)} model(s)")
    return results
