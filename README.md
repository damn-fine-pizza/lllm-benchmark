# ollama-benchmark

Benchmarks every locally installed Ollama model and records the results as
JSONL. For each model it measures generation throughput (**tokens/s**) and GPU
**memory usage** (via `rocm-smi`), alongside power, temperature, utilization and
the model's declared **capabilities**.

## Platforms (local vs remote)

Benchmark different runtimes via `--platform`:

| platform | transport | metrics |
|----------|-----------|---------|
| `ollama` (default) | local native client | full: native tokens/s + GPU/CPU/RAM sampling |
| `openai` | OpenAI-compatible API (`--base-url`) — LM Studio, vLLM, llama.cpp, remote APIs | tokens/s computed (tokens ÷ wall-time); hardware = **n/d** unless local |
| `fastflowllm` | FTL's OpenAI-style server (NPU) | tokens/s computed; NPU **power** via `xrt-smi` when present (Strix NPUs+), rest **n/d** |

```bash
# local Ollama (default)
scripts/run_benchmark.sh

# LM Studio / local OpenAI-compatible server (samples local GPU too)
scripts/run_benchmark.sh --platform openai --base-url http://localhost:1234/v1

# a remote machine we can only reach by API (no hardware sampling)
scripts/run_benchmark.sh --platform openai --base-url https://host/v1 \
    --api-key "$KEY" --mode remote

# FastFlowLM
scripts/run_benchmark.sh --platform fastflowllm --base-url http://localhost:11434/v1
```

`--mode auto|local|remote` controls hardware sampling (auto = on only for a
localhost server). Remote records set `gpu`/`cpu` to `null`, `local_metrics:false`,
and `performance.tokens_per_s_method: "wall"` (vs `"native"` for Ollama).

## How it works

- **Discovery** uses the Ollama CLI/API (`ollama ls`, `/api/tags`, `/api/show`)
  to enumerate models and read their capabilities, family, quantization, etc.
- **Timed generation** uses the local HTTP API (`/api/generate`), which returns
  exact token counts and nanosecond timings — the same numbers
  `ollama run --verbose` prints — so tokens/s is measured reliably.
- **GPU metrics** come from a background thread polling the GPU (AMD via
  `rocm-smi`, NVIDIA via `nvidia-smi` — auto-detected; force with the
  `GPU_VENDOR=amd|nvidia` env var). It captures VRAM used/peak/delta, power
  avg/peak, temp peak and utilization per card while the model generates.
- **NPU metrics** (`--platform fastflowllm` only) come from `xrt-smi` when it's
  on `PATH` (AMD XDNA / Ryzen AI). It only exposes an estimated power draw
  (Watts, Strix NPUs and newer) — VRAM/temp/utilization aren't reported by the
  tool, so those fields stay `n/d`.
- **CPU / RAM metrics** are sampled from `/proc` (no extra dependency): system
  CPU load and RAM used/peak/delta.
- Embedding models (no `completion` capability) are exercised via `/api/embed`.
- Each model is unloaded with `ollama stop` between runs so VRAM measurements
  don't bleed into the next model.

## Live dashboard

While running on a terminal, a `rich` TUI shows (instead of a scrolling log)
a fixed top row of panels — `current model | CPU | GPU0 | GPU1 | …` — plus a
flexible, scrollable **completed** table below:

- a **header** with overall progress, elapsed time, ok/fail counts;
- a **current model** panel (flexible width) with live metrics (streaming
  response preview, running tokens/s, capabilities, quantization, ctx +
  architectural max, CPU/GPU placement);
- a **CPU** panel (system CPU load + RAM, with bars);
- one **GPU** panel per card (VRAM used/peak with a bar, power, temp, util);
- a **completed** table summarizing every finished model (sortable/scrollable).

Force it with `--ui rich`, disable with `--ui plain` (auto-detects a TTY).

**Interactive keys** (rich dashboard, on a terminal):

| key | action |
|-----|--------|
| `SHIFT`+`M P Q C D G T V W H L S` | sort the completed table by that column (press again to flip) |
| `r` | reverse the current sort |
| `↑`/`↓` or `j`/`k` | scroll the completed table |
| `SPACE`, `PgUp`/`PgDn` | page the completed table |
| `g` / `b` | jump to top / bottom (bottom re-enables auto-follow) |
| `f` | toggle auto-follow (stick to the newest row) |
| `q` | quit — asks `y`/`n`; on `y` it stops cleanly and keeps partial results |

Embedding-only models are skipped by default; add `--include-embeddings` to
benchmark them too.

## Usage

```bash
scripts/setup.sh                 # create .venv and install deps
scripts/run_benchmark.sh         # benchmark all models -> results/benchmark-<stamp>.jsonl
scripts/summarize.sh             # pretty table from results/latest.jsonl
```

Options (forwarded to the Python module):

```bash
scripts/run_benchmark.sh --only gemma3:1b granite4.1:3b   # subset
scripts/run_benchmark.sh -n 256                            # 256 tokens/model
scripts/run_benchmark.sh -ocs 1M                           # force num_ctx = 1,000,000
scripts/run_benchmark.sh --override-ctx-size 256k          # force num_ctx = 256,000
scripts/run_benchmark.sh --prompt "Explain TCP in depth."  # custom prompt (inline)
scripts/run_benchmark.sh --prompt-file ./myprompt.txt      # custom prompt (from file)
scripts/run_benchmark.sh --card card0 --interval 0.2       # GPU sampling
```

`-ocs` / `--override-ctx-size` sets the context window (`num_ctx`) used to load
every model — accepts `1M`, `256k` or a raw integer (decimal suffixes: `k`=×1000,
`m`=×1000000). Larger context means a larger KV cache and more VRAM, so this is
the main knob for stress-testing memory. Results are written to
`results/benchmark@<hostname>-<YYYY.MM.DD>-<HH:MM:SS>.jsonl`.

The `ctx` column shows the **runtime** context actually loaded (from `ollama ps`),
not the model's architectural maximum; the max is still kept in the JSONL as
`details.context_length`.

Requires a running Ollama server (`ollama serve`) and, for GPU metrics,
`rocm-smi` on `PATH` (AMD ROCm).

## Output

One JSON object per line in `results/*.jsonl`, e.g.:

```json
{
  "model": "gemma3:1b",
  "capabilities": ["completion"],
  "workload": "completion",
  "performance": {"tokens_per_s": 92.4, "eval_count": 128, "load_duration_s": 0.7, ...},
  "gpu": {"vram_delta_mb": 1180.0, "vram_peak_mb": 2733.0, "power_peak_w": 61.0, ...},
  "details": {"family": "gemma3", "parameter_size": "999.89M", "quantization_level": "Q4_K_M", ...},
  "ok": true
}
```
