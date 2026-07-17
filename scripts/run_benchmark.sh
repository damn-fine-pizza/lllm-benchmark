#!/usr/bin/env bash
# Run the Ollama benchmark over every locally available model.
#
# Usage:
#   scripts/run_benchmark.sh                       # benchmark all models
#   scripts/run_benchmark.sh --only gemma3:1b      # benchmark a subset
#   scripts/run_benchmark.sh -n 256                 # generate 256 tokens/model
#
# Any extra arguments are forwarded to the Python module.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[error] virtualenv missing. Run scripts/setup.sh first." >&2
  exit 1
fi

# Detect whether we're targeting the local Ollama platform (default). The
# ollama-specific preamble is skipped for openai/fastflowllm platforms.
PLATFORM="ollama"
BASE_URL=""
prev=""
for i in "$@"; do
  [[ "$prev" == "--platform" ]] && PLATFORM="$i"
  [[ "$prev" == "--base-url" ]] && BASE_URL="$i"
  prev="$i"
done

HOST="${OLLAMA_HOST:-http://localhost:11434}"
if [[ "$PLATFORM" == "ollama" ]]; then
  # Verify the Ollama server is reachable before starting.
  if ! curl -sf "${HOST}/api/tags" >/dev/null 2>&1; then
    echo "[error] Ollama API not reachable at ${HOST}." >&2
    echo "        Start it with:  ollama serve" >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ "$PLATFORM" == "fastflowllm" && -z "$BASE_URL" ]]; then
  # FLM's server port is host-specific (set via FLM_PORT/config, not fixed
  # like Ollama's 11434) and *can* collide with an Ollama instance sitting on
  # the hardcoded backend default. Auto-detect it via `flm port` instead of
  # silently querying whatever happens to be on the default port.
  if ! command -v flm >/dev/null 2>&1; then
    echo "[error] --platform fastflowllm needs the 'flm' CLI on PATH (or pass --base-url explicitly)." >&2
    exit 1
  fi
  FLM_PORT="$(flm port 2>/dev/null | grep -oP '(?<=Server Port: )\d+')"
  if [[ -z "$FLM_PORT" ]]; then
    echo "[error] could not detect FLM's server port via 'flm port'. Pass --base-url explicitly." >&2
    exit 1
  fi
  echo "[run] auto-detected FLM base URL: http://localhost:${FLM_PORT}/v1"
  EXTRA_ARGS+=(--base-url "http://localhost:${FLM_PORT}/v1")
fi

mkdir -p "$ROOT/results"
# results/benchmark@<hostname>-<YYYY.MM.DD>-<HH:MM:SS>.jsonl
MACHINE="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date +%Y.%m.%d-%H:%M:%S)"
OUT="$ROOT/results/benchmark@${MACHINE}-${STAMP}.jsonl"

if [[ "$PLATFORM" == "ollama" ]]; then
  echo "[run] ollama models:"
  ollama ls
fi

echo "[run] writing results to $OUT"
PYTHONPATH="$ROOT/src" "$VENV/bin/python" -m ollama_benchmark \
  --out "$OUT" --host "$HOST" "${EXTRA_ARGS[@]}" "$@"

# Refresh a stable 'latest' symlink for convenience.
ln -sf "$(basename "$OUT")" "$ROOT/results/latest.jsonl"
echo "[run] done -> $OUT (results/latest.jsonl updated)"
