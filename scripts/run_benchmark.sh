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
prev=""
for i in "$@"; do
  [[ "$prev" == "--platform" ]] && PLATFORM="$i"
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
  --out "$OUT" --host "$HOST" "$@"

# Refresh a stable 'latest' symlink for convenience.
ln -sf "$(basename "$OUT")" "$ROOT/results/latest.jsonl"
echo "[run] done -> $OUT (results/latest.jsonl updated)"
