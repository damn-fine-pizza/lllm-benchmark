#!/usr/bin/env bash
# Create the Python virtualenv and install dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  echo "[setup] creating virtualenv at $VENV"
  "$PY" -m venv "$VENV"
fi

echo "[setup] installing dependencies"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$ROOT/requirements.txt"

echo "[setup] done. Run scripts/run_benchmark.sh to start the benchmark."
