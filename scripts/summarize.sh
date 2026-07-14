#!/usr/bin/env bash
# Print a compact table from a benchmark JSONL file (default: results/latest.jsonl).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILE="${1:-$ROOT/results/latest.jsonl}"

if [[ ! -f "$FILE" ]]; then
  echo "[error] no such file: $FILE" >&2
  exit 1
fi

"$ROOT/.venv/bin/python" - "$FILE" <<'PY'
import json, sys
path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]

def num(v, fmt="{:.0f}"):
    return fmt.format(v) if v is not None else "-"

hdr = (f"{'MODEL':<32}{'PARAMS':>9}  {'QUANT':<7}{'CTX':>7}  {'DISK_MB':>8}  "
       f"{'PLACEMENT':<16}{'TOK/S':>7}  {'VRAM_MB':>8}  {'RAM_MB':>8}  {'PWR_W':>6}  {'°C':>4}  CAPS")
print(hdr); print("-" * len(hdr))
for r in sorted(rows, key=lambda x: -((x.get('performance') or {}).get('tokens_per_s') or 0)):
    p = r.get('performance') or {}
    g = r.get('gpu') or {}
    cp = r.get('cpu') or {}
    d = r.get('details') or {}
    pl = r.get('placement') or {}
    place = pl.get('processor') or '-'
    ctx = pl.get('context') or d.get('context_length')  # runtime, else max
    tok = p.get('tokens_per_s')
    tok = f"{tok:.1f}" if tok else ("-" if r.get('ok') else "FAIL")
    caps = ",".join(r.get('capabilities') or [])
    print(f"{r['model']:<32}{str(d.get('parameter_size') or '-'):>9}  "
          f"{str(d.get('quantization_level') or '-'):<7}{str(ctx or '-'):>7}  "
          f"{num(r.get('size_mb')):>8}  {place:<16}{tok:>7}  "
          f"{num(g.get('vram_delta_mb')):>8}  {num(cp.get('ram_delta_mb')):>8}  "
          f"{num(g.get('power_peak_w')):>6}  {num(g.get('temp_peak_c')):>4}  {caps}")
PY
