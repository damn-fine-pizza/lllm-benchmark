"""CLI entrypoint: ``python -m ollama_benchmark``."""

from __future__ import annotations

import argparse
import sys

from .backends import PLATFORMS, make_backend
from .client import DEFAULT_HOST
from .runner import run
from .ui import make_reporter


def parse_ctx_size(value: str) -> int:
    """Parse a context size like ``1M``, ``256k``, ``256000`` into an int.

    Suffixes are decimal: ``k`` = ×1000, ``m`` = ×1_000_000 (so ``256k`` ==
    ``256000`` and ``1M`` == ``1000000``).
    """
    s = str(value).strip().lower().replace("_", "")
    mult = 1
    if s.endswith("k"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    try:
        n = int(round(float(s) * mult))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid context size: {value!r} (use e.g. 1M, 256k, 256000)")
    if n <= 0:
        raise argparse.ArgumentTypeError("context size must be positive")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollama_benchmark",
        description="Benchmark every local Ollama model (tokens/s + GPU memory via rocm-smi).",
    )
    p.add_argument("-o", "--out", default="results/benchmark.jsonl",
                   help="Output JSONL path (default: results/benchmark.jsonl)")
    p.add_argument("--platform", choices=PLATFORMS, default="ollama",
                   help="Runtime to benchmark (default: ollama)")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"Ollama host for --platform ollama (default: {DEFAULT_HOST})")
    p.add_argument("--base-url",
                   help="Base URL for openai/fastflowllm platforms, "
                        "e.g. http://localhost:1234/v1")
    p.add_argument("--api-key", help="API key for the openai platform (if required)")
    p.add_argument("--mode", choices=["auto", "local", "remote"], default="auto",
                   help="Sample local hardware? auto = on only for a localhost "
                        "OpenAI server (default: auto)")
    p.add_argument("-n", "--num-predict", type=int, default=128,
                   help="Tokens to generate per completion benchmark (default: 128)")
    p.add_argument("--card", default="card0",
                   help="rocm-smi card id to sample (default: card0)")
    p.add_argument("--interval", type=float, default=0.25,
                   help="GPU sampling interval in seconds (default: 0.25)")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="Per-request timeout in seconds (default: 900)")
    p.add_argument("--only", nargs="+", default=None,
                   help="Benchmark only these model names (space-separated)")
    p.add_argument("--ui", choices=["auto", "rich", "plain"], default="auto",
                   help="Progress display: rich dashboard, plain log, or auto (default)")
    p.add_argument("--include-embeddings", action="store_true",
                   help="Also benchmark embedding-only models (skipped by default)")
    p.add_argument("-ocs", "--override-ctx-size", type=parse_ctx_size, default=None,
                   metavar="N",
                   help="Override the context window (num_ctx) for every model, "
                        "e.g. 1M, 256k, 256000 (default: each model's own setting)")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--prompt", metavar="TEXT",
                     help="Prompt string for the completion benchmark "
                          "(overrides the built-in default)")
    grp.add_argument("--prompt-file", metavar="PATH",
                     help="Read the completion prompt from this text file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as fh:
                prompt = fh.read()
        except OSError as exc:
            print(f"[error] cannot read prompt file: {exc}", file=sys.stderr)
            return 1

    try:
        backend = make_backend(args.platform, host=args.host,
                               base_url=args.base_url, api_key=args.api_key,
                               mode=args.mode, card=args.card, timeout=args.timeout)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    reporter = make_reporter(args.ui, args.card)
    try:
        run(backend, args.out, reporter, num_predict=args.num_predict,
            card=args.card, sample_interval=args.interval, only=args.only,
            include_embeddings=args.include_embeddings,
            num_ctx=args.override_ctx_size, prompt=prompt)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        reporter.__exit__(None, None, None)
        print("\n[info] interrupted; partial results saved.", file=sys.stderr)
        return 130
    finally:
        reporter.__exit__(None, None, None)
    print(f"[done] results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
