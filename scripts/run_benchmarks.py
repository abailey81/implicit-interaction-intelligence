"""Run the I³ benchmark suite and write a fresh report.

Usage::

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --suite latency
    python scripts/run_benchmarks.py --no-server  # offline / in-process

Writes ``reports/benchmarks/<timestamp>.{json,md}`` plus four SVG plots.
A ``latest.json`` / ``latest.md`` copy is also produced so the
Benchmarks UI tab has a stable URL to fetch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# CPU-only by spec; the GPU is reserved for training.  Setting this
# *before* torch/transformers import means the runtime's CUDA probe
# always returns False — no fights over the device with the trainer.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from i3.benchmarks.runner import BenchmarkRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("latency", "perplexity", "coherence", "adaptation", "memory", "all"),
        default="all",
        help="Run only one suite (default: all)",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="Live server URL for the latency suite (use --no-server to skip)",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Force the in-process pipeline mode for the latency suite",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=20,
        help="Number of latency prompts to send (default: 20)",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help=(
            "Skip suites that require an in-process pipeline (latency "
            "fall-back, coherence audit, adaptation faithfulness). "
            "Useful when only the perplexity / memory snapshot is needed."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    runner = BenchmarkRunner(
        server_url=None if args.no_server else args.server_url,
        n_latency_prompts=args.n_prompts,
        skip_pipeline=args.skip_pipeline,
    )

    if args.suite == "all":
        report = runner.run_all()
        print(json.dumps(report["headline"], indent=2))
        print("\nReport written to reports/benchmarks/")
    else:
        if args.suite == "latency":
            rows, _plot = runner.run_latency()
        elif args.suite == "perplexity":
            rows, _plot = runner.run_perplexity()
        elif args.suite == "coherence":
            rows, _plot = runner.run_coherence()
        elif args.suite == "adaptation":
            rows, _plot = runner.run_adaptation_faithfulness()
        elif args.suite == "memory":
            rows = runner.run_memory()
        else:
            print(f"unknown suite: {args.suite}", file=sys.stderr)
            return 2
        print(json.dumps([r.to_dict() for r in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
