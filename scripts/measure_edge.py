#!/usr/bin/env python3
"""Thin wrapper around :class:`i3.edge.profiler.EdgeProfiler`.

Runs the edge-deployment measurement suite against the default v1
checkpoints and writes ``reports/edge_profile.json`` +
``reports/edge_profile.md``.

CPU-only.  Typical runtime: ~10-30 s depending on CPU.

Invoke from the repo root:

.. code-block:: console

   $ poetry run python scripts/measure_edge.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    root = _repo_root()
    # Ensure imports resolve regardless of where the script is launched
    # from — the orchestrator sometimes spawns scripts without the repo
    # root on PYTHONPATH.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from i3.edge.profiler import EdgeProfiler

    slm_ckpt = root / "checkpoints" / "slm" / "best_model.pt"
    tcn_ckpt = root / "checkpoints" / "encoder" / "best_model.pt"
    tok_path = root / "checkpoints" / "slm" / "tokenizer.json"

    for label, path in (
        ("SLM", slm_ckpt),
        ("TCN", tcn_ckpt),
        ("tokenizer", tok_path),
    ):
        if not path.is_file():
            print(f"error: {label} checkpoint missing at {path}", file=sys.stderr)
            return 2

    profiler = EdgeProfiler(slm_ckpt, tcn_ckpt, tok_path, device="cpu")
    report = profiler.measure()

    # Pretty-print the final report to stdout so CI and humans can
    # see the measurements without opening the JSON file.
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
