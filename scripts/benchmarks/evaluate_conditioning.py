#!/usr/bin/env python
"""CLI wrapper for the conditioning-sensitivity evaluation.

Loads an :class:`AdaptiveSLM` checkpoint (optional — defaults to a
freshly-initialised model, useful for smoke tests), runs the
conditioning-sensitivity test over a set of prompts, and writes both a
JSON report and a Markdown summary.

Usage::

    python scripts/evaluate_conditioning.py \\
        --checkpoint checkpoints/slm/best.pt \\
        --tokenizer checkpoints/slm/tokenizer.json \\
        --prompts data/eval/conditioning_prompts.txt \\
        --out reports/conditioning.json

If ``--prompts`` is omitted a small built-in set of prompts is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger("i3.scripts.evaluate_conditioning")


_DEFAULT_PROMPTS: tuple[str, ...] = (
    "Can you help me understand how I'm feeling today?",
    "Please describe the architecture of a transformer block.",
    "Submit the final report with all appendices attached.",
    "I'm struggling to focus, can you give me a short pep talk?",
    "Explain INT4 weight-only quantization and its trade-offs.",
)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_conditioning",
        description=(
            "Measure cross-attention conditioning sensitivity of the "
            "Adaptive SLM via pairwise KL between next-token "
            "distributions under different AdaptationVectors."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional .pt checkpoint path. If omitted a freshly-"
        "initialised AdaptiveSLM is used.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to a tokenizer .json file (SimpleTokenizer format).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Plain-text file with one prompt per line.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/conditioning.json"),
        help="Output JSON path (default: reports/conditioning.json).",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional Markdown summary path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch device ('auto', 'cpu', 'cuda', 'cuda:N', 'mps'). "
            "Default 'auto' picks CUDA when available (fallback: MPS, then "
            "CPU) via i3.runtime.device.pick_device."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompts(path: Optional[Path]) -> list[str]:
    if path is None:
        return list(_DEFAULT_PROMPTS)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or list(_DEFAULT_PROMPTS)


def _load_model(checkpoint: Optional[Path], device: str) -> Any:
    from i3.runtime.device import enable_cuda_optimizations, pick_device
    from i3.slm.model import AdaptiveSLM

    # PERF: TF32 matmul + cuDNN benchmark when CUDA is visible.  Safe
    # no-op on CPU-only wheels.
    enable_cuda_optimizations()
    resolved = pick_device(device)
    model = AdaptiveSLM().to(resolved).eval()
    if checkpoint is not None:
        state: Any = torch.load(str(checkpoint), map_location=resolved, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    return model


def _load_tokenizer(path: Optional[Path]) -> Any:
    from i3.slm.tokenizer import SimpleTokenizer

    if path is not None and path.exists():
        return SimpleTokenizer.load(str(path))
    # Fallback: a tiny default tokenizer so the CLI works end-to-end
    # even without a trained vocabulary. We seed it with the default
    # prompts so encode() returns meaningful ids.
    tokenizer = SimpleTokenizer(vocab_size=1024)
    corpus = "\n".join(_DEFAULT_PROMPTS)
    build_fn = getattr(tokenizer, "build_vocab", None) or getattr(
        tokenizer, "train", None
    )
    if callable(build_fn):
        try:
            build_fn(corpus)
        except TypeError:
            build_fn([corpus])
    return tokenizer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _build_parser().parse_args(argv)

    from i3.eval.conditioning_sensitivity import (
        measure_conditioning_sensitivity,
        report_markdown,
        save_report,
        standard_adaptation_vectors,
    )

    from i3.runtime.device import pick_device

    prompts = _load_prompts(args.prompts)
    resolved_device = pick_device(args.device)
    model = _load_model(args.checkpoint, args.device)
    tokenizer = _load_tokenizer(args.tokenizer)
    logger.info("Conditioning eval running on device=%s", resolved_device)

    results = measure_conditioning_sensitivity(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        adaptation_vectors=standard_adaptation_vectors(),
        device=str(resolved_device),
    )

    out_path = save_report(results, args.out)
    print(f"Wrote JSON report -> {out_path}")

    if args.markdown is not None:
        md = report_markdown(results)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md, encoding="utf-8")
        print(f"Wrote Markdown summary -> {args.markdown}")
    else:
        print()
        print(report_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
