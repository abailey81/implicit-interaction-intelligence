#!/usr/bin/env python
"""CLI wrapper for the ExecuTorch export pipelines.

Usage examples::

    # Export the SLM with INT8 quantization
    python scripts/export_executorch.py slm \\
        --checkpoint checkpoints/slm/best.pt \\
        --out checkpoints/slm/slm.pte \\
        --quantization int8

    # Export the TCN encoder (FP32)
    python scripts/export_executorch.py tcn \\
        --checkpoint checkpoints/encoder/best.pt \\
        --out checkpoints/encoder/tcn.pte \\
        --quantization none

Requires ``executorch`` (optionally ``torchao`` for quantization):

    pip install executorch torchao
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger("i3.scripts.export_executorch")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI arg tree."""
    parser = argparse.ArgumentParser(
        prog="export_executorch",
        description=(
            "Export I3 models to ExecuTorch .pte for on-device inference."
        ),
    )
    sub = parser.add_subparsers(dest="which", required=True)

    p_slm = sub.add_parser("slm", help="Export the Adaptive SLM.")
    p_slm.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional .pt checkpoint. If omitted a freshly-initialised "
        "AdaptiveSLM is exported (useful for shape tests).",
    )
    p_slm.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .pte path.",
    )
    p_slm.add_argument(
        "--quantization",
        choices=("none", "int8", "int4"),
        default="int8",
        help="Quantization level (default: int8).",
    )
    p_slm.add_argument(
        "--batch-size", type=int, default=1, help="Example batch size."
    )
    p_slm.add_argument(
        "--seq-len", type=int, default=16, help="Example sequence length."
    )

    p_tcn = sub.add_parser("tcn", help="Export the TCN user-state encoder.")
    p_tcn.add_argument("--checkpoint", type=Path, default=None)
    p_tcn.add_argument("--out", type=Path, required=True)
    p_tcn.add_argument(
        "--quantization",
        choices=("none", "int8"),
        default="int8",
        help="Quantization level (default: int8).",
    )
    p_tcn.add_argument("--batch-size", type=int, default=1)
    p_tcn.add_argument("--seq-len", type=int, default=100)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint_weights(model: torch.nn.Module, path: Path) -> None:
    """Load a ``state_dict`` from *path* into *model* (strict=False)."""
    state: Any = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %d", len(missing))
    if unexpected:
        logger.warning(
            "Unexpected keys when loading checkpoint: %d", len(unexpected)
        )


def _export_slm(args: argparse.Namespace) -> int:
    """Handle the ``slm`` subcommand."""
    from i3.edge.executorch_export import export_slm_to_executorch
    from i3.slm.model import AdaptiveSLM

    model = AdaptiveSLM().eval()
    if args.checkpoint is not None:
        _load_checkpoint_weights(model, args.checkpoint)

    example = (
        torch.zeros(args.batch_size, args.seq_len, dtype=torch.long),
        torch.zeros(args.batch_size, 8),
        torch.zeros(args.batch_size, 64),
    )
    pte = export_slm_to_executorch(
        model=model,
        example_inputs=example,
        out_path=args.out,
        quantization=args.quantization,
    )
    print(f"Wrote SLM .pte -> {pte}")
    return 0


def _export_tcn(args: argparse.Namespace) -> int:
    """Handle the ``tcn`` subcommand."""
    from i3.edge.tcn_executorch_export import export_tcn_to_executorch
    from i3.encoder.tcn import TemporalConvNet

    model = TemporalConvNet().eval()
    if args.checkpoint is not None:
        _load_checkpoint_weights(model, args.checkpoint)

    example = (torch.zeros(args.batch_size, args.seq_len, 32),)
    pte = export_tcn_to_executorch(
        model=model,
        example_inputs=example,
        out_path=args.out,
        quantization=args.quantization,
    )
    print(f"Wrote TCN .pte -> {pte}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.which == "slm":
        return _export_slm(args)
    if args.which == "tcn":
        return _export_tcn(args)
    parser.error("Unknown subcommand.")
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
