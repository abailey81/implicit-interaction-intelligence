"""CLI wrapper for exporting I3 models to ONNX.

Usage::

    python -m scripts.export_onnx \\
        --encoder checkpoints/encoder/best.pt \\
        --slm     checkpoints/slm/best.pt \\
        --out     exports/

The script is deliberately tolerant:

* Missing checkpoints fall back to a freshly-constructed random-weight
  model and a warning is logged -- useful for CI smoke tests.
* Missing optional dependencies (``onnx``, ``onnxruntime``) trigger
  ``SystemExit(2)`` with a clear actionable error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable when invoked as `python scripts/export_onnx.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("i3.scripts.export_onnx")


def _load_state(checkpoint: Path) -> dict:
    """Load a state_dict from a verified checkpoint.

    Falls back to ``torch.load(..., weights_only=True)`` when the file
    is not under :mod:`i3.mlops.checkpoint` management (no sidecar).

    Args:
        checkpoint: Path to the ``.pt`` file.

    Returns:
        Parsed state dict.
    """
    import torch  # type: ignore[import-not-found]

    try:
        from i3.mlops.checkpoint import load_verified

        state = load_verified(checkpoint, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_verified failed for %s (%s); falling back to torch.load",
            checkpoint,
            exc,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def main() -> int:
    """Parse CLI args and drive the export.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Export I3 TCN encoder and Adaptive SLM to ONNX."
    )
    parser.add_argument("--encoder", type=Path, default=None)
    parser.add_argument("--slm", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("exports"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip PyTorch/ONNXRuntime parity check after export.",
    )
    parser.add_argument(
        "--skip-encoder", action="store_true", help="Skip TCN encoder export."
    )
    parser.add_argument(
        "--skip-slm", action="store_true", help="Skip Adaptive SLM export."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports so this CLI loads quickly on --help.
    from i3.encoder.tcn import TemporalConvNet
    from i3.mlops.export import export_encoder, export_slm
    from i3.slm.model import AdaptiveSLM

    if not args.skip_encoder:
        encoder = TemporalConvNet()
        if args.encoder is not None and args.encoder.exists():
            encoder.load_state_dict(_load_state(args.encoder), strict=False)
            logger.info("Loaded TCN checkpoint from %s", args.encoder)
        else:
            logger.warning(
                "TCN checkpoint %s missing -- exporting random-init model.",
                args.encoder,
            )
        export_encoder(
            encoder,
            out_dir / "encoder.onnx",
            opset=args.opset,
            verify=not args.no_verify,
        )

    if not args.skip_slm:
        slm = AdaptiveSLM()
        if args.slm is not None and args.slm.exists():
            slm.load_state_dict(_load_state(args.slm), strict=False)
            logger.info("Loaded SLM checkpoint from %s", args.slm)
        else:
            logger.warning(
                "SLM checkpoint %s missing -- exporting random-init model.",
                args.slm,
            )
        export_slm(
            slm,
            out_dir / "slm.onnx",
            opset=args.opset,
            verify=not args.no_verify,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
