"""Generate an edge-device feasibility report for I3 models.

Thin wrapper around :class:`i3.profiling.report.EdgeProfiler`.  Loads
(or constructs) the encoder and SLM, runs the profiler, and writes a
Markdown report to ``reports/edge_profile_YYYY-MM-DD.md``.

Usage::

    python scripts/profile_edge.py
    python scripts/profile_edge.py --encoder checkpoints/encoder/best.pt \\
                                   --slm     checkpoints/slm/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("i3.scripts.profile_edge")


def _load_state(checkpoint: Path | None) -> dict | None:
    """Safely load a verified state_dict or return ``None``.

    Args:
        checkpoint: Path to the ``.pt`` file, or ``None``.

    Returns:
        State dict or ``None`` when the checkpoint is missing.
    """
    if checkpoint is None or not checkpoint.exists():
        return None
    import torch  # type: ignore[import-not-found]

    try:
        from i3.mlops.checkpoint import load_verified

        state = load_verified(checkpoint, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_verified failed (%s); falling back to torch.load", exc)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def main() -> int:
    """CLI entry point.

    Returns:
        0 on success, non-zero on unrecoverable error.
    """
    parser = argparse.ArgumentParser(description="Edge feasibility profiler.")
    parser.add_argument("--encoder", type=Path, default=None)
    parser.add_argument("--slm", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output Markdown path.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    import torch  # type: ignore[import-not-found]

    from i3.encoder.tcn import TemporalConvNet
    from i3.profiling.report import EdgeProfiler
    from i3.slm.model import AdaptiveSLM

    encoder = TemporalConvNet().eval()
    enc_state = _load_state(args.encoder)
    if enc_state is not None:
        encoder.load_state_dict(enc_state, strict=False)

    slm = AdaptiveSLM().eval()
    slm_state = _load_state(args.slm)
    if slm_state is not None:
        slm.load_state_dict(slm_state, strict=False)

    profiler = EdgeProfiler()
    encoder_report = profiler.profile_model(
        encoder,
        "TCN Encoder",
        torch.randn(1, 10, getattr(encoder, "input_dim", 32)),
    )
    slm_report = profiler.profile_model(
        slm,
        "Adaptive SLM",
        torch.randint(0, max(slm.vocab_size, 1), (1, 16), dtype=torch.long),
    )

    out_path = args.output or (
        _PROJECT_ROOT / "reports" / f"edge_profile_{date.today().isoformat()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_parts: list[str] = [
        f"# I3 Edge Profile -- {date.today().isoformat()}",
        "",
        encoder_report.to_markdown(),
        "",
        slm_report.to_markdown(),
    ]
    out_path.write_text("\n".join(md_parts), encoding="utf-8")
    logger.info("wrote %s", out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
