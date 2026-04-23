"""Wrap the I3 training CLIs with MLflow experiment tracking.

This script does **not** modify the existing trainers.  Instead it
shells out to :func:`training.train_encoder.main` or
:func:`training.train_slm.main` inside a managed
:class:`i3.mlops.tracking.ExperimentTracker` context, then best-effort
logs the resulting checkpoints to MLflow.

Usage::

    python scripts/train_with_tracking.py encoder --epochs 100
    python scripts/train_with_tracking.py slm     --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.mlops.tracking import ExperimentTracker

logger = logging.getLogger("i3.scripts.train_with_tracking")


def _dispatch_training(target: str, forwarded: Sequence[str]) -> int:
    """Invoke the matching training entrypoint with ``forwarded`` argv.

    Args:
        target: Either ``"encoder"`` or ``"slm"``.
        forwarded: Extra argv items to pass along.

    Returns:
        ``0`` on success; non-zero propagated from the underlying main.
    """
    original_argv = sys.argv[:]
    try:
        if target == "encoder":
            from training.train_encoder import main as train_main
        elif target == "slm":
            from training.train_slm import main as train_main
        else:
            raise ValueError(f"unknown target: {target}")
        sys.argv = [f"train_{target}", *forwarded]
        train_main()
        return 0
    finally:
        sys.argv = original_argv


def _log_artifacts(tracker: ExperimentTracker, ckpt_dir: Path) -> None:
    """Best-effort upload of every file in ``ckpt_dir``.

    Args:
        tracker: Active :class:`ExperimentTracker`.
        ckpt_dir: Directory to mirror.
    """
    if not ckpt_dir.exists():
        return
    for item in ckpt_dir.iterdir():
        if item.is_file():
            tracker.log_artifact(item, artifact_path="checkpoints")


def main() -> int:
    """Parse args, open a tracked run, and dispatch to the trainer.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Train an I3 model with MLflow tracking enabled."
    )
    parser.add_argument("target", choices=["encoder", "slm"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory to upload when training finishes.",
    )
    parser.add_argument("--tag", action="append", default=[], metavar="k=v")
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    tags: dict[str, str] = {"i3.target": args.target}
    for t in args.tag:
        if "=" not in t:
            logger.warning("ignoring malformed tag %r (expected key=value)", t)
            continue
        k, v = t.split("=", 1)
        tags[k.strip()] = v.strip()

    run_name = args.run_name or f"{args.target}-training"
    tracker = ExperimentTracker()

    with tracker.run(run_name, tags=tags):
        tracker.log_params({"target": args.target, "forwarded_args": " ".join(extra)})
        rc = _dispatch_training(args.target, extra)

        ckpt_dir = args.checkpoint_dir
        if ckpt_dir is None:
            # Default: match the trainer's default directories.
            if args.target == "encoder":
                ckpt_dir = _PROJECT_ROOT / "models" / "encoder"
            else:
                ckpt_dir = _PROJECT_ROOT / "models" / "slm"
        _log_artifacts(tracker, ckpt_dir)
        tracker.set_tag("i3.exit_code", str(rc))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
