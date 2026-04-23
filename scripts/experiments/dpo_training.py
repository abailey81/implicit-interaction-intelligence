"""CLI driver for offline reward-model training on a preference dataset.

Loads a :class:`~i3.router.preference_learning.PreferenceDataset` from
an optional SQLite file (falls back to synthetic data when no dataset
exists), trains a :class:`BradleyTerryRewardModel` using
:class:`DPOPreferenceOptimizer`, saves the model weights to
``checkpoints/preference/reward_model.pt`` and writes a markdown report
to ``reports/dpo_training_<timestamp>.md``.

Usage::

    python scripts/run_dpo_training.py --dataset data/preference.sqlite
    python scripts/run_dpo_training.py --synthetic --n-pairs 64

Design
------

* The script is self-contained — no FastAPI imports, no model-signing
  hooks.  It is safe to run offline / in CI.
* All paths are pure ``pathlib.Path``; the output directory is created
  if it does not exist.
* The script seeds every RNG (torch + numpy + python) so repeated runs
  on the same dataset produce identical checkpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from i3.router.preference_learning import (  # noqa: E402
    BradleyTerryRewardModel,
    DPOFitReport,
    DPOPreferenceOptimizer,
    PreferenceDataset,
    PreferencePair,
    build_response_features,
)

logger = logging.getLogger("run_dpo_training")


_DEFAULT_CHECKPOINT: Path = Path("checkpoints/preference/reward_model.pt")
_DEFAULT_REPORT_DIR: Path = Path("reports")


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _synthetic_dataset(n_pairs: int, seed: int) -> PreferenceDataset:
    """Build a deterministic easy-decision synthetic dataset.

    Response B always has longer length + higher latency + slightly
    lower confidence — emulating the "local SLM vs cloud" contrast.
    Winner is drawn from a noisy rule so the dataset is not trivially
    separable: the MLP should still reach ~0.8 accuracy.
    """
    rng = np.random.default_rng(seed)
    ds = PreferenceDataset()
    for _ in range(n_pairs):
        ctx = rng.uniform(0.0, 1.0, size=12).tolist()
        feat_a = build_response_features(
            length_tokens=float(rng.uniform(20.0, 120.0)),
            latency_ms=float(rng.uniform(100.0, 400.0)),
            model_confidence=float(rng.uniform(0.5, 0.9)),
        )
        feat_b = build_response_features(
            length_tokens=float(rng.uniform(150.0, 400.0)),
            latency_ms=float(rng.uniform(600.0, 1400.0)),
            model_confidence=float(rng.uniform(0.6, 0.95)),
        )
        # Rule: prefer A when context[0] (user_patience proxy) is high.
        winner = "a" if (ctx[4] + rng.normal(0.0, 0.1)) > 0.5 else "b"
        ds.append(
            PreferencePair(
                prompt="synthetic prompt",
                response_a="short response",
                response_b="long response",
                winner=winner,
                context=ctx,
                response_a_features=feat_a,
                response_b_features=feat_b,
                user_id="synthetic",
            )
        )
    return ds


async def _load_dataset(dataset_path: Path) -> PreferenceDataset:
    """Load a SQLite-backed dataset when aiosqlite is available."""
    ds = PreferenceDataset(db_path=dataset_path)
    await ds.load()
    return ds


def _write_report(
    report_dir: Path, report: DPOFitReport, dataset_desc: str
) -> Path:
    """Emit a markdown summary next to the checkpoint."""
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"dpo_training_{ts}.md"
    lines = [
        "# DPO Training Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"**Dataset:** {dataset_desc}",
        "",
        "## Metrics",
        "",
        f"- n_pairs: {report.n_pairs}",
        f"- n_train: {report.n_train}",
        f"- n_val: {report.n_val}",
        f"- train_loss: {report.train_loss:.4f}",
        f"- val_accuracy: {report.val_accuracy:.4f}",
        f"- epochs_run: {report.epochs_run}",
        f"- elapsed_seconds: {report.elapsed_seconds:.2f}",
        "",
        "## References",
        "",
        "- Rafailov et al. 2023, *Direct Preference Optimization*, NeurIPS.",
        "- Mehta et al. 2025, *Active Learning for DPO*, ICLR.",
        "- Bradley & Terry 1952, *Rank analysis of incomplete block designs*.",
        "- Azar et al. 2023, *A General Theoretical Paradigm* (IPO).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Train a Bradley-Terry reward model from a collected "
            "PreferenceDataset (SQLite) or a synthetic dataset."
        )
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the SQLite preference dataset (soft-import).",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic dataset instead of loading from disk.",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=64,
        help="Number of pairs when --synthetic is used (default: 64).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility (default: 0).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help=f"Output checkpoint path (default: {_DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=_DEFAULT_REPORT_DIR,
        help=f"Output report directory (default: {_DEFAULT_REPORT_DIR}).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit DEBUG-level logs.",
    )
    return p.parse_args()


def main() -> int:
    """Entry point."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _seed_everything(int(args.seed))

    if args.synthetic or args.dataset is None:
        logger.info("Building synthetic dataset (n_pairs=%d)", args.n_pairs)
        dataset = _synthetic_dataset(int(args.n_pairs), int(args.seed))
        dataset_desc = f"synthetic (n_pairs={args.n_pairs}, seed={args.seed})"
    else:
        logger.info("Loading dataset from %s", args.dataset)
        dataset = asyncio.run(_load_dataset(args.dataset))
        dataset_desc = f"sqlite: {args.dataset}"

    if len(dataset) == 0:
        logger.error("Dataset is empty — aborting.")
        return 2

    model = BradleyTerryRewardModel()
    optim = DPOPreferenceOptimizer(model, learning_rate=1e-3)
    logger.info("Training on %d pairs for %d epochs...", len(dataset), args.epochs)
    report = optim.fit(dataset, n_epochs=int(args.epochs), seed=int(args.seed))
    logger.info(
        "Training complete: val_accuracy=%.4f loss=%.4f in %.2fs",
        report.val_accuracy,
        report.train_loss,
        report.elapsed_seconds,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    logger.info("Checkpoint saved to %s", args.checkpoint)

    report_path = _write_report(args.report_dir, report, dataset_desc)
    logger.info("Report written to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
