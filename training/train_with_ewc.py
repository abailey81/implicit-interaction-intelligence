"""Sequential-task TCN-encoder training with Elastic Weight Consolidation.

Given a list of persona-specific training shards (each a directory of
``.pt`` files, each containing a ``TensorDataset``-compatible tuple
``(X, y)``), this script trains the shared TCN encoder on each shard in
turn and applies an EWC consolidation step between tasks so future
training preserves the parameters critical to earlier personas.

After every task the script evaluates the encoder on *every earlier*
task's validation set, producing a retention curve. The curve is written
to ``reports/ewc_training_<ts>.{json,md}``.

Example::

    python -m training.train_with_ewc \
        --shard data/shards/fresh_user \
        --shard data/shards/fatigued_developer \
        --shard data/shards/motor_impaired_user \
        --epochs 5 --lambda-ewc 1000.0

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.continual.ewc import ElasticWeightConsolidation, OnlineEWC
from i3.encoder.tcn import TemporalConvNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_EPOCHS: int = 5
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_LR: float = 1e-3
_DEFAULT_LAMBDA: float = 1000.0
_DEFAULT_FISHER_SAMPLES: int = 200
_DEFAULT_REPORT_DIR: Path = _PROJECT_ROOT / "reports"
_DEFAULT_SEQ_LEN: int = 16
_DEFAULT_INPUT_DIM: int = 32
_DEFAULT_EMBEDDING_DIM: int = 64


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskShard:
    """A single training task.

    Attributes:
        name: Short identifier (usually the persona name).
        train: Training dataset.
        val: Validation dataset.
    """

    name: str
    train: TensorDataset
    val: TensorDataset


@dataclass
class RetentionRow:
    """Row in the retention curve.

    Attributes:
        task_trained: Index of the task that was just trained.
        task_evaluated: Index of the earlier task being measured.
        task_name: Name of the evaluated task.
        loss: Validation loss (MSE on targets) on that task.
        retention: ``1 − (loss − baseline) / max(baseline, ε)`` clipped
            to ``[0, 1]``; ``1.0`` == original accuracy retained.
    """

    task_trained: int
    task_evaluated: int
    task_name: str
    loss: float
    retention: float


@dataclass
class EWCTrainingReport:
    """Full report written to disk."""

    timestamp: str
    lambda_ewc: float
    online_ewc: bool
    epochs_per_task: int
    batch_size: int
    learning_rate: float
    task_names: list[str]
    retention_curve: list[RetentionRow] = field(default_factory=list)
    baseline_losses: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


def _load_shard(path: Path, *, seq_len: int, input_dim: int) -> TaskShard:
    """Load a shard directory into a :class:`TaskShard`.

    A shard is a directory containing a ``train.pt`` and ``val.pt`` each
    storing a ``(X, y)`` tuple saved via :func:`torch.save`. When no
    such files exist, the shard is synthesised with random normal data
    seeded on the directory name so the CLI still runs on empty disks
    (useful for smoke tests).

    Args:
        path: Shard directory.
        seq_len: Sequence length for synthesised data.
        input_dim: Input feature dim for synthesised data.

    Returns:
        A :class:`TaskShard` ready for training.
    """
    train_file = path / "train.pt"
    val_file = path / "val.pt"
    name = path.name

    if train_file.exists() and val_file.exists():
        train_x, train_y = torch.load(train_file, map_location="cpu")
        val_x, val_y = torch.load(val_file, map_location="cpu")
    else:
        logger.warning(
            "Shard %s missing files; synthesising random data.", path
        )
        g = torch.Generator().manual_seed(
            abs(hash(name)) % (2**31 - 1)
        )
        n_train, n_val = 64, 16
        train_x = torch.randn(n_train, seq_len, input_dim, generator=g)
        val_x = torch.randn(n_val, seq_len, input_dim, generator=g)
        train_y = torch.randn(n_train, _DEFAULT_EMBEDDING_DIM, generator=g)
        val_y = torch.randn(n_val, _DEFAULT_EMBEDDING_DIM, generator=g)

    return TaskShard(
        name=name,
        train=TensorDataset(train_x, train_y),
        val=TensorDataset(val_x, val_y),
    )


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _task_loss(model: nn.Module, batch: tuple) -> torch.Tensor:
    """Simple MSE loss between the encoder output and the target."""
    x, y = batch
    out = model(x)
    if out.shape != y.shape:
        y = y.reshape(out.shape)
    return torch.nn.functional.mse_loss(out, y)


def _train_on_task(
    model: nn.Module,
    ewc: ElasticWeightConsolidation,
    task: TaskShard,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> float:
    """Train ``model`` for ``epochs`` on ``task.train`` with EWC penalty."""
    loader = DataLoader(task.train, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last_loss = float("nan")
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for batch in loader:
            optim.zero_grad(set_to_none=True)
            task_loss = _task_loss(model, batch)
            penalty = ewc.penalty_loss()
            loss = task_loss + penalty
            loss.backward()
            optim.step()
            running += float(loss.item())
            n += 1
        last_loss = running / max(n, 1)
        logger.info(
            "  [task=%s epoch=%d/%d] loss=%.4f",
            task.name,
            epoch + 1,
            epochs,
            last_loss,
        )
    return last_loss


@torch.no_grad()
def _evaluate(model: nn.Module, task: TaskShard, *, batch_size: int) -> float:
    """Return mean MSE on ``task.val``."""
    loader = DataLoader(task.val, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        total += float(_task_loss(model, batch).item())
        n += 1
    return total / max(n, 1)


def _retention(loss: float, baseline: float, epsilon: float = 1e-6) -> float:
    """Convert a validation loss into a retention score in ``[0, 1]``."""
    baseline = max(baseline, epsilon)
    deg = (loss - baseline) / baseline
    return max(0.0, min(1.0, 1.0 - deg))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_sequential_training(
    shards: list[TaskShard],
    *,
    lambda_ewc: float = _DEFAULT_LAMBDA,
    online_ewc: bool = False,
    fisher_samples: int = _DEFAULT_FISHER_SAMPLES,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    lr: float = _DEFAULT_LR,
    model_builder: Optional[Callable[[], nn.Module]] = None,
) -> EWCTrainingReport:
    """Train sequentially on ``shards`` with EWC and return a report."""
    if not shards:
        raise ValueError("At least one shard is required")

    encoder: nn.Module
    if model_builder is None:
        encoder = TemporalConvNet(
            input_dim=_DEFAULT_INPUT_DIM, embedding_dim=_DEFAULT_EMBEDDING_DIM
        )
    else:
        encoder = model_builder()

    if online_ewc:
        ewc: ElasticWeightConsolidation = OnlineEWC(
            encoder,
            lambda_ewc=lambda_ewc,
            fisher_estimation_samples=fisher_samples,
            loss_closure=_task_loss,
        )
    else:
        ewc = ElasticWeightConsolidation(
            encoder,
            lambda_ewc=lambda_ewc,
            fisher_estimation_samples=fisher_samples,
            loss_closure=_task_loss,
        )

    report = EWCTrainingReport(
        timestamp=time.strftime("%Y%m%dT%H%M%S"),
        lambda_ewc=lambda_ewc,
        online_ewc=online_ewc,
        epochs_per_task=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        task_names=[s.name for s in shards],
    )

    baselines: list[float] = []
    for idx, shard in enumerate(shards):
        logger.info("=== Training task %d: %s ===", idx, shard.name)
        _train_on_task(
            encoder, ewc, shard,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )
        baseline_loss = _evaluate(encoder, shard, batch_size=batch_size)
        baselines.append(baseline_loss)
        report.baseline_losses.append(baseline_loss)
        logger.info(
            "   task=%s baseline val-loss=%.4f", shard.name, baseline_loss
        )

        # Evaluate retention on *earlier* tasks.
        for prev_idx in range(idx + 1):
            prev = shards[prev_idx]
            val_loss = _evaluate(encoder, prev, batch_size=batch_size)
            retention = _retention(val_loss, baselines[prev_idx])
            report.retention_curve.append(
                RetentionRow(
                    task_trained=idx,
                    task_evaluated=prev_idx,
                    task_name=prev.name,
                    loss=val_loss,
                    retention=retention,
                )
            )

        # Consolidate for next round.
        consolidation_loader = DataLoader(
            shard.train, batch_size=batch_size, shuffle=True
        )
        ewc.consolidate(consolidation_loader)
        logger.info("   EWC consolidated (num_tasks=%d)", ewc.num_tasks_consolidated)

    return report


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


def _write_report(report: EWCTrainingReport, report_dir: Path) -> tuple[Path, Path]:
    """Write JSON + Markdown versions of the report. Returns both paths."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ewc_training_{report.timestamp}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"

    payload = {
        "timestamp": report.timestamp,
        "lambda_ewc": report.lambda_ewc,
        "online_ewc": report.online_ewc,
        "epochs_per_task": report.epochs_per_task,
        "batch_size": report.batch_size,
        "learning_rate": report.learning_rate,
        "task_names": report.task_names,
        "baseline_losses": report.baseline_losses,
        "retention_curve": [asdict(r) for r in report.retention_curve],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    lines: list[str] = []
    lines.append(f"# EWC Sequential-Task Training Report\n")
    lines.append(f"- timestamp: `{report.timestamp}`")
    lines.append(f"- lambda_ewc: `{report.lambda_ewc}`")
    lines.append(f"- online_ewc: `{report.online_ewc}`")
    lines.append(
        f"- epochs: `{report.epochs_per_task}` batch: "
        f"`{report.batch_size}` lr: `{report.learning_rate}`"
    )
    lines.append("")
    lines.append("## Task order\n")
    for i, n in enumerate(report.task_names):
        base = (
            report.baseline_losses[i]
            if i < len(report.baseline_losses)
            else float("nan")
        )
        lines.append(f"{i}. **{n}** — baseline val-loss `{base:.4f}`")
    lines.append("")
    lines.append("## Retention curve\n")
    lines.append("| trained | evaluated | task | loss | retention |")
    lines.append("|---------|-----------|------|------|-----------|")
    for row in report.retention_curve:
        lines.append(
            f"| {row.task_trained} | {row.task_evaluated} | "
            f"{row.task_name} | {row.loss:.4f} | {row.retention:.3f} |"
        )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Sequential-task TCN encoder training with Elastic Weight "
            "Consolidation."
        ),
    )
    parser.add_argument(
        "--shard",
        action="append",
        dest="shards",
        default=[],
        help="Path to a shard directory. Pass once per task.",
    )
    parser.add_argument(
        "--lambda-ewc",
        type=float,
        default=_DEFAULT_LAMBDA,
        help="EWC penalty strength λ (default: 1000.0)",
    )
    parser.add_argument(
        "--online-ewc",
        action="store_true",
        help="Use OnlineEWC (Schwarz 2018) instead of vanilla EWC.",
    )
    parser.add_argument(
        "--fisher-samples",
        type=int,
        default=_DEFAULT_FISHER_SAMPLES,
        help="Number of minibatches used to estimate the Fisher diagonal.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_DEFAULT_EPOCHS,
        help="Epochs per task.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_DEFAULT_LR,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=_DEFAULT_SEQ_LEN,
        help="Sequence length for synthesised shards (fallback).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=_DEFAULT_INPUT_DIM,
        help="Input feature dim for synthesised shards (fallback).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_DEFAULT_REPORT_DIR,
        help="Directory to write reports to.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.shards:
        parser.error("Provide at least one --shard")

    shards = [
        _load_shard(Path(p), seq_len=args.seq_len, input_dim=args.input_dim)
        for p in args.shards
    ]

    report = run_sequential_training(
        shards,
        lambda_ewc=args.lambda_ewc,
        online_ewc=args.online_ewc,
        fisher_samples=args.fisher_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    json_path, md_path = _write_report(report, args.report_dir)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
