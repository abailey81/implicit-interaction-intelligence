"""EWC demo on a synthetic sequential-persona stream.

Generates a three-task curriculum from the existing persona library:

1. :data:`~i3.eval.simulation.FRESH_USER`,
2. :data:`~i3.eval.simulation.FATIGUED_DEVELOPER`,
3. :data:`~i3.eval.simulation.MOTOR_IMPAIRED_USER`,

trains the TCN encoder sequentially with and without EWC, and reports
the retention of task-1 knowledge after tasks 2 and 3. The demo prints
a compact table and writes a Markdown + JSON report to ``reports/``.

Expected behaviour: the retention of FRESH_USER after learning
MOTOR_IMPAIRED_USER should be higher when EWC is active (Kirkpatrick
2017 shows roughly 45 % forgetting reduction in comparable settings).

Example::

    python scripts/run_ewc_demo.py --lambda-ewc 2000 --epochs 3

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.eval.simulation.personas import (
    FATIGUED_DEVELOPER,
    FRESH_USER,
    MOTOR_IMPAIRED_USER,
    HCIPersona,
)
from i3.eval.simulation.user_simulator import UserSimulator
from training.train_with_ewc import (
    TaskShard,
    _write_report as write_ewc_report,  # re-use report formatter
    run_sequential_training,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthesising persona shards
# ---------------------------------------------------------------------------


_DEFAULT_SEQ_LEN: int = 16
_DEFAULT_INPUT_DIM: int = 32
_DEFAULT_EMBEDDING_DIM: int = 64


def _persona_signature(persona: HCIPersona) -> np.ndarray:
    """Build a deterministic 64-dim ground-truth signature per persona.

    The signature is used as the regression target so the encoder must
    learn a persona-specific mapping. Personas with more divergent
    typing + linguistic profiles get more different signatures, which
    is exactly what makes catastrophic forgetting visible.
    """
    tp = persona.typing_profile
    lp = persona.linguistic_profile
    base = np.array(
        [
            tp.inter_key_interval_ms[0] / 500.0,
            tp.burst_ratio[0],
            tp.pause_ratio[0],
            tp.correction_rate[0],
            tp.typing_speed_cpm[0] / 500.0,
            lp.flesch_kincaid_target / 20.0,
            lp.formality_target,
            lp.verbosity_mean / 50.0,
            lp.sentiment_baseline,
        ],
        dtype=np.float32,
    )
    # Deterministic projection into 64-dim via hashed noise.
    rng = np.random.default_rng(abs(hash(persona.name)) % (2**31 - 1))
    proj = rng.normal(size=(base.size, _DEFAULT_EMBEDDING_DIM)).astype(np.float32)
    signature = base @ proj
    signature /= np.linalg.norm(signature) + 1e-6
    return signature.astype(np.float32)


def _synthesise_shard(
    persona: HCIPersona,
    *,
    n_train: int,
    n_val: int,
    seq_len: int,
    input_dim: int,
    seed: int,
) -> TaskShard:
    """Synthesise a :class:`TaskShard` for a persona.

    Uses the :class:`UserSimulator` to emit plausible inter-key-interval
    streams, then encodes each message into a feature window with simple
    summary statistics padded to ``input_dim`` channels.
    """
    sim = UserSimulator(persona, seed=seed)
    msgs = sim.run_session(n_train + n_val)
    xs: list[np.ndarray] = []
    for msg in msgs:
        iki = np.asarray(msg.keystroke_intervals_ms, dtype=np.float32)
        if iki.size == 0:
            iki = np.zeros(seq_len, dtype=np.float32)
        # Window of size ``seq_len`` -- pad / truncate.
        if iki.size < seq_len:
            iki = np.pad(iki, (0, seq_len - iki.size), mode="edge")
        else:
            iki = iki[-seq_len:]
        # Broadcast to ``input_dim`` channels with a deterministic per-
        # channel gain so the encoder has non-trivial structure to learn.
        rng = np.random.default_rng(abs(hash(msg.text)) % (2**31 - 1))
        gain = rng.normal(size=(input_dim,)).astype(np.float32)
        window = np.outer(iki, gain) / 500.0
        xs.append(window)

    X = torch.from_numpy(np.stack(xs, axis=0)).float()
    signature = torch.from_numpy(_persona_signature(persona)).float()
    y = signature.expand(X.shape[0], -1).contiguous()
    # Add a small persona-specific noise to the targets.
    rng_torch = torch.Generator().manual_seed(seed)
    y = y + 0.01 * torch.randn(y.shape, generator=rng_torch)

    X_train, X_val = X[:n_train], X[n_train : n_train + n_val]
    y_train, y_val = y[:n_train], y[n_train : n_train + n_val]
    return TaskShard(
        name=persona.name,
        train=TensorDataset(X_train, y_train),
        val=TensorDataset(X_val, y_val),
    )


def _build_shards(
    personas: Sequence[HCIPersona],
    *,
    n_train: int,
    n_val: int,
    seq_len: int,
    input_dim: int,
    seed: int,
) -> list[TaskShard]:
    return [
        _synthesise_shard(
            p,
            n_train=n_train,
            n_val=n_val,
            seq_len=seq_len,
            input_dim=input_dim,
            seed=seed + i,
        )
        for i, p in enumerate(personas)
    ]


# ---------------------------------------------------------------------------
# Comparative summary
# ---------------------------------------------------------------------------


@dataclass
class RetentionComparison:
    """Retention of the first task after each subsequent task."""

    task_trained: int
    task_name: str
    retention_without_ewc: float
    retention_with_ewc: float
    improvement: float


def _extract_first_task_retention(
    report_rows: list,
) -> dict[int, float]:
    """Pull out retention of task 0 from a report.

    Returns a mapping ``{trained_idx: retention_of_task_0}``.
    """
    out: dict[int, float] = {}
    for row in report_rows:
        if row.task_evaluated == 0:
            out[row.task_trained] = row.retention
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "EWC demo on a three-persona sequential stream: "
            "FRESH_USER -> FATIGUED_DEVELOPER -> MOTOR_IMPAIRED_USER."
        )
    )
    p.add_argument("--lambda-ewc", type=float, default=2000.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=_DEFAULT_SEQ_LEN)
    p.add_argument("--input-dim", type=int, default=_DEFAULT_INPUT_DIM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--report-dir",
        type=Path,
        default=_PROJECT_ROOT / "reports",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--online-ewc",
        action="store_true",
        help="Use Online EWC variant (Schwarz 2018).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    personas = (FRESH_USER, FATIGUED_DEVELOPER, MOTOR_IMPAIRED_USER)
    logger.info("Running EWC demo on: %s", [p.name for p in personas])

    # We need two parallel runs that see *exactly* the same shards so
    # only the EWC setting differs.
    shards_no_ewc = _build_shards(
        personas,
        n_train=args.n_train,
        n_val=args.n_val,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        seed=args.seed,
    )
    shards_with_ewc = _build_shards(
        personas,
        n_train=args.n_train,
        n_val=args.n_val,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        seed=args.seed,
    )

    logger.info("=== Sequential training WITHOUT EWC (λ=0) ===")
    report_no_ewc = run_sequential_training(
        shards_no_ewc,
        lambda_ewc=0.0,
        online_ewc=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    logger.info("=== Sequential training WITH EWC (λ=%.1f) ===", args.lambda_ewc)
    report_with_ewc = run_sequential_training(
        shards_with_ewc,
        lambda_ewc=args.lambda_ewc,
        online_ewc=args.online_ewc,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Compare FRESH_USER retention.
    ret_no = _extract_first_task_retention(report_no_ewc.retention_curve)
    ret_yes = _extract_first_task_retention(report_with_ewc.retention_curve)
    comparisons: list[RetentionComparison] = []
    for task_idx in sorted(ret_yes):
        comparisons.append(
            RetentionComparison(
                task_trained=task_idx,
                task_name=report_with_ewc.task_names[task_idx],
                retention_without_ewc=ret_no.get(task_idx, float("nan")),
                retention_with_ewc=ret_yes[task_idx],
                improvement=(
                    ret_yes[task_idx] - ret_no.get(task_idx, float("nan"))
                ),
            )
        )

    # Print a compact table to stdout.
    print()
    print("FRESH_USER retention after each subsequent task:")
    print(f"{'trained':>8} {'task':<24} {'no-EWC':>8} {'EWC':>8} {'Δ':>8}")
    for c in comparisons:
        print(
            f"{c.task_trained:>8} {c.task_name:<24} "
            f"{c.retention_without_ewc:>8.3f} "
            f"{c.retention_with_ewc:>8.3f} "
            f"{c.improvement:>+8.3f}"
        )
    print()

    # Persist both reports + comparison.
    args.report_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    demo_json = args.report_dir / f"ewc_demo_{ts}.json"
    demo_md = args.report_dir / f"ewc_demo_{ts}.md"

    payload = {
        "timestamp": ts,
        "lambda_ewc": args.lambda_ewc,
        "online_ewc": args.online_ewc,
        "personas": [p.name for p in personas],
        "comparison": [asdict(c) for c in comparisons],
        "report_no_ewc": {
            "baseline_losses": report_no_ewc.baseline_losses,
            "retention_curve": [asdict(r) for r in report_no_ewc.retention_curve],
        },
        "report_with_ewc": {
            "baseline_losses": report_with_ewc.baseline_losses,
            "retention_curve": [asdict(r) for r in report_with_ewc.retention_curve],
        },
    }
    with open(demo_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    lines: list[str] = [
        "# EWC demo: FRESH -> FATIGUED -> MOTOR_IMPAIRED\n",
        f"- timestamp: `{ts}`",
        f"- lambda_ewc: `{args.lambda_ewc}`",
        f"- online_ewc: `{args.online_ewc}`",
        "",
        "## FRESH_USER retention comparison\n",
        "| task trained | name | no-EWC | EWC | Δ |",
        "|--------------|------|--------|-----|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.task_trained} | {c.task_name} | "
            f"{c.retention_without_ewc:.3f} | "
            f"{c.retention_with_ewc:.3f} | "
            f"{c.improvement:+.3f} |"
        )
    lines.append("")
    with open(demo_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also write the detailed per-run reports.
    write_ewc_report(report_no_ewc, args.report_dir)
    write_ewc_report(report_with_ewc, args.report_dir)

    logger.info("Wrote %s", demo_json)
    logger.info("Wrote %s", demo_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
