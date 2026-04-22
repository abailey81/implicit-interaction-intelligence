"""CLI for meta-training the User-State Encoder with MAML.

Usage::

    # Second-order MAML on all-but-one persona, held-out for evaluation:
    python training/train_maml.py --held-out fatigued_developer

    # Cheap FO-MAML variant:
    python training/train_maml.py --first-order --held-out energetic_user

Outputs:
    * Checkpoint ``checkpoints/meta/maml_encoder.pt``.
    * Markdown report ``reports/maml_training_<timestamp>.md`` comparing
      the meta-trained encoder's few-shot adaptation performance
      against the non-meta-trained baseline.

This script is entirely offline: it operates on synthetic tasks drawn
from :class:`~i3.meta_learning.task_generator.PersonaTaskGenerator` and
never touches the Huawei cloud.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Put the project root on sys.path for absolute imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.adaptation.types import AdaptationVector
from i3.encoder.tcn import TemporalConvNet
from i3.eval.simulation.personas import ALL_PERSONAS, HCIPersona
from i3.meta_learning.few_shot_adapter import FewShotAdapter
from i3.meta_learning.maml import MAMLTrainer
from i3.meta_learning.task_generator import PersonaTaskGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Few-shot evaluation helpers
# ---------------------------------------------------------------------------


def adaptation_error(
    predicted: torch.Tensor, target: AdaptationVector
) -> float:
    """Return the L2 distance between a prediction and a target vector.

    Args:
        predicted: Either a ``[1, 8]`` or ``[8]`` tensor.
        target: The ground-truth :class:`AdaptationVector`.
    """
    pred = predicted.detach().view(-1)
    tgt = target.to_tensor().view(-1)
    return float(torch.linalg.norm(pred - tgt).item())


def evaluate_warmup(
    adapter: FewShotAdapter,
    head: torch.nn.Linear,
    persona: HCIPersona,
    max_messages: int = 8,
    adaptation_threshold: float = 0.3,
    seed: int = 0,
) -> dict[str, float]:
    """Report how many support messages suffice to reach the threshold.

    Args:
        adapter: The few-shot adapter wrapping the encoder to evaluate.
        head: The meta-trained 64->8 adaptation head.
        persona: Held-out persona to evaluate on.
        max_messages: Upper bound on the number of support messages
            to try. Must be at least one.
        adaptation_threshold: L2 distance below which we declare the
            adaptation "successful".
        seed: Deterministic seed for the evaluation session.

    Returns:
        A dict with ``"messages_to_threshold"`` (``max_messages + 1``
        if the threshold was never reached) and ``"final_error"``.
    """
    generator = PersonaTaskGenerator(
        personas=[persona], support_size=max_messages, query_size=1, seed=seed
    )
    task = generator.generate_task(persona=persona)

    messages_to_threshold = max_messages + 1
    last_error = float("inf")
    for k in range(1, max_messages + 1):
        support_k = task.support_set[:k]
        adapted = adapter.adapt_to_user(
            support_k, target_hint=task.target_adaptation
        )
        query_seq = torch.stack(
            [m.to_tensor() for m in task.query_set], dim=0
        ).unsqueeze(0)
        adapted.eval()
        with torch.no_grad():
            embedding = adapted(query_seq)
            prediction = head(embedding)
        err = adaptation_error(prediction, task.target_adaptation)
        last_error = err
        if err <= adaptation_threshold and messages_to_threshold > max_messages:
            messages_to_threshold = k
    return {
        "messages_to_threshold": float(messages_to_threshold),
        "final_error": last_error,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    path: Path,
    *,
    meta_stats: dict[str, float],
    baseline_stats: dict[str, float],
    held_out: str,
    first_order: bool,
    n_outer_steps: int,
    history_tail: list[dict[str, float]],
) -> None:
    """Persist a human-readable Markdown report of the run."""
    lines: list[str] = []
    lines.append("# MAML meta-training report\n")
    lines.append(f"- Held-out persona: **{held_out}**")
    lines.append(f"- Algorithm: **{'FO-MAML' if first_order else 'MAML (second-order)'}**")
    lines.append(f"- Outer steps: {n_outer_steps}")
    lines.append("")
    lines.append("## Few-shot adaptation on the held-out persona")
    lines.append("")
    lines.append("| Encoder | Messages to threshold | Final error |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Meta-trained | {meta_stats['messages_to_threshold']:.0f} | "
        f"{meta_stats['final_error']:.4f} |"
    )
    lines.append(
        f"| Baseline (non-meta-trained) | "
        f"{baseline_stats['messages_to_threshold']:.0f} | "
        f"{baseline_stats['final_error']:.4f} |"
    )
    lines.append("")
    lines.append("## Tail of outer-loop training history")
    lines.append("")
    lines.append("| Step | meta_loss | support_loss |")
    lines.append("| --- | --- | --- |")
    for entry in history_tail:
        lines.append(
            f"| {int(entry.get('step', 0))} | "
            f"{entry.get('meta_loss', 0.0):.4f} | "
            f"{entry.get('mean_support_loss', 0.0):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Meta-train the TCN encoder with MAML (Batch G5)."
    )
    parser.add_argument(
        "--held-out",
        type=str,
        default="fatigued_developer",
        help="Name of the persona to hold out for evaluation.",
    )
    parser.add_argument(
        "--n-outer-steps",
        type=int,
        default=200,
        help="Number of outer-loop meta-updates (default: 200).",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=3,
        help="Inner-loop steps per task (default: 3).",
    )
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=0.01,
        help="Inner-loop learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--outer-lr",
        type=float,
        default=1e-3,
        help="Outer-loop (Adam) learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--support-size", type=int, default=3, help="Support-set size."
    )
    parser.add_argument(
        "--query-size", type=int, default=5, help="Query-set size."
    )
    parser.add_argument(
        "--meta-batch-size", type=int, default=4, help="Tasks per outer step."
    )
    parser.add_argument(
        "--first-order",
        action="store_true",
        help="Use FO-MAML (no second-order gradients).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Meta-training seed."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/meta/maml_encoder.pt",
        help="Where to save the meta-trained encoder.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory for the Markdown report.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # --- Select personas -------------------------------------------------
    held_out_name = args.held_out
    train_personas: list[HCIPersona] = [
        p for p in ALL_PERSONAS if p.name != held_out_name
    ]
    held_out_persona: Optional[HCIPersona] = next(
        (p for p in ALL_PERSONAS if p.name == held_out_name), None
    )
    if held_out_persona is None:
        parser.error(
            f"--held-out {held_out_name!r} is not the name of a canonical "
            f"persona. Pick one of "
            f"{[p.name for p in ALL_PERSONAS]}."
        )
        return 2
    if not train_personas:
        parser.error("At least one persona must remain after holding out.")
        return 2

    # --- Build models & trainer ------------------------------------------
    torch.manual_seed(args.seed)
    encoder = TemporalConvNet()
    trainer = MAMLTrainer(
        encoder,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        first_order=bool(args.first_order),
    )
    task_gen = PersonaTaskGenerator(
        personas=train_personas,
        support_size=args.support_size,
        query_size=args.query_size,
        seed=args.seed,
    )

    from collections.abc import Iterator as _Iterator

    from i3.meta_learning.maml import MetaBatch

    def task_iter() -> _Iterator[MetaBatch]:
        while True:
            yield task_gen.generate_batch(
                meta_batch_size=args.meta_batch_size
            )

    # --- Meta-train ------------------------------------------------------
    t0 = time.time()
    history = trainer.meta_train(
        task_iter(),
        n_outer_steps=args.n_outer_steps,
        log_every=max(1, args.n_outer_steps // 10),
    )
    elapsed = time.time() - t0
    logger.info(
        "Meta-training finished in %.1fs over %d outer steps.",
        elapsed,
        args.n_outer_steps,
    )

    # --- Evaluate on the held-out persona --------------------------------
    meta_adapter = FewShotAdapter(
        encoder,
        n_adaptation_steps=args.inner_steps,
        adaptation_lr=args.inner_lr,
        adaptation_head=trainer.head,
    )
    meta_stats = evaluate_warmup(
        meta_adapter, trainer.head, held_out_persona, seed=args.seed
    )

    # Baseline: fresh random encoder + random head.
    baseline_encoder = TemporalConvNet()
    baseline_head = torch.nn.Linear(
        trainer.embedding_dim, trainer.adaptation_dim
    )
    torch.nn.init.xavier_uniform_(baseline_head.weight)
    torch.nn.init.zeros_(baseline_head.bias)
    baseline_adapter = FewShotAdapter(
        baseline_encoder,
        n_adaptation_steps=args.inner_steps,
        adaptation_lr=args.inner_lr,
        adaptation_head=baseline_head,
    )
    baseline_stats = evaluate_warmup(
        baseline_adapter, baseline_head, held_out_persona, seed=args.seed + 1
    )

    # --- Persist checkpoint + report -------------------------------------
    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), ckpt_path)
    logger.info("Saved meta-trained encoder to %s.", ckpt_path)

    report_path = (
        Path(args.report_dir)
        / f"maml_training_{int(time.time())}.md"
    )
    tail = history[-10:] if len(history) > 10 else history
    _write_report(
        report_path,
        meta_stats=meta_stats,
        baseline_stats=baseline_stats,
        held_out=held_out_name,
        first_order=bool(args.first_order),
        n_outer_steps=args.n_outer_steps,
        history_tail=tail,
    )
    logger.info("Saved report to %s.", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
