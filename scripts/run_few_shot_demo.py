"""Few-shot adaptation demo / evaluation (Batch G5).

Loads the meta-trained encoder saved by ``training/train_maml.py`` and
evaluates its adaptation error at 1-, 2-, 3-, and 5-shot for every
canonical :class:`~i3.eval.simulation.personas.HCIPersona`, comparing
with a non-meta-trained baseline encoder.

Outputs a Markdown retention curve at
``reports/few_shot_eval_<timestamp>.md``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.encoder.tcn import TemporalConvNet
from i3.eval.simulation.personas import ALL_PERSONAS, HCIPersona
from i3.meta_learning.few_shot_adapter import FewShotAdapter
from i3.meta_learning.task_generator import PersonaTaskGenerator

logger = logging.getLogger(__name__)

SHOTS: tuple[int, ...] = (1, 2, 3, 5)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------


def _evaluate_shots(
    encoder: torch.nn.Module,
    head: torch.nn.Linear,
    persona: HCIPersona,
    *,
    shots: tuple[int, ...] = SHOTS,
    adaptation_lr: float = 0.01,
    n_adaptation_steps: int = 3,
    seed: int = 0,
) -> dict[int, float]:
    """Return the per-shot L2 adaptation error for a single persona."""
    gen = PersonaTaskGenerator(
        personas=[persona],
        support_size=max(shots),
        query_size=5,
        seed=seed,
    )
    task = gen.generate_task(persona=persona)
    adapter = FewShotAdapter(
        encoder,
        n_adaptation_steps=n_adaptation_steps,
        adaptation_lr=adaptation_lr,
        adaptation_head=head,
    )
    query_tensor = torch.stack(
        [m.to_tensor() for m in task.query_set], dim=0
    ).unsqueeze(0)
    target = task.target_adaptation.to_tensor()
    out: dict[int, float] = {}
    for k in shots:
        support = task.support_set[:k]
        adapted = adapter.adapt_to_user(
            support, target_hint=task.target_adaptation
        )
        adapted.eval()
        with torch.no_grad():
            prediction = head(adapted(query_tensor))
        err = float(
            torch.linalg.norm(prediction.view(-1) - target.view(-1)).item()
        )
        out[k] = err
    return out


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _render_report(
    meta_results: dict[str, dict[int, float]],
    baseline_results: dict[str, dict[int, float]],
) -> str:
    """Render the Markdown retention curve."""
    lines: list[str] = []
    lines.append("# Few-shot adaptation evaluation\n")
    lines.append("L2 distance between adapted prediction and ground-truth "
                 "`AdaptationVector`. Lower is better.\n")
    lines.append("## Meta-trained encoder\n")
    header = (
        "| Persona | "
        + " | ".join(f"{k}-shot" for k in SHOTS)
        + " |"
    )
    divider = "| --- | " + " | ".join("---" for _ in SHOTS) + " |"
    lines.append(header)
    lines.append(divider)
    for name, res in meta_results.items():
        row = f"| {name} | " + " | ".join(
            f"{res[k]:.4f}" for k in SHOTS
        ) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Baseline encoder (random init)\n")
    lines.append(header)
    lines.append(divider)
    for name, res in baseline_results.items():
        row = f"| {name} | " + " | ".join(
            f"{res[k]:.4f}" for k in SHOTS
        ) + " |"
        lines.append(row)
    lines.append("")
    lines.append("## Retention delta (baseline - meta)\n")
    lines.append(header)
    lines.append(divider)
    for name, meta_res in meta_results.items():
        base_res = baseline_results[name]
        row = f"| {name} | " + " | ".join(
            f"{base_res[k] - meta_res[k]:+.4f}" for k in SHOTS
        ) + " |"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Few-shot adaptation demo (Batch G5)."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/meta/maml_encoder.pt",
        help="Path to the MAML checkpoint to load.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory where the Markdown report will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Evaluation seed."
    )
    parser.add_argument(
        "--adaptation-lr",
        type=float,
        default=0.01,
        help="Learning rate used during few-shot adaptation.",
    )
    parser.add_argument(
        "--n-adaptation-steps",
        type=int,
        default=3,
        help="Number of gradient steps per few-shot adaptation.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ckpt_path = Path(args.checkpoint_path)
    meta_encoder = TemporalConvNet()
    embedding_dim = meta_encoder.embedding_dim
    meta_head = torch.nn.Linear(embedding_dim, 8)

    if ckpt_path.exists():
        state = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        if "encoder" in state:
            meta_encoder.load_state_dict(state["encoder"])
        if "head" in state:
            meta_head.load_state_dict(state["head"])
        logger.info("Loaded meta-trained checkpoint from %s.", ckpt_path)
    else:
        logger.warning(
            "Checkpoint %s not found; meta encoder falls back to random "
            "initialisation. Run training/train_maml.py first for a real "
            "comparison.",
            ckpt_path,
        )

    baseline_encoder = TemporalConvNet()
    baseline_head = torch.nn.Linear(embedding_dim, 8)
    torch.nn.init.xavier_uniform_(baseline_head.weight)
    torch.nn.init.zeros_(baseline_head.bias)

    meta_results: dict[str, dict[int, float]] = {}
    baseline_results: dict[str, dict[int, float]] = {}
    for persona in ALL_PERSONAS:
        meta_results[persona.name] = _evaluate_shots(
            meta_encoder,
            meta_head,
            persona,
            adaptation_lr=args.adaptation_lr,
            n_adaptation_steps=args.n_adaptation_steps,
            seed=args.seed,
        )
        baseline_results[persona.name] = _evaluate_shots(
            baseline_encoder,
            baseline_head,
            persona,
            adaptation_lr=args.adaptation_lr,
            n_adaptation_steps=args.n_adaptation_steps,
            seed=args.seed + 1,
        )

    report = _render_report(meta_results, baseline_results)
    report_path = (
        Path(args.report_dir)
        / f"few_shot_eval_{int(time.time())}.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved few-shot evaluation to %s.", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
