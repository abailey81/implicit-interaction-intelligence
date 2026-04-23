"""CLI: run the MC-Dropout estimator + counterfactual explainer on a sample window.

Usage::

    python scripts/run_uncertainty_demo.py [--seed 42] [--samples 30]
                                           [--threshold 0.15] [--top-k 3]

The script:

1. Instantiates a randomly-initialised :class:`TemporalConvNet` and a
   matching :class:`AdaptationController` on the default config.
2. Generates a synthetic feature window of shape ``[16, 32]``.
3. Runs :class:`MCDropoutAdaptationEstimator.estimate` and prints the
   resulting :class:`UncertainAdaptationVector`.
4. Runs :class:`CounterfactualExplainer.explain` with a linear
   surrogate and prints the top-``k`` counterfactuals.
5. Emits a Markdown table suitable for copy-pasting into
   ``docs/research/uncertainty_and_counterfactuals.md``.

No checkpoints are required — every call falls back to a random-init
network when the trained weights are not on disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import torch

from i3.adaptation.controller import AdaptationController
from i3.adaptation.types import AdaptationVector
from i3.adaptation.uncertainty import (
    ADAPTATION_DIMS,
    MCDropoutAdaptationEstimator,
    UncertainAdaptationVector,
    confidence_threshold_policy,
    refuse_when_unsure_mask,
)
from i3.config import load_config
from i3.encoder.tcn import TemporalConvNet
from i3.interpretability.counterfactuals import (
    Counterfactual,
    CounterfactualExplainer,
)
from i3.interpretability.feature_attribution import LinearFeatureAdapter

logger = logging.getLogger("i3.uncertainty.demo")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argv slice. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the MC-Dropout uncertainty estimator + counterfactual "
            "explainer on a synthetic feature window."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Synthetic feature-window length.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the I3 YAML config.",
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def _build_pipeline(
    args: argparse.Namespace,
) -> tuple[TemporalConvNet, AdaptationController]:
    """Construct a random-init encoder + adaptation controller."""
    config = load_config(str(args.config))
    encoder = TemporalConvNet(
        input_dim=32,
        embedding_dim=64,
        dropout=0.1,
    )
    controller = AdaptationController(config.adaptation)
    return encoder, controller


def _synthetic_window(seq_len: int, seed: int) -> torch.Tensor:
    """Return a deterministic feature window of shape ``[seq_len, 32]``."""
    gen = torch.Generator().manual_seed(int(seed))
    return torch.rand(seq_len, 32, generator=gen, dtype=torch.float32)


def _markdown_uncertainty_table(uncertain: UncertainAdaptationVector) -> str:
    """Render an UncertainAdaptationVector as a Markdown table."""
    rows = [
        "| Dimension | Mean | Std | CI 2.5% | CI 97.5% |",
        "|-----------|-----:|----:|--------:|---------:|",
    ]
    mean_vec = uncertain.mean_vector()
    mean_tensor = mean_vec.to_tensor()
    for i, name in enumerate(ADAPTATION_DIMS):
        rows.append(
            "| `{name}` | {m:.3f} | {s:.3f} | {lo:.3f} | {hi:.3f} |".format(
                name=name,
                m=float(mean_tensor[i].item()),
                s=float(uncertain.std[i]),
                lo=float(uncertain.ci[i].lower),
                hi=float(uncertain.ci[i].upper),
            )
        )
    return "\n".join(rows)


def _markdown_counterfactuals_table(counterfactuals: list[Counterfactual]) -> str:
    """Render a counterfactual list as a Markdown table."""
    rows = [
        "| Feature | Current | Counterfactual | Dimension | Current | Counterfactual | Sensitivity |",
        "|---------|--------:|---------------:|-----------|--------:|---------------:|------------:|",
    ]
    for cf in counterfactuals:
        rows.append(
            "| `{f}` | {cv:.3f} | {cf_v:.3f} | `{d}` | {cd:.3f} | {cc:.3f} | {s:.3f} |".format(
                f=cf.feature_name,
                cv=cf.current_value,
                cf_v=cf.counterfactual_value,
                d=cf.dimension_affected,
                cd=cf.current_dimension,
                cc=cf.counterfactual_dimension,
                s=cf.sensitivity,
            )
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI demo.

    Args:
        argv: Optional argv slice. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (``0`` on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    torch.manual_seed(int(args.seed))

    encoder, controller = _build_pipeline(args)
    estimator = MCDropoutAdaptationEstimator(
        encoder=encoder,
        controller=controller,
        n_samples=int(args.samples),
        dropout_p=0.1,
    )

    window = _synthetic_window(int(args.seq_len), int(args.seed))
    uncertain = estimator.estimate(window)

    # Refuse-when-unsure application.
    refused: AdaptationVector = refuse_when_unsure_mask(
        uncertain, threshold=float(args.threshold)
    )
    confident = confidence_threshold_policy(
        uncertain, threshold=float(args.threshold)
    )

    # Counterfactual explanation with a linear surrogate.
    explainer = CounterfactualExplainer(
        mapping_fn=LinearFeatureAdapter(32, 8),
        target_delta=0.2,
    )
    mean_vec = uncertain.mean_vector()
    cfs = explainer.explain(
        feature_window=window,
        adaptation=mean_vec.to_tensor(),
        k=int(args.top_k),
    )

    logger.info("# MC Dropout uncertainty summary\n")
    logger.info("%s\n", _markdown_uncertainty_table(uncertain))
    logger.info(
        "All dimensions confident (threshold=%.3f): **%s**\n",
        float(args.threshold),
        "yes" if confident else "no",
    )

    logger.info("# Refuse-when-unsure mask\n")
    logger.info("```json\n%s\n```\n", refused.to_dict())

    logger.info("# Counterfactuals (top-%d)\n", int(args.top_k))
    logger.info("%s\n", _markdown_counterfactuals_table(cfs))

    logger.info("# Natural-language explanations\n")
    for cf in cfs:
        logger.info("- %s", CounterfactualExplainer.to_natural_language(cf))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
