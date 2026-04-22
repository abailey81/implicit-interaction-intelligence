"""CLI entry point for the Batch A cross-attention ablation study.

Runs the pre-registered experiment defined in
``docs/experiments/preregistration.md``, emits a JSON dump of the full
result, and a Markdown report with condition tables, pairwise effect
sizes, interpretation, and threats to validity.

The script can be run without a trained checkpoint (the primary mode for
Batch A — we measure responsiveness, not quality). If the environment
variable ``I3_CHECKPOINT_PATH`` is set and points at a ``.pt`` state dict
the script will load it via ``AblationExperiment(checkpoint_path=...)``.

Usage::

    python scripts/run_ablation_study.py --seed 42 --n-prompts 50 \\
        --out reports/ablation_study.json \\
        --out-md reports/ablation_study.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running the script directly without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.eval.ablation_experiment import (  # noqa: E402  (sys.path mutation above)
    AblationExperiment,
    AblationResult,
    canonical_archetypes,
    canonical_prompts,
)


logger = logging.getLogger("run_ablation_study")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the Batch A cross-attention ablation study.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42, per pre-registration).",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        help="Number of prompts to evaluate (default: 50, the full canonical set).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (default: cpu, required for the H3 latency claim).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=f"reports/ablation_study_{ts}.json",
        help="Output path for the JSON result dump.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=f"reports/ablation_study_{ts}.md",
        help="Output path for the Markdown report.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging to stderr.",
    )
    return parser.parse_args()


def _git_sha(repo_root: Path) -> str:
    """Return the current HEAD commit SHA, or ``"unknown"`` on failure.

    This is the ONE place the script shells out — the analysis-code-hash
    commitment in the pre-registration requires a real SHA from the git
    index at run time, which cannot be inferred from the Python import
    graph.

    Args:
        repo_root: Path to the repository root (where ``.git`` lives).

    Returns:
        A 40-character hex SHA on success, ``"unknown"`` otherwise.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


def _result_to_json(result: AblationResult, git_sha: str, seed: int) -> dict[str, object]:
    """Serialise an :class:`AblationResult` plus run metadata to a dict.

    Args:
        result: The experiment result.
        git_sha: Repo HEAD commit SHA.
        seed: Seed used for this run.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "schema_version": 1,
        "run_metadata": {
            "git_sha": git_sha,
            "seed": seed,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "preregistration": "docs/experiments/preregistration.md",
        },
        "summary": {
            "conditions": result.conditions,
            "archetype_labels": result.archetype_labels,
            "n_prompts": result.n_prompts,
            "n_pairs_per_condition": result.n_pairs_per_condition,
            "condition_kl_means": result.condition_kl_means,
            "condition_kl_cis": {
                k: list(v) for k, v in result.condition_kl_cis.items()
            },
            "condition_style_fidelity": result.condition_style_fidelity,
            "condition_style_fidelity_cis": {
                k: list(v) for k, v in result.condition_style_fidelity_cis.items()
            },
            "condition_latency_ms_p50": result.condition_latency_ms_p50,
            "condition_latency_ms_p95": result.condition_latency_ms_p95,
            "condition_latency_ms_p99": result.condition_latency_ms_p99,
            "pairwise_cohens_d": result.pairwise_cohens_d,
            "pairwise_sign_p": result.pairwise_sign_p,
            "pairwise_effect_label": result.pairwise_effect_label,
        },
        "per_pair_records": result.to_dataframe_rows(),
    }


def _format_markdown(
    result: AblationResult,
    git_sha: str,
    seed: int,
    checkpoint_path: str | None,
) -> str:
    """Render the :class:`AblationResult` as a Markdown report.

    Args:
        result: The experiment result.
        git_sha: Repo HEAD commit SHA.
        seed: Seed used for the run.
        checkpoint_path: Checkpoint path if supplied, else ``None``.

    Returns:
        Markdown string.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    checkpoint_line = (
        f"- **Checkpoint:** `{checkpoint_path}`"
        if checkpoint_path
        else "- **Checkpoint:** none — **random-init model** (H1/H2 measure responsiveness, not quality)."
    )

    lines: list[str] = []
    lines.append("# Batch A — Cross-Attention Conditioning Ablation (Results)")
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Git SHA:** `{git_sha}`")
    lines.append(f"- **Seed:** `{seed}`")
    lines.append(f"- **n_prompts:** `{result.n_prompts}`")
    lines.append(f"- **n_pairs_per_condition:** `{result.n_pairs_per_condition}`")
    lines.append(checkpoint_line)
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "This report is the empirical output of the pre-registered Batch A "
        "ablation study documented at `docs/experiments/preregistration.md`. "
        "Three conditions — no conditioning (`none`), prompt-based conditioning "
        "(`prompt`), and architectural cross-attention conditioning "
        "(`cross_attn`) — are applied to an `AdaptiveSLM` instance, and the "
        "pairwise symmetric KL divergence between next-token distributions for "
        "eight archetype `AdaptationVector`s is computed over a fixed 50-prompt "
        "test set. The pre-registration commits to 1200 total forward passes "
        "and the exact statistical plan below; any deviations are flagged in "
        "the **Deviations** section at the end."
    )
    lines.append("")

    # -- Main results table --
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Condition | KL mean (nats) | 95% CI | Style fidelity | Style CI | "
        "Latency P50 (ms) | P95 (ms) | P99 (ms) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|"
    )
    for c in result.conditions:
        kl_mean = result.condition_kl_means[c]
        kl_lo, kl_hi = result.condition_kl_cis[c]
        sty_mean = result.condition_style_fidelity[c]
        sty_lo, sty_hi = result.condition_style_fidelity_cis[c]
        lines.append(
            f"| `{c}` | {kl_mean:.5f} | [{kl_lo:.5f}, {kl_hi:.5f}] | "
            f"{sty_mean:.3f} | [{sty_lo:.3f}, {sty_hi:.3f}] | "
            f"{result.condition_latency_ms_p50[c]:.2f} | "
            f"{result.condition_latency_ms_p95[c]:.2f} | "
            f"{result.condition_latency_ms_p99[c]:.2f} |"
        )
    lines.append("")

    # -- Pairwise table --
    lines.append("### Pairwise comparisons")
    lines.append("")
    lines.append("| Comparison | Cohen's d | Interpretation | Sign-test p |")
    lines.append("|---|---|---|---|")
    for key in ["cross_attn_vs_prompt", "cross_attn_vs_none", "prompt_vs_none"]:
        d = result.pairwise_cohens_d.get(key, 0.0)
        p = result.pairwise_sign_p.get(key, 1.0)
        lab = result.pairwise_effect_label.get(key, "negligible")
        lines.append(f"| `{key}` | {d:+.3f} | {lab} | {p:.4g} |")
    lines.append("")

    # -- Latency overhead vs H3 --
    none_p50 = result.condition_latency_ms_p50.get("none", float("nan"))
    cross_p50 = result.condition_latency_ms_p50.get("cross_attn", float("nan"))
    if none_p50 and none_p50 > 0:
        overhead_pct = 100.0 * (cross_p50 - none_p50) / none_p50
    else:
        overhead_pct = float("nan")
    lines.append(
        f"**H3 latency overhead (cross_attn vs none, P50):** {overhead_pct:+.1f} %. "
        f"Threshold: < 15 %."
    )
    lines.append("")

    # -- Interpretation --
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The KL means are read as *responsiveness* scores: how much does the "
        "next-token distribution move when the `AdaptationVector` is swapped "
        "between two archetypes, under this condition? Under a RANDOM-INIT "
        "model — the default mode for Batch A — the **absolute** magnitudes "
        "are expected to be small, because untrained cross-attention weights "
        "are not yet tuned to amplify the conditioning signal. The load-bearing "
        "claim is the **direction**: cross-attention responsiveness should "
        "exceed prompt-based responsiveness, which should exceed no-conditioning "
        "responsiveness (floor ≈ 0). A directionally-correct ordering on a "
        "random-init model is strong evidence that the *architectural capacity* "
        "for responsiveness is real and not an artefact of learned weights."
    )
    lines.append("")
    lines.append(
        "The style-fidelity metric is a length-distribution match against a "
        "verbosity-conditioned Gaussian target; again under random-init the "
        "values are noisy, and a higher mean for `cross_attn` is a necessary "
        "but not sufficient indicator that the architecture *can* track style "
        "once trained."
    )
    lines.append("")

    # -- Threats to validity --
    lines.append("## Threats to validity")
    lines.append("")
    lines.append(
        "1. **Random-init SLM limits external validity.** The forward pass is "
        "deterministic given the input, but the weights have never seen "
        "natural-language supervision, so the KL numbers cannot be compared "
        "to a trained baseline in the field. They measure architectural "
        "*plumbing*, not *learning*."
    )
    lines.append(
        "2. **Synthetic archetype AdaptationVectors.** The eight archetypes "
        "are hand-designed corners of the adaptation space, not empirically "
        "sampled user profiles; they may over- or under-emphasise any single "
        "dimension."
    )
    lines.append(
        "3. **Short context windows.** All forward passes are capped at 32 "
        "prompt tokens to keep the latency claim interpretable. Attention "
        "dilution over longer contexts (the effect that motivates the "
        "architectural approach in the first place) is not tested here."
    )
    lines.append(
        "4. **Single-seed run.** The pre-registration fixes seed = 42. "
        "Variance across seeds is not reported; readers should treat the "
        "confidence intervals as *within-sample* bootstrap intervals, not "
        "cross-run confidence."
    )
    lines.append("")

    # -- Deviations --
    lines.append("## Deviations from pre-registration")
    lines.append("")
    lines.append("None recorded automatically. If any deviation is introduced "
                 "after the fact, append it here in a dated sub-section.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run the ablation study. Returns the shell exit code.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    checkpoint_env = os.environ.get("I3_CHECKPOINT_PATH")
    checkpoint_path: str | None = checkpoint_env if checkpoint_env else None
    if checkpoint_path:
        logger.info("Loading checkpoint from I3_CHECKPOINT_PATH=%s", checkpoint_path)

    # Surface the canonical dataset in logs for traceability.
    logger.info(
        "Canonical set: %d prompts, %d archetypes",
        len(canonical_prompts()),
        len(canonical_archetypes()),
    )

    experiment = AblationExperiment(
        seed=args.seed,
        n_prompts=args.n_prompts,
        device=args.device,
        checkpoint_path=checkpoint_path,
    )
    result = experiment.run()

    git_sha = _git_sha(_ROOT)

    # -- Write JSON --
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = _result_to_json(result, git_sha=git_sha, seed=args.seed)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote JSON result: %s", out_json)

    # -- Write Markdown --
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = _format_markdown(
        result,
        git_sha=git_sha,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
    )
    out_md.write_text(md, encoding="utf-8")
    logger.info("wrote Markdown report: %s", out_md)

    print(f"JSON:     {out_json}")
    print(f"Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
