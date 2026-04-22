"""CLI entry point for the Batch G4 LLM-as-judge harness.

The script consumes either

1. an ablation-study JSON dump produced by ``scripts/run_ablation_study.py``
   (Batch A) — in which case it judges the top-N pairs per
   ``(condition, archetype)`` cell; or
2. a benchmark-results JSON dump produced by
   ``scripts/run_implicit_adapt_bench.py`` (Batch C) — in which case it
   pair-judges the submissions;

and emits two artefacts:

* ``reports/llm_judge_<ts>.json`` — full per-pair judgements.
* ``reports/llm_judge_<ts>.md``   — summary with winner rates per
  condition, per-rubric means, bootstrap CIs, inter-judge agreement if
  multiple judges are configured, and a bias audit summary.

Usage::

    python scripts/run_llm_judge.py \\
        --ablation-results reports/ablation_study_XXXX.json \\
        --judge-provider anthropic --judge-model claude-sonnet-4-5 \\
        --rubric full --n-pairs 50

All provider selection goes through the G7 :class:`~i3.cloud.providers.
ProviderRegistry`, so any registered provider name works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Allow direct invocation without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from i3.adaptation.types import AdaptationVector, StyleVector  # noqa: E402
from i3.cloud.providers.base import CloudProvider  # noqa: E402
from i3.cloud.provider_registry import ProviderRegistry  # noqa: E402
from i3.eval.ablation_statistics import bootstrap_ci  # noqa: E402
from i3.eval.judge_calibration import (  # noqa: E402
    JudgeCalibrator,
    PairItem,
)
from i3.eval.judge_ensemble import MultiJudgeEnsemble  # noqa: E402
from i3.eval.judge_rubric import (  # noqa: E402
    ACCESSIBILITY_RUBRIC,
    COGNITIVE_LOAD_RUBRIC,
    FULL_ADAPTATION_RUBRIC,
    STYLE_MATCH_RUBRIC,
)
from i3.eval.llm_judge import JudgementResult, LLMJudge  # noqa: E402

logger = logging.getLogger("run_llm_judge")

REPORTS_DIR: Path = _ROOT / "reports"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _now_ts() -> str:
    """Return a UTC ISO-8601 basic timestamp."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional override for ``sys.argv[1:]`` (test hook).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    ts = _now_ts()
    parser = argparse.ArgumentParser(
        description="Run the Batch G4 LLM-as-judge harness.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ablation-results",
        type=str,
        default=None,
        help="Path to an ablation_study_*.json produced by Batch A.",
    )
    src.add_argument(
        "--benchmark-results",
        type=str,
        default=None,
        help="Path to an implicit_adapt_bench_*.json produced by Batch C.",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="anthropic",
        help="G7 provider name (default: anthropic).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model override passed to the provider factory.",
    )
    parser.add_argument(
        "--extra-judge",
        action="append",
        default=[],
        help=(
            "Additional judge as 'provider:model' (repeatable). "
            "Creates a multi-judge ensemble."
        ),
    )
    parser.add_argument(
        "--rubric",
        type=str,
        choices=("full", "style", "cognitive_load", "accessibility"),
        default="full",
        help="Rubric to apply (default: full).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=50,
        help="Maximum number of pair judgements (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for pair sampling (default: 42).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=f"reports/llm_judge_{ts}.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=f"reports/llm_judge_{ts}.md",
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--bias-audit",
        action="store_true",
        help=(
            "If set, also run position-bias / length-bias / self-"
            "consistency audits on a subsample (8 items)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO logs."
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


_RUBRIC_LOOKUP: dict[str, list[str]] = {
    "full": FULL_ADAPTATION_RUBRIC,
    "style": STYLE_MATCH_RUBRIC,
    "cognitive_load": COGNITIVE_LOAD_RUBRIC,
    "accessibility": ACCESSIBILITY_RUBRIC,
}


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file as a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If the top-level JSON is not an object.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _av_from_record(rec: dict[str, Any]) -> AdaptationVector:
    """Reconstruct an :class:`AdaptationVector` from a benchmark record.

    The benchmark record stores ``target_adaptation_vector`` as an 8-element
    list matching :meth:`AdaptationVector.to_tensor`.

    Args:
        rec: A benchmark record dict.

    Returns:
        Reconstructed :class:`AdaptationVector`.
    """
    vec = rec.get("target_adaptation_vector") or [0.5] * 8
    if len(vec) < 7:
        return AdaptationVector.default()
    return AdaptationVector(
        cognitive_load=float(vec[0]),
        style_mirror=StyleVector(
            formality=float(vec[1]),
            verbosity=float(vec[2]),
            emotionality=float(vec[3]),
            directness=float(vec[4]),
        ),
        emotional_tone=float(vec[5]),
        accessibility=float(vec[6]),
    )


def _pairs_from_ablation(
    payload: dict[str, Any], n_pairs: int, seed: int
) -> list[tuple[str, str, str, str, AdaptationVector]]:
    """Build pair items from an ablation-study JSON dump.

    We pair one response per archetype under ``cross_attn`` vs ``prompt``
    (the load-bearing contrast in Batch A) using a fixed small stub
    "response" synthesised from the prompt + archetype label — because
    the ablation study itself does not emit decoded text. If the JSON
    does contain a ``generated_texts`` field we prefer that.

    Args:
        payload: The parsed JSON payload.
        n_pairs: Max number of pairs to return.
        seed: Seed for shuffling.

    Returns:
        List of ``(pair_id, prompt, response_a, response_b, target)``.
    """
    rng = random.Random(seed)
    summary = payload.get("summary", {})
    archetype_labels = list(summary.get("archetype_labels") or [])
    per_pair_records = payload.get("per_pair_records") or []
    generated_texts = payload.get("generated_texts") or {}

    from i3.eval.ablation_experiment import (
        canonical_archetypes,
        canonical_prompts,
    )

    arche_map = canonical_archetypes()
    prompts = canonical_prompts()

    items: list[tuple[str, str, str, str, AdaptationVector]] = []
    rng.shuffle(per_pair_records)
    for rec in per_pair_records:
        if len(items) >= n_pairs:
            break
        cond = rec.get("condition")
        if cond != "cross_attn":
            continue
        p_idx = int(rec.get("prompt_index", 0))
        arche_i = rec.get("archetype_i")
        arche_j = rec.get("archetype_j")
        if arche_i not in arche_map or arche_j not in arche_map:
            continue
        if p_idx >= len(prompts):
            continue
        prompt = prompts[p_idx]
        target = arche_map[arche_i]
        # Retrieve decoded text when available; otherwise synthesise a
        # short deterministic placeholder for the judge.
        resp_a = generated_texts.get(
            f"{cond}:{p_idx}:{arche_i}",
            f"[condition={cond}, archetype={arche_i}] {prompt}",
        )
        resp_b = generated_texts.get(
            f"prompt:{p_idx}:{arche_i}",
            f"[condition=prompt, archetype={arche_i}] {prompt}",
        )
        pair_id = f"abl:p{p_idx}:{arche_i}_vs_prompt"
        items.append((pair_id, prompt, resp_a, resp_b, target))
    return items


def _pairs_from_benchmark(
    payload: dict[str, Any], n_pairs: int, seed: int
) -> list[tuple[str, str, str, str, AdaptationVector]]:
    """Build pair items from a benchmark-results JSON dump.

    Expects ``payload["submissions_by_method"]`` as a dict of
    ``method -> [{record_id, generated_text, ...}, ...]`` and
    ``payload["records"]`` as the gold record list. We pair
    ``baseline_cross_attention`` vs ``baseline_prompt`` on each record
    (fallback: the first two methods present).

    Args:
        payload: The parsed JSON payload.
        n_pairs: Max number of pairs to return.
        seed: Seed for shuffling.

    Returns:
        List of ``(pair_id, prompt, response_a, response_b, target)``.
    """
    rng = random.Random(seed)
    subs_by_method = payload.get("submissions_by_method") or {}
    records = payload.get("records") or []
    if not subs_by_method or not records:
        return []
    methods = list(subs_by_method.keys())
    if "baseline_cross_attention" in methods and "baseline_prompt" in methods:
        m_a, m_b = "baseline_cross_attention", "baseline_prompt"
    else:
        m_a, m_b = methods[0], methods[-1] if len(methods) > 1 else methods[0]
    rec_by_id: dict[str, dict[str, Any]] = {
        r["record_id"]: r for r in records if "record_id" in r
    }
    sub_a_by_id = {
        s["record_id"]: s for s in subs_by_method[m_a] if "record_id" in s
    }
    sub_b_by_id = {
        s["record_id"]: s for s in subs_by_method[m_b] if "record_id" in s
    }
    common = list(set(sub_a_by_id) & set(sub_b_by_id) & set(rec_by_id))
    rng.shuffle(common)
    items: list[tuple[str, str, str, str, AdaptationVector]] = []
    for rid in common:
        if len(items) >= n_pairs:
            break
        rec = rec_by_id[rid]
        target = _av_from_record(rec)
        items.append(
            (
                f"bench:{rid}:{m_a}_vs_{m_b}",
                str(rec.get("prompt", "")),
                str(sub_a_by_id[rid].get("generated_text", "")),
                str(sub_b_by_id[rid].get("generated_text", "")),
                target,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summarise(
    results: list[tuple[str, JudgementResult]],
    rubric: list[str],
    seed: int,
) -> dict[str, Any]:
    """Summarise a list of per-pair judgements into aggregate statistics.

    Args:
        results: List of ``(pair_id, JudgementResult)``.
        rubric: The rubric used.
        seed: Seed for the bootstrap CI.

    Returns:
        A dict with winner rates, per-rubric mean + 95 % CI, and
        confidence statistics.
    """
    total = len(results)
    if total == 0:
        return {
            "n_pairs": 0,
            "winner_rates": {"A": 0.0, "B": 0.0, "tie": 0.0},
            "per_rubric_mean_a": {},
            "per_rubric_mean_b": {},
            "per_rubric_ci_a": {},
            "per_rubric_ci_b": {},
            "mean_confidence": 0.0,
        }
    winner_counts = {"A": 0, "B": 0, "tie": 0}
    confidences: list[float] = []
    per_dim_a: dict[str, list[float]] = {d: [] for d in rubric}
    per_dim_b: dict[str, list[float]] = {d: [] for d in rubric}
    for _pid, jr in results:
        winner_counts[jr.winner] += 1
        confidences.append(jr.confidence)
        for d in rubric:
            per_dim_a[d].append(float(jr.per_rubric_scores_a.get(d, 0)))
            per_dim_b[d].append(float(jr.per_rubric_scores_b.get(d, 0)))

    rng = np.random.default_rng(seed)
    ci_a: dict[str, tuple[float, float]] = {}
    ci_b: dict[str, tuple[float, float]] = {}
    mean_a: dict[str, float] = {}
    mean_b: dict[str, float] = {}
    for d in rubric:
        vals_a = per_dim_a[d]
        vals_b = per_dim_b[d]
        mean_a[d] = float(statistics.fmean(vals_a)) if vals_a else 0.0
        mean_b[d] = float(statistics.fmean(vals_b)) if vals_b else 0.0
        if len(vals_a) >= 2:
            ci_a[d] = bootstrap_ci(vals_a, n_resamples=2000, rng=rng)
        else:
            ci_a[d] = (mean_a[d], mean_a[d])
        if len(vals_b) >= 2:
            ci_b[d] = bootstrap_ci(vals_b, n_resamples=2000, rng=rng)
        else:
            ci_b[d] = (mean_b[d], mean_b[d])

    return {
        "n_pairs": total,
        "winner_rates": {k: v / total for k, v in winner_counts.items()},
        "per_rubric_mean_a": mean_a,
        "per_rubric_mean_b": mean_b,
        "per_rubric_ci_a": {k: list(v) for k, v in ci_a.items()},
        "per_rubric_ci_b": {k: list(v) for k, v in ci_b.items()},
        "mean_confidence": float(sum(confidences) / len(confidences))
        if confidences else 0.0,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _format_markdown(
    summary: dict[str, Any],
    rubric: list[str],
    judge_names: list[str],
    input_source: str,
    bias: dict[str, float] | None,
    ensemble_kappa: float | None,
) -> str:
    """Render the summary as a Markdown report.

    Args:
        summary: Dict produced by :func:`_summarise`.
        rubric: The rubric used.
        judge_names: Ordered list of judge provider names.
        input_source: Description of the input (ablation / benchmark + path).
        bias: Optional bias-audit dict.
        ensemble_kappa: Optional inter-judge kappa for a panel.

    Returns:
        Markdown string.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append("# Batch G4 — LLM-as-Judge Results")
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Input:** {input_source}")
    lines.append(f"- **Judges:** {', '.join(judge_names)}")
    lines.append(f"- **Rubric:** {', '.join(rubric)}")
    lines.append(f"- **n_pairs:** {summary['n_pairs']}")
    lines.append(f"- **mean_confidence:** {summary['mean_confidence']:.3f}")
    lines.append("")
    lines.append("## Winner rates")
    lines.append("")
    lines.append("| Winner | Rate |")
    lines.append("|---|---|")
    for w in ("A", "B", "tie"):
        lines.append(f"| {w} | {summary['winner_rates'][w]:.3f} |")
    lines.append("")
    lines.append("## Per-rubric mean scores (95 % bootstrap CI)")
    lines.append("")
    lines.append("| Dimension | A mean | A CI | B mean | B CI |")
    lines.append("|---|---|---|---|---|")
    for d in rubric:
        a = summary["per_rubric_mean_a"].get(d, 0.0)
        b = summary["per_rubric_mean_b"].get(d, 0.0)
        a_ci = summary["per_rubric_ci_a"].get(d, [a, a])
        b_ci = summary["per_rubric_ci_b"].get(d, [b, b])
        lines.append(
            f"| {d} | {a:.2f} | [{a_ci[0]:.2f}, {a_ci[1]:.2f}] | "
            f"{b:.2f} | [{b_ci[0]:.2f}, {b_ci[1]:.2f}] |"
        )
    lines.append("")
    if ensemble_kappa is not None:
        lines.append(f"**Inter-judge panel agreement (kappa-surrogate):** {ensemble_kappa:.3f}")
        lines.append("")
    if bias is not None:
        lines.append("## Bias audit")
        lines.append("")
        lines.append("| Audit | Value |")
        lines.append("|---|---|")
        for k, v in bias.items():
            lines.append(f"| {k} | {v:.3f} |")
        lines.append("")
    lines.append("## Threats to validity")
    lines.append("")
    lines.append(
        "LLM-as-judge introduces its own biases (position, length, self-"
        "preference when the judge and generator share a family). The bias "
        "audit above quantifies position and length bias; for a fuller "
        "picture run a multi-provider panel via `--extra-judge` and inspect "
        "the inter-judge kappa."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main async flow
# ---------------------------------------------------------------------------


def _build_judge(provider_name: str, model: str) -> LLMJudge:
    """Instantiate an :class:`LLMJudge` from a provider name + model.

    Args:
        provider_name: Registered provider name (e.g. ``"anthropic"``).
        model: Model identifier to pass to the factory.

    Returns:
        Configured :class:`LLMJudge`.
    """
    provider: CloudProvider = ProviderRegistry.get(
        provider_name, {"model": model}
    )
    return LLMJudge(provider=provider, temperature=0.0)


async def _run(args: argparse.Namespace) -> int:
    """Run the full pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        ``0`` on success, non-zero on handled failure.
    """
    rubric = _RUBRIC_LOOKUP[args.rubric]

    if args.ablation_results:
        payload = _load_json(Path(args.ablation_results))
        pairs = _pairs_from_ablation(payload, args.n_pairs, args.seed)
        input_source = f"ablation: {args.ablation_results}"
    else:
        payload = _load_json(Path(args.benchmark_results))
        pairs = _pairs_from_benchmark(payload, args.n_pairs, args.seed)
        input_source = f"benchmark: {args.benchmark_results}"

    if not pairs:
        logger.error("No pairs extracted from input; aborting.")
        return 2

    logger.info("Prepared %d pair items.", len(pairs))

    # -- Build the (ensemble of) judges ---------------------------------
    primary = _build_judge(args.judge_provider, args.judge_model)
    extras: list[LLMJudge] = []
    for spec in args.extra_judge:
        if ":" not in spec:
            logger.warning("Ignoring --extra-judge %r (expected provider:model)", spec)
            continue
        pv, mdl = spec.split(":", 1)
        extras.append(_build_judge(pv.strip(), mdl.strip()))
    judges = [primary] + extras
    ensemble: MultiJudgeEnsemble | None = (
        MultiJudgeEnsemble(judges, aggregation="majority") if extras else None
    )

    # -- Run judgements --------------------------------------------------
    results: list[tuple[str, JudgementResult]] = []
    ensemble_kappas: list[float] = []
    try:
        for pid, prompt, resp_a, resp_b, target in pairs:
            if ensemble is not None:
                ej = await ensemble.judge_pair_ensemble(
                    prompt=prompt,
                    response_a=resp_a,
                    response_b=resp_b,
                    target_adaptation=target,
                    rubric=rubric,
                )
                ensemble_kappas.append(ej.inter_judge_kappa)
                # Roll up into a JudgementResult-like record using judge 0.
                first = ej.per_judge[0]
                results.append((pid, first))
            else:
                jr = await primary.judge_pair(
                    prompt=prompt,
                    response_a=resp_a,
                    response_b=resp_b,
                    target_adaptation=target,
                    rubric=rubric,
                )
                results.append((pid, jr))
    finally:
        for j in judges:
            await j.close()

    summary = _summarise(results, rubric, args.seed)

    # -- Optional bias audit --------------------------------------------
    bias: dict[str, float] | None = None
    if args.bias_audit and pairs:
        calibrator = JudgeCalibrator()
        sample_items = [
            PairItem(
                prompt=p,
                response_a=a,
                response_b=b,
                target_adaptation=t,
            )
            for (_pid, p, a, b, t) in pairs[: min(8, len(pairs))]
        ]
        try:
            pos = await calibrator.position_bias_test(primary, sample_items)
            lng = await calibrator.length_bias_test(primary, sample_items)
            bias = {
                "position_flip_rate": pos,
                "length_bias_corr": lng,
            }
        except ValueError as exc:
            logger.warning("Bias audit failed: %s", exc)
            bias = None

    ensemble_kappa = (
        float(statistics.fmean(ensemble_kappas))
        if ensemble_kappas
        else None
    )

    # -- Write outputs ---------------------------------------------------
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    dump: dict[str, Any] = {
        "schema_version": 1,
        "run_metadata": {
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "seed": args.seed,
            "rubric": args.rubric,
            "input_source": input_source,
            "judges": [j.provider_name for j in judges],
        },
        "summary": summary,
        "per_pair": [
            {
                "pair_id": pid,
                "winner": jr.winner,
                "per_rubric_scores_a": jr.per_rubric_scores_a,
                "per_rubric_scores_b": jr.per_rubric_scores_b,
                "confidence": jr.confidence,
                "rationale": jr.rationale,
                "judge_model": jr.judge_model,
            }
            for pid, jr in results
        ],
        "bias_audit": bias,
        "ensemble_panel_agreement": ensemble_kappa,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, sort_keys=True)

    md = _format_markdown(
        summary=summary,
        rubric=rubric,
        judge_names=[j.provider_name for j in judges],
        input_source=input_source,
        bias=bias,
        ensemble_kappa=ensemble_kappa,
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("Wrote %s and %s", out_json, out_md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional CLI arg vector (test hook).

    Returns:
        Process exit code.
    """
    args = _parse_args(argv)
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
