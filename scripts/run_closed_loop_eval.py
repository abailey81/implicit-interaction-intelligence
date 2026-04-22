"""CLI entry point for the closed-loop persona-simulation evaluation.

Drives the full I3 :class:`~i3.pipeline.engine.Pipeline` through the eight
canonical :class:`~i3.eval.simulation.personas.HCIPersona` instances,
scores persona recovery / adaptation-vector error / convergence speed /
router bias, and emits a JSON result and a Markdown report.

Example::

    python scripts/run_closed_loop_eval.py \\
        --config configs/default.yaml \\
        --out reports/closed_loop_eval_20240422.json \\
        --out-md reports/closed_loop_eval_20240422.md

The script pins the HEAD commit SHA into the JSON payload (for research
traceability) and ensures the output directory exists before writing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running the script directly without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.config import load_config  # noqa: E402  (sys.path mutation above)
from i3.eval.simulation import (  # noqa: E402
    ALL_PERSONAS,
    ClosedLoopEvaluator,
    ClosedLoopResult,
    HCIPersona,
)
from i3.pipeline.engine import Pipeline  # noqa: E402


logger = logging.getLogger("run_closed_loop_eval")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Run the closed-loop persona-simulation evaluation and emit a "
            "JSON + Markdown report."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Pipeline configuration YAML (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help=(
            "Comma-separated persona names to include (default: all 8). "
            "Unknown names are rejected with a clear error message."
        ),
    )
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=5,
        help="Number of simulated sessions per persona (default: 5).",
    )
    parser.add_argument(
        "--n-messages",
        type=int,
        default=15,
        help="Messages per session (default: 15).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="L2-error threshold below which the system is considered "
        "converged (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global deterministic seed (default: 42).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=f"reports/closed_loop_eval_{ts}.json",
        help="Output path for the JSON result dump.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=f"reports/closed_loop_eval_{ts}.md",
        help="Output path for the Markdown report.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging to stderr.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Persona filtering
# ---------------------------------------------------------------------------


def _select_personas(spec: str) -> list[HCIPersona]:
    """Resolve the ``--personas`` CLI string to a list of personas.

    Args:
        spec: Empty string or a comma-separated persona-name list.

    Returns:
        The filtered persona list, in the order supplied by the user
        (or the default order if ``spec`` is empty).

    Raises:
        ValueError: If any supplied name does not match a canonical
            persona.
    """
    if not spec.strip():
        return list(ALL_PERSONAS)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    index = {p.name: p for p in ALL_PERSONAS}
    unknown = [n for n in wanted if n not in index]
    if unknown:
        raise ValueError(
            f"Unknown persona name(s): {unknown!r}. "
            f"Valid names: {sorted(index)}"
        )
    return [index[n] for n in wanted]


# ---------------------------------------------------------------------------
# Git metadata
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    """Return the current HEAD SHA, or ``"unknown"`` on failure.

    This is the one place the script shells out -- pinning the analysis
    code hash into each report allows downstream consumers to trace a
    result back to the exact source tree that produced it.

    Args:
        repo_root: Directory that contains ``.git``.

    Returns:
        A 40-char hex SHA, or ``"unknown"`` if git is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _result_to_json(
    result: ClosedLoopResult,
    *,
    git_sha: str,
    seed: int,
    config_path: str,
) -> dict[str, Any]:
    """Serialise the result plus run metadata to a JSON-friendly dict.

    Args:
        result: A populated :class:`ClosedLoopResult`.
        git_sha: The HEAD SHA at run time.
        seed: Seed used for this run.
        config_path: Path to the YAML config in use.

    Returns:
        JSON-serialisable dict.
    """
    return {
        "schema_version": 1,
        "run_metadata": {
            "git_sha": git_sha,
            "seed": seed,
            "config": config_path,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        },
        "summary": {
            "persona_order": result.persona_order,
            "per_persona_recovery_rate": result.per_persona_recovery_rate,
            "per_persona_recovery_ci": {
                k: list(v) for k, v in result.per_persona_recovery_ci.items()
            },
            "per_persona_adaptation_error": (
                result.per_persona_adaptation_error
            ),
            "per_persona_adaptation_error_ci": {
                k: list(v)
                for k, v in result.per_persona_adaptation_error_ci.items()
            },
            "per_persona_error_by_message": (
                result.per_persona_error_by_message
            ),
            "convergence_speeds": result.convergence_speeds,
            "persona_confusion_matrix": result.persona_confusion_matrix,
            "aggregate_recovery_rate": result.aggregate_recovery_rate,
            "aggregate_recovery_rate_ci": list(
                result.aggregate_recovery_rate_ci
            ),
            "aggregate_adaptation_error": result.aggregate_adaptation_error,
            "aggregate_adaptation_error_ci": list(
                result.aggregate_adaptation_error_ci
            ),
            "router_bias": result.router_bias,
            "n_sessions_per_persona": result.n_sessions_per_persona,
            "n_messages_per_session": result.n_messages_per_session,
            "adapt_converged_threshold": result.adapt_converged_threshold,
            "wall_clock_seconds": result.wall_clock_seconds,
        },
        "per_message_records": [rec.model_dump() for rec in result.per_message_records],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _ascii_sparkline(values: list[float], width: int = 48) -> str:
    """Render a numeric series as an ASCII sparkline with 8 glyph buckets.

    Args:
        values: Numeric series (all finite).
        width: Target line width; the series is sub-sampled by stride
            (never interpolated) when longer than ``width``.

    Returns:
        An ASCII sparkline string of length at most ``width``.
    """
    if not values:
        return ""
    if len(values) > width:
        stride = max(1, len(values) // width)
        values = values[::stride][:width]
    glyphs = " ▁▂▃▄▅▆▇█"
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng <= 1e-12:
        return glyphs[1] * len(values)
    out = []
    for v in values:
        idx = int(round((v - lo) / rng * (len(glyphs) - 1)))
        idx = max(0, min(len(glyphs) - 1, idx))
        out.append(glyphs[idx])
    return "".join(out)


def _format_markdown(
    result: ClosedLoopResult,
    *,
    git_sha: str,
    seed: int,
    config_path: str,
    personas: list[HCIPersona],
) -> str:
    """Render the :class:`ClosedLoopResult` as a human-readable report.

    Args:
        result: The evaluation result.
        git_sha: The HEAD SHA at run time.
        seed: Seed used for the run.
        config_path: YAML config path.
        personas: Evaluated persona list (used for descriptions).

    Returns:
        A Markdown string.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append("# Closed-Loop Persona Simulation Evaluation")
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Git SHA:** `{git_sha}`")
    lines.append(f"- **Seed:** `{seed}`")
    lines.append(f"- **Config:** `{config_path}`")
    lines.append(
        f"- **Sessions per persona:** `{result.n_sessions_per_persona}`"
    )
    lines.append(
        f"- **Messages per session:** `{result.n_messages_per_session}`"
    )
    lines.append(
        f"- **Convergence threshold:** `{result.adapt_converged_threshold}`"
    )
    lines.append(
        f"- **Wall-clock:** `{result.wall_clock_seconds:.1f} s`"
    )
    lines.append("")

    # -- 1. Methodology --
    lines.append("## 1. Methodology")
    lines.append("")
    lines.append(
        "This report is the empirical output of the closed-loop evaluation "
        "harness documented at `docs/research/closed_loop_evaluation.md`. "
        "For each of the eight canonical HCI personas (see "
        "`i3/eval/simulation/personas.py`) we run "
        f"{result.n_sessions_per_persona} independent simulated sessions of "
        f"{result.n_messages_per_session} messages each, drawn from a seeded "
        "`UserSimulator`. Every message is pushed through the full I3 "
        "`Pipeline` and the resulting `AdaptationVector` is scored against "
        "the persona's ground-truth `expected_adaptation` via Euclidean "
        "distance. 1-NN persona recovery, L2 adaptation error, convergence "
        "speed, and persona-conditional router bias are reported with "
        "95 % bootstrap confidence intervals (10 000 resamples)."
    )
    lines.append("")

    # -- 2. Per-persona recovery rates --
    lines.append("## 2. Per-persona recovery rates")
    lines.append("")
    lines.append("| Persona | Recovery rate | 95 % CI |")
    lines.append("|---|---|---|")
    for p in personas:
        rate = result.per_persona_recovery_rate.get(p.name, 0.0)
        lo, hi = result.per_persona_recovery_ci.get(p.name, (0.0, 0.0))
        lines.append(f"| `{p.name}` | {rate:.3f} | [{lo:.3f}, {hi:.3f}] |")
    agg_lo, agg_hi = result.aggregate_recovery_rate_ci
    lines.append(
        f"| **aggregate** | **{result.aggregate_recovery_rate:.3f}** | "
        f"[{agg_lo:.3f}, {agg_hi:.3f}] |"
    )
    lines.append("")

    # -- 3. Per-persona adaptation error by message --
    lines.append("## 3. Per-persona adaptation error by message index")
    lines.append("")
    lines.append(
        "Each sparkline shows the mean L2 error at each message index "
        "(0 = first message of session, N-1 = last)."
    )
    lines.append("")
    lines.append("| Persona | Error trace |  L2 mean | 95 % CI |")
    lines.append("|---|---|---|---|")
    for p in personas:
        trace = result.per_persona_error_by_message.get(p.name, [])
        spark = _ascii_sparkline(trace, width=48) if trace else ""
        mean_err = result.per_persona_adaptation_error.get(p.name, 0.0)
        lo, hi = result.per_persona_adaptation_error_ci.get(p.name, (0.0, 0.0))
        lines.append(
            f"| `{p.name}` | `{spark}` | {mean_err:.4f} | "
            f"[{lo:.4f}, {hi:.4f}] |"
        )
    lines.append("")

    # -- 4. Convergence speeds --
    lines.append("## 4. Convergence speeds")
    lines.append("")
    lines.append(
        "Mean message index at which the inferred adaptation vector first "
        f"fell below the convergence threshold "
        f"(`{result.adapt_converged_threshold}`). A value of `None` means "
        "no session converged within the window."
    )
    lines.append("")
    lines.append("| Persona | Mean message to convergence |")
    lines.append("|---|---|")
    for p in personas:
        cs = result.convergence_speeds.get(p.name)
        cell = f"{cs:.2f}" if cs is not None else "(no convergence)"
        lines.append(f"| `{p.name}` | {cell} |")
    lines.append("")

    # -- 5. Persona confusion matrix --
    lines.append("## 5. Persona confusion matrix (1-NN recovery, final message)")
    lines.append("")
    lines.append(
        "Row = true persona, column = nearest-neighbour persona at the "
        "final message. Rows sum to the number of sessions per persona."
    )
    lines.append("")
    header = "| True \\ Inferred | " + " | ".join(
        f"`{n}`" for n in result.persona_order
    ) + " |"
    sep = "|---|" + "---|" * len(result.persona_order)
    lines.append(header)
    lines.append(sep)
    for i, true_name in enumerate(result.persona_order):
        row = result.persona_confusion_matrix[i]
        row_str = " | ".join(str(v) for v in row)
        lines.append(f"| `{true_name}` | {row_str} |")
    lines.append("")
    lines.append("### ASCII heatmap (darker = more hits)")
    lines.append("")
    lines.append("```")
    heat_glyphs = " .:-=+*#%@"
    flat = [v for row in result.persona_confusion_matrix for v in row]
    hi = max(flat) if flat else 1
    for i, true_name in enumerate(result.persona_order):
        row = result.persona_confusion_matrix[i]
        glyphs = []
        for v in row:
            idx = 0 if hi <= 0 else int(round(v / hi * (len(heat_glyphs) - 1)))
            glyphs.append(heat_glyphs[idx] * 2)
        lines.append(f"{true_name:<28s} " + " ".join(glyphs))
    lines.append("```")
    lines.append("")

    # -- 6. Router bias --
    lines.append("## 6. Router-bias check (accessibility personas vs. baseline)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in sorted(result.router_bias):
        lines.append(f"| `{key}` | {result.router_bias[key]:.3f} |")
    lines.append("")
    delta = result.router_bias.get("accessibility_vs_baseline_delta", 0.0)
    lines.append(
        f"Accessibility personas (`motor_impaired_user`, `low_vision_user`) "
        f"routed locally at a rate **{delta:+.3f}** above the baseline "
        f"(`fresh_user` or non-accessibility mean). Positive deltas indicate "
        "the accessibility prior is translating into privacy-preserving "
        "local routing."
    )
    lines.append("")

    # -- 7. Threats to validity --
    lines.append("## 7. Threats to validity")
    lines.append("")
    lines.append(
        "1. **Synthetic keystroke signatures.** The `UserSimulator` draws "
        "inter-key intervals from persona-specific Gaussians grounded in "
        "published HCI literature, but no live-user calibration study has "
        "tuned these distributions. Mean and spread may drift from real "
        "populations."
    )
    lines.append(
        "2. **Canonical prompt pool.** Message text is drawn from a fixed "
        "24-prompt library rewritten to match each persona's linguistic "
        "profile. It does not cover code-switched, multi-turn, or long-form "
        "composition."
    )
    lines.append(
        "3. **Ground-truth adaptation vectors are researcher-chosen.** The "
        "`expected_adaptation` for each persona is a literature-informed "
        "best guess rather than an externally validated target. The harness "
        "measures *consistency with the researcher's model*, not clinical "
        "truth."
    )
    lines.append(
        "4. **Single random seed, single pipeline instance.** Variance "
        "across seeds is not reported; readers should treat the confidence "
        "intervals as within-sample bootstrap intervals, not cross-run "
        "confidence."
    )
    lines.append(
        "5. **Router cold-start.** The contextual bandit router is "
        "freshly initialised at the start of each run, so the "
        "persona-conditional routing signal reflects the *prior* rather "
        "than the learned policy. Long-run behaviour may differ."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run_async(args: argparse.Namespace) -> ClosedLoopResult:
    """Build the pipeline, run the evaluator, return the result."""
    config = load_config(args.config, set_seeds=True)
    pipeline = Pipeline(config)
    await pipeline.initialize()
    try:
        personas = _select_personas(args.personas)
        evaluator = ClosedLoopEvaluator(
            pipeline=pipeline,
            personas=personas,
            n_sessions_per_persona=args.n_sessions,
            n_messages_per_session=args.n_messages,
            adapt_converged_threshold=args.threshold,
            seed=args.seed,
        )
        return await evaluator.run()
    finally:
        await pipeline.shutdown()


def main() -> int:
    """Script entry point. Returns the shell exit code.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    personas = _select_personas(args.personas)
    logger.info(
        "Closed-loop eval: %d personas x %d sessions x %d messages",
        len(personas),
        args.n_sessions,
        args.n_messages,
    )

    result = asyncio.run(_run_async(args))
    git_sha = _git_sha(_ROOT)

    # -- Write JSON --
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = _result_to_json(
        result,
        git_sha=git_sha,
        seed=args.seed,
        config_path=args.config,
    )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote JSON result: %s", out_json)

    # -- Write Markdown --
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    md = _format_markdown(
        result,
        git_sha=git_sha,
        seed=args.seed,
        config_path=args.config,
        personas=personas,
    )
    out_md.write_text(md, encoding="utf-8")
    logger.info("wrote Markdown report: %s", out_md)

    print(f"JSON:     {out_json}")
    print(f"Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
