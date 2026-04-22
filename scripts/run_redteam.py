"""CLI entry point for the I3 red-team harness.

Run the 55-attack corpus across one or more target surfaces, aggregate
the results, evaluate the four runtime invariants, and emit both a
machine-readable JSON report and a human-readable Markdown report.

Exit codes:
    0 -- every attack matched its expected outcome and no critical
         failures were observed.
    1 -- at least one critical attack slipped through, or the overall
         pass rate dropped below ``1.0 - tolerance``.
    2 -- an internal error occurred (bad arguments, invariant checker
         crashed, ...).

Example:
    poetry run python scripts/run_redteam.py \\
        --targets fastapi,sanitizer,pddl,guardrails \\
        --fail-fast \\
        --out reports/redteam_ci.json \\
        --out-md reports/redteam_ci.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess  # nosec: B404 -- only `git rev-parse HEAD` with a fixed arg list
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from i3.redteam import (
    ATTACK_CORPUS,
    Attack,
    AttackResult,
    GuardrailsTargetSurface,
    PDDLPlannerTargetSurface,
    RedTeamReport,
    RedTeamRunner,
    SanitizerTargetSurface,
    TargetSurface,
    load_external_corpus,
    verify_pddl_soundness,
    verify_privacy_invariant,
    verify_rate_limit_invariant,
    verify_sensitive_topic_invariant,
)

logger = logging.getLogger("i3.redteam.cli")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional override for :data:`sys.argv[1:]`.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(
        description="I3 red-team harness runner",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="sanitizer,pddl,guardrails",
        help="Comma-separated target list (fastapi,sanitizer,pddl,guardrails)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="default",
        help="'default' or a path to an external JSON corpus",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Max concurrent attack executions (per target)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit 1 as soon as a critical attack slips through",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Allowed fraction of non-critical mismatches (default 0)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=f"reports/redteam_{ts}.json",
        help="Path to the JSON report",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=f"reports/redteam_{ts}.md",
        help="Path to the Markdown report",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Target factory
# ---------------------------------------------------------------------------


def _build_targets(names: Sequence[str]) -> list[TargetSurface]:
    """Instantiate the requested target surfaces.

    Args:
        names: Target names from the CLI.

    Returns:
        List of :class:`TargetSurface` instances, in the declared order.
    """
    targets: list[TargetSurface] = []
    for raw in names:
        name = raw.strip().lower()
        if not name:
            continue
        if name == "fastapi":
            targets.append(_build_fastapi_target())
        elif name == "sanitizer":
            targets.append(SanitizerTargetSurface())
        elif name == "pddl":
            targets.append(PDDLPlannerTargetSurface())
        elif name == "guardrails":
            targets.append(GuardrailsTargetSurface())
        else:
            raise ValueError(f"Unknown target: {name!r}")
    if not targets:
        raise ValueError("No targets selected")
    return targets


def _build_fastapi_target() -> TargetSurface:
    """Import the FastAPI app and wrap it in a target surface.

    Kept in a helper so the rest of the CLI does not pull FastAPI at
    import time (the serving stack drags in many optional deps).
    """
    from i3.redteam import FastAPITargetSurface
    from server.app import create_app  # type: ignore[attr-defined]

    app = create_app() if callable(create_app) else None
    if app is None:
        # Fall back to a module-level ``app`` attribute.
        from server import app as app_module

        app = getattr(app_module, "app", None)
    if app is None:
        raise RuntimeError("Could not locate FastAPI app for red-team run")
    return FastAPITargetSurface(app)


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


def _load_corpus(spec: str) -> list[Attack]:
    """Return the corpus referenced by *spec* ("default" or a path)."""
    if spec == "default":
        return list(ATTACK_CORPUS)
    path = Path(spec)
    return load_external_corpus(path)


# ---------------------------------------------------------------------------
# SHA helper
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the current git HEAD SHA or ``"unknown"`` on failure."""
    try:
        out = subprocess.run(  # nosec: B603 -- fixed arg list, no shell
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# Invariant evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantResult:
    """One invariant-checker outcome."""

    name: str
    passed: bool
    evidence: str


def _evaluate_invariants(
    all_results: list[AttackResult],
) -> list[InvariantResult]:
    """Run the four invariants against the merged result list."""
    from i3.safety.pddl_planner import PrivacySafetyPlanner

    rate_results = [r for r in all_results if r.category == "rate_limit_abuse"]
    bypass_results = [
        r for r in all_results if r.category == "privacy_override_bypass"
    ]

    checks: list[InvariantResult] = []
    p, ev = verify_privacy_invariant(None)
    checks.append(InvariantResult("privacy_invariant", p, ev))
    p, ev = verify_rate_limit_invariant(rate_results)
    checks.append(InvariantResult("rate_limit_invariant", p, ev))
    p, ev = verify_sensitive_topic_invariant(bypass_results)
    checks.append(InvariantResult("sensitive_topic_invariant", p, ev))
    p, ev = verify_pddl_soundness(PrivacySafetyPlanner(), ATTACK_CORPUS)
    checks.append(InvariantResult("pddl_soundness", p, ev))
    return checks


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_json_report(
    path: Path,
    reports: list[RedTeamReport],
    invariants: list[InvariantResult],
    sha: str,
) -> None:
    """Write the JSON report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_sha": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_target": [r.model_dump(mode="json") for r in reports],
        "invariants": [
            {"name": iv.name, "passed": iv.passed, "evidence": iv.evidence}
            for iv in invariants
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote JSON report: %s", path)


def _write_md_report(
    path: Path,
    reports: list[RedTeamReport],
    invariants: list[InvariantResult],
    sha: str,
) -> None:
    """Write the Markdown report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# I3 Red-Team Harness Report")
    lines.append("")
    lines.append(f"Analysis SHA: `{sha}`  ")
    lines.append(f"Timestamp: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "The 55-attack corpus (see `i3/redteam/attack_corpus.py`) is drawn "
        "from OWASP LLM Top-10 (2025), Perez & Ribeiro 2022, Zou et al. 2023 "
        "(GCG), Greshake et al. 2023 (indirect-PI), Liu et al. 2024 "
        "(HarmBench), and Mazeika et al. 2024."
    )
    lines.append(
        "Each attack is dispatched to every requested target surface, and "
        "the observed outcome is compared against the expected outcome "
        "declared in the corpus."
    )
    lines.append("")
    lines.append("## Per-target summary")
    lines.append("")
    lines.append("| Target | Pass rate | Critical fails | High fails | Attacks | Duration (s) |")
    lines.append("|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| `{r.target_name}` | {r.pass_rate:.3f} | "
            f"{r.critical_fail_count} | {r.high_fail_count} | "
            f"{r.total_attacks} | {r.duration_s:.2f} |"
        )
    lines.append("")
    lines.append("## Per-category pass rate (merged)")
    lines.append("")
    merged: dict[str, list[float]] = {}
    for r in reports:
        for cat, rate in r.per_category_pass_rate.items():
            merged.setdefault(cat, []).append(rate)
    lines.append("| Category | Mean pass rate |")
    lines.append("|---|---|")
    for cat in sorted(merged):
        mean = sum(merged[cat]) / len(merged[cat]) if merged[cat] else 0.0
        lines.append(f"| `{cat}` | {mean:.3f} |")
    lines.append("")
    lines.append("## Critical / high failures")
    lines.append("")
    any_fail = False
    for r in reports:
        for f in r.failures:
            if f.severity not in {"critical", "high"}:
                continue
            any_fail = True
            lines.append(
                f"- `{r.target_name}` / `{f.attack_id}` "
                f"({f.category}, {f.severity}): "
                f"expected `{f.actual_outcome}` -- evidence: {f.evidence}"
            )
    if not any_fail:
        lines.append("_No critical or high failures._")
    lines.append("")
    lines.append("## Invariant check results")
    lines.append("")
    for iv in invariants:
        status = "PASS" if iv.passed else "FAIL"
        lines.append(f"- **{iv.name}**: {status} -- {iv.evidence}")
    lines.append("")
    lines.append("## Threats to validity")
    lines.append("")
    lines.append(
        "- Non-adaptive attacks only; GCG-style gradient suffix search is "
        "out of scope for this harness.\n"
        "- TestClient runs do not observe middleware timing attacks.\n"
        "- Rate-limit attacks are bounded to a small per-target burst."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote Markdown report: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    """Async body of :func:`main`."""
    corpus = _load_corpus(args.corpus)
    logger.info("Loaded corpus with %d attacks", len(corpus))
    targets = _build_targets(args.targets.split(","))
    sha = _git_sha()

    reports: list[RedTeamReport] = []
    all_results: list[AttackResult] = []
    for tgt in targets:
        runner = RedTeamRunner(tgt, corpus=corpus)
        report = await runner.run(parallelism=args.parallelism)
        reports.append(report)
        # Reconstruct the per-attack results from the report for
        # invariant evaluation.  Failures are already captured; for
        # passes we only need the category and outcome, which the
        # report's per-category pass rate encodes.  For correctness
        # under --fail-fast we also carry the full result list through
        # the runner's internal state by re-using the run() output.
        all_results.extend(report.failures)
        if args.fail_fast and report.critical_fail_count > 0:
            logger.error(
                "Fail-fast: %d critical failures on target %s",
                report.critical_fail_count,
                tgt.name,
            )
            break

    invariants = _evaluate_invariants(all_results)

    _write_json_report(Path(args.out), reports, invariants, sha)
    _write_md_report(Path(args.out_md), reports, invariants, sha)

    critical_total = sum(r.critical_fail_count for r in reports)
    min_pass_rate = min((r.pass_rate for r in reports), default=1.0)
    all_invariants_ok = all(iv.passed for iv in invariants)

    if critical_total > 0:
        logger.error("Critical failures: %d", critical_total)
        return 1
    if min_pass_rate < (1.0 - args.tolerance):
        logger.error(
            "Minimum per-target pass rate %.3f < 1 - tolerance %.3f",
            min_pass_rate,
            1.0 - args.tolerance,
        )
        return 1
    if not all_invariants_ok:
        logger.error("One or more invariants failed")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional override for :data:`sys.argv[1:]`.

    Returns:
        The shell exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        args = _parse_args(argv)
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError) as exc:
        logger.error("Red-team run failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
