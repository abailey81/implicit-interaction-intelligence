#!/usr/bin/env python3
"""Generate an I³ per-archetype fairness report.

Reads the interaction diary via :class:`~i3.diary.store.DiaryStore`,
computes the per-archetype adaptation-bias report (see
:mod:`i3.fairness.subgroup_metrics`), and writes a Markdown document under
``reports/fairness_report_<YYYY-MM-DD>.md`` with per-archetype tables,
confidence intervals, disparity flags, and plain-English interpretation.

Usage::

    python scripts/fairness_report.py --user-id demo_user
    python scripts/fairness_report.py --user-id demo_user --db data/diary.db --threshold 0.2
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import sys
from pathlib import Path

from i3.diary.store import DiaryStore
from i3.fairness.subgroup_metrics import (
    EPP_VIZER_ZIMMERMANN_ARCHETYPES,
    FairnessReport,
    compute_per_archetype_adaptation_bias,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fairness_report")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def render_markdown(report: FairnessReport, user_id: str) -> str:
    """Render a :class:`FairnessReport` as a standalone Markdown document."""
    today = datetime.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# I³ Fairness Report — {today}")
    lines.append("")
    lines.append(f"**User:** `{user_id}`  ")
    lines.append(f"**Total exchanges analysed:** {report.total_exchanges}  ")
    lines.append(f"**Disparity threshold:** {report.threshold:.3f}")
    lines.append("")

    # Plain-English headline
    if report.flagged_dimensions:
        dims = ", ".join(f"`{d}`" for d in report.flagged_dimensions)
        lines.append(
            f"> **Headline.** {len(report.flagged_dimensions)} adaptation "
            f"dimension(s) exceed the disparity threshold "
            f"of {report.threshold:.2f}: {dims}.  "
            "These dimensions should be audited for systematic bias across "
            "archetypes before the adaptation controller is deployed."
        )
    else:
        lines.append(
            "> **Headline.** No adaptation dimension exceeds the disparity "
            "threshold.  Subject to sample-size caveats in the per-archetype "
            "table, the adaptation controller appears to behave equitably "
            "across the 8 Epp/Vizer/Zimmermann archetypes."
        )
    lines.append("")

    lines.append("## Per-archetype mean adaptation (95% bootstrap CI)")
    lines.append("")
    lines.append(
        "| Archetype | n | cognitive_load | style_formality | style_verbosity | style_emotionality | style_directness | emotional_tone | accessibility |"
    )
    lines.append(
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    )
    dims = [
        "cognitive_load",
        "style_formality",
        "style_verbosity",
        "style_emotionality",
        "style_directness",
        "emotional_tone",
        "accessibility",
    ]
    for am in report.per_archetype:
        if am.n_exchanges == 0:
            cells = ["—"] * len(dims)
        else:
            cells = [
                f"{am.mean_adaptation[d]:.2f} ({am.ci_lower[d]:.2f}-{am.ci_upper[d]:.2f})"
                for d in dims
            ]
        lines.append(f"| `{am.archetype}` | {am.n_exchanges} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Disparity across archetypes")
    lines.append("")
    lines.append("| Dimension | max − min | Flagged |")
    lines.append("|:---|:---:|:---:|")
    for dim in dims:
        d = report.disparity.get(dim, 0.0)
        flag = "**yes**" if dim in report.flagged_dimensions else "no"
        lines.append(f"| `{dim}` | {d:.3f} | {flag} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- *cognitive_load* disparity of 0.1–0.15 is typically expected "
        "because the TCN encoder legitimately responds to typing speed "
        "differences between archetypes.  Disparities above 0.15 deserve a "
        "closer look — the adaptation controller may be compounding small "
        "encoder biases into larger behavioural differences."
    )
    lines.append(
        "- *accessibility* disparity above the threshold is the single "
        "clearest fairness risk.  The accessibility adapter should trigger "
        "most strongly for the `motor_difficulty` archetype; if it also "
        "fires disproportionately for, say, `distracted_multitasking`, the "
        "adapter is mistaking distraction for motor difficulty."
    )
    lines.append(
        "- Confidence intervals are 95% bootstrap (Efron 1979) over 2,000 "
        "resamples. Interpret CIs wider than ±0.1 with caution; they "
        "usually indicate too few exchanges for that archetype."
    )
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append(
        "- Efron, B. (1979). *Bootstrap methods: another look at the "
        "jackknife.* Annals of Statistics 7(1)."
    )
    lines.append(
        "- Epp, C., Lippold, M., Mandryk, R. L. (2011). *Identifying "
        "emotional states using keystroke dynamics.* CHI."
    )
    lines.append(
        "- Barocas, S., Hardt, M., Narayanan, A. (2019). *Fairness and "
        "Machine Learning.* fairmlbook.org."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main_async(args: argparse.Namespace) -> int:
    store = DiaryStore(db_path=args.db)
    await store.initialize()

    report = await compute_per_archetype_adaptation_bias(
        diary=store,
        archetypes=list(EPP_VIZER_ZIMMERMANN_ARCHETYPES),
        user_id=args.user_id,
        disparity_threshold=args.threshold,
        num_resamples=args.num_resamples,
        seed=args.seed,
        max_sessions=args.max_sessions,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    out_path = out_dir / f"fairness_report_{today}.md"
    out_path.write_text(render_markdown(report, args.user_id), encoding="utf-8")
    logger.info("Wrote fairness report to %s", out_path)

    if report.flagged_dimensions and args.fail_on_flag:
        logger.error(
            "Fairness flags raised on dimensions: %s",
            ", ".join(report.flagged_dimensions),
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="I³ fairness report generator")
    parser.add_argument("--user-id", required=True, help="User id to audit.")
    parser.add_argument("--db", default="data/diary.db", help="SQLite diary path.")
    parser.add_argument("--out-dir", default="reports", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Disparity threshold.")
    parser.add_argument("--num-resamples", type=int, default=2000)
    parser.add_argument("--max-sessions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--fail-on-flag",
        action="store_true",
        help="Exit non-zero if any dimension is flagged — for CI use.",
    )
    args = parser.parse_args()

    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
