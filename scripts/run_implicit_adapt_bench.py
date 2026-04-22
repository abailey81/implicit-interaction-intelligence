"""CLI entry point for ImplicitAdaptBench.

Three operating modes:

* ``--generate-data`` — deterministically write the four synthetic splits
  (``train``, ``dev``, ``test``, ``held_out_human``) into
  ``benchmarks/implicit_adapt_bench/data/``.
* ``--run-baselines`` — run ``baseline_none``, ``baseline_prompt``, and
  ``baseline_cross_attention`` against the dev split; write their
  submissions to ``reports/implicit_adapt_bench_<ts>/<method>.jsonl`` and
  score them.
* ``--score PATH`` — score an external submission file against the dev
  split (or the split selected via ``--records-path``).

Every mode writes a Markdown summary to
``reports/implicit_adapt_bench_<ts>.md``.

Usage::

    python scripts/run_implicit_adapt_bench.py --generate-data
    python scripts/run_implicit_adapt_bench.py --run-baselines
    python scripts/run_implicit_adapt_bench.py --score my_submission.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running the script directly without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.implicit_adapt_bench.baselines import (  # noqa: E402
    run_baseline_cross_attention,
    run_baseline_none,
    run_baseline_prompt,
)
from benchmarks.implicit_adapt_bench.data_generator import (  # noqa: E402
    generate_synthetic_split,
    read_benchmark_jsonl,
    write_benchmark_jsonl,
)
from benchmarks.implicit_adapt_bench.data_schema import (  # noqa: E402
    BenchmarkRecord,
    BenchmarkScore,
    BenchmarkSubmission,
)
from benchmarks.implicit_adapt_bench.scoring import (  # noqa: E402
    score_submission,
    score_submissions_in_memory,
    write_score_json,
)

logger = logging.getLogger("run_implicit_adapt_bench")


DATA_DIR: Path = _ROOT / "benchmarks" / "implicit_adapt_bench" / "data"
REPORTS_DIR: Path = _ROOT / "reports"


def _now_ts() -> str:
    """Return a UTC ISO-8601 basic timestamp like ``20260422T120000Z``."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Run ImplicitAdaptBench: generate data, run baselines, or score.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate the synthetic train/dev/test/held_out_human splits.",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run the three reference baselines against the dev split.",
    )
    parser.add_argument(
        "--score",
        type=str,
        default=None,
        help="Path to a submission JSONL to score. Uses --records-path as gold.",
    )
    parser.add_argument(
        "--records-path",
        type=str,
        default=str(DATA_DIR / "dev.jsonl"),
        help="Path to the gold records JSONL (default: dev split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for all deterministic operations (default: 42).",
    )
    parser.add_argument(
        "--n-per-archetype",
        type=int,
        default=4,
        help="Records per archetype for each synthetic split (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string for the baselines (default: cpu).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Generation budget per record (default: 32).",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Optional path for the Markdown summary. Defaults to "
        "reports/implicit_adapt_bench_<ts>.md.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mode: generate data
# ---------------------------------------------------------------------------


def _cmd_generate_data(
    *, n_per_archetype: int, seed: int
) -> dict[str, Path]:
    """Write the four synthetic splits to ``DATA_DIR``.

    Args:
        n_per_archetype: Records per archetype per split.
        seed: Base seed.

    Returns:
        Dict of ``split_name -> output_path``.
    """
    out: dict[str, Path] = {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test", "held_out_human"):
        records = generate_synthetic_split(
            n_records_per_archetype=n_per_archetype,
            split=split,  # type: ignore[arg-type]
            seed=seed,
        )
        path = DATA_DIR / f"{split}.jsonl"
        write_benchmark_jsonl(records, path)
        logger.info("Wrote %d records to %s", len(records), path)
        out[split] = path
    return out


# ---------------------------------------------------------------------------
# Mode: run baselines
# ---------------------------------------------------------------------------


def _write_submissions_jsonl(
    submissions: list[BenchmarkSubmission], path: Path
) -> None:
    """Write a list of submissions to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for sub in submissions:
            fh.write(sub.model_dump_json())
            fh.write("\n")


def _cmd_run_baselines(
    *,
    records_path: Path,
    device: str,
    seed: int,
    max_new_tokens: int,
    run_dir: Path,
) -> dict[str, BenchmarkScore]:
    """Run the three baselines against ``records_path`` and score each.

    Args:
        records_path: Path to the gold records JSONL.
        device: Torch device string.
        seed: Torch seed.
        max_new_tokens: Generation budget per record.
        run_dir: Directory to write the per-baseline submission JSONLs into.

    Returns:
        Dict mapping method name to :class:`BenchmarkScore`.
    """
    records: list[BenchmarkRecord] = read_benchmark_jsonl(records_path)
    scores: dict[str, BenchmarkScore] = {}

    for method_name, runner in (
        ("baseline_none", run_baseline_none),
        ("baseline_prompt", run_baseline_prompt),
        ("baseline_cross_attention", run_baseline_cross_attention),
    ):
        logger.info("Running %s over %d records ...", method_name, len(records))
        submissions = runner(
            records,
            device=device,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        sub_path = run_dir / f"{method_name}.jsonl"
        _write_submissions_jsonl(submissions, sub_path)
        score = score_submissions_in_memory(submissions, records)
        scores[method_name] = score
        write_score_json(score, run_dir / f"{method_name}.score.json")
        logger.info(
            "  %s: aggregate=%.3f, style_match=%.3f",
            method_name,
            score.aggregate,
            score.per_metric.get("style_match", 0.0),
        )

    return scores


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _render_markdown(
    *,
    mode: str,
    timestamp: str,
    generated: dict[str, Path] | None,
    scores: dict[str, BenchmarkScore] | None,
    single_score: BenchmarkScore | None,
    extra: dict[str, str] | None = None,
) -> str:
    """Render a human-readable Markdown summary.

    Args:
        mode: The command-line mode ("generate-data", "run-baselines",
            "score").
        timestamp: A UTC timestamp string.
        generated: Optional dict of generated split -> path.
        scores: Optional dict of method -> score.
        single_score: Optional single score (for ``--score`` mode).
        extra: Optional freeform key-value pairs to render as a section.

    Returns:
        The Markdown string.
    """
    lines: list[str] = []
    lines.append("# ImplicitAdaptBench run")
    lines.append("")
    lines.append(f"- timestamp (UTC): `{timestamp}`")
    lines.append(f"- mode: `{mode}`")
    lines.append("")

    if generated:
        lines.append("## Generated splits")
        lines.append("")
        lines.append("| split | path |")
        lines.append("|-------|------|")
        for name, p in generated.items():
            lines.append(f"| `{name}` | `{p.as_posix()}` |")
        lines.append("")

    if scores:
        lines.append("## Baseline scores")
        lines.append("")
        metric_names = sorted(next(iter(scores.values())).per_metric.keys())
        header = "| method | aggregate | " + " | ".join(metric_names) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(metric_names) + 2))
        for method, sc in scores.items():
            row_cells = [f"{sc.per_metric.get(m, 0.0):.3f}" for m in metric_names]
            lines.append(
                f"| `{method}` | {sc.aggregate:.3f} | " + " | ".join(row_cells) + " |"
            )
        lines.append("")

    if single_score is not None:
        lines.append("## Submission score")
        lines.append("")
        lines.append(f"- method: `{single_score.method_name}`")
        lines.append(f"- n_records: {single_score.n_records}")
        lines.append(f"- aggregate: `{single_score.aggregate:.3f}`")
        if single_score.notes:
            lines.append(f"- notes: {single_score.notes}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|--------|-------|")
        for name, val in sorted(single_score.per_metric.items()):
            lines.append(f"| `{name}` | {val:.3f} |")
        lines.append("")

    if extra:
        lines.append("## Notes")
        lines.append("")
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Report produced by `scripts/run_implicit_adapt_bench.py`. "
        "See `benchmarks/implicit_adapt_bench/README.md` for task "
        "definition and `docs/research/implicit_adapt_bench.md` for "
        "the full paper-style spec."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point — returns a process exit code.

    Returns:
        ``0`` on success, ``2`` on CLI misuse.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()

    if not (args.generate_data or args.run_baselines or args.score):
        logger.error(
            "at least one of --generate-data, --run-baselines, --score PATH is required."
        )
        return 2

    ts = _now_ts()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = Path(args.out_md) if args.out_md else (
        REPORTS_DIR / f"implicit_adapt_bench_{ts}.md"
    )

    generated: dict[str, Path] | None = None
    scores: dict[str, BenchmarkScore] | None = None
    single_score: BenchmarkScore | None = None
    mode_parts: list[str] = []

    if args.generate_data:
        mode_parts.append("generate-data")
        generated = _cmd_generate_data(
            n_per_archetype=args.n_per_archetype, seed=args.seed
        )

    if args.run_baselines:
        mode_parts.append("run-baselines")
        run_dir = REPORTS_DIR / f"implicit_adapt_bench_{ts}"
        # If the dev split doesn't exist yet, generate it on the fly.
        dev_path = Path(args.records_path)
        if not dev_path.exists():
            logger.info("dev split not found at %s; generating now.", dev_path)
            generated = _cmd_generate_data(
                n_per_archetype=args.n_per_archetype, seed=args.seed
            )
            dev_path = DATA_DIR / "dev.jsonl"
        scores = _cmd_run_baselines(
            records_path=dev_path,
            device=args.device,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            run_dir=run_dir,
        )

    if args.score is not None:
        mode_parts.append("score")
        single_score = score_submission(
            submission_path=Path(args.score),
            records_path=Path(args.records_path),
        )
        # Echo as JSON for pipe-friendly use.
        sys.stdout.write(json.dumps(single_score.model_dump(), indent=2))
        sys.stdout.write("\n")

    markdown = _render_markdown(
        mode="+".join(mode_parts),
        timestamp=ts,
        generated=generated,
        scores=scores,
        single_score=single_score,
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote Markdown report to %s", md_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
