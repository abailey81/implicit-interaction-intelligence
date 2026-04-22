"""CLI-usable scoring for ImplicitAdaptBench.

The scoring module loads a submission JSONL and a gold record JSONL, runs
the metric bundle from :mod:`benchmarks.implicit_adapt_bench.metrics`, and
returns a :class:`BenchmarkScore`.

Example::

    from pathlib import Path
    from benchmarks.implicit_adapt_bench.scoring import score_submission

    score = score_submission(
        submission_path=Path("submissions/my_method.jsonl"),
        records_path=Path("benchmarks/implicit_adapt_bench/data/dev.jsonl"),
    )
    print(score.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from benchmarks.implicit_adapt_bench.data_generator import read_benchmark_jsonl
from benchmarks.implicit_adapt_bench.data_schema import (
    BenchmarkRecord,
    BenchmarkScore,
    BenchmarkSubmission,
)
from benchmarks.implicit_adapt_bench.metrics import (
    DEFAULT_METRIC_WEIGHTS,
    aggregate_score,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)


def read_submissions_jsonl(path: Path) -> list[BenchmarkSubmission]:
    """Read a list of :class:`BenchmarkSubmission` objects from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of validated submissions.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or does not conform to the
            :class:`BenchmarkSubmission` schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    submissions: list[BenchmarkSubmission] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {i + 1} of {path}: {exc}"
                ) from exc
            submissions.append(BenchmarkSubmission.model_validate(payload))
    return submissions


def _pick_method_name(submissions: Sequence[BenchmarkSubmission]) -> str:
    """Return the most common ``method_name`` (or ``'unknown'`` if empty).

    The scorer tolerates multiple method names in a single file but flags the
    mix via a notes string.

    Args:
        submissions: Candidate submissions.

    Returns:
        A method name string.
    """
    if not submissions:
        return "unknown"
    counter = Counter(s.method_name for s in submissions)
    method, _ = counter.most_common(1)[0]
    return method


def score_submission(
    submission_path: Path,
    records_path: Path,
    *,
    weights: dict[str, float] | None = None,
    budget_p95_ms: float = 200.0,
) -> BenchmarkScore:
    """Score a submission JSONL against a gold record JSONL.

    Args:
        submission_path: Path to the candidate's submission JSONL.
        records_path: Path to the gold record JSONL.
        weights: Optional custom metric weights; defaults to
            :data:`DEFAULT_METRIC_WEIGHTS`.
        budget_p95_ms: Latency budget for
            :func:`runtime_budget_compliance`.

    Returns:
        A :class:`BenchmarkScore` with per-metric and aggregate scalars.
    """
    submissions = read_submissions_jsonl(submission_path)
    records = read_benchmark_jsonl(records_path)

    # -- Coverage sanity check --------------------------------------------
    record_ids: set[str] = {r.record_id for r in records}
    submission_ids: set[str] = {s.record_id for s in submissions}
    missing = sorted(record_ids - submission_ids)
    extra = sorted(submission_ids - record_ids)
    notes_parts: list[str] = []
    if missing:
        notes_parts.append(f"{len(missing)} record_ids missing from submission.")
    if extra:
        notes_parts.append(f"{len(extra)} submission record_ids not in gold.")

    method_names = {s.method_name for s in submissions}
    if len(method_names) > 1:
        notes_parts.append(
            f"Submission contains multiple method_name values: {sorted(method_names)}."
        )

    per_metric = compute_all_metrics(
        submissions, records, budget_p95_ms=budget_p95_ms
    )
    agg = aggregate_score(per_metric, weights)

    return BenchmarkScore(
        method_name=_pick_method_name(submissions),
        n_records=len(submissions),
        per_metric=per_metric,
        aggregate=agg,
        notes=" ".join(notes_parts),
    )


def score_submissions_in_memory(
    submissions: Sequence[BenchmarkSubmission],
    records: Sequence[BenchmarkRecord],
    *,
    weights: dict[str, float] | None = None,
    budget_p95_ms: float = 200.0,
) -> BenchmarkScore:
    """Score an in-memory set of submissions against in-memory records.

    This avoids the round-trip through JSONL for programmatic callers (tests,
    CLI pipelines).

    Args:
        submissions: Candidate submissions.
        records: Gold records.
        weights: Optional custom metric weights.
        budget_p95_ms: Latency budget for the runtime metric.

    Returns:
        A :class:`BenchmarkScore`.
    """
    per_metric = compute_all_metrics(
        submissions, records, budget_p95_ms=budget_p95_ms
    )
    agg = aggregate_score(per_metric, weights)
    return BenchmarkScore(
        method_name=_pick_method_name(submissions),
        n_records=len(submissions),
        per_metric=per_metric,
        aggregate=agg,
        notes="",
    )


def score_to_json(score: BenchmarkScore) -> str:
    """Serialise a :class:`BenchmarkScore` to pretty JSON.

    Args:
        score: The score object.

    Returns:
        A UTF-8 JSON string (2-space indent).
    """
    return score.model_dump_json(indent=2)


def write_score_json(score: BenchmarkScore, path: Path) -> None:
    """Write a :class:`BenchmarkScore` to ``path`` as pretty JSON.

    Args:
        score: The score object.
        path: Output path (parents created if missing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(score_to_json(score), encoding="utf-8")


__all__ = [
    "DEFAULT_METRIC_WEIGHTS",
    "read_submissions_jsonl",
    "score_submission",
    "score_submissions_in_memory",
    "score_to_json",
    "write_score_json",
]
