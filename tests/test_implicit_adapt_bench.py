"""Unit tests for ImplicitAdaptBench (Batch C).

Covers:

* Pydantic validation on malformed records / submissions.
* Metric math correctness (known-answer tests).
* Aggregate-score monotonicity under metric improvements.
* Data-generator reproducibility under a fixed seed.
* Baseline runnability on a random-init :class:`AdaptiveSLM`.

The tests are intentionally fast — the baseline runnability tests cap
generation at a handful of tokens so each test finishes in seconds on a
laptop CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.implicit_adapt_bench.data_generator import (
    generate_synthetic_split,
    read_benchmark_jsonl,
    write_benchmark_jsonl,
)
from benchmarks.implicit_adapt_bench.data_schema import (
    BehaviouralWindow,
    BenchmarkRecord,
    BenchmarkSubmission,
)
from benchmarks.implicit_adapt_bench.metrics import (
    DEFAULT_METRIC_WEIGHTS,
    accessibility_mode_appropriateness,
    aggregate_score,
    cognitive_load_fidelity,
    compute_all_metrics,
    preference_rate,
    runtime_budget_compliance,
    style_match_score,
)
from benchmarks.implicit_adapt_bench.scoring import score_submissions_in_memory


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------


def _make_minimal_record(**overrides: object) -> BenchmarkRecord:
    """Build a minimal valid BenchmarkRecord with optional overrides."""
    data: dict[str, object] = {
        "record_id": "dev-0001",
        "behavioural_window": BehaviouralWindow(
            feature_vector=[0.0] * 32,
            raw_keystroke_intervals_ms=[100.0, 120.0],
            baseline_deviation_metrics={"iki_deviation": 0.1},
        ),
        "prompt": "Hello",
        "target_archetype": "neutral",
        "target_adaptation_vector": [0.5] * 8,
        "reference_style_label": "neutral_neutral_medium",
        "reference_length_bucket": "medium",
        "reference_formality_bucket": "neutral",
    }
    data.update(overrides)
    return BenchmarkRecord.model_validate(data)


def test_behavioural_window_rejects_wrong_length() -> None:
    """A feature_vector with != 32 elements must raise."""
    with pytest.raises(ValidationError):
        BehaviouralWindow(feature_vector=[0.0] * 31)
    with pytest.raises(ValidationError):
        BehaviouralWindow(feature_vector=[0.0] * 33)


def test_behavioural_window_rejects_negative_iki() -> None:
    """Negative keystroke intervals must raise."""
    with pytest.raises(ValidationError):
        BehaviouralWindow(
            feature_vector=[0.0] * 32,
            raw_keystroke_intervals_ms=[-1.0, 100.0],
        )


def test_benchmark_record_rejects_wrong_adaptation_vector_length() -> None:
    """target_adaptation_vector must have exactly 8 elements."""
    with pytest.raises(ValidationError):
        _make_minimal_record(target_adaptation_vector=[0.5] * 7)


def test_benchmark_record_rejects_out_of_range_adaptation_values() -> None:
    """An adaptation value well outside [0, 1] is rejected."""
    bad = [0.5] * 8
    bad[3] = 2.0  # clearly outside the 1.1 slack
    with pytest.raises(ValidationError):
        _make_minimal_record(target_adaptation_vector=bad)


def test_benchmark_record_human_preference_range() -> None:
    """human_preference_score must be in [0, 1] when provided."""
    with pytest.raises(ValidationError):
        _make_minimal_record(human_preference_score=1.5)


def test_submission_p95_must_be_at_least_p50() -> None:
    """runtime_ms_p95 < runtime_ms_p50 is rejected."""
    with pytest.raises(ValidationError):
        BenchmarkSubmission(
            record_id="dev-0001",
            generated_text="ok",
            method_name="m",
            runtime_ms_p50=100.0,
            runtime_ms_p95=50.0,
        )


# ---------------------------------------------------------------------------
# Metric math — known-answer tests
# ---------------------------------------------------------------------------


def test_runtime_budget_compliance_perfect_and_none() -> None:
    """All under-budget -> 1.0; all over-budget -> 0.0."""
    passing = [
        BenchmarkSubmission(
            record_id=f"r{i}",
            generated_text="x",
            method_name="m",
            runtime_ms_p50=10.0,
            runtime_ms_p95=50.0,
        )
        for i in range(5)
    ]
    assert runtime_budget_compliance(passing, budget_p95_ms=200.0) == 1.0

    failing = [
        BenchmarkSubmission(
            record_id=f"r{i}",
            generated_text="x",
            method_name="m",
            runtime_ms_p50=10.0,
            runtime_ms_p95=500.0,
        )
        for i in range(3)
    ]
    assert runtime_budget_compliance(failing, budget_p95_ms=200.0) == 0.0


def test_runtime_budget_compliance_empty_is_vacuously_true() -> None:
    """Empty input returns 1.0 (vacuously satisfied)."""
    assert runtime_budget_compliance([], budget_p95_ms=200.0) == 1.0


def test_runtime_budget_compliance_rejects_nonpositive_budget() -> None:
    """Budget must be > 0."""
    with pytest.raises(ValueError):
        runtime_budget_compliance([], budget_p95_ms=0.0)


def test_cognitive_load_fidelity_known_answers() -> None:
    """FK-grade in the target band -> 1.0."""
    # A very simple kindergarten-level sentence, target_load=0.0.
    simple = "The cat sat on the mat. The dog ran. The sun is hot."
    assert cognitive_load_fidelity(simple, 0.1) >= 0.5
    # A long technical sentence likely triggers a higher FK grade.
    tech = (
        "Gradient descent minimises an objective function by iteratively "
        "computing gradients, adjusting parameters, and evaluating "
        "convergence with respect to a chosen learning rate and schedule."
    )
    # We expect the technical sentence to do better on high target load
    # than the kindergarten one.
    assert cognitive_load_fidelity(tech, 0.9) >= cognitive_load_fidelity(
        simple, 0.9
    )


def test_cognitive_load_fidelity_rejects_out_of_range() -> None:
    """target_load far outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        cognitive_load_fidelity("hi", 2.0)


def test_accessibility_mode_appropriateness_when_mode_off() -> None:
    """target_accessibility <= 0.7 always scores 1.0."""
    assert accessibility_mode_appropriateness("anything", 0.2) == 1.0


def test_accessibility_mode_appropriateness_perfect_and_worst() -> None:
    """A targeted accessible response should outscore a long idiomatic one."""
    good = "Let me explain. This is simple. Can I continue?"
    bad = (
        "Gradient descent is a piece of cake, really, and you'll find that "
        "once you bite the bullet the whole optimisation business costs "
        "an arm and a leg to understand at first glance."
    )
    assert accessibility_mode_appropriateness(good, 0.9) > (
        accessibility_mode_appropriateness(bad, 0.9)
    )


def test_style_match_score_bounded() -> None:
    """Score must lie in [0, 1] on arbitrary inputs."""
    for text in ("", "yes.", "A formal response." * 50):
        for label in (
            "warm_casual_short",
            "objective_formal_long",
            "reserved_neutral_medium",
        ):
            s = style_match_score(text, label)
            assert 0.0 <= s <= 1.0


def test_preference_rate_only_counts_labelled_records() -> None:
    """Records with a human_preference_score contribute; others are skipped."""
    labelled = _make_minimal_record(
        record_id="held-1", human_preference_score=0.8
    )
    unlabelled = _make_minimal_record(record_id="dev-1")
    subs = [
        BenchmarkSubmission(
            record_id="held-1",
            generated_text="x",
            method_name="m",
            runtime_ms_p50=1.0,
            runtime_ms_p95=2.0,
        ),
        BenchmarkSubmission(
            record_id="dev-1",
            generated_text="x",
            method_name="m",
            runtime_ms_p50=1.0,
            runtime_ms_p95=2.0,
        ),
    ]
    assert preference_rate(subs, [labelled, unlabelled]) == pytest.approx(0.8)


def test_preference_rate_empty_when_no_overlap() -> None:
    """No matching ids -> 0.0."""
    labelled = _make_minimal_record(
        record_id="held-1", human_preference_score=0.8
    )
    subs = [
        BenchmarkSubmission(
            record_id="other",
            generated_text="x",
            method_name="m",
            runtime_ms_p50=1.0,
            runtime_ms_p95=2.0,
        ),
    ]
    assert preference_rate(subs, [labelled]) == 0.0


# ---------------------------------------------------------------------------
# Aggregate monotonicity
# ---------------------------------------------------------------------------


def test_aggregate_score_is_monotone_in_each_metric() -> None:
    """Increasing any single metric cannot decrease the aggregate."""
    base = {name: 0.5 for name in DEFAULT_METRIC_WEIGHTS}
    base_agg = aggregate_score(base)
    for name in DEFAULT_METRIC_WEIGHTS:
        improved = dict(base)
        improved[name] = 0.9
        assert aggregate_score(improved) >= base_agg


def test_aggregate_score_bounded_01() -> None:
    """Aggregate must lie in [0, 1]."""
    all_zero = {name: 0.0 for name in DEFAULT_METRIC_WEIGHTS}
    all_one = {name: 1.0 for name in DEFAULT_METRIC_WEIGHTS}
    assert aggregate_score(all_zero) == 0.0
    assert aggregate_score(all_one) == pytest.approx(1.0)


def test_aggregate_score_rejects_zero_weights() -> None:
    """All-zero weights raise ValueError."""
    with pytest.raises(ValueError):
        aggregate_score({"a": 0.5}, weights={"a": 0.0})


# ---------------------------------------------------------------------------
# Data generator reproducibility
# ---------------------------------------------------------------------------


def test_data_generator_reproducible_under_fixed_seed() -> None:
    """Two runs with the same seed produce identical records."""
    a = generate_synthetic_split(
        n_records_per_archetype=2, split="dev", seed=42
    )
    b = generate_synthetic_split(
        n_records_per_archetype=2, split="dev", seed=42
    )
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert x.model_dump() == y.model_dump()


def test_data_generator_differs_across_splits() -> None:
    """Different splits produced from the same base seed are not identical."""
    train = generate_synthetic_split(
        n_records_per_archetype=2, split="train", seed=42
    )
    dev = generate_synthetic_split(
        n_records_per_archetype=2, split="dev", seed=42
    )
    # Record ids encode the split prefix, so at minimum those differ.
    assert {r.record_id for r in train}.isdisjoint(
        {r.record_id for r in dev}
    )
    # And the behavioural windows should not collide.
    assert any(
        r1.behavioural_window.feature_vector
        != r2.behavioural_window.feature_vector
        for r1, r2 in zip(train, dev)
    )


def test_held_out_human_populates_preference() -> None:
    """The held_out_human split always has a human_preference_score."""
    records = generate_synthetic_split(
        n_records_per_archetype=1, split="held_out_human", seed=0
    )
    assert all(r.human_preference_score is not None for r in records)


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """write_benchmark_jsonl / read_benchmark_jsonl round-trip losslessly."""
    records = generate_synthetic_split(
        n_records_per_archetype=1, split="dev", seed=7
    )
    path = tmp_path / "dev.jsonl"
    write_benchmark_jsonl(records, path)
    loaded = read_benchmark_jsonl(path)
    assert [r.model_dump() for r in records] == [r.model_dump() for r in loaded]


def test_generator_rejects_zero_records() -> None:
    """n_records_per_archetype <= 0 raises."""
    with pytest.raises(ValueError):
        generate_synthetic_split(
            n_records_per_archetype=0, split="dev", seed=1
        )


# ---------------------------------------------------------------------------
# Baseline runnability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "baseline_name",
    ["baseline_none", "baseline_prompt", "baseline_cross_attention"],
)
def test_baseline_runs_on_random_init(baseline_name: str) -> None:
    """Each baseline returns exactly one submission per record.

    The test pulls in torch + the SLM lazily; we use a very small split and
    a short generation budget so the test finishes quickly.
    """
    torch = pytest.importorskip("torch")
    _ = torch  # imported to signal availability; baselines will use it internally.

    # Import lazily so the rest of the test module doesn't depend on torch.
    from benchmarks.implicit_adapt_bench.baselines import (
        run_baseline_cross_attention,
        run_baseline_none,
        run_baseline_prompt,
    )

    runners = {
        "baseline_none": run_baseline_none,
        "baseline_prompt": run_baseline_prompt,
        "baseline_cross_attention": run_baseline_cross_attention,
    }
    records = generate_synthetic_split(
        n_records_per_archetype=1, split="dev", seed=1
    )[:2]
    submissions = runners[baseline_name](
        records, device="cpu", seed=0, max_new_tokens=4
    )
    assert len(submissions) == len(records)
    for sub in submissions:
        assert sub.method_name == baseline_name
        assert sub.runtime_ms_p95 >= sub.runtime_ms_p50 >= 0.0
        assert isinstance(sub.generated_text, str)


# ---------------------------------------------------------------------------
# End-to-end scoring smoke
# ---------------------------------------------------------------------------


def test_compute_all_metrics_returns_canonical_keys() -> None:
    """compute_all_metrics returns exactly the default-weighted metric names."""
    records = generate_synthetic_split(
        n_records_per_archetype=1, split="dev", seed=3
    )[:2]
    subs = [
        BenchmarkSubmission(
            record_id=r.record_id,
            generated_text="A short answer.",
            method_name="dummy",
            runtime_ms_p50=10.0,
            runtime_ms_p95=30.0,
        )
        for r in records
    ]
    metrics = compute_all_metrics(subs, records)
    assert set(metrics.keys()) == set(DEFAULT_METRIC_WEIGHTS.keys())


def test_score_submissions_in_memory_smoke() -> None:
    """End-to-end programmatic scoring returns a valid BenchmarkScore."""
    records = generate_synthetic_split(
        n_records_per_archetype=1, split="dev", seed=3
    )[:2]
    subs = [
        BenchmarkSubmission(
            record_id=r.record_id,
            generated_text="A short answer.",
            method_name="dummy",
            runtime_ms_p50=10.0,
            runtime_ms_p95=30.0,
        )
        for r in records
    ]
    score = score_submissions_in_memory(subs, records)
    assert score.method_name == "dummy"
    assert 0.0 <= score.aggregate <= 1.0
    assert score.n_records == len(subs)
