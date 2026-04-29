"""Tests for ``i3.interaction.features.BaselineTracker``.

Covers:

* Welford's online algorithm correctness (mean / variance match
  numpy reference).
* Bessel-corrected sample variance (n-1 denominator) — iter 2 fix.
* Warm-up gate: deviation returns 0.0 until ``warmup`` observations.
* Extreme-value clamping of z-scores to [-1, 1].
* Reset clears all state.
* Defensive behaviour with degenerate inputs (constant feature ⇒ std=0).
"""

from __future__ import annotations

import math
import statistics

import pytest

from i3.interaction.features import BaselineTracker
from i3.interaction.types import InteractionFeatureVector


def _fv(**overrides: float) -> InteractionFeatureVector:
    """Build an InteractionFeatureVector with all-zero defaults + overrides."""
    base: dict[str, float] = {
        # keystroke
        "mean_iki": 0.0,
        "std_iki": 0.0,
        "mean_burst_length": 0.0,
        "mean_pause_duration": 0.0,
        "backspace_ratio": 0.0,
        "composition_speed": 0.0,
        "pause_before_send": 0.0,
        "editing_effort": 0.0,
        # message content
        "message_length": 0.0,
        "type_token_ratio": 0.0,
        "mean_word_length": 0.0,
        "flesch_kincaid": 0.0,
        "question_ratio": 0.0,
        "formality": 0.0,
        "emoji_density": 0.0,
        "sentiment_valence": 0.0,
        # session
        "length_trend": 0.0,
        "latency_trend": 0.0,
        "vocab_trend": 0.0,
        "engagement_velocity": 0.0,
        "topic_coherence": 0.0,
        "session_progress": 0.0,
        "time_deviation": 0.0,
        "response_depth": 0.0,
        # deviations
        "iki_deviation": 0.0,
        "length_deviation": 0.0,
        "vocab_deviation": 0.0,
        "formality_deviation": 0.0,
        "speed_deviation": 0.0,
        "engagement_deviation": 0.0,
        "complexity_deviation": 0.0,
        "pattern_deviation": 0.0,
    }
    base.update(overrides)
    return InteractionFeatureVector(**base)


# ---------------------------------------------------------------------------
# Welford correctness
# ---------------------------------------------------------------------------


def test_welford_mean_matches_python_statistics() -> None:
    """Online mean must match the offline reference."""
    bt = BaselineTracker(warmup=2)
    samples = [0.10, 0.30, 0.55, 0.80, 0.42, 0.18]
    for s in samples:
        bt.update(_fv(mean_iki=s))

    assert bt.count == len(samples)
    assert bt.get_mean("mean_iki") == pytest.approx(
        statistics.fmean(samples), abs=1e-12
    )


def test_welford_unbiased_variance_matches_python_statistics() -> None:
    """Iter 2: sample variance (Bessel-corrected) matches statistics.variance."""
    bt = BaselineTracker(warmup=2)
    samples = [0.10, 0.30, 0.55, 0.80, 0.42, 0.18]
    for s in samples:
        bt.update(_fv(mean_iki=s))

    # statistics.variance uses (n-1) denominator — the same Bessel
    # correction we want.
    expected_var = statistics.variance(samples)
    expected_std = math.sqrt(expected_var)

    assert bt.get_std("mean_iki") == pytest.approx(expected_std, abs=1e-12)


# ---------------------------------------------------------------------------
# Warm-up gate
# ---------------------------------------------------------------------------


def test_deviation_zero_until_warmup_reached() -> None:
    bt = BaselineTracker(warmup=5)
    for _ in range(4):
        bt.update(_fv(mean_iki=0.5))
    # Only 4 observations — under warmup.
    assert bt.deviation("mean_iki", 0.9) == 0.0
    assert bt.is_established is False

    bt.update(_fv(mean_iki=0.5))
    assert bt.is_established is True
    # All five observations were 0.5; the variance is zero so deviation
    # also returns 0.0 (degenerate-distribution guard).
    assert bt.deviation("mean_iki", 0.9) == 0.0


# ---------------------------------------------------------------------------
# Extreme-value clamping
# ---------------------------------------------------------------------------


def test_deviation_clamps_to_negative_one_one_range() -> None:
    """Z-scores beyond ±3σ clamp to ±1 after the (z/3) normalisation."""
    bt = BaselineTracker(warmup=5)
    # Tight cluster of identical-but-perturbed values (avoid std=0).
    samples = [0.10, 0.11, 0.10, 0.11, 0.10, 0.11]
    for s in samples:
        bt.update(_fv(mean_iki=s))

    # An extreme value is clamped to the closed [-1, 1] interval.
    assert bt.deviation("mean_iki", 100.0) == pytest.approx(1.0)
    assert bt.deviation("mean_iki", -100.0) == pytest.approx(-1.0)


def test_deviation_returns_zero_when_std_collapses() -> None:
    """Constant feature ⇒ std=0 ⇒ deviation defined as 0.0 (not NaN)."""
    bt = BaselineTracker(warmup=3)
    for _ in range(5):
        bt.update(_fv(formality=0.5))
    assert bt.is_established is True
    assert bt.get_std("formality") == 0.0
    assert bt.deviation("formality", 1.0) == 0.0


# ---------------------------------------------------------------------------
# Bessel correction precision (iter 2)
# ---------------------------------------------------------------------------


def test_bessel_correction_z_score_matches_unbiased_reference() -> None:
    """Iter 2: z-score uses sample (n-1) variance, not population (n)."""
    bt = BaselineTracker(warmup=3)
    samples = [0.20, 0.40, 0.60, 0.80, 1.00]
    for s in samples:
        bt.update(_fv(message_length=s))

    expected_mean = statistics.fmean(samples)
    expected_std = math.sqrt(statistics.variance(samples))
    probe = 1.20
    expected_z = (probe - expected_mean) / expected_std
    expected_clamped = max(-1.0, min(1.0, expected_z / 3.0))

    actual = bt.deviation("message_length", probe)
    assert actual == pytest.approx(expected_clamped, abs=1e-12)


def test_bessel_at_minimum_sample_size_does_not_divide_by_zero() -> None:
    """count=2 should produce a finite (or zero) deviation, never NaN/inf."""
    bt = BaselineTracker(warmup=2)
    bt.update(_fv(mean_iki=0.10))
    bt.update(_fv(mean_iki=0.20))

    # is_established=True at count=2 with warmup=2.
    dev = bt.deviation("mean_iki", 0.30)
    assert math.isfinite(dev)
    # Should be a positive deviation (probe is above the mean).
    assert dev > 0.0


def test_get_std_at_count_one_returns_zero_not_nan() -> None:
    """A single observation has no defined sample std; must not blow up."""
    bt = BaselineTracker()
    bt.update(_fv(mean_iki=0.42))
    assert bt.count == 1
    std = bt.get_std("mean_iki")
    assert math.isfinite(std)
    assert std == 0.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_all_state() -> None:
    bt = BaselineTracker(warmup=2)
    for s in [0.10, 0.30, 0.55]:
        bt.update(_fv(mean_iki=s))
    assert bt.is_established is True

    bt.reset()
    assert bt.count == 0
    assert bt.is_established is False
    assert bt.get_mean("mean_iki") == 0.0
    assert bt.get_std("mean_iki") == 0.0
    assert bt.deviation("mean_iki", 5.0) == 0.0


# ---------------------------------------------------------------------------
# Unknown-feature defensive behaviour
# ---------------------------------------------------------------------------


def test_unknown_feature_returns_zero() -> None:
    bt = BaselineTracker(warmup=2)
    for s in [0.1, 0.2, 0.3]:
        bt.update(_fv(mean_iki=s))

    # Feature name that wasn't tracked — returns 0.0 cleanly.
    assert bt.get_mean("nonexistent_feature") == 0.0
    assert bt.get_std("nonexistent_feature") == 0.0
    assert bt.deviation("nonexistent_feature", 0.5) == 0.0


# ---------------------------------------------------------------------------
# Numerical stability over a long stream
# ---------------------------------------------------------------------------


def test_nan_feature_does_not_corrupt_baseline() -> None:
    """Iter 33: a single NaN feature value must not poison the
    running mean / variance for subsequent updates.  The whole turn
    is silently dropped instead."""
    bt = BaselineTracker(warmup=2)

    bt.update(_fv(mean_iki=0.3))
    # NaN turn — silently dropped wholesale.
    bt.update(_fv(mean_iki=float("nan"), formality=0.5))
    bt.update(_fv(mean_iki=0.4))
    bt.update(_fv(mean_iki=0.5))

    # Only 3 finite turns contributed: mean = 0.4
    assert bt.count == 3
    assert bt.get_mean("mean_iki") == pytest.approx(0.4, abs=1e-9)
    assert math.isfinite(bt.get_std("mean_iki"))
    # Formality also reflects only the 3 finite turns.
    assert math.isfinite(bt.get_mean("formality"))


def test_inf_feature_does_not_corrupt_baseline() -> None:
    """+inf and -inf are also dropped at the turn level."""
    bt = BaselineTracker(warmup=2)
    bt.update(_fv(mean_iki=0.3))
    bt.update(_fv(mean_iki=float("inf")))
    bt.update(_fv(mean_iki=float("-inf")))
    bt.update(_fv(mean_iki=0.5))

    # Only 0.3 and 0.5 contributed: mean = 0.4
    assert bt.count == 2
    assert bt.get_mean("mean_iki") == pytest.approx(0.4, abs=1e-9)
    assert math.isfinite(bt.get_std("mean_iki"))


def test_long_stream_remains_numerically_stable() -> None:
    """5 000 samples at high magnitude — no precision drift."""
    bt = BaselineTracker(warmup=2)
    n = 5_000
    base = 1_000_000.0
    samples = [base + (i % 7) * 0.1 for i in range(n)]
    for s in samples:
        bt.update(_fv(message_length=s))

    expected_mean = statistics.fmean(samples)
    expected_std = math.sqrt(statistics.variance(samples))

    assert bt.get_mean("message_length") == pytest.approx(expected_mean, rel=1e-9)
    # Welford with sum-of-squares is stable; we accept 1e-6 relative
    # error vs the offline reference at this magnitude.
    assert bt.get_std("message_length") == pytest.approx(expected_std, rel=1e-6)
