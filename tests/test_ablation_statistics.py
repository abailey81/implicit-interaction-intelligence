"""Unit tests for ``i3.eval.ablation_statistics``.

Covers the four helpers used by the Batch A ablation:

* :func:`i3.eval.ablation_statistics.bootstrap_ci` — coverage property +
  basic shape / input-validation tests.
* :func:`i3.eval.ablation_statistics.cohens_d` — recovery of a known
  Gaussian-vs-shifted-Gaussian effect size.
* :func:`i3.eval.ablation_statistics.paired_sign_test` — ``p ≈ 0.5`` under
  exchangeability; ``p < 0.05`` under a strong signed shift.
* :func:`i3.eval.ablation_statistics.effect_size_interpretation` — Cohen
  (1988) threshold bucketing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from i3.eval.ablation_statistics import (
    bootstrap_ci,
    cohens_d,
    effect_size_interpretation,
    paired_sign_test,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_mean_for_known_sample() -> None:
    """CI must bracket the sample mean for a reasonably-sized sample."""
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=5.0, scale=2.0, size=200)
    lo, hi = bootstrap_ci(samples, n_resamples=4_000, ci=0.95, rng=rng)
    assert lo < float(samples.mean()) < hi


def test_bootstrap_ci_tightens_with_larger_sample() -> None:
    """The width of a 95 % bootstrap CI should shrink with ``sqrt(n)``."""
    rng_small = np.random.default_rng(7)
    rng_large = np.random.default_rng(7)
    small = rng_small.normal(size=50)
    large = rng_large.normal(size=5_000)
    small_lo, small_hi = bootstrap_ci(small, n_resamples=2_000, rng=rng_small)
    large_lo, large_hi = bootstrap_ci(large, n_resamples=2_000, rng=rng_large)
    assert (large_hi - large_lo) < (small_hi - small_lo)


def test_bootstrap_ci_rejects_empty_input() -> None:
    """An empty sample must raise a ``ValueError``."""
    with pytest.raises(ValueError):
        bootstrap_ci([], n_resamples=100)


def test_bootstrap_ci_rejects_bad_ci() -> None:
    """CI outside ``(0, 1)`` must raise a ``ValueError``."""
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], ci=1.5)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0, 3.0], ci=0.0)


@pytest.mark.parametrize("n", [30, 100, 500])
def test_bootstrap_ci_coverage_parametrised(n: int) -> None:
    """Parametrised deterministic coverage check.

    We repeatedly draw Gaussian samples and check that the bootstrap 95 %
    CI contains the *true* mean (0.0) in at least 85 % of repetitions —
    a loose but honest coverage floor that tolerates bootstrap bias in
    small samples.
    """
    covered = 0
    trials = 40
    rng = np.random.default_rng(101)
    for _ in range(trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=n)
        # Use a fresh per-trial RNG so each bootstrap is deterministic but
        # the trials are independent.
        boot_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
        lo, hi = bootstrap_ci(sample, n_resamples=2_000, ci=0.95, rng=boot_rng)
        if lo <= 0.0 <= hi:
            covered += 1
    assert covered / trials >= 0.85


def test_bootstrap_ci_hypothesis_coverage_property() -> None:
    """Property-based coverage test when Hypothesis is available.

    Skipped when the ``hypothesis`` package is not installed.
    """
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @given(
        loc=st.floats(min_value=-10.0, max_value=10.0),
        scale=st.floats(min_value=0.1, max_value=5.0),
        size=st.integers(min_value=30, max_value=500),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=10, deadline=None)
    def _inner(loc: float, scale: float, size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        sample = rng.normal(loc=loc, scale=scale, size=size)
        lo, hi = bootstrap_ci(sample, n_resamples=1_000, rng=rng)
        # The CI must bracket the *sample* mean (always true for a
        # percentile bootstrap when n_resamples is large enough).
        assert lo <= float(sample.mean()) <= hi

    _inner()
    assert hypothesis.__name__ == "hypothesis"


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


def test_cohens_d_recovers_known_shift() -> None:
    """For two unit-variance Gaussians with a known shift, d ≈ shift."""
    rng = np.random.default_rng(2026)
    a = rng.normal(loc=1.0, scale=1.0, size=2_000)
    b = rng.normal(loc=0.0, scale=1.0, size=2_000)
    d = cohens_d(a, b)
    # Known effect size is 1.0; allow 0.15 tolerance for finite samples.
    assert math.isclose(d, 1.0, abs_tol=0.15)


def test_cohens_d_zero_when_identical() -> None:
    """Equal-distribution samples should yield d ≈ 0."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    assert abs(cohens_d(a, b)) < 0.2


def test_cohens_d_sign_follows_mean_difference() -> None:
    """``mean(a) > mean(b)`` must produce a positive d."""
    a = [2.0, 2.5, 3.0, 2.0, 3.0, 2.5]
    b = [0.0, 0.5, 1.0, 0.0, 1.0, 0.5]
    assert cohens_d(a, b) > 0.0
    assert cohens_d(b, a) < 0.0


def test_cohens_d_rejects_degenerate_sample_size() -> None:
    """A sample of size < 2 must raise ``ValueError``."""
    with pytest.raises(ValueError):
        cohens_d([1.0], [1.0, 2.0])


def test_cohens_d_zero_variance_zero_diff_returns_zero() -> None:
    """Identical constants give d = 0 rather than NaN."""
    a = [1.0, 1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0, 1.0]
    assert cohens_d(a, b) == 0.0


# ---------------------------------------------------------------------------
# paired_sign_test
# ---------------------------------------------------------------------------


def test_paired_sign_test_exchangeable_is_near_half() -> None:
    """Exchangeable pairs should yield p ≈ 0.5."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    p = paired_sign_test(a, b)
    # Loose band: exchangeability + 200 samples → p generally in [0.1, 1.0].
    assert 0.05 < p <= 1.0


def test_paired_sign_test_strongly_signed_is_significant() -> None:
    """A systematic positive shift yields p < 0.05."""
    rng = np.random.default_rng(17)
    b = rng.normal(size=100)
    a = b + 1.0  # every pair has positive difference → minimal p-value
    p = paired_sign_test(a, b)
    assert p < 0.05


def test_paired_sign_test_shape_mismatch_raises() -> None:
    """Mismatched lengths must raise ``ValueError``."""
    with pytest.raises(ValueError):
        paired_sign_test([1.0, 2.0], [1.0])


def test_paired_sign_test_all_ties_returns_one() -> None:
    """All-zero differences → no evidence against null → p = 1.0."""
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    assert paired_sign_test(a, b) == 1.0


def test_paired_sign_test_empty_raises() -> None:
    """Empty pairs must raise ``ValueError``."""
    with pytest.raises(ValueError):
        paired_sign_test([], [])


# ---------------------------------------------------------------------------
# effect_size_interpretation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, expected",
    [
        (0.0, "negligible"),
        (0.1, "negligible"),
        (-0.19, "negligible"),
        (0.2, "small"),
        (0.4, "small"),
        (-0.49, "small"),
        (0.5, "medium"),
        (0.7, "medium"),
        (-0.79, "medium"),
        (0.8, "large"),
        (2.5, "large"),
        (-3.0, "large"),
    ],
)
def test_effect_size_interpretation_thresholds(d: float, expected: str) -> None:
    """Threshold bucketing must follow Cohen (1988)."""
    assert effect_size_interpretation(d) == expected
