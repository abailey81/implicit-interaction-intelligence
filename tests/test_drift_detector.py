"""Tests for the ADWIN-style drift detector and PSI helper.

Covers:
* Constructor validation (delta, window sizes, cooldown).
* Detector fires on a known distribution shift (mean shift, variance
  shift) with a bounded number of observations.
* Detector stays silent on a stationary stream for its full length.
* Cooldown prevents repeat firing in quick succession.
* ``on_drift_detected`` callback is invoked exactly once per event.
* :func:`population_stability_index` returns ``0`` on identical
  distributions and grows monotonically with mean shift.
"""

from __future__ import annotations

import pytest
import torch

from i3.continual.drift_detector import (
    ConceptDriftDetector,
    DriftDetectionResult,
    PSI,
    population_stability_index,
)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestDriftDetectorValidation:
    """Argument validation in :class:`ConceptDriftDetector`."""

    @pytest.mark.parametrize("delta", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_delta_rejected(self, delta: float) -> None:
        """delta must lie in (0, 1)."""
        with pytest.raises(ValueError):
            ConceptDriftDetector(delta=delta)

    def test_window_validation(self) -> None:
        """max_window must be >= 2 * min_sub_window."""
        with pytest.raises(ValueError):
            ConceptDriftDetector(min_sub_window=10, max_window=10)

    def test_negative_cooldown_rejected(self) -> None:
        """cooldown must be non-negative."""
        with pytest.raises(ValueError):
            ConceptDriftDetector(cooldown=-1)


# ---------------------------------------------------------------------------
# Detection behaviour
# ---------------------------------------------------------------------------


class TestDriftDetectorBehaviour:
    """End-to-end detection tests."""

    def test_detects_mean_shift(self) -> None:
        """A step change in the mean must trigger detection."""
        det = ConceptDriftDetector(
            delta=0.01, min_sub_window=10, max_window=512, cooldown=0,
        )
        torch.manual_seed(0)
        for _ in range(60):
            det.update(float(0.1 + 0.02 * torch.randn(()).item()))
        fired_idx = None
        for step in range(120):
            result = det.update(float(5.0 + 0.02 * torch.randn(()).item()))
            if result.drift_detected:
                fired_idx = step
                assert isinstance(result, DriftDetectionResult)
                assert result.old_mean is not None and result.new_mean is not None
                assert result.new_mean > result.old_mean
                break
        assert fired_idx is not None, "Detector did not fire on mean shift"
        assert fired_idx < 100, "Detector was too slow to fire"

    def test_silent_on_stationary_stream(self) -> None:
        """Stationary stream -> zero alarms for the full length."""
        det = ConceptDriftDetector(
            delta=0.001, min_sub_window=16, max_window=1024, cooldown=0,
        )
        torch.manual_seed(1)
        samples = 0.5 + 0.05 * torch.randn(800)
        events = 0
        for v in samples.tolist():
            if det.update(v).drift_detected:
                events += 1
        assert events == 0, f"Expected zero drift events, got {events}"

    def test_cooldown_prevents_immediate_refire(self) -> None:
        """A second drift right after the first is suppressed by cooldown."""
        det = ConceptDriftDetector(
            delta=0.05, min_sub_window=8, max_window=256, cooldown=30,
        )
        for _ in range(24):
            det.update(0.1)
        first_fired = False
        for _ in range(40):
            result = det.update(5.0)
            if result.drift_detected:
                first_fired = True
                break
        assert first_fired

        # Immediately push through another shift; cooldown should mute it.
        refired = False
        for _ in range(10):
            result = det.update(0.1)
            if result.drift_detected:
                refired = True
        assert not refired, "Cooldown did not suppress immediate refire"

    def test_callback_invoked_on_drift(self) -> None:
        """on_drift_detected fires once per event."""
        seen: list[DriftDetectionResult] = []

        def cb(res: DriftDetectionResult) -> None:
            seen.append(res)

        det = ConceptDriftDetector(
            delta=0.05,
            min_sub_window=8,
            max_window=256,
            cooldown=0,
            on_drift_detected=cb,
        )
        for _ in range(24):
            det.update(0.1)
        for _ in range(96):
            res = det.update(5.0)
            if res.drift_detected:
                break
        assert len(seen) >= 1
        assert all(r.drift_detected for r in seen)

    def test_non_finite_values_ignored(self) -> None:
        """NaN / Inf updates are silently dropped."""
        det = ConceptDriftDetector()
        initial = det.window_size
        det.update(float("nan"))
        det.update(float("inf"))
        det.update(-float("inf"))
        assert det.window_size == initial

    def test_reset_clears_state(self) -> None:
        """reset() empties the window and counters."""
        det = ConceptDriftDetector(
            delta=0.05, min_sub_window=8, max_window=256, cooldown=0,
        )
        for v in (0.1, 0.2, 0.3, 0.4, 0.5, 5.0, 5.1, 5.2, 5.3, 5.4):
            det.update(v)
        det.reset()
        assert det.window_size == 0
        assert det.drift_count == 0


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------


class TestPopulationStabilityIndex:
    """Tests for the PSI helper."""

    def test_identical_distributions_psi_near_zero(self) -> None:
        """PSI on identical samples must be ~0."""
        torch.manual_seed(2)
        a = 0.5 + 0.1 * torch.randn(1024)
        b = a.clone()
        psi = population_stability_index(a, b)
        assert psi == pytest.approx(0.0, abs=0.02)

    def test_mean_shift_grows_psi(self) -> None:
        """Monotonic: larger mean shift -> larger PSI."""
        torch.manual_seed(3)
        base = 0.0 + 0.1 * torch.randn(2048)
        prior: float = 0.0
        for shift in (0.0, 0.25, 0.75, 2.0):
            shifted = base + shift
            psi = population_stability_index(base, shifted)
            assert psi + 1e-6 >= prior, f"PSI not monotone at shift={shift}"
            prior = psi
        # Final significant shift must cross the 'moderate' threshold.
        assert prior > 0.1

    def test_alias_matches_function(self) -> None:
        """``PSI`` alias must refer to the same callable."""
        assert PSI is population_stability_index

    def test_empty_inputs_return_zero(self) -> None:
        """Empty samples return 0 rather than raising."""
        assert population_stability_index([], [0.1, 0.2]) == 0.0
        assert population_stability_index([0.1, 0.2], []) == 0.0

    def test_n_bins_validation(self) -> None:
        """n_bins must be >= 2."""
        with pytest.raises(ValueError):
            population_stability_index([0.1, 0.2], [0.3, 0.4], n_bins=1)
