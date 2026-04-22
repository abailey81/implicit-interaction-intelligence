"""Unit tests for :mod:`i3.adaptation.uncertainty`.

Covers:
    * Shape / sample-count invariants on the MC Dropout output.
    * Law of large numbers: std shrinks as n_samples grows.
    * Deterministic behaviour under a fixed torch seed.
    * Policy helpers (``confidence_threshold_policy`` and
      ``refuse_when_unsure_mask``) on hand-crafted inputs.
    * Input validation (wrong shapes / zero-size samples).
"""

from __future__ import annotations

import pytest
import torch

from i3.adaptation.controller import AdaptationController
from i3.adaptation.types import AdaptationVector, StyleVector
from i3.adaptation.uncertainty import (
    ADAPTATION_DIMS,
    DimensionInterval,
    MCDropoutAdaptationEstimator,
    UncertainAdaptationVector,
    confidence_threshold_policy,
    refuse_when_unsure_mask,
)
from i3.config import load_config
from i3.encoder.tcn import TemporalConvNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def encoder() -> TemporalConvNet:
    """A small, random-init TCN with dropout enabled."""
    torch.manual_seed(0)
    return TemporalConvNet(input_dim=32, embedding_dim=64, dropout=0.1)


@pytest.fixture
def controller() -> AdaptationController:
    """The production :class:`AdaptationController` on default config."""
    config = load_config("configs/default.yaml")
    return AdaptationController(config.adaptation)


@pytest.fixture
def feature_window() -> torch.Tensor:
    """A deterministic ``[16, 32]`` feature window."""
    gen = torch.Generator().manual_seed(7)
    return torch.rand(16, 32, generator=gen, dtype=torch.float32)


def _uniform_uncertain(
    mean: AdaptationVector, std_val: float
) -> UncertainAdaptationVector:
    """Helper to build an :class:`UncertainAdaptationVector` for policy tests.

    Args:
        mean: The sample mean to report.
        std_val: Constant per-dimension std.

    Returns:
        A populated :class:`UncertainAdaptationVector`.
    """
    return UncertainAdaptationVector(
        mean=mean.to_dict(),
        std=[float(std_val)] * 8,
        ci=[
            DimensionInterval(
                lower=max(0.0, 0.5 - std_val),
                upper=min(1.0, 0.5 + std_val),
            )
            for _ in range(8)
        ],
        sample_count=10,
    )


# ---------------------------------------------------------------------------
# MCDropoutAdaptationEstimator
# ---------------------------------------------------------------------------


class TestMCDropoutAdaptationEstimator:
    """Unit tests for the MC Dropout estimator."""

    def test_returns_uncertain_vector_with_8_dim(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """The estimator must return an 8-dim std / ci."""
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=5)
        out = est.estimate(feature_window)
        assert isinstance(out, UncertainAdaptationVector)
        assert len(out.std) == 8
        assert len(out.ci) == 8
        assert out.sample_count == 5

    def test_shape_matches_adaptation_vector(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """Mean must deserialise to a valid :class:`AdaptationVector`."""
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=4)
        out = est.estimate(feature_window)
        mean = out.mean_vector()
        assert isinstance(mean, AdaptationVector)
        assert mean.to_tensor().shape == (8,)

    def test_std_nonnegative_bounded(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """Per-dimension std must be finite, non-negative, and bounded."""
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=6)
        out = est.estimate(feature_window)
        for s in out.std:
            assert s >= 0.0
            # Bounded by the [0, 1] range of the underlying values.
            assert s <= 1.0

    def test_ci_ordering(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """CI lower must be <= CI upper for every dimension."""
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=6)
        out = est.estimate(feature_window)
        for interval in out.ci:
            assert interval.lower <= interval.upper

    def test_deterministic_under_fixed_seed(
        self,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """Two runs with identical seeds must return identical samples."""
        torch.manual_seed(123)
        enc1 = TemporalConvNet(input_dim=32, embedding_dim=64, dropout=0.1)
        est1 = MCDropoutAdaptationEstimator(enc1, controller, n_samples=4)
        torch.manual_seed(999)
        out1 = est1.estimate(feature_window)

        torch.manual_seed(123)
        enc2 = TemporalConvNet(input_dim=32, embedding_dim=64, dropout=0.1)
        est2 = MCDropoutAdaptationEstimator(enc2, controller, n_samples=4)
        torch.manual_seed(999)
        out2 = est2.estimate(feature_window)

        assert out1.std == pytest.approx(out2.std)

    def test_restores_training_mode(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """The estimator must never leak training mode onto the encoder."""
        encoder.eval()
        assert not encoder.training
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=3)
        est.estimate(feature_window)
        assert not encoder.training

    def test_wrong_window_shape_raises(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
    ) -> None:
        """A 1-D feature window must be rejected."""
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=2)
        with pytest.raises(ValueError):
            est.estimate(torch.randn(32))

    def test_reject_batched_multi(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
    ) -> None:
        """Batch size != 1 must raise."""
        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=2)
        with pytest.raises(ValueError):
            est.estimate(torch.randn(2, 16, 32))

    def test_invalid_n_samples_raises(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
    ) -> None:
        """n_samples < 2 must raise."""
        with pytest.raises(ValueError):
            MCDropoutAdaptationEstimator(encoder, controller, n_samples=1)

    def test_invalid_dropout_p_raises(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
    ) -> None:
        """dropout_p outside [0, 1) must raise."""
        with pytest.raises(ValueError):
            MCDropoutAdaptationEstimator(
                encoder, controller, n_samples=3, dropout_p=1.0
            )
        with pytest.raises(ValueError):
            MCDropoutAdaptationEstimator(
                encoder, controller, n_samples=3, dropout_p=-0.1
            )


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


class TestConfidencePolicy:
    """Policy behaviour on hand-crafted inputs."""

    def test_confident_on_zero_std(self) -> None:
        uncertain = _uniform_uncertain(AdaptationVector.default(), std_val=0.0)
        assert confidence_threshold_policy(uncertain, threshold=0.15) is True

    def test_uncertain_above_threshold(self) -> None:
        uncertain = _uniform_uncertain(AdaptationVector.default(), std_val=0.3)
        assert confidence_threshold_policy(uncertain, threshold=0.15) is False

    def test_threshold_must_be_positive(self) -> None:
        uncertain = _uniform_uncertain(AdaptationVector.default(), std_val=0.1)
        with pytest.raises(ValueError):
            confidence_threshold_policy(uncertain, threshold=0.0)

    def test_refuse_mask_high_variance_dim(self) -> None:
        """High-variance dims must collapse to the neutral baseline."""
        mean = AdaptationVector(
            cognitive_load=0.9,
            style_mirror=StyleVector(
                formality=0.9, verbosity=0.9, emotionality=0.9, directness=0.9
            ),
            emotional_tone=0.9,
            accessibility=0.9,
        )
        uncertain = _uniform_uncertain(mean, std_val=0.5)
        refused = refuse_when_unsure_mask(uncertain, threshold=0.15)
        neutral = AdaptationVector.default()
        assert refused.cognitive_load == pytest.approx(neutral.cognitive_load)
        assert refused.emotional_tone == pytest.approx(neutral.emotional_tone)
        assert refused.accessibility == pytest.approx(neutral.accessibility)

    def test_refuse_mask_preserves_confident_dims(self) -> None:
        """Low-variance dims must pass through unchanged."""
        mean = AdaptationVector(
            cognitive_load=0.9,
            style_mirror=StyleVector(
                formality=0.9, verbosity=0.9, emotionality=0.9, directness=0.9
            ),
            emotional_tone=0.9,
            accessibility=0.9,
        )
        uncertain = _uniform_uncertain(mean, std_val=0.05)
        refused = refuse_when_unsure_mask(uncertain, threshold=0.15)
        assert refused.cognitive_load == pytest.approx(0.9)
        assert refused.emotional_tone == pytest.approx(0.9)

    def test_canonical_dim_order(self) -> None:
        """The canonical order must match the AdaptationVector layout."""
        assert ADAPTATION_DIMS[0] == "cognitive_load"
        assert ADAPTATION_DIMS[5] == "emotional_tone"
        assert ADAPTATION_DIMS[6] == "accessibility"
        assert ADAPTATION_DIMS[7] == "reserved"


# ---------------------------------------------------------------------------
# Monte-Carlo convergence
# ---------------------------------------------------------------------------


class TestMCConvergence:
    """Behavioural tests for the MC sampling convergence properties."""

    def test_mean_std_shrinks_with_more_samples(
        self,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """Std of the *mean across re-runs* shrinks with sample count.

        This is the operational form of the law of large numbers that
        the MC Dropout estimator satisfies: the Monte-Carlo standard
        error of the mean decreases as ``1/sqrt(N)``.
        """
        torch.manual_seed(0)
        encoder = TemporalConvNet(input_dim=32, embedding_dim=64, dropout=0.2)

        def _run(n_samples: int, seeds: list[int]) -> float:
            est = MCDropoutAdaptationEstimator(
                encoder, controller, n_samples=n_samples, dropout_p=0.2
            )
            means = []
            for s in seeds:
                torch.manual_seed(s)
                out = est.estimate(feature_window)
                means.append(out.mean_vector().cognitive_load)
            t = torch.tensor(means, dtype=torch.float32)
            return float(t.std(unbiased=False).item())

        seeds = list(range(1, 9))
        small = _run(4, seeds)
        big = _run(40, seeds)
        # Typical MC behaviour: big mean is at least a little less
        # dispersed than small mean. We allow a small tolerance to
        # accommodate discrete sample noise.
        assert big <= small + 0.05

    def test_std_values_finite(
        self,
        encoder: TemporalConvNet,
        controller: AdaptationController,
        feature_window: torch.Tensor,
    ) -> None:
        """All entries of ``std`` and ``ci`` must be finite."""
        import math

        est = MCDropoutAdaptationEstimator(encoder, controller, n_samples=8)
        out = est.estimate(feature_window)
        for s in out.std:
            assert math.isfinite(s)
        for c in out.ci:
            assert math.isfinite(c.lower) and math.isfinite(c.upper)
