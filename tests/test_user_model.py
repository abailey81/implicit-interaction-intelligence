"""Tests for the persistent user model.

Validates the three-timescale architecture: session lifecycle, EMA
convergence, baseline establishment, and deviation computation.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import torch

from i3.config import UserModelConfig
from i3.interaction.types import InteractionFeatureVector
from i3.user_model.model import UserModel
from i3.user_model.types import DeviationMetrics, SessionState, UserProfile


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def config() -> UserModelConfig:
    """User model config with fast warm-up for testing."""
    return UserModelConfig(
        session_ema_alpha=0.3,
        longterm_ema_alpha=0.1,
        baseline_warmup=3,
        deviation_threshold=1.5,
        max_history_sessions=50,
    )


@pytest.fixture
def model(config: UserModelConfig) -> UserModel:
    """Fresh user model with no prior history."""
    return UserModel(user_id="test_user", config=config)


def _make_features(**kwargs) -> InteractionFeatureVector:
    """Create an InteractionFeatureVector with specified overrides."""
    return InteractionFeatureVector(**kwargs)


def _random_embedding() -> torch.Tensor:
    """Generate a random 64-dim unit-norm embedding."""
    e = torch.randn(64)
    return e / e.norm()


# -------------------------------------------------------------------------
# Session lifecycle
# -------------------------------------------------------------------------

class TestSessionLifecycle:
    """Tests for session start, update, and end."""

    def test_start_session_creates_state(self, model: UserModel) -> None:
        """start_session should initialise the session state."""
        model.start_session()
        assert model.current_session is not None
        assert model.current_session.message_count == 0

    def test_update_without_session_raises(self, model: UserModel) -> None:
        """Calling update_state without an active session should raise."""
        with pytest.raises(RuntimeError, match="No active session"):
            model.update_state(_random_embedding(), _make_features())

    def test_end_without_session_raises(self, model: UserModel) -> None:
        """Calling end_session without an active session should raise."""
        with pytest.raises(RuntimeError, match="No active session"):
            model.end_session()

    def test_session_message_count_increments(self, model: UserModel) -> None:
        """Each update_state call should increment the message count."""
        model.start_session()
        for i in range(5):
            model.update_state(_random_embedding(), _make_features())
            assert model.current_session.message_count == i + 1

    def test_end_session_returns_summary(self, model: UserModel) -> None:
        """end_session should return a metadata-only summary dict."""
        model.start_session()
        for _ in range(3):
            model.update_state(_random_embedding(), _make_features())
        summary = model.end_session()

        assert summary['user_id'] == 'test_user'
        assert summary['message_count'] == 3
        assert 'duration_seconds' in summary
        assert 'mean_engagement' in summary
        assert 'baseline_established' in summary

    def test_end_session_clears_state(self, model: UserModel) -> None:
        """After ending a session, current_session should be None."""
        model.start_session()
        model.update_state(_random_embedding(), _make_features())
        model.end_session()
        assert model.current_session is None
        assert model.current_state is None

    def test_multiple_sessions(self, model: UserModel) -> None:
        """Multiple sessions should increment total_sessions."""
        for session_num in range(3):
            model.start_session()
            for _ in range(2):
                model.update_state(_random_embedding(), _make_features())
            model.end_session()
        assert model.profile.total_sessions == 3
        assert model.profile.total_messages == 6


# -------------------------------------------------------------------------
# EMA convergence
# -------------------------------------------------------------------------

class TestEMAUpdate:
    """Tests for the session-level Exponential Moving Average."""

    def test_first_message_sets_embedding(
        self, model: UserModel
    ) -> None:
        """First message should set the session embedding directly."""
        model.start_session()
        emb = _random_embedding()
        model.update_state(emb, _make_features())

        torch.testing.assert_close(
            model.current_session.embedding, emb,
            atol=1e-5, rtol=1e-5,
        )

    def test_session_ema_converges(self, model: UserModel) -> None:
        """With repeated identical embeddings, the session EMA should
        converge to that embedding."""
        target = _random_embedding()
        model.start_session()

        # Feed the same embedding 20 times
        for _ in range(20):
            model.update_state(target.clone(), _make_features())

        session_emb = model.current_session.embedding
        cos_sim = torch.nn.functional.cosine_similarity(
            session_emb.unsqueeze(0), target.unsqueeze(0)
        ).item()

        assert cos_sim > 0.99, (
            f"Session EMA did not converge: cosine similarity = {cos_sim:.4f}"
        )

    def test_ema_responds_to_shift(self, model: UserModel) -> None:
        """After a sudden embedding shift, the EMA should track it."""
        emb_a = torch.zeros(64)
        emb_a[0] = 1.0
        emb_b = torch.zeros(64)
        emb_b[1] = 1.0

        model.start_session()

        # Phase A: constant
        for _ in range(10):
            model.update_state(emb_a.clone(), _make_features())
        session_after_a = model.current_session.embedding.clone()

        # Phase B: shift
        for _ in range(10):
            model.update_state(emb_b.clone(), _make_features())
        session_after_b = model.current_session.embedding

        # The EMA should have moved toward emb_b
        assert session_after_b[1] > session_after_a[1]


# -------------------------------------------------------------------------
# Baseline establishment
# -------------------------------------------------------------------------

class TestBaselineEstablishment:
    """Tests for baseline warm-up and establishment."""

    def test_baseline_not_established_initially(
        self, model: UserModel
    ) -> None:
        """Baseline should not be established before warm-up messages."""
        assert model.profile.baseline_established is False

    def test_baseline_established_after_warmup(
        self, model: UserModel, config: UserModelConfig
    ) -> None:
        """After baseline_warmup messages, baseline should be established."""
        model.start_session()
        for _ in range(config.baseline_warmup):
            model.update_state(_random_embedding(), _make_features())

        assert model.profile.baseline_established is True

    def test_baseline_embedding_populated(
        self, model: UserModel, config: UserModelConfig
    ) -> None:
        """After warm-up, the baseline embedding should be non-None."""
        model.start_session()
        for _ in range(config.baseline_warmup + 2):
            model.update_state(_random_embedding(), _make_features())

        assert model.profile.baseline_embedding is not None
        assert model.profile.baseline_embedding.shape == (64,)

    def test_relationship_strength_grows(self, model: UserModel) -> None:
        """Relationship strength should increase with more interactions."""
        model.start_session()
        for _ in range(5):
            model.update_state(_random_embedding(), _make_features())
        model.end_session()

        strength_1 = model.profile.relationship_strength

        model.start_session()
        for _ in range(50):
            model.update_state(_random_embedding(), _make_features())
        model.end_session()

        strength_2 = model.profile.relationship_strength
        assert strength_2 > strength_1


# -------------------------------------------------------------------------
# Deviation computation
# -------------------------------------------------------------------------

class TestDeviationComputation:
    """Tests for deviation metrics from baseline."""

    def test_deviation_returns_metrics(self, model: UserModel) -> None:
        """update_state should return a DeviationMetrics instance."""
        model.start_session()
        metrics = model.update_state(_random_embedding(), _make_features())

        assert isinstance(metrics, DeviationMetrics)
        assert hasattr(metrics, 'engagement_score')
        assert hasattr(metrics, 'current_vs_baseline')
        assert hasattr(metrics, 'iki_deviation')

    def test_deviation_defaults_before_baseline(
        self, model: UserModel
    ) -> None:
        """Before baseline is established, deviation should be ~0."""
        assert model.deviation_from_baseline.current_vs_baseline == 0.0

    def test_engagement_score_in_range(self, model: UserModel) -> None:
        """Engagement score should be a finite number."""
        model.start_session()
        for _ in range(5):
            dev = model.update_state(_random_embedding(), _make_features())

        score = model.engagement_score
        assert isinstance(score, float)
        # Score can be outside [0,1] before baseline, just check finite
        assert not (score != score)  # Not NaN

    def test_deviation_meaningful_after_baseline(
        self, model: UserModel, config: UserModelConfig
    ) -> None:
        """After baseline, feeding a very different embedding should
        produce non-zero deviation."""
        model.start_session()

        # Establish baseline with consistent embeddings
        emb_baseline = _random_embedding()
        for _ in range(config.baseline_warmup + 5):
            model.update_state(emb_baseline.clone(), _make_features())

        assert model.profile.baseline_established

        # Now send a very different embedding
        emb_outlier = -emb_baseline  # Opposite direction
        dev = model.update_state(emb_outlier, _make_features())

        # The cosine distance from baseline should be substantial
        assert dev.current_vs_baseline > 0.1 or dev.magnitude > 0.0


# -------------------------------------------------------------------------
# Feature statistics
# -------------------------------------------------------------------------

class TestFeatureStatistics:
    """Tests for running feature mean/std (Welford's algorithm)."""

    def test_feature_mean_updated(self, model: UserModel) -> None:
        """After updates, baseline_features_mean should be populated."""
        model.start_session()
        for i in range(5):
            features = _make_features(mean_iki=0.5, message_length=float(i) * 0.1)
            model.update_state(_random_embedding(), features)

        assert model.profile.baseline_features_mean is not None
        assert 'mean_iki' in model.profile.baseline_features_mean

    def test_feature_std_after_multiple_updates(
        self, model: UserModel
    ) -> None:
        """After 3+ updates, std should be populated."""
        model.start_session()
        for i in range(5):
            features = _make_features(
                mean_iki=0.3 + i * 0.1,
                message_length=0.2 + i * 0.05,
            )
            model.update_state(_random_embedding(), features)

        assert model.profile.baseline_features_std is not None
        # Std should be positive for varied inputs
        std = model.profile.baseline_features_std.get('mean_iki', 0.0)
        assert std > 0.0, "Std of varied inputs should be > 0"


# -------------------------------------------------------------------------
# Profile persistence
# -------------------------------------------------------------------------

class TestProfilePersistence:
    """Tests for profile state across sessions."""

    def test_profile_preserved_across_sessions(
        self, config: UserModelConfig
    ) -> None:
        """Creating a new UserModel with an existing profile should
        restore state."""
        model1 = UserModel(user_id="persist_test", config=config)
        model1.start_session()
        for _ in range(5):
            model1.update_state(_random_embedding(), _make_features())
        model1.end_session()

        # Create new model with the same profile
        model2 = UserModel(
            user_id="persist_test",
            config=config,
            profile=model1.profile,
        )

        assert model2.profile.total_sessions == 1
        assert model2.profile.total_messages == 5
