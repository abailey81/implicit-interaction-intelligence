"""Tests for the contextual Thompson sampling bandit.

Validates initialisation, arm selection, posterior updates, convergence
behaviour, and the cold-start Beta fallback mechanism.
"""

from __future__ import annotations

import pytest
import numpy as np

from i3.router.bandit import ContextualThompsonBandit


class TestBanditInitialization:
    """Tests for bandit construction and parameter validation."""

    def test_default_dimensions(self) -> None:
        """Default bandit: 2 arms, 12-dim context."""
        bandit = ContextualThompsonBandit()
        assert bandit.n_arms == 2
        assert bandit.context_dim == 12

    def test_custom_dimensions(self) -> None:
        """Custom arm and context dimensions."""
        bandit = ContextualThompsonBandit(n_arms=3, context_dim=8)
        assert bandit.n_arms == 3
        assert bandit.context_dim == 8
        assert len(bandit.weight_means) == 3
        assert len(bandit.weight_covs) == 3
        assert bandit.weight_means[0].shape == (8,)
        assert bandit.weight_covs[0].shape == (8, 8)

    def test_initial_posteriors(self) -> None:
        """Initial weight means should be zero; covs should be scaled identity."""
        bandit = ContextualThompsonBandit(
            n_arms=2, context_dim=4, prior_precision=2.0
        )
        np.testing.assert_array_equal(
            bandit.weight_means[0], np.zeros(4)
        )
        expected_cov = np.eye(4) / 2.0
        np.testing.assert_array_almost_equal(
            bandit.weight_covs[0], expected_cov
        )

    def test_initial_beta_params(self) -> None:
        """Initial Beta(1, 1) is uniform prior."""
        bandit = ContextualThompsonBandit(n_arms=2)
        assert bandit.alpha == [1.0, 1.0]
        assert bandit.beta_param == [1.0, 1.0]

    def test_invalid_n_arms_raises(self) -> None:
        with pytest.raises(ValueError, match="n_arms"):
            ContextualThompsonBandit(n_arms=0)

    def test_invalid_context_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="context_dim"):
            ContextualThompsonBandit(context_dim=0)

    def test_invalid_prior_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="prior_precision"):
            ContextualThompsonBandit(prior_precision=-1.0)


class TestSelectArm:
    """Tests for arm selection."""

    def test_returns_valid_arm(self) -> None:
        """select_arm must return an integer in [0, n_arms)."""
        bandit = ContextualThompsonBandit(n_arms=3, context_dim=4)
        ctx = np.random.randn(4)
        arm, confidence = bandit.select_arm(ctx)

        assert isinstance(arm, int)
        assert 0 <= arm < 3

    def test_returns_confidence_dict(self) -> None:
        """Confidence dict must have keys for each arm and sum to ~1."""
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)
        _, confidence = bandit.select_arm(ctx)

        assert 'arm_0' in confidence
        assert 'arm_1' in confidence
        total = sum(confidence.values())
        assert abs(total - 1.0) < 0.01, f"Confidence sum = {total}"

    def test_wrong_context_dim_raises(self) -> None:
        """Context with wrong dimensionality should raise ValueError."""
        bandit = ContextualThompsonBandit(context_dim=8)
        with pytest.raises(ValueError, match="context of dim"):
            bandit.select_arm(np.zeros(5))

    def test_deterministic_seed(self) -> None:
        """With a fixed seed, selection should be reproducible."""
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.array([0.1, 0.2, 0.3, 0.4])

        np.random.seed(42)
        arm1, _ = bandit.select_arm(ctx)

        np.random.seed(42)
        arm2, _ = bandit.select_arm(ctx)

        assert arm1 == arm2


class TestUpdate:
    """Tests for posterior updates."""

    def test_beta_update_success(self) -> None:
        """Reward > 0.5 should add fractional evidence to alpha.

        The implementation uses continuous-reward Beta updates per
        ``i3/router/bandit.py:317-318``: ``alpha += reward`` and
        ``beta += 1 - reward``.  This is a strict generalisation of the
        textbook discrete +1 Bernoulli update — it preserves
        Beta-conjugacy and is more informative on smoothly-graded
        rewards (a 0.49 vs 0.51 distinction matters; the discrete
        threshold form would discard it).  See SEC note in the source.
        """
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)

        bandit.update(0, ctx, reward=0.8)
        assert bandit.alpha[0] == pytest.approx(1.8, abs=0.01)
        assert bandit.beta_param[0] == pytest.approx(1.2, abs=0.01)

    def test_beta_update_failure(self) -> None:
        """Reward <= 0.5 should add fractional evidence to beta.

        Same fractional-reward contract — see ``test_beta_update_success``
        for the rationale.
        """
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)

        bandit.update(0, ctx, reward=0.3)
        assert bandit.alpha[0] == pytest.approx(1.3, abs=0.01)
        assert bandit.beta_param[0] == pytest.approx(1.7, abs=0.01)

    def test_history_recorded(self) -> None:
        """Each update should add an observation to the arm's history."""
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)

        bandit.update(0, ctx, reward=0.9)
        assert len(bandit.history[0]) == 1
        assert bandit.total_pulls[0] == 1
        assert bandit.total_rewards[0] == pytest.approx(0.9)

    def test_posteriors_change_after_update(self) -> None:
        """After enough updates, the Laplace posterior should shift."""
        bandit = ContextualThompsonBandit(
            n_arms=2, context_dim=4, refit_interval=5
        )

        initial_mean = bandit.weight_means[0].copy()
        ctx = np.array([1.0, 0.0, 0.0, 0.0])

        # Feed 10 positive updates to arm 0
        for _ in range(10):
            bandit.update(0, ctx, reward=0.9)

        # After 10 updates (2 refits at interval=5), mean should shift
        assert not np.allclose(bandit.weight_means[0], initial_mean), (
            "Posterior mean did not change after updates."
        )

    def test_invalid_arm_raises(self) -> None:
        """Out-of-range arm index should raise ValueError."""
        bandit = ContextualThompsonBandit(n_arms=2)
        with pytest.raises(ValueError, match="arm must be"):
            bandit.update(5, np.zeros(12), reward=0.5)

    def test_wrong_context_dim_in_update_raises(self) -> None:
        """Context with wrong dimensionality in update should raise."""
        bandit = ContextualThompsonBandit(context_dim=8)
        with pytest.raises(ValueError, match="context of dim"):
            bandit.update(0, np.zeros(3), reward=0.5)


class TestConvergence:
    """Tests for learning behaviour over many interactions."""

    def test_learns_better_arm(self) -> None:
        """After many updates, the bandit should prefer the consistently
        rewarded arm."""
        np.random.seed(123)
        bandit = ContextualThompsonBandit(
            n_arms=2, context_dim=4, refit_interval=10
        )
        ctx = np.array([1.0, 0.5, 0.2, 0.1])

        # Arm 0: consistently high reward; Arm 1: low reward
        for _ in range(50):
            bandit.update(0, ctx, reward=np.random.uniform(0.7, 1.0))
            bandit.update(1, ctx, reward=np.random.uniform(0.0, 0.3))

        # After training, arm 0 should be selected most of the time
        selections = [bandit.select_arm(ctx)[0] for _ in range(100)]
        arm0_frac = selections.count(0) / len(selections)
        assert arm0_frac > 0.6, (
            f"Arm 0 selected only {arm0_frac*100:.0f}% of the time "
            f"(expected > 60%)."
        )


class TestColdStart:
    """Tests for the Beta-Bernoulli cold-start fallback."""

    def test_uses_beta_before_threshold(self) -> None:
        """With fewer than COLD_START_PULLS, should use Beta sampling."""
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)

        # Only 1 pull -- should be in cold-start mode
        bandit.update(0, ctx, reward=1.0)
        assert bandit.total_pulls[0] == 1  # Below threshold (5)

        # Selection should still work (via Beta)
        arm, confidence = bandit.select_arm(ctx)
        assert isinstance(arm, int)
        assert 0 <= arm < 2

    def test_transitions_to_logistic_after_threshold(self) -> None:
        """After COLD_START_PULLS, should use logistic regression sampling."""
        bandit = ContextualThompsonBandit(
            n_arms=2, context_dim=4, refit_interval=5
        )
        ctx = np.random.randn(4)

        # Feed 6 pulls to arm 0 (above cold start threshold of 5)
        for _ in range(6):
            bandit.update(0, ctx, reward=0.8)

        assert bandit.total_pulls[0] == 6
        # Should work without errors
        arm, confidence = bandit.select_arm(ctx)
        assert isinstance(arm, int)


class TestReset:
    """Tests for the reset method."""

    def test_reset_clears_state(self) -> None:
        """After reset, all state should return to the initial prior."""
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)

        for _ in range(20):
            bandit.update(0, ctx, reward=0.9)

        bandit.reset()

        assert bandit.total_pulls == [0, 0]
        assert bandit.total_rewards == [0.0, 0.0]
        assert bandit.alpha == [1.0, 1.0]
        assert bandit.beta_param == [1.0, 1.0]
        assert all(len(h) == 0 for h in bandit.history)
        np.testing.assert_array_equal(
            bandit.weight_means[0], np.zeros(4)
        )


class TestDiagnostics:
    """Tests for get_arm_stats."""

    def test_arm_stats_structure(self) -> None:
        """get_arm_stats should return well-structured diagnostics."""
        bandit = ContextualThompsonBandit(n_arms=2, context_dim=4)
        ctx = np.random.randn(4)
        bandit.update(0, ctx, reward=0.7)

        stats = bandit.get_arm_stats()
        assert stats['n_arms'] == 2
        assert stats['context_dim'] == 4
        assert stats['total_observations'] == 1
        assert len(stats['arms']) == 2
        assert stats['arms'][0]['pulls'] == 1
        assert stats['arms'][0]['mean_reward'] == pytest.approx(0.7)
