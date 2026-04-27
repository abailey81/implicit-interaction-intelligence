"""Iter 89 — ContextualThompsonBandit invariant tests.

Pins the invariants of the LinUCB-style bandit + Beta-Bernoulli
fallback that drives edge/cloud routing.
"""
from __future__ import annotations

import numpy as np
import pytest

from i3.router.bandit import ContextualThompsonBandit


@pytest.fixture
def bandit():
    return ContextualThompsonBandit(n_arms=2, context_dim=12)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_n_arms_must_be_positive():
    with pytest.raises(ValueError):
        ContextualThompsonBandit(n_arms=0)


def test_context_dim_must_be_positive():
    with pytest.raises(ValueError):
        ContextualThompsonBandit(context_dim=0)


def test_prior_precision_must_be_positive():
    with pytest.raises(ValueError):
        ContextualThompsonBandit(prior_precision=0)


# ---------------------------------------------------------------------------
# select_arm + update behaviour
# ---------------------------------------------------------------------------

def test_select_arm_returns_valid_index(bandit):
    ctx = np.zeros(12)
    arm, conf = bandit.select_arm(ctx)
    assert 0 <= arm < bandit.n_arms
    # conf can be a float or a dict {arm_<i>: float}; both shapes valid.
    if isinstance(conf, dict):
        for v in conf.values():
            assert 0.0 <= v <= 1.0
    else:
        assert 0.0 <= float(conf) <= 1.0


def test_update_increments_total_pulls(bandit):
    ctx = np.zeros(12)
    bandit.update(0, ctx, reward=1.0)
    assert bandit.total_pulls[0] == 1
    assert bandit.total_pulls[1] == 0


def test_update_accumulates_rewards(bandit):
    ctx = np.ones(12)
    for _ in range(5):
        bandit.update(0, ctx, reward=0.5)
    assert bandit.total_pulls[0] == 5
    assert bandit.total_rewards[0] == pytest.approx(2.5, abs=0.01)


def test_select_arm_after_clear_evidence_picks_winner(bandit):
    ctx = np.ones(12)
    # Make arm 0 reliably good, arm 1 reliably bad.
    for _ in range(40):
        bandit.update(0, ctx, reward=1.0)
        bandit.update(1, ctx, reward=0.0)
    # Now sample many times and check the winner is preferred.
    counts = [0, 0]
    for _ in range(50):
        arm, _ = bandit.select_arm(ctx)
        counts[arm] += 1
    # Arm 0 should be picked the majority of the time.
    assert counts[0] > counts[1], \
        f"after evidence, arm 0 picked {counts[0]} vs arm 1 {counts[1]}"


def test_beta_alpha_increases_on_positive_reward(bandit):
    ctx = np.zeros(12)
    a0 = bandit.alpha[0]
    bandit.update(0, ctx, reward=1.0)
    assert bandit.alpha[0] > a0


def test_beta_beta_increases_on_negative_reward(bandit):
    ctx = np.zeros(12)
    b0 = bandit.beta_param[0]
    bandit.update(0, ctx, reward=0.0)
    assert bandit.beta_param[0] > b0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_select_arm_handles_zero_context(bandit):
    arm, conf = bandit.select_arm(np.zeros(12))
    assert 0 <= arm < bandit.n_arms
    assert isinstance(conf, (float, dict))


def test_select_arm_handles_extreme_context(bandit):
    """Extreme context values should not produce NaN in the conf
    payload."""
    arm, conf = bandit.select_arm(np.full(12, 1e6))
    if isinstance(conf, dict):
        for v in conf.values():
            assert not (np.isnan(v) or np.isinf(v))
    else:
        assert not (np.isnan(conf) or np.isinf(conf))


def test_update_with_extreme_reward(bandit):
    ctx = np.zeros(12)
    # Reward outside [0,1] is unusual; bandit should at least not crash.
    bandit.update(0, ctx, reward=2.0)
    assert bandit.total_pulls[0] == 1


def test_update_invalid_arm_raises_or_ignores(bandit):
    """Out-of-range arm should either raise or no-op, not silently
    corrupt arm 0."""
    pulls0_before = bandit.total_pulls[0]
    try:
        bandit.update(99, np.zeros(12), reward=1.0)
    except (ValueError, IndexError):
        pass  # acceptable
    # Either way, arm 0 must not have moved.
    assert bandit.total_pulls[0] == pulls0_before
