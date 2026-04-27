"""Iter 111 — ContextualThompsonBandit convergence + posterior shape."""
from __future__ import annotations

import numpy as np
import pytest

from i3.router.bandit import ContextualThompsonBandit


def test_beta_alpha_grows_unbounded_with_positive_rewards():
    b = ContextualThompsonBandit(n_arms=2, context_dim=4)
    ctx = np.zeros(4)
    a0 = b.alpha[0]
    for _ in range(50):
        b.update(0, ctx, reward=1.0)
    assert b.alpha[0] > a0 + 10  # grew substantially


def test_beta_beta_grows_with_negative_rewards():
    b = ContextualThompsonBandit(n_arms=2, context_dim=4)
    ctx = np.zeros(4)
    bp0 = b.beta_param[0]
    for _ in range(50):
        b.update(0, ctx, reward=0.0)
    assert b.beta_param[0] > bp0 + 10


def test_total_pulls_aggregate_correctly():
    b = ContextualThompsonBandit(n_arms=2, context_dim=4)
    ctx = np.zeros(4)
    for _ in range(7):
        b.update(0, ctx, reward=1.0)
    for _ in range(13):
        b.update(1, ctx, reward=0.0)
    assert b.total_pulls == [7, 13]


def test_total_rewards_aggregate_correctly():
    b = ContextualThompsonBandit(n_arms=2, context_dim=4)
    ctx = np.zeros(4)
    for r in [0.5, 0.7, 0.3]:
        b.update(0, ctx, reward=r)
    assert b.total_rewards[0] == pytest.approx(1.5, abs=0.01)


def test_select_arm_does_not_mutate_arm_state(tmp_path):
    """Two select_arm() calls without an intervening update() should
    leave the posterior parameters identical (sampling is read-only)."""
    b = ContextualThompsonBandit(n_arms=2, context_dim=4)
    ctx = np.ones(4) * 0.5
    means_before = [m.copy() for m in b.weight_means]
    pulls_before = list(b.total_pulls)
    for _ in range(10):
        b.select_arm(ctx)
    for m_b, m_a in zip(means_before, b.weight_means):
        assert np.allclose(m_b, m_a)
    assert b.total_pulls == pulls_before


def test_select_arm_returns_balanced_arms_under_zero_evidence():
    """With no evidence, both arms should have similar selection
    probability (within statistical noise)."""
    b = ContextualThompsonBandit(n_arms=2, context_dim=4)
    ctx = np.zeros(4)
    counts = [0, 0]
    for _ in range(500):
        arm, _ = b.select_arm(ctx)
        counts[arm] += 1
    # Either arm should be picked between 30% and 70% of the time.
    ratio = counts[0] / sum(counts)
    assert 0.30 <= ratio <= 0.70, f"unbalanced cold-start: {counts}"


def test_history_capped():
    """Iter-51 audit added a deque(maxlen=...) so per-arm history
    can't grow unbounded.  Push way more than the cap and check."""
    b = ContextualThompsonBandit(n_arms=1, context_dim=4)
    ctx = np.zeros(4)
    for _ in range(50_000):
        b.update(0, ctx, reward=0.5)
    # The cap is bounded — history shouldn't be 50k.
    assert len(b.history[0]) < 50_000
