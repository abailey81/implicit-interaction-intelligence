"""Hypothesis property tests for the contextual Thompson bandit.

Invariants:
    * **Beta-posterior monotonicity** — after a success, ``alpha`` grows
      monotonically; after a failure, ``beta`` grows monotonically.
    * **Arm symmetry** — starting from the prior, two identical update
      sequences against two different arms yield identical posterior
      statistics.
    * **Context validation** — passing a wrongly-shaped context raises.
    * **Convergence** — under deterministic rewards (arm 0 always wins)
      the bandit eventually preferentially picks arm 0.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from i3.router.bandit import ContextualThompsonBandit


_CTX_DIM = 4


def _bandit(context_dim: int = _CTX_DIM) -> ContextualThompsonBandit:
    return ContextualThompsonBandit(
        n_arms=2,
        context_dim=context_dim,
        prior_precision=1.0,
        exploration_bonus=0.0,
        refit_interval=5,
    )


_CTX_STRATEGY = arrays(
    dtype=np.float64,
    shape=(_CTX_DIM,),
    elements=st.floats(
        min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False,
    ),
)

_REWARD_STRATEGY = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False,
)


# ─────────────────────────────────────────────────────────────────────────
#  Posterior monotonicity
# ─────────────────────────────────────────────────────────────────────────


class TestBetaPosteriorMonotonicity:
    @given(ctx=_CTX_STRATEGY, reward=_REWARD_STRATEGY)
    @settings(max_examples=100, deadline=None)
    def test_alpha_plus_beta_monotone(
        self, ctx: np.ndarray, reward: float
    ) -> None:
        """The pseudo-count ``alpha + beta`` grows by exactly 1 per update."""
        bandit = _bandit()
        arm = 0
        before = bandit.alpha[arm] + bandit.beta_param[arm]
        bandit.update(arm, ctx, reward=reward)
        after = bandit.alpha[arm] + bandit.beta_param[arm]
        assert math.isclose(after - before, 1.0, abs_tol=1e-9)

    @given(
        rewards=st.lists(_REWARD_STRATEGY, min_size=1, max_size=20),
        ctx=_CTX_STRATEGY,
    )
    @settings(max_examples=50, deadline=None)
    def test_alpha_monotone_nondecreasing(
        self, rewards: list[float], ctx: np.ndarray
    ) -> None:
        """``alpha`` is non-decreasing over any reward sequence in [0,1]."""
        bandit = _bandit()
        arm = 0
        prev_alpha = bandit.alpha[arm]
        for r in rewards:
            bandit.update(arm, ctx, reward=r)
            assert bandit.alpha[arm] >= prev_alpha - 1e-12
            prev_alpha = bandit.alpha[arm]


# ─────────────────────────────────────────────────────────────────────────
#  Arm symmetry under exchangeable data
# ─────────────────────────────────────────────────────────────────────────


class TestArmSymmetry:
    @given(
        ctxs=st.lists(_CTX_STRATEGY, min_size=1, max_size=10),
        rewards=st.lists(_REWARD_STRATEGY, min_size=1, max_size=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_identical_sequences_give_identical_posterior(
        self, ctxs: list[np.ndarray], rewards: list[float]
    ) -> None:
        """Arm 0 and arm 1 fed the SAME data have the SAME sufficient stats."""
        bandit = _bandit()
        n = min(len(ctxs), len(rewards))
        for ctx, reward in zip(ctxs[:n], rewards[:n]):
            bandit.update(0, ctx, reward=reward)
            bandit.update(1, ctx, reward=reward)

        assert math.isclose(bandit.alpha[0], bandit.alpha[1], abs_tol=1e-9)
        assert math.isclose(
            bandit.beta_param[0], bandit.beta_param[1], abs_tol=1e-9
        )
        assert bandit.total_pulls[0] == bandit.total_pulls[1]
        assert math.isclose(
            bandit.total_rewards[0], bandit.total_rewards[1], abs_tol=1e-9
        )


# ─────────────────────────────────────────────────────────────────────────
#  Context validation
# ─────────────────────────────────────────────────────────────────────────


class TestContextValidation:
    @given(
        bad_dim=st.integers(min_value=1, max_value=20).filter(
            lambda d: d != _CTX_DIM
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_wrong_context_dim_raises(self, bad_dim: int) -> None:
        bandit = _bandit()
        bad_ctx = np.zeros(bad_dim, dtype=np.float64)
        with pytest.raises(ValueError):
            bandit.select_arm(bad_ctx)
        with pytest.raises(ValueError):
            bandit.update(0, bad_ctx, reward=1.0)

    @given(
        arr=arrays(
            dtype=np.float64,
            shape=(_CTX_DIM,),
            elements=st.one_of(
                st.just(float("nan")),
                st.just(float("inf")),
                st.just(-float("inf")),
            ),
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_nonfinite_context_does_not_crash(self, arr: np.ndarray) -> None:
        """NaN / Inf contexts are sanitised rather than propagated."""
        bandit = _bandit()
        arm, confidence = bandit.select_arm(arr)
        assert arm in (0, 1)
        assert sum(confidence.values()) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────
#  Convergence (weaker — we only require the posterior mean to move)
# ─────────────────────────────────────────────────────────────────────────


class TestBanditConvergence:
    def test_winning_arm_dominates_posterior(self) -> None:
        """After biased feedback (arm 0 wins, arm 1 loses) the Beta
        posterior mean of arm 0 exceeds that of arm 1."""
        rng = np.random.RandomState(0)
        bandit = _bandit()
        for _ in range(60):
            ctx = rng.randn(_CTX_DIM)
            bandit.update(0, ctx, reward=1.0)
            bandit.update(1, ctx, reward=0.0)
        # Beta(1+n, 1) mean > Beta(1, 1+n) mean
        mean0 = bandit.alpha[0] / (bandit.alpha[0] + bandit.beta_param[0])
        mean1 = bandit.alpha[1] / (bandit.alpha[1] + bandit.beta_param[1])
        assert mean0 > 0.9, mean0
        assert mean1 < 0.1, mean1
