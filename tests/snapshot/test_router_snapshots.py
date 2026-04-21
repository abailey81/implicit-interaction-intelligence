"""Snapshot tests for the router's public outputs.

We pin:
    * The exact :class:`RoutingContext.to_vector()` layout for three
      canonical scenarios (warmup, steady-state, high-sensitivity).
    * The bandit's confidence dict shape after a deterministic sequence
      of seeded updates.

Run ``pytest --snapshot-update tests/snapshot/test_router_snapshots.py`` to
refresh the committed snapshots after a deliberate change.
"""

from __future__ import annotations

import pytest


syrupy = pytest.importorskip("syrupy")


@pytest.fixture(autouse=True)
def _seed_rngs():
    import random

    import numpy as np

    random.seed(42)
    np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────
#  RoutingContext snapshot
# ─────────────────────────────────────────────────────────────────────────


def _round_list(xs, ndigits=6):
    return [round(float(x), ndigits) for x in xs]


class TestRoutingContextSnapshots:
    def test_warmup_context(self, snapshot) -> None:
        from i3.router.types import RoutingContext

        ctx = RoutingContext(
            user_state_compressed=[0.0, 0.0, 0.0, 0.0],
            query_complexity=0.2,
            topic_sensitivity=0.0,
            user_patience=0.5,
            session_progress=0.05,
            baseline_established=False,
            previous_route=-1,
            previous_engagement=0.0,
            time_of_day=0.5,
            message_count=1,
            cloud_latency_est=0.3,
            slm_confidence=0.5,
        )
        assert _round_list(ctx.to_vector()) == snapshot

    def test_steady_state_context(self, snapshot) -> None:
        from i3.router.types import RoutingContext

        ctx = RoutingContext(
            user_state_compressed=[0.1, -0.2, 0.3, -0.4],
            query_complexity=0.6,
            topic_sensitivity=0.2,
            user_patience=0.7,
            session_progress=0.5,
            baseline_established=True,
            previous_route=1,
            previous_engagement=0.8,
            time_of_day=0.25,
            message_count=25,
            cloud_latency_est=0.4,
            slm_confidence=0.6,
        )
        assert _round_list(ctx.to_vector()) == snapshot

    def test_high_sensitivity_context(self, snapshot) -> None:
        from i3.router.types import RoutingContext

        ctx = RoutingContext(
            user_state_compressed=[0.05, 0.05, 0.05, 0.05],
            query_complexity=0.4,
            topic_sensitivity=0.95,  # personal / sensitive topic
            user_patience=0.5,
            session_progress=0.3,
            baseline_established=True,
            previous_route=0,
            previous_engagement=0.5,
            time_of_day=0.75,
            message_count=12,
            cloud_latency_est=0.3,
            slm_confidence=0.7,
        )
        assert _round_list(ctx.to_vector()) == snapshot


# ─────────────────────────────────────────────────────────────────────────
#  Bandit confidence / posterior snapshot
# ─────────────────────────────────────────────────────────────────────────


class TestBanditSnapshots:
    def test_initial_confidence(self, snapshot) -> None:
        """With no data, both arms should yield a confidence near 0.5."""
        import numpy as np

        from i3.router.bandit import ContextualThompsonBandit

        np.random.seed(2024)
        bandit = ContextualThompsonBandit(
            n_arms=2, context_dim=4,
            prior_precision=1.0, exploration_bonus=0.0,
        )
        ctx = np.zeros(4, dtype=np.float64)
        _, confidence = bandit.select_arm(ctx)
        rounded = {k: round(v, 3) for k, v in confidence.items()}
        assert rounded == snapshot

    def test_posterior_shape_after_updates(self, snapshot) -> None:
        """After a fixed sequence of updates the posterior statistics are
        deterministic to 6 decimal places."""
        import numpy as np

        from i3.router.bandit import ContextualThompsonBandit

        rng = np.random.RandomState(7)
        bandit = ContextualThompsonBandit(
            n_arms=2, context_dim=4,
            prior_precision=1.0, exploration_bonus=0.0,
            refit_interval=3,
        )
        # Deterministic biased feedback
        for _ in range(12):
            ctx = rng.randn(4)
            bandit.update(0, ctx, reward=0.9)
            bandit.update(1, ctx, reward=0.1)

        stats = bandit.get_arm_stats()
        # Normalise floats before snapshotting.
        rounded = {
            "n_arms": stats["n_arms"],
            "context_dim": stats["context_dim"],
            "total_observations": stats["total_observations"],
            "arms": [
                {
                    "arm": a["arm"],
                    "pulls": a["pulls"],
                    "mean_reward": round(a["mean_reward"], 6),
                    "beta_alpha": round(a["beta_alpha"], 6),
                    "beta_beta": round(a["beta_beta"], 6),
                }
                for a in stats["arms"]
            ],
        }
        assert rounded == snapshot
