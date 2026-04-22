"""Unit tests for :mod:`i3.router.adaptive_compute`.

Covers the third-arm escalation logic, privacy-override dominance,
pass-through behaviour on high-confidence contexts, and the extended
:class:`AdaptiveRoutingDecision` shape.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from i3.config import load_config
from i3.router.adaptive_compute import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    AdaptiveComputeRouter,
    AdaptiveRoutingDecision,
)
from i3.router.router import IntelligentRouter
from i3.router.types import RouteChoice, RoutingContext, RoutingDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Default config for the router subsystem."""
    return load_config("configs/default.yaml")


@pytest.fixture
def base_router(config) -> IntelligentRouter:
    """A stock two-arm :class:`IntelligentRouter`."""
    return IntelligentRouter(config)


@pytest.fixture
def adaptive(base_router: IntelligentRouter) -> AdaptiveComputeRouter:
    """A :class:`AdaptiveComputeRouter` wrapping the base router."""
    return AdaptiveComputeRouter(base_router=base_router)


def _make_context(
    *,
    topic_sensitivity: float = 0.0,
    slm_confidence: float = 0.9,
    query_complexity: float = 0.3,
) -> RoutingContext:
    """Helper: build a :class:`RoutingContext` with sensible defaults."""
    return RoutingContext(
        user_state_compressed=[0.1, -0.1, 0.2, 0.0],
        query_complexity=query_complexity,
        topic_sensitivity=topic_sensitivity,
        user_patience=0.7,
        session_progress=0.2,
        baseline_established=True,
        previous_route=0,
        previous_engagement=0.8,
        time_of_day=0.5,
        message_count=3,
        cloud_latency_est=0.3,
        slm_confidence=slm_confidence,
    )


def _force_base_decision(
    route: RouteChoice,
    *,
    ctx: RoutingContext,
    was_privacy_override: bool = False,
):
    """Build a canned :class:`RoutingDecision`. Used with ``patch.object``."""
    return RoutingDecision(
        chosen_route=route,
        confidence={
            RouteChoice.LOCAL_SLM.value: 0.8 if route == RouteChoice.LOCAL_SLM else 0.2,
            RouteChoice.CLOUD_LLM.value: 0.2 if route == RouteChoice.LOCAL_SLM else 0.8,
        },
        context=ctx,
        was_privacy_override=was_privacy_override,
        reasoning=(
            f"canned base decision: {route.value}, "
            f"privacy_override={was_privacy_override}"
        ),
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for constructor validation."""

    def test_extended_context_dim(
        self, adaptive: AdaptiveComputeRouter, base_router: IntelligentRouter
    ) -> None:
        """Extended context is base + 2."""
        assert adaptive.extended_context_dim == base_router.bandit.context_dim + 2

    def test_invalid_threshold_raises(
        self, base_router: IntelligentRouter
    ) -> None:
        """Confidence threshold outside [0, 1] must raise."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            AdaptiveComputeRouter(
                base_router=base_router, confidence_threshold=1.5
            )
        with pytest.raises(ValueError, match="confidence_threshold"):
            AdaptiveComputeRouter(
                base_router=base_router, confidence_threshold=-0.1
            )

    def test_invalid_multipliers_raise(
        self, base_router: IntelligentRouter
    ) -> None:
        """Non-positive reflect multipliers must raise."""
        with pytest.raises(ValueError, match="max_new_tokens_multiplier"):
            AdaptiveComputeRouter(
                base_router=base_router,
                reflect_max_new_tokens_multiplier=0.0,
            )
        with pytest.raises(ValueError, match="top_k_multiplier"):
            AdaptiveComputeRouter(
                base_router=base_router, reflect_top_k_multiplier=-1.0
            )
        with pytest.raises(ValueError, match="extra_sampling_rounds"):
            AdaptiveComputeRouter(
                base_router=base_router, reflect_extra_sampling_rounds=-1
            )

    def test_three_arms_in_adaptive_bandit(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """The adaptive layer's dedicated bandit must have exactly three arms."""
        assert adaptive._adaptive_bandit.n_arms == 3


# ---------------------------------------------------------------------------
# 2. Escalation to local_reflect
# ---------------------------------------------------------------------------


class TestEscalation:
    """Tests for the third-arm escalation policy."""

    def test_low_confidence_triggers_escalation(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Low SLM confidence on a local_slm decision triggers escalation."""
        ctx = _make_context(slm_confidence=0.3, topic_sensitivity=0.0)
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)

        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route(
                "some moderately hard query", ctx,
                prior_query_difficulty_estimate=0.7,
            )

        assert isinstance(decision, AdaptiveRoutingDecision)
        assert decision.compute_budget == "heavy"
        assert decision.escalated is True
        assert decision.chosen_route == RouteChoice.LOCAL_SLM
        assert "max_new_tokens_multiplier" in decision.reflect_params

    def test_exactly_at_threshold_does_not_escalate(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Strict inequality: at exactly the threshold, no escalation."""
        ctx = _make_context(slm_confidence=DEFAULT_CONFIDENCE_THRESHOLD)
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)

        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)

        assert decision.escalated is False
        assert decision.compute_budget == "standard"

    def test_high_confidence_preserves_base(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """High-confidence local_slm stays local_slm with standard budget."""
        ctx = _make_context(slm_confidence=0.95)
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)

        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)

        assert decision.escalated is False
        assert decision.compute_budget == "standard"
        assert decision.chosen_route == RouteChoice.LOCAL_SLM

    def test_cloud_winner_does_not_escalate(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Cloud-winning decisions are never escalated to local_reflect."""
        ctx = _make_context(slm_confidence=0.2)  # low, but cloud won
        base = _force_base_decision(RouteChoice.CLOUD_LLM, ctx=ctx)

        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)

        assert decision.escalated is False
        assert decision.compute_budget == "standard"
        assert decision.chosen_route == RouteChoice.CLOUD_LLM


# ---------------------------------------------------------------------------
# 3. Privacy override dominance
# ---------------------------------------------------------------------------


class TestPrivacyOverride:
    """Privacy override must strictly dominate the adaptive layer."""

    def test_privacy_override_suppresses_escalation(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """A privacy override with low confidence does NOT escalate."""
        ctx = _make_context(
            topic_sensitivity=0.9, slm_confidence=0.2
        )
        base = _force_base_decision(
            RouteChoice.LOCAL_SLM, ctx=ctx, was_privacy_override=True
        )

        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("sensitive query", ctx)

        assert decision.was_privacy_override is True
        assert decision.escalated is False
        assert decision.compute_budget == "standard"
        assert decision.chosen_route == RouteChoice.LOCAL_SLM


# ---------------------------------------------------------------------------
# 4. RoutingDecision shape
# ---------------------------------------------------------------------------


class TestDecisionShape:
    """Tests for the :class:`AdaptiveRoutingDecision` surface."""

    def test_decision_has_compute_budget(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """``compute_budget`` field exists and is one of the 3 literals."""
        ctx = _make_context()
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)
        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)
        assert decision.compute_budget in ("light", "standard", "heavy")

    def test_decision_delegates_fields(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Delegation properties return the base decision's fields."""
        ctx = _make_context()
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)
        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)

        assert decision.chosen_route == base.chosen_route
        assert decision.confidence == base.confidence
        assert decision.context is base.context
        assert decision.was_privacy_override == base.was_privacy_override
        # combined reasoning contains both layers
        assert base.reasoning in decision.reasoning

    def test_reflect_params_shape_on_escalation(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """When escalated, reflect_params contains the expected keys."""
        ctx = _make_context(slm_confidence=0.2)
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)
        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)

        assert decision.escalated is True
        assert set(decision.reflect_params.keys()) == {
            "max_new_tokens_multiplier",
            "top_k_multiplier",
            "extra_sampling_rounds",
        }
        assert decision.reflect_params["max_new_tokens_multiplier"] > 1.0
        assert decision.reflect_params["top_k_multiplier"] > 1.0
        assert decision.reflect_params["extra_sampling_rounds"] >= 0


# ---------------------------------------------------------------------------
# 5. Reward update
# ---------------------------------------------------------------------------


class TestRewardUpdate:
    """The update path feeds both bandits."""

    def test_update_feeds_both_bandits(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """A reward update must increment pull counts in base + adaptive bandits."""
        ctx = _make_context(slm_confidence=0.2)
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)

        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route(
                "q", ctx, prior_query_difficulty_estimate=0.5
            )

        # pre-update pull counts
        base_pre = sum(adaptive.base_router.bandit.total_pulls)
        adapt_pre = sum(adaptive._adaptive_bandit.total_pulls)

        adaptive.update_reward(
            decision, engagement=0.75,
            prior_query_difficulty_estimate=0.5,
        )

        base_post = sum(adaptive.base_router.bandit.total_pulls)
        adapt_post = sum(adaptive._adaptive_bandit.total_pulls)
        assert base_post == base_pre + 1
        assert adapt_post == adapt_pre + 1

    def test_update_clips_invalid_engagement(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Non-finite engagement must be clamped, not crash."""
        ctx = _make_context()
        base = _force_base_decision(RouteChoice.LOCAL_SLM, ctx=ctx)
        with patch.object(adaptive.base_router, "route", return_value=base):
            decision = adaptive.route("q", ctx)
        # Should not raise
        adaptive.update_reward(decision, engagement=float("nan"))


# ---------------------------------------------------------------------------
# 6. Stats
# ---------------------------------------------------------------------------


class TestStats:
    """Diagnostics surface."""

    def test_stats_contains_adaptive_and_base(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """``get_stats`` must surface both bandits' summaries."""
        stats = adaptive.get_stats()
        assert "adaptive_bandit" in stats
        assert "bandit" in stats  # from the base router
        assert stats["adaptive_bandit"]["n_arms"] == 3
        assert stats["extended_context_dim"] == adaptive.extended_context_dim
        assert stats["confidence_threshold"] == pytest.approx(
            adaptive.confidence_threshold
        )

    def test_stats_arm_names(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Each arm in the adaptive stats carries a human-readable ``route``."""
        stats = adaptive.get_stats()
        routes = {a["route"] for a in stats["adaptive_bandit"]["arms"]}
        assert routes == {"local_slm", "cloud_llm", "local_reflect"}


# ---------------------------------------------------------------------------
# 7. Extended context plumbing
# ---------------------------------------------------------------------------


class TestExtendedContext:
    """The extended context vector has the right shape and contents."""

    def test_extended_context_shape(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        ctx = _make_context()
        ext = adaptive._extended_context(
            ctx=ctx,
            prior_query_difficulty_estimate=0.7,
            prior_slm_self_confidence=0.4,
        )
        assert ext.shape == (adaptive.extended_context_dim,)
        # Last two entries are the new features.
        assert ext[-2] == pytest.approx(0.7)
        assert ext[-1] == pytest.approx(0.4)

    def test_extended_context_clips_tail(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """Out-of-range tail values are clipped to [0, 1]."""
        ctx = _make_context()
        ext = adaptive._extended_context(
            ctx=ctx,
            prior_query_difficulty_estimate=5.0,
            prior_slm_self_confidence=-1.0,
        )
        assert ext[-2] == 1.0
        assert ext[-1] == 0.0

    def test_extended_context_defaults_to_ctx_confidence(
        self, adaptive: AdaptiveComputeRouter
    ) -> None:
        """If no prior confidence is supplied, we use ``ctx.slm_confidence``."""
        ctx = _make_context(slm_confidence=0.33)
        ext = adaptive._extended_context(
            ctx=ctx,
            prior_query_difficulty_estimate=0.5,
            prior_slm_self_confidence=None,
        )
        assert ext[-1] == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# 8. Base router immutability
# ---------------------------------------------------------------------------


class TestBaseRouterUntouched:
    """The wrapped base router's behaviour must not be altered."""

    def test_base_router_not_mutated(
        self, base_router: IntelligentRouter
    ) -> None:
        """Wrapping does not change the base router's attribute values."""
        before_override = base_router.privacy_override
        before_floor = base_router.min_cloud_complexity
        before_threshold = base_router.sensitivity_threshold

        _ = AdaptiveComputeRouter(base_router=base_router)

        assert base_router.privacy_override == before_override
        assert base_router.min_cloud_complexity == before_floor
        assert base_router.sensitivity_threshold == before_threshold

    def test_base_bandit_independent_of_adaptive_bandit(
        self, base_router: IntelligentRouter,
    ) -> None:
        """The two bandits are distinct Python instances."""
        wrapped = AdaptiveComputeRouter(base_router=base_router)
        assert wrapped._adaptive_bandit is not base_router.bandit
        # Different arm counts prove they are not the same object.
        assert wrapped._adaptive_bandit.n_arms != base_router.bandit.n_arms
