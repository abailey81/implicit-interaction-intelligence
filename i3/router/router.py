"""High-level Intelligent Router combining the contextual bandit with
query analysis components.

The :class:`IntelligentRouter` is the main entry point for the routing
subsystem.  It:

1. Analyses the query for topic sensitivity (privacy).
2. Estimates query complexity.
3. Optionally overrides the bandit when privacy demands local processing.
4. Delegates the arm-selection decision to the contextual Thompson
   Sampling bandit.
5. Returns a fully documented :class:`RoutingDecision`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from i3.config import Config
from i3.router.bandit import ContextualThompsonBandit
from i3.router.complexity import QueryComplexityEstimator
from i3.router.sensitivity import TopicSensitivityDetector
from i3.router.types import RouteChoice, RoutingContext, RoutingDecision

logger = logging.getLogger(__name__)

# Arm index mapping (must match RouteChoice ordering)
_ARM_SLM = 0
_ARM_CLOUD = 1

_ROUTE_BY_ARM: dict[int, RouteChoice] = {
    _ARM_SLM: RouteChoice.LOCAL_SLM,
    _ARM_CLOUD: RouteChoice.CLOUD_LLM,
}


class IntelligentRouter:
    """Orchestrates the routing decision by combining privacy detection,
    complexity estimation, and contextual Thompson Sampling.

    The router supports three routing modes:

    - **Privacy override**: If the query is sensitive and
      ``privacy_override`` is enabled, the query is always routed to
      the local SLM regardless of the bandit's preference.
    - **Complexity floor**: If the query complexity is below
      ``min_cloud_complexity``, the cloud arm is penalised to favour
      local processing for simple queries.
    - **Bandit selection**: For all other cases, the contextual Thompson
      Sampling bandit selects the optimal arm.

    Args:
        config: The global I3 configuration.  The router reads settings
            from ``config.router``.

    Example::

        from i3.config import load_config
        from i3.router import IntelligentRouter, RoutingContext

        cfg = load_config("configs/default.yaml")
        router = IntelligentRouter(cfg)

        ctx = RoutingContext(
            user_state_compressed=[0.1, -0.2, 0.3, 0.0],
            query_complexity=0.6,
            topic_sensitivity=0.1,
            user_patience=0.7,
            session_progress=0.3,
            baseline_established=True,
            previous_route=0,
            previous_engagement=0.8,
            time_of_day=0.5,
            message_count=5,
            cloud_latency_est=0.3,
            slm_confidence=0.6,
        )

        decision = router.route("Explain the transformer architecture", ctx=ctx)
        print(decision.chosen_route, decision.reasoning)
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # SEC (H-9, 2026-04-23 audit): previously this constructor
        # passed ``config.router.prior_alpha`` (a Beta prior) to the
        # bandit's ``prior_precision`` slot (a Gaussian weight
        # precision).  They are distinct statistical quantities.  The
        # config now carries both fields explicitly so operators can
        # tune them independently.
        self.bandit = ContextualThompsonBandit(
            n_arms=2,
            context_dim=config.router.context_dim,
            prior_precision=config.router.prior_precision,
            exploration_bonus=config.router.exploration_bonus,
        )
        self.complexity_estimator = QueryComplexityEstimator()
        self.sensitivity_detector = TopicSensitivityDetector()

        # Router behaviour flags
        self.privacy_override: bool = config.router.privacy_override
        self.min_cloud_complexity: float = config.router.min_cloud_complexity
        self.sensitivity_threshold: float = 0.5  # Sensitivity above this triggers override

    def route(
        self,
        text: str,
        ctx: RoutingContext,
        *,
        user_state: np.ndarray | None = None,
    ) -> RoutingDecision:
        """Make a routing decision for the given query.

        Args:
            text: The raw user query text.
            ctx: Pre-built routing context.  The ``query_complexity`` and
                ``topic_sensitivity`` fields may be overwritten by the
                router's own estimators if they are set to their defaults.
            user_state: Optional full 64-dim user state.  Not used directly
                by the router (the compressed version is in ``ctx``), but
                available for future extensions.

        Returns:
            A :class:`RoutingDecision` describing the chosen route,
            confidence levels, and reasoning.
        """
        # -----------------------------------------------------------------
        # Step 1: Analyse query for sensitivity and complexity
        # -----------------------------------------------------------------
        sensitivity = self.sensitivity_detector.detect(text)
        complexity = self.complexity_estimator.estimate(text)

        logger.debug(
            "Query analysis: sensitivity=%.3f, complexity=%.3f",
            sensitivity,
            complexity,
        )

        # -----------------------------------------------------------------
        # Step 2: Privacy override check
        # -----------------------------------------------------------------
        if self.privacy_override and sensitivity > self.sensitivity_threshold:
            reasoning = (
                f"Privacy override: sensitivity={sensitivity:.2f} exceeds "
                f"threshold={self.sensitivity_threshold:.2f}. "
                f"Routing to local SLM to protect user privacy."
            )
            logger.info(reasoning)
            return RoutingDecision(
                chosen_route=RouteChoice.LOCAL_SLM,
                confidence={
                    RouteChoice.LOCAL_SLM.value: 1.0,
                    RouteChoice.CLOUD_LLM.value: 0.0,
                },
                context=ctx,
                was_privacy_override=True,
                reasoning=reasoning,
            )

        # -----------------------------------------------------------------
        # Step 3: Contextual bandit selection
        # -----------------------------------------------------------------
        context_vec = ctx.to_vector()
        arm, raw_confidence = self.bandit.select_arm(context_vec)
        chosen_route = _ROUTE_BY_ARM[arm]

        # Build human-readable confidence dict with route names
        confidence = {
            RouteChoice.LOCAL_SLM.value: raw_confidence.get("arm_0", 0.5),
            RouteChoice.CLOUD_LLM.value: raw_confidence.get("arm_1", 0.5),
        }

        # -----------------------------------------------------------------
        # Step 4: Complexity floor -- prefer SLM for very simple queries
        # -----------------------------------------------------------------
        if (
            chosen_route == RouteChoice.CLOUD_LLM
            and complexity < self.min_cloud_complexity
        ):
            reasoning = (
                f"Complexity floor: complexity={complexity:.2f} < "
                f"min_cloud_complexity={self.min_cloud_complexity:.2f}. "
                f"Bandit preferred cloud (conf={confidence[RouteChoice.CLOUD_LLM.value]:.2f}) "
                f"but overriding to local SLM for efficiency."
            )
            logger.info(reasoning)
            return RoutingDecision(
                chosen_route=RouteChoice.LOCAL_SLM,
                confidence=confidence,
                context=ctx,
                was_privacy_override=False,
                reasoning=reasoning,
            )

        # -----------------------------------------------------------------
        # Step 5: Standard bandit decision
        # -----------------------------------------------------------------
        reasoning = (
            f"Bandit selected {chosen_route.value} "
            f"(confidence: SLM={confidence[RouteChoice.LOCAL_SLM.value]:.2f}, "
            f"Cloud={confidence[RouteChoice.CLOUD_LLM.value]:.2f}). "
            f"Query complexity={complexity:.2f}, sensitivity={sensitivity:.2f}."
        )
        logger.debug(reasoning)

        return RoutingDecision(
            chosen_route=chosen_route,
            confidence=confidence,
            context=ctx,
            was_privacy_override=False,
            reasoning=reasoning,
        )

    def update_reward(
        self,
        decision: RoutingDecision,
        engagement: float,
    ) -> None:
        """Feed the observed engagement reward back to the bandit.

        This closes the feedback loop: after the chosen model generates a
        response and the user interacts with it, the engagement signal
        (derived from implicit interaction features) is used to update
        the bandit's posterior.

        Args:
            decision: The original routing decision (contains context and
                chosen route).
            engagement: Observed engagement reward in [0, 1].  Higher
                values indicate the user was satisfied with the response.

        Note:
            Privacy-overridden decisions are still updated so the bandit
            can learn that sensitive contexts tend to yield high engagement
            when handled locally.
        """
        arm = _ARM_CLOUD if decision.chosen_route == RouteChoice.CLOUD_LLM else _ARM_SLM
        context_vec = decision.context.to_vector()

        # SEC: Clip and sanitize engagement before forwarding to the bandit.
        # The bandit also clips, but doing it here ensures the logger reflects
        # the value actually used.
        if not np.isfinite(engagement):
            logger.warning(
                "Non-finite engagement (%r) — replacing with 0.0", engagement
            )
            engagement = 0.0
        engagement = float(np.clip(engagement, 0.0, 1.0))

        self.bandit.update(arm, context_vec, engagement)
        logger.debug(
            "Reward update: arm=%d (%s), engagement=%.3f",
            arm,
            decision.chosen_route.value,
            engagement,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return combined statistics from the bandit and router.

        Returns:
            A dict with bandit arm stats and router configuration.
        """
        bandit_stats = self.bandit.get_arm_stats()

        # Rename arms with human-readable labels
        for arm_stat in bandit_stats["arms"]:
            arm_idx = arm_stat["arm"]
            arm_stat["route"] = _ROUTE_BY_ARM[arm_idx].value

        return {
            "bandit": bandit_stats,
            "router_config": {
                "privacy_override": self.privacy_override,
                "min_cloud_complexity": self.min_cloud_complexity,
                "sensitivity_threshold": self.sensitivity_threshold,
                "context_dim": self.config.router.context_dim,
            },
        }

    def save_state(self, path: str) -> None:
        """Persist the bandit state to disk.

        Args:
            path: Filesystem path for the JSON state file.
        """
        self.bandit.save_state(path)

    def load_state(self, path: str) -> None:
        """Restore the bandit state from disk.

        Args:
            path: Filesystem path to the JSON state file.
        """
        self.bandit.load_state(path)

    def reset(self) -> None:
        """Reset the bandit to its initial prior, clearing all history."""
        self.bandit.reset()
