"""AI-Glasses third routing arm: ``paired_phone_inference``.

The brief (§11) identifies a natural L3 extension: when the current device
is AI-Glasses-class, the router should have a *third* option — delegate
inference to the paired smartphone rather than run locally or in the cloud.

This module exposes :class:`PairedDeviceRouter`, which **wraps** — and does
not modify — the existing :class:`~i3.router.router.IntelligentRouter`.  The
design preserves backward compatibility: callers on phone-class devices see
the original 2-arm router unchanged; callers on AI-Glasses-class devices
get the 3-arm extended decision surface.

The third arm is an :class:`AIGlassesArm` ``Enum`` value extending
:class:`RouteChoice`.  The wrapping router intercepts the decision:
when the arm score for ``paired_phone_inference`` would win under a simple
utility rule, the wrapper returns a routing decision flagged with
``chosen_route="paired_phone_inference"`` — but the underlying bandit's
posterior is untouched.  This keeps the extension **a config change, not an
arch change**, which is how the brief frames it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from i3.crossdevice.device_registry import DeviceClass, DeviceInfo, DeviceRegistry
from i3.router.types import RouteChoice, RoutingContext, RoutingDecision

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from i3.router.router import IntelligentRouter


class AIGlassesArm(str, Enum):
    """Extension arms available when the current device is AI-Glasses-class."""

    PAIRED_PHONE_INFERENCE = "paired_phone_inference"


@dataclass
class ExtendedRoutingDecision:
    """A routing decision that may include the AI-Glasses extension arm.

    We do **not** inherit from :class:`RoutingDecision` to avoid polluting
    the existing type surface — consumers already written against
    :class:`RoutingDecision` keep working, and new consumers of the 3-arm
    extension can handle this new type explicitly.

    Attributes:
        base_decision: The underlying 2-arm bandit decision.
        final_route: The resolved route label, potentially
            ``"paired_phone_inference"``.
        paired_device_id: ``device_id`` of the peer phone (when the extended
            arm fires).
    """

    base_decision: RoutingDecision
    final_route: str
    paired_device_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Utility: arm score for the extended arm
# ---------------------------------------------------------------------------

def paired_phone_inference_arm(
    context: RoutingContext,
    peer: DeviceInfo,
) -> float:
    """Score function for the ``paired_phone_inference`` arm.

    The score is a bounded approximation of the user-perceived utility of
    offloading inference to the paired phone.  Conceptually it mirrors the
    rule the brief's L1–L5 framework gives for AI-Glasses offload:

    * High *query_complexity* favours offload (phone has the SLM).
    * Low *user_patience* penalises offload (adds a round-trip hop).
    * High *slm_confidence* penalises offload (glasses' own encoder is
      already confident — no benefit from phone's richer inference).
    * *baseline_established* gives a small reward (the phone has more
      long-term user-profile context than the glasses do).

    Args:
        context: The same :class:`RoutingContext` the existing bandit sees.
        peer: The paired phone's :class:`DeviceInfo`.

    Returns:
        A score in ``[0, 1]``; higher means more desirable.
    """
    if not peer.supports_slm:
        return 0.0

    complexity_pref = float(context.query_complexity)
    patience_penalty = 1.0 - float(context.user_patience)
    confidence_penalty = float(context.slm_confidence)
    baseline_bonus = 0.1 if context.baseline_established else 0.0

    score = (
        0.5 * complexity_pref
        - 0.3 * patience_penalty
        - 0.3 * confidence_penalty
        + baseline_bonus
    )
    return float(max(0.0, min(1.0, score + 0.3)))  # re-centre to midpoint


# ---------------------------------------------------------------------------
# Wrapping router
# ---------------------------------------------------------------------------

class PairedDeviceRouter:
    """Wraps an existing :class:`IntelligentRouter` with the third arm.

    The inner router is unmodified — this is an **outer decorator**.  On
    phone-class devices the wrapper returns the original
    :class:`RoutingDecision` verbatim; on AI-Glasses-class devices, the
    wrapper computes the arm score via :func:`paired_phone_inference_arm`
    and returns :class:`ExtendedRoutingDecision` with the extension route
    when the arm wins.

    Args:
        inner_router: The canonical 2-arm router.
        current_device_class: Class of the device the router is running on.
        registry: Paired-device registry.
        decision_threshold: The extension arm fires only when its score
            exceeds the best of the inner router's arms *and* this
            threshold.  Keeps the arm from firing on low-confidence scores.
    """

    def __init__(
        self,
        inner_router: "IntelligentRouter",
        current_device_class: DeviceClass,
        registry: DeviceRegistry,
        decision_threshold: float = 0.55,
    ) -> None:
        self._inner = inner_router
        self._current_class = current_device_class
        self._registry = registry
        self._threshold = decision_threshold

    # ------------------------------------------------------------------
    # Decision API
    # ------------------------------------------------------------------
    def route(
        self,
        context: RoutingContext,
        query_text: Optional[str] = None,
    ) -> RoutingDecision | ExtendedRoutingDecision:
        """Pass-through to the inner router; post-hoc override on AI-Glasses.

        Args:
            context: Routing context.
            query_text: Optional raw query text.  Forwarded to the inner
                router's privacy hooks; never stored here.

        Returns:
            A :class:`RoutingDecision` (phone path) or
            :class:`ExtendedRoutingDecision` (AI-Glasses path).
        """
        base = self._inner.decide(context, query_text=query_text) if hasattr(
            self._inner, "decide"
        ) else self._inner.route(context, query_text=query_text)  # type: ignore[attr-defined]

        if self._current_class != DeviceClass.AI_GLASSES:
            return base

        if base.was_privacy_override:
            # Never override a privacy veto — the glasses should still
            # force-local in that case (or, equivalently, decline to
            # transmit to the phone at all).
            logger.debug(
                "PairedDeviceRouter: privacy override active — "
                "skipping paired-phone arm"
            )
            return base

        peer = self._registry.find_paired_phone()
        if peer is None:
            return base

        arm_score = paired_phone_inference_arm(context, peer)
        inner_best = max(base.confidence.values()) if base.confidence else 0.5
        if arm_score > max(inner_best, self._threshold):
            logger.info(
                "PairedDeviceRouter: extending to paired_phone_inference "
                "(score=%.2f > inner_best=%.2f, peer=%s)",
                arm_score,
                inner_best,
                peer.display_name,
            )
            return ExtendedRoutingDecision(
                base_decision=base,
                final_route=AIGlassesArm.PAIRED_PHONE_INFERENCE.value,
                paired_device_id=peer.device_id,
            )
        return base

    # ------------------------------------------------------------------
    # Pass-through helpers (update, persistence, etc.)
    # ------------------------------------------------------------------
    def update(
        self,
        chosen_route: str,
        context: RoutingContext,
        reward: float,
    ) -> None:
        """Update the underlying bandit's posterior.

        The extension arm's rewards are **not** fed back into the inner
        bandit's posterior — they are summarised separately (see the
        L4 handover design in ``docs/huawei/l1_l5_framework.md §4``).
        When ``chosen_route`` is the extension arm, we short-circuit.
        """
        if chosen_route == AIGlassesArm.PAIRED_PHONE_INFERENCE.value:
            logger.debug(
                "PairedDeviceRouter.update: skipping inner bandit update "
                "for extension arm (reward=%.3f)",
                reward,
            )
            return
        if hasattr(self._inner, "update"):
            arm = 0 if chosen_route == RouteChoice.LOCAL_SLM.value else 1
            ctx_vec = context.to_vector()
            self._inner.update(arm, ctx_vec, reward)  # type: ignore[attr-defined]
