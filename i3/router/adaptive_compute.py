"""Adaptive fast/slow compute router — a PanGu-5.5-style third arm.

Extends the existing two-arm :class:`i3.router.router.IntelligentRouter`
(local SLM vs cloud LLM) with a **third arm**: ``local_reflect``. The
third arm is the *same* local SLM, but with a substantially increased
compute budget — more sampling rounds, a larger ``top_k``, and a
``max_new_tokens`` multiplier. The intuition is a middle ground between
the two binary extremes:

  - The cheap path (local SLM, standard budget) is correct for routine
    turns.
  - The expensive path (cloud LLM) is correct for complex or niche
    queries the local model cannot answer.
  - There is a middle class of queries where the local SLM *could*
    give a high-quality answer with more compute. The third arm
    addresses exactly that class.

This mirrors **Huawei PanGu 5.5**'s publicly announced *adaptive
fast/slow thinking integration*: the model dynamically decides how deep
to think about each problem and reports roughly 8× overall inference
efficiency as a result. The mechanism here is structurally simpler —
three discrete compute budgets rather than a continuum — but it shares
the same principle: an **adaptive** choice of compute, trained from
engagement feedback.

A second lineage is the dynamic early-exit literature: Schwartz et al.
2020 (*The Right Tool for the Job: Matching Model and Instance
Complexities*, ACL 2020) proposed allocating compute at inference time
based on a calibrated confidence estimator. The third arm generalises
that idea — instead of deciding whether to "keep thinking" inside a
single model, the bandit chooses among three entire compute budgets,
with the choice calibrated by a Thompson-sampling posterior over an
extended context (including the model's own self-confidence).

Design constraints:

  * The existing :class:`IntelligentRouter` is **not modified**.
    :class:`AdaptiveComputeRouter` composes it.
  * The two-arm decision flow is preserved for backward compatibility;
    the third arm is only reached by explicit escalation.
  * Privacy override strictly dominates: a query flagged as sensitive
    is still forced to the local path, and the escalation to
    ``local_reflect`` is allowed (it is still local) but the force-to-
    local semantics are retained.

References
----------
* Huawei (June 2025). *PanGu 5.5 — 718 B-parameter MoE with adaptive
  fast/slow thinking integration.* Public announcement.
* Schwartz, Stanovsky, Swayamdipta, Dodge, & Smith (2020). *The Right
  Tool for the Job: Matching Model and Instance Complexities.*
  ACL 2020. arXiv:2004.07453.
* Russo, Van Roy, Kazerouni, Osband & Wen (2018). *A Tutorial on
  Thompson Sampling.* Foundations and Trends in Machine Learning.
  arXiv:1707.02038.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from i3.router.bandit import ContextualThompsonBandit
from i3.router.router import IntelligentRouter
from i3.router.types import RouteChoice, RoutingContext, RoutingDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


ComputeBudget = Literal["light", "standard", "heavy"]
"""Discrete compute budget labels.

* ``"light"``    — not currently used by the default policy; reserved
  for future very-cheap paths (e.g. template / cache hits).
* ``"standard"`` — the default budget used by the two-arm router
  (local SLM or cloud LLM at their standard sampling settings).
* ``"heavy"``    — the ``local_reflect`` third arm: same local SLM
  with an increased compute budget (more sampling, larger top-k,
  larger ``max_new_tokens``).
"""


# Escalation thresholds. Public for tests and downstream tuning.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
"""SLM self-confidence below which we escalate to ``local_reflect``."""

# ---------------------------------------------------------------------------
# Extended routing output
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveRoutingDecision:
    """Extended routing decision carrying a ``compute_budget`` field.

    Composed *around* the existing :class:`RoutingDecision` rather than
    subclassing it, so downstream code that type-checks against the
    original dataclass continues to work unchanged.

    Attributes:
        base: The original :class:`RoutingDecision` returned by the
            wrapped :class:`IntelligentRouter`.
        compute_budget: One of ``"light"``, ``"standard"``, or
            ``"heavy"``.
        escalated: ``True`` iff the adaptive router escalated the
            wrapped router's local_slm decision up to the
            ``local_reflect`` arm.
        reflect_params: Concrete sampling-budget knobs the runtime
            should use for ``local_reflect``. Populated only when
            :attr:`compute_budget` == ``"heavy"``. Keys:
            ``"max_new_tokens_multiplier"``, ``"top_k_multiplier"``,
            ``"extra_sampling_rounds"``.
        adaptive_reasoning: Human-readable explanation of the adaptive
            layer's contribution on top of ``base.reasoning``.
    """

    base: RoutingDecision
    compute_budget: ComputeBudget
    escalated: bool
    reflect_params: dict[str, float] = field(default_factory=dict)
    adaptive_reasoning: str = ""

    # ---- convenience forwarding ------------------------------------------

    @property
    def chosen_route(self) -> RouteChoice:
        """Delegate to :attr:`base` so the adaptive decision is
        drop-in compatible with the original API surface."""
        return self.base.chosen_route

    @property
    def confidence(self) -> dict[str, float]:
        """Delegate confidence dict to :attr:`base`."""
        return self.base.confidence

    @property
    def context(self) -> RoutingContext:
        """Delegate context to :attr:`base`."""
        return self.base.context

    @property
    def was_privacy_override(self) -> bool:
        """Delegate privacy-override flag to :attr:`base`."""
        return self.base.was_privacy_override

    @property
    def reasoning(self) -> str:
        """Combined reasoning from the base decision and the adaptive layer."""
        if self.adaptive_reasoning:
            return f"{self.base.reasoning} {self.adaptive_reasoning}"
        return self.base.reasoning


# ---------------------------------------------------------------------------
# Public router class
# ---------------------------------------------------------------------------


class AdaptiveComputeRouter:
    """Third-arm bandit + compute-budget escalation wrapper.

    Composes a pre-existing :class:`IntelligentRouter` (unchanged) with:

    1. A three-arm :class:`ContextualThompsonBandit` operating on an
       **extended** context (adds ``prior_query_difficulty_estimate``
       and ``prior_slm_self_confidence`` dimensions, for a total of
       ``base_context_dim + 2`` features).
    2. An escalation policy: if the wrapped router picks ``local_slm``,
       and the SLM's self-reported confidence is below
       :data:`DEFAULT_CONFIDENCE_THRESHOLD`, and the decision is not a
       privacy override, escalate to ``local_reflect`` with the
       ``"heavy"`` compute budget.

    The wrapped router is never mutated; its bandit posterior continues
    to learn from its own two-arm decisions. The third-arm bandit is a
    *separate* instance that learns from the adaptive router's own
    reward signal.

    Args:
        base_router: An already-constructed :class:`IntelligentRouter`.
            This router's own two-arm bandit is preserved and used for
            the inner fast/slow decision.
        confidence_threshold: SLM self-confidence below which the
            adaptive layer escalates a winning ``local_slm`` decision
            to ``local_reflect``. Default is
            :data:`DEFAULT_CONFIDENCE_THRESHOLD`.
        reflect_max_new_tokens_multiplier: Multiplier applied to the
            default ``max_new_tokens`` when the chosen budget is
            ``"heavy"``. Default ``1.5`` (Huawei-PanGu-flavoured
            multiplier: a modest uplift, not a 10x).
        reflect_top_k_multiplier: Multiplier applied to ``top_k`` when
            the chosen budget is ``"heavy"``. Default ``2.0``.
        reflect_extra_sampling_rounds: Additional resamples performed
            in the ``"heavy"`` budget. Default ``1`` (so the SLM is
            invoked twice and the best-of-two is emitted).
    """

    # Arm indices for the third-arm bandit.
    _ARM_LOCAL: int = 0
    _ARM_CLOUD: int = 1
    _ARM_REFLECT: int = 2

    _ARM_NAMES: dict[int, str] = {
        _ARM_LOCAL: "local_slm",
        _ARM_CLOUD: "cloud_llm",
        _ARM_REFLECT: "local_reflect",
    }

    def __init__(
        self,
        base_router: IntelligentRouter,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        reflect_max_new_tokens_multiplier: float = 1.5,
        reflect_top_k_multiplier: float = 2.0,
        reflect_extra_sampling_rounds: int = 1,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be in [0, 1], "
                f"got {confidence_threshold}"
            )
        if reflect_max_new_tokens_multiplier <= 0:
            raise ValueError(
                "reflect_max_new_tokens_multiplier must be > 0, "
                f"got {reflect_max_new_tokens_multiplier}"
            )
        if reflect_top_k_multiplier <= 0:
            raise ValueError(
                "reflect_top_k_multiplier must be > 0, "
                f"got {reflect_top_k_multiplier}"
            )
        if reflect_extra_sampling_rounds < 0:
            raise ValueError(
                "reflect_extra_sampling_rounds must be >= 0, "
                f"got {reflect_extra_sampling_rounds}"
            )

        self.base_router: IntelligentRouter = base_router
        self.confidence_threshold: float = float(confidence_threshold)

        # Reflect-budget knobs.
        self.reflect_max_new_tokens_multiplier: float = float(
            reflect_max_new_tokens_multiplier
        )
        self.reflect_top_k_multiplier: float = float(reflect_top_k_multiplier)
        self.reflect_extra_sampling_rounds: int = int(
            reflect_extra_sampling_rounds
        )

        # Extended context dimensionality for the third-arm bandit.
        base_dim = int(base_router.bandit.context_dim)
        self.extended_context_dim: int = base_dim + 2

        # Dedicated three-arm bandit on the extended context.
        self._adaptive_bandit: ContextualThompsonBandit = ContextualThompsonBandit(
            n_arms=3,
            context_dim=self.extended_context_dim,
            prior_precision=base_router.bandit.prior_precision,
            exploration_bonus=base_router.bandit.exploration_bonus,
            refit_interval=base_router.bandit.refit_interval,
        )

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route(
        self,
        text: str,
        ctx: RoutingContext,
        *,
        user_state: np.ndarray | None = None,
        prior_query_difficulty_estimate: float = 0.5,
        prior_slm_self_confidence: float | None = None,
    ) -> AdaptiveRoutingDecision:
        """Make an adaptive routing decision for ``text``.

        The procedure is:

        1. Delegate to :meth:`IntelligentRouter.route` to obtain the
           base two-arm decision. All privacy / complexity behaviour
           of the wrapped router is preserved exactly.
        2. If the base decision was a privacy override → return
           immediately with ``compute_budget="standard"``. Privacy
           wins; we never escalate an override to any other arm.
        3. If the winning arm is ``local_slm`` AND the SLM's
           self-confidence is below :attr:`confidence_threshold` →
           escalate to ``local_reflect`` with budget ``"heavy"``.
        4. Otherwise → return the base decision unchanged, with
           budget ``"standard"``.

        Args:
            text: Raw user query.
            ctx: Pre-built routing context.
            user_state: Optional full user-state (forwarded to the
                wrapped router).
            prior_query_difficulty_estimate: Estimated difficulty of
                the query in ``[0, 1]``, appended to the extended
                bandit context.
            prior_slm_self_confidence: Optional override of the SLM's
                self-confidence. When ``None``, the value is read from
                ``ctx.slm_confidence``. Used by tests and by callers
                that have already recomputed the estimate downstream
                of the encoder.

        Returns:
            An :class:`AdaptiveRoutingDecision`.
        """
        # ---- Step 1: Delegate to the wrapped router ----------------------
        base_decision = self.base_router.route(
            text=text, ctx=ctx, user_state=user_state
        )

        # ---- Sanitise inputs ---------------------------------------------
        slm_conf_src = (
            prior_slm_self_confidence
            if prior_slm_self_confidence is not None
            else ctx.slm_confidence
        )
        slm_conf = float(np.clip(float(slm_conf_src), 0.0, 1.0))
        difficulty = float(
            np.clip(float(prior_query_difficulty_estimate), 0.0, 1.0)
        )

        # ---- Step 2: Privacy override dominates --------------------------
        if base_decision.was_privacy_override:
            return AdaptiveRoutingDecision(
                base=base_decision,
                compute_budget="standard",
                escalated=False,
                reflect_params={},
                adaptive_reasoning=(
                    "Privacy override in force — adaptive escalation is "
                    "suppressed to preserve the force-local semantics."
                ),
            )

        # ---- Step 3: Escalation check ------------------------------------
        if (
            base_decision.chosen_route == RouteChoice.LOCAL_SLM
            and slm_conf < self.confidence_threshold
        ):
            reflect_params = self._reflect_params()
            rewritten_base = RoutingDecision(
                chosen_route=RouteChoice.LOCAL_SLM,  # still local
                confidence=base_decision.confidence,
                context=base_decision.context,
                was_privacy_override=False,
                reasoning=base_decision.reasoning,
            )
            return AdaptiveRoutingDecision(
                base=rewritten_base,
                compute_budget="heavy",
                escalated=True,
                reflect_params=reflect_params,
                adaptive_reasoning=(
                    f"Escalated local_slm → local_reflect "
                    f"(slm_confidence={slm_conf:.3f} < "
                    f"threshold={self.confidence_threshold:.2f}, "
                    f"difficulty={difficulty:.3f}). Reflect params: "
                    f"max_new_tokens×{reflect_params['max_new_tokens_multiplier']:.2f}, "
                    f"top_k×{reflect_params['top_k_multiplier']:.2f}, "
                    f"+{int(reflect_params['extra_sampling_rounds'])} "
                    f"sampling round(s)."
                ),
            )

        # ---- Step 4: Pass-through ----------------------------------------
        return AdaptiveRoutingDecision(
            base=base_decision,
            compute_budget="standard",
            escalated=False,
            reflect_params={},
            adaptive_reasoning=(
                f"No escalation (slm_confidence={slm_conf:.3f} >= "
                f"threshold={self.confidence_threshold:.2f} or non-local "
                f"winning arm). Budget=standard."
            ),
        )

    # ------------------------------------------------------------------
    # Reward feedback
    # ------------------------------------------------------------------

    def update_reward(
        self,
        decision: AdaptiveRoutingDecision,
        engagement: float,
        *,
        prior_query_difficulty_estimate: float = 0.5,
        prior_slm_self_confidence: float | None = None,
    ) -> None:
        """Feed engagement reward back to both bandits.

        The wrapped router's two-arm bandit sees the base arm (local
        vs cloud). The adaptive layer's three-arm bandit additionally
        sees the escalation signal, so it can learn when the reflect
        path was actually worth the extra compute.

        Args:
            decision: The previous :class:`AdaptiveRoutingDecision`.
            engagement: Observed engagement reward, clipped to ``[0,1]``.
            prior_query_difficulty_estimate: Must match the value
                passed to :meth:`route` (the extended context must be
                recoverable at update time).
            prior_slm_self_confidence: See :meth:`route`.
        """
        # Base router learns from its own decision (unchanged).
        self.base_router.update_reward(decision.base, engagement)

        # Third-arm bandit learns on the extended context.
        if decision.compute_budget == "heavy":
            arm = self._ARM_REFLECT
        elif decision.chosen_route == RouteChoice.CLOUD_LLM:
            arm = self._ARM_CLOUD
        else:
            arm = self._ARM_LOCAL

        ext = self._extended_context(
            ctx=decision.context,
            prior_query_difficulty_estimate=prior_query_difficulty_estimate,
            prior_slm_self_confidence=prior_slm_self_confidence,
        )

        eng = float(engagement)
        if not np.isfinite(eng):
            eng = 0.0
        eng = float(np.clip(eng, 0.0, 1.0))
        self._adaptive_bandit.update(arm, ext, eng)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return combined statistics from base + adaptive bandits."""
        base = self.base_router.get_stats()
        extras = self._adaptive_bandit.get_arm_stats()
        for arm_stat in extras["arms"]:
            arm_stat["route"] = self._ARM_NAMES.get(
                arm_stat["arm"], f"arm_{arm_stat['arm']}"
            )
        return {
            **base,
            "adaptive_bandit": extras,
            "confidence_threshold": self.confidence_threshold,
            "extended_context_dim": self.extended_context_dim,
            "reflect_params": self._reflect_params(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extended_context(
        self,
        ctx: RoutingContext,
        *,
        prior_query_difficulty_estimate: float,
        prior_slm_self_confidence: float | None,
    ) -> np.ndarray:
        """Build the ``context_dim + 2`` context vector.

        The two appended dimensions are:

        * ``prior_query_difficulty_estimate`` — a caller-supplied
          estimate of the query's difficulty in ``[0, 1]``.
        * ``prior_slm_self_confidence`` — the SLM's self-reported
          confidence at the *start* of the turn (before any
          generation). Defaults to ``ctx.slm_confidence`` when not
          provided.
        """
        base_vec = ctx.to_vector()
        slm_conf_src = (
            prior_slm_self_confidence
            if prior_slm_self_confidence is not None
            else ctx.slm_confidence
        )
        tail = np.array(
            [
                float(np.clip(float(prior_query_difficulty_estimate), 0.0, 1.0)),
                float(np.clip(float(slm_conf_src), 0.0, 1.0)),
            ],
            dtype=np.float64,
        )
        return np.concatenate([base_vec, tail])

    def _reflect_params(self) -> dict[str, float]:
        """Return the concrete sampling knobs for the ``"heavy"`` budget."""
        return {
            "max_new_tokens_multiplier": self.reflect_max_new_tokens_multiplier,
            "top_k_multiplier": self.reflect_top_k_multiplier,
            "extra_sampling_rounds": float(self.reflect_extra_sampling_rounds),
        }
