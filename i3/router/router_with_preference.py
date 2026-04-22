"""Preference-aware router composition.

This module wraps :class:`~i3.router.router.IntelligentRouter` (without
modifying it) so that the bandit's reward signal can be sourced from a
learned Bradley-Terry reward model instead of the hand-coded composite
heuristic.  Until the reward model has enough data to be reliable the
router falls back to the caller-supplied engagement reward — this is the
"graceful cold-start" behaviour required by Batch F-4.

Design
------

``PreferenceAwareRouter`` holds:

* The original :class:`IntelligentRouter` (composition, never subclass).
* A :class:`PreferenceDataset` (append-only pairwise preferences).
* A :class:`BradleyTerryRewardModel` (shared between the dataset-training
  pathway and the live inference pathway).
* An :class:`ActivePreferenceSelector` driving the "should we ask?" UX.

All reward model operations are optional — callers that do not want to
pay the MLP forward pass can set ``use_learned_reward=False`` on the
constructor.  The class is async-friendly so the FastAPI layer can
await it in the request path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence

import numpy as np

from i3.router.preference_learning import (
    ActivePreferenceSelector,
    BradleyTerryRewardModel,
    DPOFitReport,
    DPOPreferenceOptimizer,
    PreferenceDataset,
    PreferencePair,
)
from i3.router.router import IntelligentRouter
from i3.router.types import RoutingDecision

logger = logging.getLogger(__name__)


#: Minimum preference pairs required before the learned reward is trusted.
_MIN_PAIRS_FOR_LEARNED_REWARD: int = 8

#: Default sampling cadence for A/B preference prompts (1-in-N messages).
_DEFAULT_PROMPT_EVERY_N: int = 50

#: Information-gain threshold above which a candidate pair is shown to
#: the user even outside the normal sampling cadence.  Tuned empirically
#: to the expected Fisher scale on 12-dim features.
_DEFAULT_IG_THRESHOLD: float = 0.25


@dataclass(frozen=True)
class PreferenceQueryDecision:
    """Output of :meth:`PreferenceAwareRouter.maybe_prepare_query`.

    Attributes:
        should_query: Whether the UI should show the user a preference
            prompt for the current turn.
        reason: Human-readable justification, for logging / debugging.
        information_gain: The D-optimal score of the proposed pair
            (zero when ``should_query`` is False).
    """

    should_query: bool
    reason: str
    information_gain: float


class PreferenceAwareRouter:
    """Composition wrapper that adds preference-based reward learning.

    Args:
        router: The underlying :class:`IntelligentRouter` to delegate
            routing decisions to.
        reward_model: Optional externally-constructed
            :class:`BradleyTerryRewardModel`.  When ``None`` a fresh
            model is created with default dimensions.
        preference_dataset: Optional externally-constructed
            :class:`PreferenceDataset`.  When ``None`` an in-memory
            dataset is created.
        prompt_every_n: Sampling cadence in messages.  The router emits
            a preference-query suggestion roughly every ``N`` turns.
        information_gain_threshold: Active-learning threshold above which
            a preference query is emitted regardless of the cadence.
        use_learned_reward: When False the learned reward is never used
            (useful for ablation).
    """

    def __init__(
        self,
        router: IntelligentRouter,
        *,
        reward_model: BradleyTerryRewardModel | None = None,
        preference_dataset: PreferenceDataset | None = None,
        prompt_every_n: int = _DEFAULT_PROMPT_EVERY_N,
        information_gain_threshold: float = _DEFAULT_IG_THRESHOLD,
        use_learned_reward: bool = True,
    ) -> None:
        if prompt_every_n < 1:
            raise ValueError(
                f"prompt_every_n must be >= 1, got {prompt_every_n}"
            )
        if information_gain_threshold < 0.0:
            raise ValueError(
                f"information_gain_threshold must be >= 0, "
                f"got {information_gain_threshold}"
            )
        self.router = router
        self.reward_model: BradleyTerryRewardModel = (
            reward_model if reward_model is not None else BradleyTerryRewardModel()
        )
        self.dataset: PreferenceDataset = (
            preference_dataset
            if preference_dataset is not None
            else PreferenceDataset()
        )
        self.selector: ActivePreferenceSelector = ActivePreferenceSelector(
            self.reward_model
        )
        self.prompt_every_n: int = int(prompt_every_n)
        self.information_gain_threshold: float = float(information_gain_threshold)
        self.use_learned_reward: bool = bool(use_learned_reward)

        self._turn_counter: int = 0
        #: Counters exposed to ``/api/preference/stats``.
        self.learned_reward_uses: int = 0
        self.fallback_reward_uses: int = 0

        # Optional hook for a UI callback — the frontend can register an
        # async coroutine that actually shows the A/B prompt.  Default is
        # a no-op so the router remains fully usable on the server side.
        self._preference_hook: Optional[
            Callable[[PreferencePair], Awaitable[None]]
        ] = None

    # ------------------------------------------------------------------
    # Routing delegation
    # ------------------------------------------------------------------

    def route(self, text: str, ctx: Any) -> RoutingDecision:
        """Delegate to the underlying router and increment the turn counter."""
        self._turn_counter += 1
        return self.router.route(text, ctx=ctx)

    # ------------------------------------------------------------------
    # Reward translation
    # ------------------------------------------------------------------

    def update_reward_from_engagement(
        self,
        decision: RoutingDecision,
        engagement: float,
        response_features: Sequence[float] | None = None,
    ) -> float:
        """Convert an engagement reward into a bandit update.

        If the learned reward is ready, the engagement value is
        *replaced* by the Bradley-Terry score on ``response_features``.
        Otherwise the engagement is passed through.  Either way the
        bandit update goes via :meth:`IntelligentRouter.update_reward`.

        Args:
            decision: The original :class:`RoutingDecision`.
            engagement: Baseline engagement reward in ``[0, 1]``.
            response_features: Feature vector describing the response
                that was produced.  When ``None`` only the engagement
                fallback is used.

        Returns:
            The reward value that was actually sent to the bandit (in
            ``[0, 1]``).  Useful for logging.
        """
        reward = float(np.clip(float(engagement), 0.0, 1.0))
        using_learned = False
        if (
            self.use_learned_reward
            and response_features is not None
            and self._reward_model_ready()
        ):
            try:
                raw = self.reward_model.score(
                    decision.context.to_vector().tolist(), response_features
                )
                # Squash the raw reward into [0, 1] via a logistic so the
                # bandit's expected-reward model stays on its usual scale.
                reward = float(1.0 / (1.0 + np.exp(-raw)))
                using_learned = True
            except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Learned-reward scoring failed; falling back to engagement: %s",
                    exc,
                )
        if using_learned:
            self.learned_reward_uses += 1
        else:
            self.fallback_reward_uses += 1
        self.router.update_reward(decision, reward)
        return reward

    # ------------------------------------------------------------------
    # Active preference queries
    # ------------------------------------------------------------------

    def register_preference_hook(
        self, hook: Callable[[PreferencePair], Awaitable[None]]
    ) -> None:
        """Register a coroutine that is invoked when a query is emitted."""
        self._preference_hook = hook

    async def request_preference_from_user(self, pair: PreferencePair) -> None:
        """Trigger the UI's A/B preference prompt for ``pair``.

        When no hook is registered this is a no-op.  Hooks must be
        async — synchronous callables are rejected.
        """
        pair.validate()
        if self._preference_hook is None:
            logger.debug("request_preference_from_user: no hook registered")
            return
        await self._preference_hook(pair)

    def maybe_prepare_query(
        self, candidate_pairs: Sequence[PreferencePair]
    ) -> PreferenceQueryDecision:
        """Decide whether to show the user a preference prompt this turn.

        The decision combines two signals:

        1. **Cadence:** ask roughly once every ``prompt_every_n`` turns.
        2. **Information gain:** override the cadence when any candidate
           pair exceeds :attr:`information_gain_threshold`.

        Args:
            candidate_pairs: Pairs the UI could show.

        Returns:
            A :class:`PreferenceQueryDecision`.
        """
        if not candidate_pairs:
            return PreferenceQueryDecision(
                should_query=False,
                reason="No candidate pairs available",
                information_gain=0.0,
            )
        best = self.selector.select_next_query(candidate_pairs, n=1)
        if not best:
            return PreferenceQueryDecision(
                should_query=False,
                reason="Selector produced no valid pair",
                information_gain=0.0,
            )
        ig = self.selector.score_pair(best[0])
        cadence_trigger = self._turn_counter % self.prompt_every_n == 0
        threshold_trigger = ig >= self.information_gain_threshold
        if cadence_trigger or threshold_trigger:
            parts = []
            if cadence_trigger:
                parts.append(f"cadence ({self.prompt_every_n}-turn)")
            if threshold_trigger:
                parts.append(
                    f"ig {ig:.3f} >= {self.information_gain_threshold:.3f}"
                )
            return PreferenceQueryDecision(
                should_query=True,
                reason="; ".join(parts),
                information_gain=float(ig),
            )
        return PreferenceQueryDecision(
            should_query=False,
            reason=(
                f"Below threshold (ig {ig:.3f} < "
                f"{self.information_gain_threshold:.3f}) and off-cadence"
            ),
            information_gain=float(ig),
        )

    # ------------------------------------------------------------------
    # Preference ingestion
    # ------------------------------------------------------------------

    def record_preference(self, pair: PreferencePair) -> None:
        """Append a labelled pair to the dataset and update the selector."""
        self.dataset.append(pair)
        self.selector.register_labelled(pair)

    def train(self, n_epochs: int = 50) -> DPOFitReport:
        """Train the reward model on the current dataset.

        Returns the :class:`DPOFitReport` emitted by the optimiser.

        Raises:
            ValueError: If the dataset has fewer than
                :data:`_MIN_PAIRS_FOR_LEARNED_REWARD` pairs.
        """
        if len(self.dataset) < _MIN_PAIRS_FOR_LEARNED_REWARD:
            raise ValueError(
                f"Need at least {_MIN_PAIRS_FOR_LEARNED_REWARD} pairs, "
                f"have {len(self.dataset)}"
            )
        optim = DPOPreferenceOptimizer(self.reward_model)
        return optim.fit(self.dataset, n_epochs=n_epochs)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of internal counters."""
        return {
            "turn_counter": self._turn_counter,
            "pairs_collected": len(self.dataset),
            "learned_reward_uses": self.learned_reward_uses,
            "fallback_reward_uses": self.fallback_reward_uses,
            "reward_model_ready": self._reward_model_ready(),
            "prompt_every_n": self.prompt_every_n,
            "information_gain_threshold": self.information_gain_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reward_model_ready(self) -> bool:
        """True when enough data has been collected to trust the reward."""
        return len(self.dataset) >= _MIN_PAIRS_FOR_LEARNED_REWARD


__all__ = [
    "PreferenceAwareRouter",
    "PreferenceQueryDecision",
]
