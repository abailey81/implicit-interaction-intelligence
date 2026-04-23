"""REST endpoints for active preference learning.

This module mounts three endpoints under ``/api/preference``:

* ``POST /api/preference/record`` — append a labelled preference pair
  to the per-user :class:`~i3.router.preference_learning.PreferenceDataset`.
* ``GET  /api/preference/query/{user_id}`` — return the most informative
  A/B pair the UI should show (may be empty).
* ``GET  /api/preference/stats/{user_id}`` — diagnostic snapshot of the
  per-user dataset and reward model state.

Design notes
------------

* All routes are per-user: state lives in an in-memory FIFO cache keyed
  by user ID (bounded to ``_MAX_USERS`` entries), mirroring
  :mod:`server.routes_explain` to avoid cross-user leakage.
* Candidate-pair construction is synthetic: when no upstream generator
  has submitted candidates yet we fabricate a pair from the last
  recorded feature vectors so the UI always has something to render
  when ``should_query`` is True.
* Body size is capped at 8 KiB — preference labels carry at most two
  response strings, and the rest of the payload is numeric.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from server.auth import require_user_identity, require_user_identity_from_body
from i3.privacy.sanitizer import PrivacySanitizer
from i3.router.preference_learning import (
    ActivePreferenceSelector,
    BradleyTerryRewardModel,
    PreferenceDataset,
    PreferencePair,
    build_response_features,
)
from i3.router.router_with_preference import PreferenceAwareRouter

# SEC (H-2, 2026-04-23 audit): all free-text fields entering the preference
# dataset and leaving through ``GET /api/preference/query/{user_id}`` MUST
# pass through the sanitiser so an attacker cannot stash PII and harvest it
# via another client's GET.  The sanitiser is stateless and module-scoped
# so the compiled regex battery is shared across requests.
_SANITIZER: PrivacySanitizer = PrivacySanitizer(enabled=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Regex mirroring :mod:`server.routes_explain` for user IDs.
USER_ID_REGEX: str = r"^[a-zA-Z0-9_-]{1,64}$"

#: 8 KiB cap on preference POST bodies.
MAX_BODY_BYTES: int = 8 * 1024

#: Maximum number of distinct users tracked in the in-memory cache.
_MAX_USERS: int = 256


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class PreferenceRecordRequest(BaseModel):
    """Body model for ``POST /api/preference/record``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    prompt: str = Field(..., min_length=1, max_length=4096)
    response_a: str = Field(..., min_length=1, max_length=4096)
    response_b: str = Field(..., min_length=1, max_length=4096)
    winner: str = Field(..., pattern=r"^(a|b|tie)$")
    context: list[float] = Field(default_factory=list, max_length=64)
    response_a_features: list[float] = Field(default_factory=list, max_length=64)
    response_b_features: list[float] = Field(default_factory=list, max_length=64)


class PreferenceRecordResponse(BaseModel):
    """Response model for ``POST /api/preference/record``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    pairs_collected: int
    accepted: bool


class PreferenceQueryResponse(BaseModel):
    """Response for ``GET /api/preference/query/{user_id}``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    should_query: bool
    prompt: str
    response_a: str
    response_b: str
    context: list[float]
    response_a_features: list[float]
    response_b_features: list[float]
    information_gain: float
    reason: str


class PreferenceStatsResponse(BaseModel):
    """Response for ``GET /api/preference/stats/{user_id}``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    pairs_collected: int
    reward_model_ready: bool
    reward_model_accuracy: float
    estimated_active_budget_remaining: int
    learned_reward_uses: int
    fallback_reward_uses: int


# ---------------------------------------------------------------------------
# Per-user state
# ---------------------------------------------------------------------------


class _UserState:
    """Per-user bundle of dataset, reward model, and selector.

    Kept small: no heavy resources (no aiosqlite connection) so that the
    bounded FIFO cache can evict entries without special cleanup.
    """

    def __init__(self) -> None:
        self.dataset: PreferenceDataset = PreferenceDataset()
        self.reward_model: BradleyTerryRewardModel = BradleyTerryRewardModel()
        self.selector: ActivePreferenceSelector = ActivePreferenceSelector(
            self.reward_model
        )
        #: Last known reward-model validation accuracy.
        self.last_accuracy: float = 0.0
        #: Target budget of labels to collect; advertised as the
        #: "remaining active budget" in :class:`PreferenceStatsResponse`.
        #: The figure matches the ~10-20 pairs per user sample-efficiency
        #: claim from Mehta et al. 2025.
        self.target_labels: int = 20
        #: Last candidate pair cached so GETs can return something useful.
        self.last_candidate: PreferencePair | None = None

    def candidate_pairs(self) -> list[PreferencePair]:
        """Return a small list of candidate pairs the UI could show.

        If we have never seen any pairs we fabricate one neutral pair so
        the UI always has something renderable.  Otherwise we reuse the
        most recent observed pair as the seed candidate.
        """
        if self.last_candidate is not None:
            return [self.last_candidate]
        # Neutral fabricated pair — all-zero features, neutral context.
        fabricated = PreferencePair(
            prompt="Which response feels more natural right now?",
            response_a="Response A",
            response_b="Response B",
            winner="tie",
            context=[0.0] * self.reward_model.context_dim,
            response_a_features=build_response_features(
                length_tokens=60.0,
                latency_ms=250.0,
                model_confidence=0.7,
                response_dim=self.reward_model.response_dim,
            ),
            response_b_features=build_response_features(
                length_tokens=180.0,
                latency_ms=800.0,
                model_confidence=0.85,
                response_dim=self.reward_model.response_dim,
            ),
            user_id="anonymous",
        )
        return [fabricated]


class _UserCache:
    """Bounded FIFO mapping of ``user_id -> _UserState``."""

    def __init__(self, max_entries: int = _MAX_USERS) -> None:
        self._max: int = int(max_entries)
        self._store: "OrderedDict[str, _UserState]" = OrderedDict()

    def get_or_create(self, user_id: str) -> _UserState:
        if user_id in self._store:
            self._store.move_to_end(user_id)
            return self._store[user_id]
        state = _UserState()
        self._store[user_id] = state
        while len(self._store) > self._max:
            evicted, _ = self._store.popitem(last=False)
            logger.debug("Evicted preference state for user_id=%s", evicted)
        return state

    def get(self, user_id: str) -> _UserState | None:
        return self._store.get(user_id)


_CACHE = _UserCache()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/api/preference", tags=["preference"])


@router.post(
    "/record",
    dependencies=[Depends(require_user_identity_from_body)],
)
async def record_preference(request: Request) -> JSONResponse:
    """Append a labelled preference pair to the user's dataset."""
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    try:
        body = PreferenceRecordRequest.model_validate_json(raw)
    except ValueError as exc:
        logger.debug("record_preference: validation failed: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid request payload") from exc

    state = _CACHE.get_or_create(body.user_id)

    # Zero-pad empty vectors to sensible defaults so the frontend doesn't
    # need to know the reward model's dimensions.
    ctx_dim = state.reward_model.context_dim
    resp_dim = state.reward_model.response_dim
    context = body.context if body.context else [0.0] * ctx_dim
    feat_a = (
        body.response_a_features
        if body.response_a_features
        else [0.0] * resp_dim
    )
    feat_b = (
        body.response_b_features
        if body.response_b_features
        else [0.0] * resp_dim
    )

    # SEC (H-2): sanitise every free-text field before it enters the
    # per-user dataset.  Anything that later flows out of
    # GET /api/preference/query/{user_id} or GET /api/preference/stats/
    # {user_id} must have been through the PII sanitiser first, otherwise
    # we become a cross-user PII harvester.
    safe_prompt = _SANITIZER.sanitize(body.prompt)
    safe_response_a = _SANITIZER.sanitize(body.response_a)
    safe_response_b = _SANITIZER.sanitize(body.response_b)

    try:
        pair = PreferencePair(
            prompt=safe_prompt,
            response_a=safe_response_a,
            response_b=safe_response_b,
            winner=body.winner,
            context=context,
            response_a_features=feat_a,
            response_b_features=feat_b,
            user_id=body.user_id,
        )
        pair.validate()
    except ValueError as exc:
        logger.debug("record_preference: pair validation failed: %s", exc)
        raise HTTPException(
            status_code=422, detail="Invalid preference pair"
        ) from exc

    state.dataset.append(pair)
    state.selector.register_labelled(pair)
    state.last_candidate = pair

    payload = PreferenceRecordResponse(
        user_id=body.user_id,
        pairs_collected=len(state.dataset),
        accepted=True,
    )
    return JSONResponse(payload.model_dump(mode="json"))


@router.get("/query/{user_id}", dependencies=[Depends(require_user_identity)])
async def query_next(
    user_id: str = Path(..., pattern=USER_ID_REGEX, min_length=1, max_length=64),
) -> JSONResponse:
    """Return the next preference pair the active learner wants labelled."""
    state = _CACHE.get_or_create(user_id)

    # Build an ephemeral PreferenceAwareRouter just to reuse the cadence
    # / threshold decision logic without allocating a full IntelligentRouter.
    selector = state.selector
    candidates = state.candidate_pairs()
    top = selector.select_next_query(candidates, n=1)
    if not top:
        payload = PreferenceQueryResponse(
            user_id=user_id,
            should_query=False,
            prompt="",
            response_a="",
            response_b="",
            context=[],
            response_a_features=[],
            response_b_features=[],
            information_gain=0.0,
            reason="No candidate pairs available",
        )
        return JSONResponse(payload.model_dump(mode="json"))

    chosen = top[0]
    ig = float(selector.score_pair(chosen))
    # Always surface the candidate; the frontend decides whether to show
    # it based on its own cadence rules.  ``should_query`` is True when
    # the information gain clears a low baseline.
    threshold = 0.01
    payload = PreferenceQueryResponse(
        user_id=user_id,
        should_query=bool(ig >= threshold),
        prompt=chosen.prompt,
        response_a=chosen.response_a,
        response_b=chosen.response_b,
        context=list(chosen.context),
        response_a_features=list(chosen.response_a_features),
        response_b_features=list(chosen.response_b_features),
        information_gain=ig,
        reason=(
            f"ig {ig:.3f} >= {threshold:.3f}"
            if ig >= threshold
            else f"ig {ig:.3f} < {threshold:.3f}"
        ),
    )
    return JSONResponse(payload.model_dump(mode="json"))


@router.get("/stats/{user_id}", dependencies=[Depends(require_user_identity)])
async def stats(
    user_id: str = Path(..., pattern=USER_ID_REGEX, min_length=1, max_length=64),
) -> JSONResponse:
    """Return a diagnostic snapshot for ``user_id``.

    Returns 404 only if the user has never interacted with the system
    AND no implicit state has been created.  In practice the per-user
    state is lazily created on any read, so this endpoint tends to
    return a zero-filled snapshot rather than 404.
    """
    state = _CACHE.get(user_id)
    if state is None:
        raise HTTPException(status_code=404, detail="No preference state for user")
    pairs = len(state.dataset)
    budget_remaining = max(0, state.target_labels - pairs)
    # Reuse the router-level counters when a PreferenceAwareRouter has
    # been attached to the app; fall back to zeros otherwise.
    learned = 0
    fallback = 0
    try:
        aware = getattr(stats, "_aware", None)
        if isinstance(aware, PreferenceAwareRouter):
            learned = aware.learned_reward_uses
            fallback = aware.fallback_reward_uses
    except AttributeError:  # pragma: no cover - defensive
        pass
    payload = PreferenceStatsResponse(
        user_id=user_id,
        pairs_collected=pairs,
        reward_model_ready=bool(pairs >= 8),
        reward_model_accuracy=float(state.last_accuracy),
        estimated_active_budget_remaining=int(budget_remaining),
        learned_reward_uses=int(learned),
        fallback_reward_uses=int(fallback),
    )
    return JSONResponse(payload.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def include_preference_routes(app: FastAPI) -> None:
    """Mount the preference router on ``app``.

    Args:
        app: The :class:`FastAPI` application from
            :func:`server.app.create_app`.
    """
    app.include_router(router)


__all__ = [
    "MAX_BODY_BYTES",
    "PreferenceQueryResponse",
    "PreferenceRecordRequest",
    "PreferenceRecordResponse",
    "PreferenceStatsResponse",
    "include_preference_routes",
    "router",
]
