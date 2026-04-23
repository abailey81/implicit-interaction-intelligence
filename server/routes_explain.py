"""Uncertainty + counterfactual-explanation endpoints.

This router mounts two routes under ``/api/explain``:

* ``POST /api/explain/adaptation`` — run Monte-Carlo-dropout on the
  encoder + controller and return the resulting
  :class:`~i3.adaptation.uncertainty.UncertainAdaptationVector`
  together with the top-``k`` counterfactual explanations from
  :class:`~i3.interpretability.counterfactuals.CounterfactualExplainer`.
* ``GET /api/explain/last-decision/{user_id}`` — fetch the most recent
  explanation payload computed for ``user_id`` (an in-memory cache,
  bounded to ``_MAX_CACHED_USERS`` entries with FIFO eviction — no
  cross-user leakage).

Design notes
------------

1. **Soft-fall-back.** Whenever the pipeline, encoder, or adaptation
   controller is not available (unit tests, or a standalone launch
   used for screenshots), the route falls back to a deterministic
   surrogate so the UI always gets a plausible payload to render.
2. **No raw text.** The route never reads or returns message text;
   only adaptation numerics are exposed. This matches the privacy
   boundary in :mod:`server.routes_tts`.
3. **Bounded body.** A 2 KB body cap rejects spraying attacks on the
   MC sampler even though the request itself is tiny.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Optional

import torch
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse

from server.auth import require_user_identity, require_user_identity_from_body
from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector
from i3.adaptation.uncertainty import (
    ADAPTATION_DIMS,
    MCDropoutAdaptationEstimator,
    UncertainAdaptationVector,
    confidence_threshold_policy,
    refuse_when_unsure_mask,
)
from i3.interpretability.counterfactuals import (
    Counterfactual,
    CounterfactualExplainer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

#: 2 KB body cap — the wire payload carries only a user id + flags.
MAX_BODY_BYTES: int = 2 * 1024

#: Default per-dimension std threshold for the "refuse when unsure"
#: policy. Must stay in sync with the UI colour-coding in
#: ``web/js/explain_panel.js``.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.15

#: Default number of counterfactuals returned per request.
DEFAULT_TOP_K: int = 3

#: FIFO cache of the most recent explanation per user id. Bounded so a
#: long-running process cannot blow up its heap on a malicious user-id
#: iteration.
_MAX_CACHED_USERS: int = 128


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    """Wire model for ``POST /api/explain/adaptation``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=8)
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        gt=0.0,
        le=1.0,
    )


class DimensionConfidence(BaseModel):
    """Confidence classification for a single adaptation dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    classification: str  # "confident" | "uncertain"


class ExplainResponse(BaseModel):
    """Wire model for both explain endpoints."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    adaptation_mean: dict[str, Any]
    adaptation_refused: dict[str, Any]
    sample_count: int
    all_dimensions_confident: bool
    per_dimension: list[DimensionConfidence]
    counterfactuals: list[dict[str, Any]]
    natural_language: str
    latency_ms: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Router + shared state
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/api/explain", tags=["explain"])


class _ExplainCache:
    """Bounded FIFO cache of the most recent explanation per user id."""

    def __init__(self, max_entries: int = _MAX_CACHED_USERS) -> None:
        self._max = int(max_entries)
        self._store: "OrderedDict[str, ExplainResponse]" = OrderedDict()

    def get(self, user_id: str) -> Optional[ExplainResponse]:
        return self._store.get(user_id)

    def set(self, user_id: str, payload: ExplainResponse) -> None:
        if user_id in self._store:
            self._store.move_to_end(user_id)
        self._store[user_id] = payload
        while len(self._store) > self._max:
            self._store.popitem(last=False)


_CACHE = _ExplainCache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_estimator(request: Request) -> Optional[MCDropoutAdaptationEstimator]:
    """Build an :class:`MCDropoutAdaptationEstimator` from live pipeline state.

    Returns ``None`` if the pipeline does not expose both an encoder
    and an adaptation controller. The fallback path in
    :func:`_build_fallback_payload` then takes over.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return None
    encoder = getattr(pipeline, "encoder", None)
    controller = getattr(pipeline, "adaptation", None)
    if encoder is None or controller is None:
        return None
    try:
        return MCDropoutAdaptationEstimator(
            encoder=encoder,
            controller=controller,
            n_samples=30,
            dropout_p=0.1,
        )
    except (TypeError, ValueError) as exc:
        logger.debug("explain: estimator unavailable: %s", exc)
        return None


def _resolve_feature_window(
    request: Request, user_id: str
) -> Optional[torch.Tensor]:
    """Extract the user's most recent feature-window tensor from the pipeline.

    Returns ``None`` if the pipeline does not have a feature-window
    cache for this user — in which case we fall back to a synthetic
    baseline window.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return None
    try:
        user_models = getattr(pipeline, "user_models", None) or {}
        um = user_models.get(user_id)
    except AttributeError:
        return None
    if um is None:
        return None
    window = getattr(um, "feature_window", None)
    if isinstance(window, torch.Tensor):
        return window
    return None


def _synthetic_feature_window() -> torch.Tensor:
    """Return a neutral-baseline feature window.

    Used when the pipeline is not live (unit tests, screenshots).
    Produces a tensor of shape ``[16, 32]`` filled with 0.5s — a
    neutral "mid-range" baseline.
    """
    return torch.full((16, 32), 0.5, dtype=torch.float32)


def _classify(std: float, threshold: float) -> str:
    """Classify a per-dimension std as ``confident`` or ``uncertain``."""
    return "uncertain" if float(std) >= float(threshold) else "confident"


#: Cached deterministic surrogate layer.  Lazily constructed once per
#: process (see :func:`_surrogate_mapping_fn`) so every explain request
#: shares the same ranking.  Building the layer is cheap (~1 KB of
#: floats) but the *previous* implementation called ``torch.manual_seed``
#: — a **global** side-effect that reseeded every other coroutine's RNG
#: on every request (H-8, 2026-04-23 audit).  We now use a scoped
#: ``torch.Generator`` so no global state is touched.
_SURROGATE_LAYER: "torch.nn.Module | None" = None


def _surrogate_mapping_fn() -> torch.nn.Module:
    """Return a deterministic linear surrogate for counterfactual explanations.

    We cannot backprop through the actual adaptation controller
    (non-differentiable thresholds in :class:`AccessibilityAdapter`),
    so we use a small bias-free linear surrogate. The weights are
    seeded deterministically via a **scoped**
    :class:`torch.Generator` — never via :func:`torch.manual_seed`,
    which would mutate every other coroutine's RNG.
    """
    global _SURROGATE_LAYER
    if _SURROGATE_LAYER is not None:
        return _SURROGATE_LAYER

    import torch.nn as nn

    gen = torch.Generator(device="cpu").manual_seed(0xA11CE)
    layer = nn.Linear(32, 8, bias=False)
    with torch.no_grad():
        for p in layer.parameters():
            # SEC (H-8): scoped generator → no global RNG side-effect.
            p.copy_(torch.randn(p.shape, generator=gen))
            p.requires_grad_(True)
    _SURROGATE_LAYER = layer
    return layer


def _build_payload(
    user_id: str,
    uncertain: UncertainAdaptationVector,
    counterfactuals: list[Counterfactual],
    threshold: float,
    latency_ms: float,
) -> ExplainResponse:
    """Assemble the outbound :class:`ExplainResponse`."""
    per_dim: list[DimensionConfidence] = []
    mean_vec = uncertain.mean_vector()
    mean_tensor = mean_vec.to_tensor()
    for i, name in enumerate(ADAPTATION_DIMS):
        per_dim.append(
            DimensionConfidence(
                dimension=name,
                mean=float(mean_tensor[i].item()),
                std=float(uncertain.std[i]),
                ci_lower=float(uncertain.ci[i].lower),
                ci_upper=float(uncertain.ci[i].upper),
                classification=_classify(uncertain.std[i], threshold),
            )
        )
    refused: AdaptationVector = refuse_when_unsure_mask(uncertain, threshold=threshold)
    all_confident = confidence_threshold_policy(uncertain, threshold=threshold)

    explainer = CounterfactualExplainer(
        mapping_fn=_surrogate_mapping_fn(),
        target_delta=0.2,
    )
    nl_parts: list[str] = []
    for cf in counterfactuals:
        nl_parts.append(explainer.to_natural_language(cf))
    if all_confident:
        nl_parts.insert(
            0,
            "The system is confident enough in every adaptation dimension "
            "to act on the mean values.",
        )
    else:
        nl_parts.insert(
            0,
            "Some adaptation dimensions are uncertain. Those dimensions "
            "have been reset to their neutral baseline.",
        )

    return ExplainResponse(
        user_id=user_id,
        adaptation_mean=mean_vec.to_dict(),
        adaptation_refused=refused.to_dict(),
        sample_count=uncertain.sample_count,
        all_dimensions_confident=all_confident,
        per_dimension=per_dim,
        counterfactuals=[cf.model_dump(mode="json") for cf in counterfactuals],
        natural_language=" ".join(nl_parts),
        latency_ms=float(latency_ms),
    )


def _build_fallback_payload(user_id: str, threshold: float) -> ExplainResponse:
    """Build a deterministic payload when the live pipeline is unavailable.

    Used by the unit tests and by the demo when the model checkpoints
    have not been loaded. The std values are set to
    ``threshold * 0.5`` so every dimension classifies as ``confident``
    — a safe default that never fires a "refuse to adapt" badge on
    the UI when there is no live signal to justify one.
    """
    start = time.perf_counter()
    default = AdaptationVector.default()
    std_val = float(threshold) * 0.5
    per_dim: list[DimensionConfidence] = []
    mean_tensor = default.to_tensor()
    for i, name in enumerate(ADAPTATION_DIMS):
        mean_i = float(mean_tensor[i].item())
        per_dim.append(
            DimensionConfidence(
                dimension=name,
                mean=mean_i,
                std=std_val,
                ci_lower=max(0.0, mean_i - std_val),
                ci_upper=min(1.0, mean_i + std_val),
                classification="confident",
            )
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ExplainResponse(
        user_id=user_id,
        adaptation_mean=default.to_dict(),
        adaptation_refused=default.to_dict(),
        sample_count=1,
        all_dimensions_confident=True,
        per_dimension=per_dim,
        counterfactuals=[],
        natural_language=(
            "The pipeline is not fully initialised; showing neutral "
            "defaults."
        ),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/adaptation",
    dependencies=[Depends(require_user_identity_from_body)],
)
async def explain_adaptation(request: Request) -> JSONResponse:
    """Run MC-Dropout + counterfactuals for the given user.

    Returns:
        A serialised :class:`ExplainResponse`.

    Raises:
        HTTPException: 413 if the body exceeds :data:`MAX_BODY_BYTES`;
            422 on validation failure.
    """
    start = time.perf_counter()
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        body = ExplainRequest.model_validate_json(raw_body)
    except ValueError as exc:
        logger.debug("explain: validation failure: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid request payload") from exc

    estimator = _resolve_estimator(request)
    window = _resolve_feature_window(request, body.user_id)
    if estimator is None:
        payload = _build_fallback_payload(body.user_id, body.confidence_threshold)
        _CACHE.set(body.user_id, payload)
        return JSONResponse(payload.model_dump(mode="json"))

    feature_window = window if window is not None else _synthetic_feature_window()

    try:
        uncertain = estimator.estimate(feature_window)
    except (ValueError, TypeError) as exc:
        logger.debug("explain: estimation failed: %s", exc)
        payload = _build_fallback_payload(body.user_id, body.confidence_threshold)
        _CACHE.set(body.user_id, payload)
        return JSONResponse(payload.model_dump(mode="json"))

    explainer = CounterfactualExplainer(
        mapping_fn=_surrogate_mapping_fn(),
        target_delta=0.2,
    )
    mean_vec = uncertain.mean_vector()
    try:
        cfs = explainer.explain(
            feature_window=feature_window,
            adaptation=mean_vec.to_tensor(),
            k=body.top_k,
        )
    except (ValueError, TypeError) as exc:
        logger.debug("explain: counterfactual failure: %s", exc)
        cfs = []

    latency_ms = (time.perf_counter() - start) * 1000.0
    payload = _build_payload(
        user_id=body.user_id,
        uncertain=uncertain,
        counterfactuals=cfs,
        threshold=body.confidence_threshold,
        latency_ms=latency_ms,
    )
    _CACHE.set(body.user_id, payload)
    return JSONResponse(payload.model_dump(mode="json"))


@router.get(
    "/last-decision/{user_id}",
    dependencies=[Depends(require_user_identity)],
)
async def last_decision(
    user_id: str = Path(..., pattern=USER_ID_REGEX, min_length=1, max_length=64),
) -> JSONResponse:
    """Return the most recent explanation cached for ``user_id``.

    Raises:
        HTTPException: 404 if no cached explanation exists for the
            given ``user_id``.
    """
    payload = _CACHE.get(user_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No cached explanation")
    return JSONResponse(payload.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def include_explain_routes(app: FastAPI) -> None:
    """Mount the explain router onto ``app``.

    Args:
        app: The :class:`FastAPI` application from
            :func:`server.app.create_app`.
    """
    app.include_router(router)


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_TOP_K",
    "DimensionConfidence",
    "ExplainRequest",
    "ExplainResponse",
    "include_explain_routes",
    "router",
]
