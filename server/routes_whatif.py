"""REST endpoints for "what-if" adaptation-override experiments.

These endpoints power the interpretability panel in the demo UI.  They
let a user (or the interviewer) ask *"what would the response look like
if the cognitive_load were 0.9?"* without having to somehow drive the
user into that state.  The endpoints:

- override the computed :class:`AdaptationVector` for a single forward
  pass through the SLM, and
- optionally fan out to several override variants in parallel so the UI
  can render a side-by-side comparison.

Security posture
----------------
These endpoints inherit the full middleware stack (rate limit, size
limit, security headers, CORS), and reuse the same ``user_id`` regex
used by :mod:`server.routes` to prevent path-traversal payloads.  They
NEVER persist state -- the override affects one forward pass and the
user's real session is untouched.

This routes module is wired into the FastAPI application by the single
call to :func:`include_whatif_routes` in :mod:`server.app`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from server.auth import require_user_identity_from_body

logger = logging.getLogger(__name__)

# SEC: Anchored user_id regex -- matches ``server.routes.USER_ID_REGEX`` so
# the two endpoint families have a uniform input-validation surface.
USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

# Hard caps on request shape to stop a malicious client from burning GPU.
MAX_MESSAGE_CHARS = 4000
MAX_VARIANTS = 4

# 8-dim AdaptationVector layout:
#   [0] cognitive_load  [1] style.formality     [2] style.verbosity
#   [3] style.emotionality [4] style.directness [5] emotional_tone
#   [6] accessibility   [7] reserved
ADAPT_DIM = 8


router = APIRouter(prefix="/whatif", tags=["whatif"])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class AdaptationOverride(BaseModel):
    """Wire-format AdaptationVector override.

    Every field is optional -- omitted fields fall back to the value of
    the corresponding dimension in the user's currently-computed
    :class:`AdaptationVector`.  All floats are clamped to ``[0, 1]``
    before use (NaN / inf safe).

    Attributes:
        cognitive_load: Cognitive load in ``[0, 1]``.
        formality: Style-mirror formality in ``[0, 1]``.
        verbosity: Style-mirror verbosity in ``[0, 1]``.
        emotionality: Style-mirror emotionality in ``[0, 1]``.
        directness: Style-mirror directness in ``[0, 1]``.
        emotional_tone: Emotional tone in ``[0, 1]``.
        accessibility: Accessibility mode in ``[0, 1]``.
    """

    cognitive_load: float | None = Field(default=None, ge=0.0, le=1.0)
    formality: float | None = Field(default=None, ge=0.0, le=1.0)
    verbosity: float | None = Field(default=None, ge=0.0, le=1.0)
    emotionality: float | None = Field(default=None, ge=0.0, le=1.0)
    directness: float | None = Field(default=None, ge=0.0, le=1.0)
    emotional_tone: float | None = Field(default=None, ge=0.0, le=1.0)
    accessibility: float | None = Field(default=None, ge=0.0, le=1.0)


class WhatIfRespondRequest(BaseModel):
    """Request body for ``POST /whatif/respond``.

    Attributes:
        user_id: Validated user identifier.
        message: Prompt text for the what-if generation.
        override_adaptation: Partial or full AdaptationVector override.
    """

    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    override_adaptation: AdaptationOverride = Field(
        default_factory=AdaptationOverride
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: str) -> str:
        """Strip surrounding whitespace and reject empty-after-strip values."""
        # SEC: ``min_length=1`` already rejects empty strings, but after
        # stripping an all-whitespace payload could still slip through.
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must contain non-whitespace text")
        return stripped


class WhatIfCompareRequest(BaseModel):
    """Request body for ``POST /whatif/compare``.

    Attributes:
        user_id: Validated user identifier.
        message: Prompt text to generate against for each variant.
        adaptation_variants: Up to four ``AdaptationOverride``
            alternatives to compare.  Excess variants past
            :data:`MAX_VARIANTS` are rejected with HTTP 422.
    """

    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    adaptation_variants: list[AdaptationOverride] = Field(
        default_factory=list, min_length=1, max_length=MAX_VARIANTS
    )

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: str) -> str:
        """Same whitespace validator as :class:`WhatIfRespondRequest`."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must contain non-whitespace text")
        return stripped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pipeline(request: Request) -> Any:
    """Fetch the pipeline from app.state or raise 503.

    SEC: matches the pattern used in :mod:`server.routes` so the two
    endpoint families never diverge on service-unavailable handling.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return pipeline


def _apply_override(
    base: Any,
    override: AdaptationOverride,
) -> tuple[torch.Tensor, Any]:
    """Produce an 8-dim ``AdaptationVector`` tensor with overrides applied.

    Args:
        base: The user's currently-computed
            :class:`~i3.adaptation.types.AdaptationVector` (may be
            ``None`` -- in which case a neutral default is used).
        override: The partial-override payload from the client.

    Returns:
        A pair ``(tensor, adaptation_vector)`` where ``tensor`` is a
        1-D ``torch.Tensor`` of shape ``[8]`` suitable for passing to
        :meth:`SLMGenerator.generate`, and ``adaptation_vector`` is the
        :class:`~i3.adaptation.types.AdaptationVector` dataclass used
        to re-serialise the clamped values for the JSON response.
    """
    from i3.adaptation.types import AdaptationVector

    if base is None or not hasattr(base, "to_tensor"):
        base = AdaptationVector.default()

    # Start from the baseline tensor then patch each provided override.
    values = base.to_tensor().tolist()  # length 8, layout documented above.

    def _set(idx: int, val: float | None) -> None:
        if val is None:
            return
        # Field validators already clamped to [0, 1]; defensive clamp here
        # guards against future model changes that might loosen the
        # validator or against NaN slipping through via an old client.
        try:
            v = float(val)
        except (TypeError, ValueError):
            return
        if v != v:  # NaN check without importing math
            return
        values[idx] = max(0.0, min(1.0, v))

    _set(0, override.cognitive_load)
    _set(1, override.formality)
    _set(2, override.verbosity)
    _set(3, override.emotionality)
    _set(4, override.directness)
    _set(5, override.emotional_tone)
    _set(6, override.accessibility)
    # idx 7 (reserved) is always zero.
    values[7] = 0.0

    tensor = torch.tensor(values, dtype=torch.float32)
    # Round-trip through AdaptationVector to re-use its clamp semantics.
    av = AdaptationVector.from_tensor(tensor)
    return av.to_tensor(), av


async def _compute_base_adaptation(pipeline: Any, user_id: str) -> Any:
    """Fetch the user's current AdaptationVector from the pipeline.

    Falls back to :meth:`AdaptationVector.default` if the user has no
    active session or anything goes wrong.
    """
    from i3.adaptation.types import AdaptationVector

    # Best-effort: use the pipeline's stored user model, if present.
    try:
        user_models = getattr(pipeline, "user_models", {})
        um = user_models.get(user_id)
        if um is not None:
            current_style = getattr(
                pipeline.adaptation, "current_style", None
            )
            # The pipeline's adaptation controller holds the *last*
            # computed style in ``_current_style``; the full vector must
            # be reconstructed from the controller + current user model
            # state.  We can't recompute without a feature vector, so
            # return a neutral default that is then overridden by the
            # client-supplied dimensions.
            if current_style is not None:
                return AdaptationVector(
                    cognitive_load=0.5,
                    style_mirror=current_style,
                    emotional_tone=0.5,
                    accessibility=0.0,
                )
    except Exception:  # pragma: no cover - defensive
        logger.exception("base-adaptation lookup failed; using default.")

    return AdaptationVector.default()


def _build_user_state(pipeline: Any, user_id: str) -> torch.Tensor:
    """Return the user's 64-dim user-state embedding, or zeros.

    Uses the pipeline's in-memory ``user_models`` when available.
    Always returns a non-``None`` tensor so downstream cross-attention
    never sees a missing key/value.
    """
    try:
        um = getattr(pipeline, "user_models", {}).get(user_id)
        if um is not None:
            state = getattr(um, "current_state_embedding", None)
            if isinstance(state, torch.Tensor) and state.numel() == 64:
                return state.detach().cpu()
    except Exception:  # pragma: no cover - defensive
        logger.exception("user-state lookup failed; using zero embedding.")
    return torch.zeros(64, dtype=torch.float32)


async def _generate_with_override(
    pipeline: Any,
    message: str,
    adaptation_override: torch.Tensor,
    adaptation_obj: Any,
    user_state: torch.Tensor,
) -> str:
    """Run the pipeline's generator with a forced AdaptationVector.

    Tries retrieval first (so the same Q→A pair the chat would have
    picked is reused as the substrate), then runs the result through
    the pipeline's ResponsePostProcessor with the forced adaptation —
    that's what makes the override *visibly* reshape the reply
    (truncation, contraction expand/contract, hedge stripping,
    accessibility simplification).

    Falls back to the local SLM cross-attention generator when no
    retrieval candidate clears the threshold.
    """
    base_text: str | None = None

    # 1. Retrieval-first: reuse the same Q→A substrate the chat path
    #    would have picked, so the diff between profiles is purely the
    #    rewriting (not generator stochasticity).
    retriever = getattr(pipeline, "_slm_retriever", None)
    if retriever is not None:
        try:
            adapt_dict = pipeline._adaptation_to_dict(adaptation_obj)
            match = retriever.best(message, adaptation=adapt_dict, min_score=0.30)
            if match is not None:
                cand, score = match
                if score >= 0.55:
                    base_text = cand
        except Exception:
            logger.debug("whatif retrieval failed; falling through to SLM.")

    # 2. SLM cross-attention generator if retrieval didn't have a
    #    confident answer.
    if base_text is None:
        slm_gen = getattr(pipeline, "_slm_generator", None)
        if slm_gen is not None:
            try:
                raw = slm_gen.generate(
                    prompt=message,
                    adaptation_vector=adaptation_override.unsqueeze(0),
                    user_state=user_state.unsqueeze(0),
                )
                base_text = pipeline._clean_slm_output(raw, prompt=message)
            except Exception:
                logger.exception(
                    "SLM generation failed in what-if; using rule-based fallback."
                )

    # 3. Apply adaptation rewriting on top of whatever substrate we got.
    if base_text:
        try:
            adapted, _log = pipeline.postprocessor.adapt_with_log(
                base_text, adaptation_obj
            )
            return adapted
        except Exception:
            logger.exception(
                "adapt_with_log failed in what-if; returning unmodified base."
            )
            return base_text

    # Rule-based fallback mirrors the pipeline's ``_fallback_response``.
    try:
        fallback = getattr(pipeline, "_fallback_response", None)
        if fallback is not None:
            fallback_text: str = fallback(adaptation_obj)
            return fallback_text
    except Exception:  # pragma: no cover - defensive
        logger.exception("Fallback response failed; returning stub.")
    return "(what-if response unavailable)"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/respond",
    dependencies=[Depends(require_user_identity_from_body)],
)
async def whatif_respond(
    request: Request,
    body: WhatIfRespondRequest,
) -> JSONResponse:
    """Generate a response with an overridden AdaptationVector.

    Args:
        request: The FastAPI request (used to access app state).
        body: Parsed :class:`WhatIfRespondRequest`.

    Returns:
        JSON with ``text`` (the response), ``latency_ms`` and
        ``adaptation_used`` (the final 8-dim vector actually passed to
        the SLM, as a dict for the front-end).
    """
    pipeline = _get_pipeline(request)
    start = time.perf_counter()
    try:
        base = await _compute_base_adaptation(pipeline, body.user_id)
        adaptation_tensor, adaptation_obj = _apply_override(
            base, body.override_adaptation
        )
        user_state = _build_user_state(pipeline, body.user_id)
        text = await _generate_with_override(
            pipeline,
            body.message,
            adaptation_tensor,
            adaptation_obj,
            user_state,
        )
    except HTTPException:
        raise
    except Exception:
        # SEC: never leak the exception type or message to the client.
        logger.exception("whatif_respond failed")
        raise HTTPException(status_code=500, detail="Internal error")

    latency_ms = (time.perf_counter() - start) * 1000.0
    return JSONResponse(
        {
            "text": text,
            "latency_ms": round(latency_ms, 2),
            "adaptation_used": adaptation_obj.to_dict(),
        }
    )


@router.post(
    "/compare",
    dependencies=[Depends(require_user_identity_from_body)],
)
async def whatif_compare(
    request: Request,
    body: WhatIfCompareRequest,
) -> JSONResponse:
    """Generate parallel responses for several adaptation variants.

    Args:
        request: The FastAPI request.
        body: Parsed :class:`WhatIfCompareRequest` -- 1 to 4 variants.

    Returns:
        JSON ``{"variants": [{"adaptation", "text", "latency_ms"}, ...]}``
        with exactly ``len(body.adaptation_variants)`` entries, preserving
        their original order.
    """
    pipeline = _get_pipeline(request)
    base = await _compute_base_adaptation(pipeline, body.user_id)
    user_state = _build_user_state(pipeline, body.user_id)

    variants_out: list[dict[str, Any]] = []
    for override in body.adaptation_variants:
        start = time.perf_counter()
        try:
            adaptation_tensor, adaptation_obj = _apply_override(base, override)
            text = await _generate_with_override(
                pipeline,
                body.message,
                adaptation_tensor,
                adaptation_obj,
                user_state,
            )
        except HTTPException:
            raise
        except Exception:
            # SEC: on per-variant failure, emit a stub with no exception
            # details in the body and keep processing the remaining
            # variants so one bad override cannot tank the whole batch.
            logger.exception("whatif_compare variant failed")
            variants_out.append(
                {
                    "adaptation": None,
                    "text": "(variant failed)",
                    "latency_ms": 0.0,
                }
            )
            continue
        latency_ms = (time.perf_counter() - start) * 1000.0
        variants_out.append(
            {
                "adaptation": adaptation_obj.to_dict(),
                "text": text,
                "latency_ms": round(latency_ms, 2),
            }
        )

    return JSONResponse({"variants": variants_out})


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------


def include_whatif_routes(app: FastAPI) -> None:
    """Mount the what-if router onto ``app`` under ``/whatif``.

    Args:
        app: The :class:`FastAPI` application instance created by
            :func:`server.app.create_app`.

    Notes:
        The router is registered under the ``/whatif`` prefix (baked
        into :data:`router`).  Including it multiple times will raise
        a FastAPI internal assertion; callers should call this exactly
        once per application.
    """
    # SEC: mounted under /api so the what-if / interpretability routes
    # live on the same namespace as the rest of the REST surface.  The
    # router's own ``/whatif`` prefix is preserved, giving the final
    # paths ``/api/whatif/respond`` and ``/api/whatif/compare`` — the
    # forms assumed by the Advanced UI client and the verification
    # harness.
    app.include_router(router, prefix="/api")


__all__ = [
    "AdaptationOverride",
    "WhatIfCompareRequest",
    "WhatIfRespondRequest",
    "include_whatif_routes",
    "router",
]
