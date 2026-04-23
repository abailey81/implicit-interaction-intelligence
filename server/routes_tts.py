"""Adaptation-conditioned text-to-speech endpoints.

This router mounts three routes under ``/api/tts``:

* ``POST /api/tts`` — synthesise arbitrary text using the user's
  currently-computed :class:`~i3.adaptation.types.AdaptationVector`
  (or a what-if override if supplied).
* ``GET /api/tts/backends`` — enumerate which TTS backends are
  installed on the server host.
* ``GET /api/tts/preview?archetype=<name>`` — render a canonical phrase
  under one of eight reference archetype vectors so the UI can let the
  user hear the difference.

The route layer is responsible for:

1. Enforcing the 8 KB hard body cap (so a rogue client cannot spend
   synthesis compute on a giant payload).
2. PII-sanitising the input text with :class:`PrivacySanitizer` — TTS
   input is free-form user content and must never be shipped to a
   third-party speech engine with raw PII in it.
3. Falling back to a neutral :class:`AdaptationVector` when the
   pipeline is not installed (standalone testing).

The actual TTS dispatch is delegated to :class:`i3.tts.TTSEngine`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from server.auth import require_user_identity_from_body
from pydantic import BaseModel, ConfigDict, Field, field_validator

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.privacy.sanitizer import PrivacySanitizer
from i3.tts import (
    TTSEngine,
    TTSOutput,
    TTSParams,
    derive_tts_params,
    explain_params,
    list_backend_statuses,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

# SEC: hard body-size cap.  TTS input is expected to be a single
# response; anything above 8 KB is rejected with 413 so synthesis
# compute is never spent on an abusively large payload.
MAX_BODY_BYTES: int = 8 * 1024

# Cap on the decoded text field — also enforced by the engine.
MAX_TEXT_CHARS: int = 2000

# Canonical phrase used by the preview endpoint (and by the demo CLI).
# Matches docs/research/adaptive_tts.md §4.
CANONICAL_PREVIEW_PHRASE: str = (
    "I've noted that you asked for a shorter response — here it is."
)

# Canonical set of archetype AdaptationVectors for the preview /
# demo-script matrix.  Each archetype isolates one dimension so users
# can hear the effect of that dimension in isolation.
ARCHETYPES: dict[str, AdaptationVector] = {
    "neutral": AdaptationVector.default(),
    "cognitive_load_high": AdaptationVector(
        cognitive_load=0.9,
        style_mirror=StyleVector.default(),
        emotional_tone=0.5,
        accessibility=0.0,
    ),
    "cognitive_load_low": AdaptationVector(
        cognitive_load=0.1,
        style_mirror=StyleVector.default(),
        emotional_tone=0.5,
        accessibility=0.0,
    ),
    "accessibility_high": AdaptationVector(
        cognitive_load=0.5,
        style_mirror=StyleVector.default(),
        emotional_tone=0.5,
        accessibility=0.85,
    ),
    "emotional_warm": AdaptationVector(
        cognitive_load=0.3,
        style_mirror=StyleVector(
            formality=0.3, verbosity=0.4, emotionality=0.8, directness=0.4
        ),
        emotional_tone=0.0,  # fully warm
        accessibility=0.0,
    ),
    "emotional_neutral": AdaptationVector(
        cognitive_load=0.3,
        style_mirror=StyleVector(
            formality=0.7, verbosity=0.4, emotionality=0.2, directness=0.7
        ),
        emotional_tone=1.0,  # fully neutral
        accessibility=0.0,
    ),
    "formal": AdaptationVector(
        cognitive_load=0.4,
        style_mirror=StyleVector(
            formality=0.95, verbosity=0.6, emotionality=0.3, directness=0.7
        ),
        emotional_tone=0.6,
        accessibility=0.0,
    ),
    "casual": AdaptationVector(
        cognitive_load=0.2,
        style_mirror=StyleVector(
            formality=0.05, verbosity=0.3, emotionality=0.6, directness=0.5
        ),
        emotional_tone=0.3,
        accessibility=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Pydantic wire models
# ---------------------------------------------------------------------------


class AdaptationOverride(BaseModel):
    """Optional override applied to the resolved :class:`AdaptationVector`.

    Identical shape to :class:`server.routes_whatif.AdaptationOverride`
    — any field ``None`` falls back to the user's live value.
    """

    model_config = ConfigDict(extra="forbid")

    cognitive_load: float | None = Field(default=None, ge=0.0, le=1.0)
    formality: float | None = Field(default=None, ge=0.0, le=1.0)
    verbosity: float | None = Field(default=None, ge=0.0, le=1.0)
    emotionality: float | None = Field(default=None, ge=0.0, le=1.0)
    directness: float | None = Field(default=None, ge=0.0, le=1.0)
    emotional_tone: float | None = Field(default=None, ge=0.0, le=1.0)
    accessibility: float | None = Field(default=None, ge=0.0, le=1.0)


class TTSRequest(BaseModel):
    """Wire model for ``POST /api/tts``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    backend_hint: str | None = Field(default=None, max_length=32)
    override_adaptation: AdaptationOverride | None = None

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        """Reject all-whitespace payloads."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must contain non-whitespace content.")
        return stripped

    @field_validator("backend_hint")
    @classmethod
    def _validate_backend_hint(cls, v: str | None) -> str | None:
        """Whitelist backend hints to the four known names."""
        if v is None:
            return None
        allowed = {"pyttsx3", "piper", "kokoro", "web_speech_api"}
        if v not in allowed:
            raise ValueError(f"backend_hint must be one of {sorted(allowed)}")
        return v


class TTSResponse(BaseModel):
    """Wire model for a successful TTS response.

    Mirrors :class:`TTSOutput` with the additional ``explanation`` field
    surfaced by :func:`i3.tts.conditioning.explain_params` and the
    privacy / adaptation context the UI uses for the caption.
    """

    model_config = ConfigDict(extra="forbid")

    audio_wav_base64: str | None = None
    directive: dict[str, Any] | None = None
    sample_rate_hz: int
    duration_ms: int
    backend_name: str
    params_used: dict[str, Any]
    adaptation_applied: dict[str, Any]
    explanation: str
    pii_redactions: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Router + shared state
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/api/tts", tags=["tts"])

# Module-level sanitiser and engine so each request reuses compiled
# regexes and backend-probe state.
_SANITIZER = PrivacySanitizer(enabled=True)
_ENGINE = TTSEngine(allow_web_speech=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_override(
    base: AdaptationVector, override: AdaptationOverride | None
) -> AdaptationVector:
    """Return a new :class:`AdaptationVector` with *override* fields applied.

    Args:
        base: Base vector (normally the pipeline-resolved one).
        override: Optional partial override from the request body.

    Returns:
        A fresh :class:`AdaptationVector`.  *base* is not mutated.
    """
    if override is None:
        return base

    def pick(a: float | None, b: float) -> float:
        return float(b if a is None else a)

    new_style = StyleVector(
        formality=pick(override.formality, base.style_mirror.formality),
        verbosity=pick(override.verbosity, base.style_mirror.verbosity),
        emotionality=pick(override.emotionality, base.style_mirror.emotionality),
        directness=pick(override.directness, base.style_mirror.directness),
    )
    return AdaptationVector(
        cognitive_load=pick(override.cognitive_load, base.cognitive_load),
        style_mirror=new_style,
        emotional_tone=pick(override.emotional_tone, base.emotional_tone),
        accessibility=pick(override.accessibility, base.accessibility),
    )


async def _resolve_adaptation(request: Request, user_id: str) -> AdaptationVector:
    """Look up the user's current :class:`AdaptationVector`.

    Matches the resolution logic in :mod:`server.routes_translate`: if
    the pipeline is not installed, fall back to
    :meth:`AdaptationVector.default`.

    Args:
        request: FastAPI request (used for app-state access).
        user_id: Already-validated user id.

    Returns:
        A usable adaptation vector.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return AdaptationVector.default()
    try:
        user_models = getattr(pipeline, "user_models", {})
        um = user_models.get(user_id)
        if um is not None:
            current_style = getattr(
                getattr(pipeline, "adaptation", object()), "current_style", None
            )
            if current_style is not None:
                return AdaptationVector(
                    cognitive_load=0.5,
                    style_mirror=current_style,
                    emotional_tone=0.5,
                    accessibility=0.0,
                )
    except AttributeError:
        logger.debug("Pipeline lacks adaptation fields; using default vector.")
    return AdaptationVector.default()


def _synthesise(
    text: str,
    adaptation: AdaptationVector,
    backend_hint: str | None,
) -> tuple[TTSOutput, TTSParams, str]:
    """Dispatch the TTS call and build the explanation string.

    Args:
        text: Sanitised text.
        adaptation: The vector to condition on.
        backend_hint: Optional backend name.

    Returns:
        A tuple ``(output, params, explanation)``.

    Raises:
        HTTPException: 503 if no TTS backend is available, 500 on
            unexpected backend errors.
    """
    params = derive_tts_params(adaptation)
    try:
        output = _ENGINE.speak(text, params, backend_hint=backend_hint)
    except RuntimeError as exc:
        # SEC: log the full exception server-side but never echo the
        # message (it may contain install paths).
        logger.warning("TTS synthesis unavailable: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="TTS backend unavailable"
        ) from exc
    except ValueError as exc:
        logger.debug("TTS validation failure: %s", type(exc).__name__)
        raise HTTPException(status_code=422, detail="Invalid TTS request") from exc
    explanation = explain_params(params, adaptation)
    return output, params, explanation


def _to_response(
    output: TTSOutput,
    adaptation: AdaptationVector,
    explanation: str,
    pii_redactions: int,
    latency_ms: float,
) -> TTSResponse:
    """Build the outbound :class:`TTSResponse`.

    Args:
        output: Backend output.
        adaptation: The vector actually used.
        explanation: Natural-language conditioning summary.
        pii_redactions: Number of PII fragments scrubbed.
        latency_ms: Round-trip latency in ms.

    Returns:
        A ready-to-serialise :class:`TTSResponse`.
    """
    return TTSResponse(
        audio_wav_base64=output.audio_wav_base64,
        directive=output.directive,
        sample_rate_hz=output.sample_rate_hz,
        duration_ms=output.duration_ms,
        backend_name=output.backend_name,
        params_used=output.params_used.model_dump(mode="json"),
        adaptation_applied=adaptation.to_dict(),
        explanation=explanation,
        pii_redactions=pii_redactions,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    dependencies=[Depends(require_user_identity_from_body)],
)
async def synthesise_tts(request: Request) -> JSONResponse:
    """Synthesise speech for arbitrary text.

    Args:
        request: The FastAPI request.  Body is parsed manually so we
            can enforce :data:`MAX_BODY_BYTES` before Pydantic runs.

    Returns:
        :class:`TTSResponse` on success, or a JSON error document.

    Raises:
        HTTPException: 413 when the body exceeds the cap, 422 on
            validation errors, 503 when no TTS backend is available.
    """
    start = time.perf_counter()
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        body = TTSRequest.model_validate_json(raw_body)
    except ValueError as exc:
        logger.debug("tts: validation failure: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid request payload") from exc

    # --- Privacy: strip PII before any synthesis step. ---
    sanitisation = _SANITIZER.sanitize(body.text)
    sanitised_text = sanitisation.sanitized_text
    pii_redactions = sanitisation.replacements_made

    # The text is user-supplied.  Enforce the char cap again after
    # sanitisation (sanitiser never grows text, but belt-and-braces).
    if len(sanitised_text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail="Invalid request payload")

    base_adaptation = await _resolve_adaptation(request, body.user_id)
    adaptation = _apply_override(base_adaptation, body.override_adaptation)

    output, _params, explanation = _synthesise(
        text=sanitised_text,
        adaptation=adaptation,
        backend_hint=body.backend_hint,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    resp = _to_response(
        output=output,
        adaptation=adaptation,
        explanation=explanation,
        pii_redactions=pii_redactions,
        latency_ms=latency_ms,
    )
    return JSONResponse(resp.model_dump(mode="json"))


@router.get("/backends")
async def list_backends() -> JSONResponse:
    """Return installation status for every known TTS backend.

    Returns:
        A JSON document with ``backends`` mapping each backend name to
        its availability + install hint.
    """
    payload = {
        "backends": [
            {
                "name": st.name,
                "display_name": st.display_name,
                "available": st.available,
                "install_hint": st.install_hint,
            }
            for st in list_backend_statuses()
        ]
    }
    return JSONResponse(payload)


@router.get("/preview")
async def preview_archetype(
    archetype: str = Query(
        ..., min_length=1, max_length=32, pattern=r"^[a-z_]+$"
    ),
) -> JSONResponse:
    """Render the canonical phrase under a named archetype vector.

    Args:
        archetype: One of the keys of :data:`ARCHETYPES`.  Only
            lowercase ASCII and underscores are accepted.

    Returns:
        :class:`TTSResponse` JSON for the archetype.

    Raises:
        HTTPException: 404 if the archetype name is unknown, 503 when
            no TTS backend is available.
    """
    start = time.perf_counter()
    if archetype not in ARCHETYPES:
        raise HTTPException(status_code=404, detail="Unknown archetype")
    adaptation = ARCHETYPES[archetype]
    sanitised_text = _SANITIZER.sanitize(CANONICAL_PREVIEW_PHRASE).sanitized_text
    output, _params, explanation = _synthesise(
        text=sanitised_text,
        adaptation=adaptation,
        backend_hint=None,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    resp = _to_response(
        output=output,
        adaptation=adaptation,
        explanation=explanation,
        pii_redactions=0,
        latency_ms=latency_ms,
    )
    return JSONResponse(resp.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def include_tts_routes(app: FastAPI) -> None:
    """Mount the TTS router onto ``app``.

    Args:
        app: The :class:`FastAPI` application from
            :func:`server.app.create_app`.
    """
    app.include_router(router)


__all__ = [
    "ARCHETYPES",
    "AdaptationOverride",
    "CANONICAL_PREVIEW_PHRASE",
    "MAX_BODY_BYTES",
    "MAX_TEXT_CHARS",
    "TTSRequest",
    "TTSResponse",
    "include_tts_routes",
    "router",
]
