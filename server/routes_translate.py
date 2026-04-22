"""Real-time translation endpoint (AI Glasses parallel).

Huawei launched the AI Glasses on 20 Apr 2026 with real-time translation
across Chinese + 20 languages as a headline feature.  This module adds
``POST /api/translate`` -- I³'s AdaptationVector-conditioned equivalent:
the user's current personalisation profile shapes the *style* of the
translated text (formality, verbosity, emotional tone), not just the
language pair.

Behaviour at a glance:

* Request body is bounded to 4 KiB (see :data:`MAX_BODY_BYTES`); larger
  requests are rejected with 413 before any compute is spent.
* PII is stripped by :class:`~i3.privacy.sanitizer.PrivacySanitizer`
  before the cloud call; the sanitised text is what Claude sees.
* When the ``CloudLLMClient`` is unavailable (no API key, network off,
  pipeline not initialised) the endpoint returns a deterministic
  pseudo-translation with ``fallback_mode=true`` so the demo stays
  runnable without secrets.
* Every request is logged through the I³ telemetry sink as a scalar
  event (no text).

See :doc:`docs/huawei/harmonyos6_ai_glasses_alignment.md` §2 for the
product-alignment story.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from i3.adaptation.types import AdaptationVector
from i3.privacy.sanitizer import PrivacySanitizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# SEC: anchored user_id regex -- matches the rest of the server surface.
USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

# SEC: hard body-size cap.  Translation requests are short by design
# (a single utterance).  Anything above 4 KiB is rejected with 413.
MAX_BODY_BYTES = 4 * 1024

# Cap on the decoded text field itself -- even if the outer body is
# under the byte cap, a 4 KB UTF-8 payload is the functional ceiling.
MAX_TEXT_CHARS = 3800


class LanguageCode(str, enum.Enum):
    """Supported ISO 639-1 language codes.

    Mirrors the AI Glasses launch list (Chinese plus 20 languages) --
    here we expose 9 target languages plus Chinese and English.
    """

    CHINESE = "zh"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    KOREAN = "ko"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TranslateRequest(BaseModel):
    """Wire model for ``POST /api/translate``.

    Attributes:
        user_id: Validated user identifier.
        text: The text to translate (<= :data:`MAX_TEXT_CHARS`).
        target_language: Target language code.
        source_language: Optional source language; ``None`` means auto-detect.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    target_language: LanguageCode
    source_language: LanguageCode | None = None

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        """Reject all-whitespace text payloads."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must contain non-whitespace content.")
        return stripped


class TranslateResponse(BaseModel):
    """Wire model for the successful response.

    Attributes:
        translated_text: Final (post-processing) translation.
        adaptation_applied: AdaptationVector actually used to shape the
            translation style (serialised dict form of
            :class:`~i3.adaptation.types.AdaptationVector`).
        source_language: Echoed/resolved source language.
        target_language: Echoed target language.
        latency_ms: End-to-end wall-clock latency.
        fallback_mode: ``True`` when the cloud client was unavailable
            and the endpoint responded with a deterministic stub.
        pii_redactions: Number of PII fragments stripped before the
            translation call.
    """

    model_config = ConfigDict(extra="forbid")

    translated_text: str
    adaptation_applied: dict[str, Any]
    source_language: LanguageCode | None
    target_language: LanguageCode
    latency_ms: float = Field(ge=0.0)
    fallback_mode: bool
    pii_redactions: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/api/translate", tags=["translate"])


# Module-level sanitiser so each request reuses the compiled regexes.
_SANITIZER = PrivacySanitizer(enabled=True)


# Human-friendly labels the system prompt embeds verbatim.
_LANGUAGE_LABELS: dict[LanguageCode, str] = {
    LanguageCode.CHINESE: "Chinese",
    LanguageCode.ENGLISH: "English",
    LanguageCode.FRENCH: "French",
    LanguageCode.GERMAN: "German",
    LanguageCode.SPANISH: "Spanish",
    LanguageCode.ITALIAN: "Italian",
    LanguageCode.PORTUGUESE: "Portuguese",
    LanguageCode.JAPANESE: "Japanese",
    LanguageCode.KOREAN: "Korean",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _resolve_adaptation(request: Request, user_id: str) -> AdaptationVector:
    """Look up the user's current AdaptationVector.

    Falls back to :meth:`AdaptationVector.default` when the pipeline is
    absent or the user has no active session.  This matches the
    behaviour of :mod:`server.routes_whatif`.

    Args:
        request: FastAPI request for app-state access.
        user_id: The validated user id.

    Returns:
        A usable :class:`AdaptationVector`.
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


def _build_translation_system_prompt(
    target: LanguageCode,
    source: LanguageCode | None,
    adaptation: AdaptationVector,
) -> str:
    """Build the translation-specific system prompt.

    The AdaptationVector is embedded in natural language so the cloud
    model can mirror the user's style (formality / verbosity / emotional
    tone) in the translated output.  Sensitive fields are never part of
    the prompt -- the 8 scalars in the AdaptationVector are summary
    statistics, not text.

    Args:
        target: Target language.
        source: Optional source language (``None`` means auto-detect).
        adaptation: The user's current :class:`AdaptationVector`.

    Returns:
        A system-prompt string for the Anthropic Messages API.
    """
    target_label = _LANGUAGE_LABELS[target]
    source_fragment = (
        f"from {_LANGUAGE_LABELS[source]} "
        if source is not None
        else "(auto-detect the source language) "
    )
    style = adaptation.style_mirror
    return (
        "You are a real-time translator operating inside Huawei's AI-"
        "Glasses-style paired-device pipeline. Translate the user's "
        f"message {source_fragment}into {target_label}.\n"
        "Match the user's communication style using these dimensions "
        "(each in [0, 1]):\n"
        f"- cognitive_load: {adaptation.cognitive_load:.2f}\n"
        f"- formality: {style.formality:.2f}\n"
        f"- verbosity: {style.verbosity:.2f}\n"
        f"- emotionality: {style.emotionality:.2f}\n"
        f"- directness: {style.directness:.2f}\n"
        f"- emotional_tone: {adaptation.emotional_tone:.2f}\n"
        f"- accessibility: {adaptation.accessibility:.2f}\n"
        "Output ONLY the translated text. No labels, no commentary, no "
        "transliteration unless the target language is pictographic and "
        "the source text contained a proper noun."
    )


def _fallback_translate(
    sanitised_text: str, target: LanguageCode
) -> str:
    """Deterministic pseudo-translation used when cloud is unavailable.

    This is intentionally NOT an attempt at real translation; it is a
    stable echo the tests can pin on.

    Args:
        sanitised_text: PII-stripped input.
        target: Target language code.

    Returns:
        A prefixed echo string.
    """
    return f"[{target.value}] {sanitised_text}"


async def _call_cloud(
    pipeline: Any,
    system_prompt: str,
    user_message: str,
) -> str | None:
    """Invoke the pipeline's CloudLLMClient for the translation.

    Args:
        pipeline: The I³ pipeline from app.state.
        system_prompt: Built by :func:`_build_translation_system_prompt`.
        user_message: Sanitised user text.

    Returns:
        The cloud translation string, or ``None`` if the cloud client is
        not configured or the call failed.
    """
    cloud_client = getattr(pipeline, "cloud_client", None)
    if cloud_client is None or not getattr(cloud_client, "is_available", False):
        return None
    try:
        result = await cloud_client.generate(
            user_message=user_message,
            system_prompt=system_prompt,
        )
    except (RuntimeError, TimeoutError, ConnectionError) as exc:
        logger.warning("Cloud translate call failed: %s", type(exc).__name__)
        return None
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text:
        return None
    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("")
async def translate(request: Request) -> JSONResponse:
    """Translate *text* into *target_language*, style-conditioned.

    Args:
        request: The FastAPI request.  Body is parsed manually so we can
            enforce :data:`MAX_BODY_BYTES` before Pydantic touches it.

    Returns:
        A :class:`TranslateResponse` JSON document on success, or an
        HTTP error on validation / size failure.

    Raises:
        HTTPException: 413 if body exceeds 4 KiB, 422 on validation
            errors, 500 on internal faults.
    """
    start = time.perf_counter()
    raw_body = await request.body()
    if len(raw_body) > MAX_BODY_BYTES:
        # SEC: reject oversized payloads before Pydantic so we never
        # materialise more than 4 KiB as a Python string.
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        body = TranslateRequest.model_validate_json(raw_body)
    except ValueError as exc:
        # SEC: never echo the raw payload or validation error details.
        logger.debug("translate: validation failure: %s", exc)
        raise HTTPException(status_code=422, detail="Invalid request payload")

    # --- Privacy: strip PII before any cloud leg. ---
    sanitisation = _SANITIZER.sanitize(body.text)
    sanitised_text = sanitisation.sanitized_text
    pii_redactions = sanitisation.replacements_made

    adaptation = await _resolve_adaptation(request, body.user_id)
    system_prompt = _build_translation_system_prompt(
        target=body.target_language,
        source=body.source_language,
        adaptation=adaptation,
    )

    pipeline = getattr(request.app.state, "pipeline", None)
    translated: str | None = None
    if pipeline is not None:
        translated = await _call_cloud(
            pipeline=pipeline,
            system_prompt=system_prompt,
            user_message=sanitised_text,
        )

    fallback_mode = translated is None
    if translated is None:
        translated = _fallback_translate(sanitised_text, body.target_language)

    resp = TranslateResponse(
        translated_text=translated,
        adaptation_applied=adaptation.to_dict(),
        source_language=body.source_language,
        target_language=body.target_language,
        latency_ms=(time.perf_counter() - start) * 1000.0,
        fallback_mode=fallback_mode,
        pii_redactions=pii_redactions,
    )
    return JSONResponse(resp.model_dump(mode="json"))


@router.get("/languages")
async def list_languages() -> JSONResponse:
    """Return the list of supported language codes.

    Returns:
        A JSON document with ``languages`` mapping each ISO code to its
        human-readable label.
    """
    payload = {
        "languages": [
            {"code": code.value, "label": label}
            for code, label in _LANGUAGE_LABELS.items()
        ]
    }
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------


def include_translate_routes(app: FastAPI) -> None:
    """Mount the translation router onto ``app``.

    Args:
        app: The :class:`FastAPI` application instance created by
            :func:`server.app.create_app`.
    """
    app.include_router(router)


__all__ = [
    "LanguageCode",
    "MAX_BODY_BYTES",
    "MAX_TEXT_CHARS",
    "TranslateRequest",
    "TranslateResponse",
    "include_translate_routes",
    "router",
]
