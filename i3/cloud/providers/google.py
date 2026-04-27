"""Google Gemini provider adapter.

Targets the ``google-generativeai`` SDK and Gemini 2.5 models
(``gemini-2.5-pro``, ``gemini-2.5-flash``).

Gemini uses a ``contents`` / ``parts`` message shape that does not
have a top-level system role; the adapter injects the I3 ``system``
prompt as a leading ``user`` turn (the most broadly compatible
workaround, consistent with Google's own documentation).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from i3.cloud.providers.base import (
    AuthError,
    CompletionRequest,
    CompletionResult,
    PermanentError,
    ProviderError,
    RateLimitedError,
    TokenUsage,
    TransientError,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The 'google-generativeai' package is required for GoogleProvider.  "
    "Install it with: pip install google-generativeai>=0.7"
)


class GoogleProvider:
    """Google Gemini (via ``google-generativeai``) provider adapter.

    Args:
        model: Gemini model id, e.g. ``gemini-2.5-pro`` or
            ``gemini-2.5-flash``.
        max_tokens: Default upper bound on generated tokens.
        timeout: Unused by the google-generativeai SDK directly, but
            kept for parity with sibling adapters (future SDK versions
            accept a per-request timeout kwarg).
    """

    provider_name: str = "google"

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-pro",
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        self._model_id = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._configured = False
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._model is not None:
            return self._model

        # Iter 51: accept either ``GOOGLE_API_KEY`` (Vertex/GCP) or the
        # AI-Studio-style ``GEMINI_API_KEY``.  We prefer the
        # AI-Studio key when both are set since AI Studio is the
        # cheaper, key-only path most internship demos use.
        api_key = (
            os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if not api_key:
            raise AuthError(
                "neither GEMINI_API_KEY nor GOOGLE_API_KEY is set",
                provider=self.provider_name,
            )

        # Prefer the new ``google-genai`` SDK (the supported route for
        # 2026+); fall back to the deprecated ``google-generativeai``
        # only when the new package isn't installed.
        try:
            from google import genai as new_genai  # type: ignore[import-not-found]
            self._client = new_genai.Client(api_key=api_key)
            self._sdk = "new"
            # The new SDK doesn't have a per-model handle — we call
            # ``client.models.generate_content(model=...)`` directly.
            # Store a sentinel so callers know ``_ensure_client`` ran.
            self._model = self._client
            self._configured = True
            return self._model
        except ImportError:
            pass

        try:
            import google.generativeai as old_genai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        if not self._configured:
            old_genai.configure(api_key=api_key)
            self._configured = True
        self._sdk = "old"
        self._model = old_genai.GenerativeModel(self._model_id)
        return self._model

    @staticmethod
    def _to_contents(
        request: CompletionRequest,
    ) -> list[dict[str, Any]]:
        """Build the ``contents`` array for Gemini.

        The first turn is the ``system`` prompt (if any) squashed in
        as a ``user`` role -- Gemini's "system_instruction" field is
        SDK-version dependent, the leading-user approach is the
        broadest-compatible path.
        """
        contents: list[dict[str, Any]] = []
        if request.system:
            contents.append(
                {"role": "user", "parts": [{"text": request.system}]}
            )
            contents.append(
                {"role": "model", "parts": [{"text": "Understood."}]}
            )
        for m in request.messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return contents

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        """Generate a response via Gemini."""
        model = self._ensure_client()
        generation_config: dict[str, Any] = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens or self._max_tokens,
        }
        if request.stop:
            generation_config["stop_sequences"] = request.stop

        start = time.monotonic()
        try:
            if getattr(self, "_sdk", "old") == "new":
                # google-genai (the supported SDK).  Different call
                # surface — ``client.models.generate_content`` takes
                # ``model=`` and a ``config=`` GenerateContentConfig.
                from google.genai import types as gtypes
                cfg_kwargs: dict = dict(
                    temperature=generation_config["temperature"],
                    max_output_tokens=generation_config["max_output_tokens"],
                    stop_sequences=generation_config.get("stop_sequences"),
                )
                # Gemini 2.5 Flash spends an opaque chunk of the token
                # budget on internal "thinking" before any output is
                # emitted.  For chat we want every token visible, so
                # disable thinking when the SDK supports the field.
                try:
                    cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(
                        thinking_budget=0,
                    )
                except Exception:
                    pass
                cfg = gtypes.GenerateContentConfig(**cfg_kwargs)
                # The new client is sync; run it off-loop so we don't
                # block the event loop on the network round-trip.
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model_id,
                    contents=self._to_contents(request),
                    config=cfg,
                )
            elif hasattr(model, "generate_content_async"):
                # google-generativeai async path (legacy).
                response = await model.generate_content_async(
                    self._to_contents(request),
                    generation_config=generation_config,
                )
            else:
                # google-generativeai sync path (legacy fallback).
                response = await asyncio.to_thread(
                    model.generate_content,
                    self._to_contents(request),
                    generation_config=generation_config,
                )
        except Exception as exc:
            msg = str(exc)
            lowered = msg.lower()
            if "unauthenticated" in lowered or "api key" in lowered:
                raise AuthError(msg, provider=self.provider_name) from exc
            if "rate" in lowered or "quota" in lowered or "429" in lowered:
                raise RateLimitedError(
                    msg, provider=self.provider_name
                ) from exc
            if "deadline" in lowered or "timeout" in lowered:
                raise TransientError(
                    msg, provider=self.provider_name
                ) from exc
            if "invalid" in lowered or "not found" in lowered:
                raise PermanentError(
                    msg, provider=self.provider_name
                ) from exc
            raise ProviderError(
                f"Google adapter failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc
        latency_ms = int((time.monotonic() - start) * 1000.0)

        # ``response.text`` is a *property* on the new SDK that raises
        # when the candidate has no text Part (safety filter, hit
        # MAX_TOKENS before any content was emitted, etc.).  Drop to
        # the underlying parts list when that happens.  Either way we
        # never let this raise out of complete() — the chain treats an
        # empty CompletionResult as a soft fallback.
        text = ""
        try:
            text = getattr(response, "text", None) or ""
        except Exception:
            cands = getattr(response, "candidates", None) or []
            for cand in cands:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        text += t
                if text:
                    break
        usage_meta = getattr(response, "usage_metadata", None)
        usage = TokenUsage(
            prompt_tokens=int(
                getattr(usage_meta, "prompt_token_count", 0) or 0
            ),
            completion_tokens=int(
                getattr(usage_meta, "candidates_token_count", 0) or 0
            ),
            total_tokens=int(
                getattr(usage_meta, "total_token_count", 0) or 0
            ),
            cached_tokens=int(
                getattr(usage_meta, "cached_content_token_count", 0) or 0
            ),
        )
        finish = "stop"
        candidates = getattr(response, "candidates", None)
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            if reason is not None:
                # Gemini uses SDK enums; stringify and lowercase.
                finish = str(reason).split(".")[-1].lower()
        return CompletionResult(
            text=text,
            provider=self.provider_name,
            model=self._model_id,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish,
        )

    async def close(self) -> None:
        """No persistent connection in the SDK; drop the model handle."""
        self._model = None


__all__ = ["GoogleProvider"]
