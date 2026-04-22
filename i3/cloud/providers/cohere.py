"""Cohere provider adapter.

Targets Cohere's ``AsyncClient`` and the Command-R family
(``command-r-plus``, ``command-r``).

Cohere's chat API uses a ``chat_history`` + ``message`` (current
turn) shape, distinct from OpenAI-style flat messages lists.
"""

from __future__ import annotations

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
    "The 'cohere' package is required for CohereProvider.  "
    "Install it with: pip install cohere>=5.0"
)


class CohereProvider:
    """Cohere provider adapter.

    Args:
        model: Cohere model id, e.g. ``command-r-plus`` or
            ``command-r``.
        max_tokens: Default upper bound on generated tokens.
        timeout: Per-request HTTP timeout in seconds.
    """

    provider_name: str = "cohere"

    def __init__(
        self,
        *,
        model: str = "command-r-plus",
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise AuthError(
                "COHERE_API_KEY is not set",
                provider=self.provider_name,
            )
        self._client = cohere.AsyncClient(
            api_key=api_key, timeout=self._timeout
        )
        return self._client

    @staticmethod
    def _split(
        request: CompletionRequest,
    ) -> tuple[str, list[dict[str, str]], str]:
        """Split the request into Cohere's (preamble, chat_history, message)."""
        preamble = request.system or ""
        non_system = [m for m in request.messages if m.role != "system"]
        if not non_system:
            return preamble, [], ""
        last = non_system[-1]
        if last.role == "user":
            current = last.content
            prior = non_system[:-1]
        else:
            current = ""
            prior = non_system

        role_map = {"user": "USER", "assistant": "CHATBOT"}
        chat_history = [
            {"role": role_map.get(m.role, "USER"), "message": m.content}
            for m in prior
        ]
        return preamble, chat_history, current

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        client = self._ensure_client()
        preamble, history, user_message = self._split(request)

        start = time.monotonic()
        try:
            response = await client.chat(
                model=self._model,
                preamble=preamble or None,
                chat_history=history or None,
                message=user_message,
                max_tokens=request.max_tokens or self._max_tokens,
                temperature=request.temperature,
                stop_sequences=request.stop,
            )
        except Exception as exc:
            lowered = str(exc).lower()
            name = type(exc).__name__.lower()
            if "unauthor" in lowered or "api key" in lowered or "401" in lowered:
                raise AuthError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "rate" in lowered or "429" in lowered or "too many" in lowered:
                raise RateLimitedError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "timeout" in name or "timeout" in lowered:
                raise TransientError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "invalid" in lowered or "400" in lowered:
                raise PermanentError(
                    str(exc), provider=self.provider_name
                ) from exc
            raise ProviderError(
                f"Cohere adapter failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000.0)
        text = getattr(response, "text", "") or ""
        finish = str(
            getattr(response, "finish_reason", "COMPLETE") or "COMPLETE"
        ).lower()
        meta = getattr(response, "meta", None)
        billed = getattr(meta, "billed_units", None) if meta else None
        p_tokens = int(getattr(billed, "input_tokens", 0) or 0) if billed else 0
        c_tokens = (
            int(getattr(billed, "output_tokens", 0) or 0) if billed else 0
        )
        usage = TokenUsage(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
        )
        return CompletionResult(
            text=text,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish,
        )

    async def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                try:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result  # type: ignore[misc]
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "cohere close() raised %s", type(exc).__name__
                    )
            self._client = None


__all__ = ["CohereProvider"]
