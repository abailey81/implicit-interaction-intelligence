"""Mistral AI provider adapter.

Targets the official ``mistralai`` Python SDK.  Models include
``mistral-large-latest`` and ``codestral-latest``.
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
    "The 'mistralai' package is required for MistralProvider.  "
    "Install it with: pip install mistralai>=1.0"
)


class MistralProvider:
    """Mistral AI provider adapter.

    Args:
        model: Mistral model id, e.g. ``mistral-large-latest`` or
            ``codestral-latest``.
        max_tokens: Default upper bound on generated tokens.
        timeout: Per-request HTTP timeout in seconds.
    """

    provider_name: str = "mistral"

    def __init__(
        self,
        *,
        model: str = "mistral-large-latest",
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
            from mistralai import Mistral  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise AuthError(
                "MISTRAL_API_KEY is not set",
                provider=self.provider_name,
            )
        self._client = Mistral(api_key=api_key, timeout_ms=int(self._timeout * 1000))
        return self._client

    @staticmethod
    def _to_messages(
        request: CompletionRequest,
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        for m in request.messages:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        client = self._ensure_client()
        start = time.monotonic()
        try:
            # Mistral SDK v1 uses ``chat.complete_async``.
            response = await client.chat.complete_async(
                model=self._model,
                messages=self._to_messages(request),
                max_tokens=request.max_tokens or self._max_tokens,
                temperature=request.temperature,
                stop=request.stop,
            )
        except Exception as exc:
            lowered = str(exc).lower()
            name = type(exc).__name__.lower()
            if "auth" in lowered or "401" in lowered or "403" in lowered:
                raise AuthError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "rate" in lowered or "429" in lowered:
                raise RateLimitedError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "timeout" in name or "timeout" in lowered:
                raise TransientError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "400" in lowered or "invalid" in lowered:
                raise PermanentError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "5" in lowered and "0" in lowered and "error" in lowered:
                raise TransientError(
                    str(exc), provider=self.provider_name
                ) from exc
            raise ProviderError(
                f"Mistral adapter failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000.0)
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        text = (
            choice.message.content
            if choice and getattr(choice.message, "content", None)
            else ""
        )
        finish = (
            getattr(choice, "finish_reason", "stop") if choice else "stop"
        )
        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(
                getattr(usage_obj, "completion_tokens", 0) or 0
            ),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        )
        return CompletionResult(
            text=text or "",
            provider=self.provider_name,
            model=getattr(response, "model", self._model),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=str(finish),
        )

    async def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                try:
                    # Mistral client's close may be sync.
                    result = close()
                    if hasattr(result, "__await__"):
                        await result  # type: ignore[misc]
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "mistral close() raised %s", type(exc).__name__
                    )
            self._client = None


__all__ = ["MistralProvider"]
