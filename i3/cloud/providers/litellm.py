"""LiteLLM universal adapter.

LiteLLM (https://github.com/BerriAI/litellm) normalises 100+
providers onto an OpenAI-compatible surface.  This adapter is the
"universal fallback" for any provider we don't ship a first-class
adapter for.

Soft-imports ``litellm``.  Call :meth:`complete` to see the install
hint when the package is missing.
"""

from __future__ import annotations

import logging
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
    "The 'litellm' package is required for LiteLLMProvider.  "
    "Install it with: pip install litellm>=1.50"
)


class LiteLLMProvider:
    """LiteLLM universal provider adapter.

    Args:
        model: LiteLLM model id.  LiteLLM uses per-provider prefixes
            like ``openai/gpt-4.1``, ``anthropic/claude-sonnet-4-5``,
            ``vertex_ai/gemini-2.5-pro``, ``together_ai/...``.
        max_tokens: Default upper bound on generated tokens.
        timeout: Per-request timeout in seconds.
        extra_kwargs: Arbitrary provider-specific kwargs forwarded to
            ``litellm.acompletion`` (e.g. ``api_base``, ``api_version``,
            ``aws_region_name``).
    """

    provider_name: str = "litellm"

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4.1",
        max_tokens: int = 512,
        timeout: float = 60.0,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._extra = dict(extra_kwargs or {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_sdk() -> Any:
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        return litellm

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
        litellm = self._ensure_sdk()
        start = time.monotonic()
        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=self._to_messages(request),
                max_tokens=request.max_tokens or self._max_tokens,
                temperature=request.temperature,
                stop=request.stop,
                timeout=self._timeout,
                **self._extra,
            )
        except Exception as exc:
            name = type(exc).__name__.lower()
            lowered = str(exc).lower()
            if "auth" in name or "401" in lowered or "403" in lowered:
                raise AuthError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "rate" in name or "429" in lowered:
                raise RateLimitedError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "timeout" in name or "timeout" in lowered:
                raise TransientError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "invalid" in name or "400" in lowered:
                raise PermanentError(
                    str(exc), provider=self.provider_name
                ) from exc
            if (
                "internal" in name
                or "service" in name
                or "server" in lowered
            ):
                raise TransientError(
                    str(exc), provider=self.provider_name
                ) from exc
            raise ProviderError(
                f"LiteLLM adapter failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000.0)
        # LiteLLM normalises the response to OpenAI shape (dict or
        # object depending on version).
        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        choices = _get(response, "choices") or []
        choice = choices[0] if choices else {}
        message = _get(choice, "message") or {}
        text = str(_get(message, "content", "") or "")
        finish = str(_get(choice, "finish_reason", "stop") or "stop")
        usage_obj = _get(response, "usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(_get(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(
                _get(usage_obj, "completion_tokens", 0) or 0
            ),
            total_tokens=int(_get(usage_obj, "total_tokens", 0) or 0),
        )
        return CompletionResult(
            text=text,
            provider=self.provider_name,
            model=str(_get(response, "model", self._model)),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish,
        )

    async def close(self) -> None:
        # LiteLLM is stateless at the module level; nothing to close.
        return None


__all__ = ["LiteLLMProvider"]
