"""OpenAI (and OpenAI-compatible) provider adapter.

Covers:

- OpenAI models: ``gpt-4.1``, ``gpt-5-turbo``, ``o3``, ``o4-mini``.
- Any OpenAI-wire-compatible endpoint via ``base_url`` override
  (Azure OpenAI uses the dedicated :mod:`i3.cloud.providers.azure`
  adapter; this one targets the public OpenAI endpoint by default).

Soft-imports the ``openai`` package: missing at import time is OK,
but calling :meth:`complete` without it raises :class:`ImportError`
with an install hint.
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
    "The 'openai' package is required for OpenAIProvider.  "
    "Install it with: pip install openai>=1.40"
)


class OpenAIProvider:
    """OpenAI / OpenAI-compatible provider adapter.

    Args:
        model: Model id, e.g. ``gpt-4.1``, ``gpt-5-turbo``, ``o3``,
            ``o4-mini``.
        max_tokens: Default upper bound on generated tokens.
        timeout: Per-request HTTP timeout in seconds.
        base_url: Override the API base (for OpenAI-compatible proxies
            such as vLLM, Together, Fireworks, ...).  ``None`` uses the
            official endpoint.
    """

    provider_name: str = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        max_tokens: int = 512,
        timeout: float = 30.0,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._base_url = base_url
        self._client: Any | None = None  # openai.AsyncOpenAI

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise AuthError(
                "OPENAI_API_KEY is not set",
                provider=self.provider_name,
            )
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    @staticmethod
    def _to_openai_messages(
        request: CompletionRequest,
    ) -> list[dict[str, str]]:
        """Flatten the I3 request into OpenAI's flat messages list."""
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
        """Run ``request`` via OpenAI's chat completions API."""
        client = self._ensure_client()
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=self._to_openai_messages(request),
                max_tokens=request.max_tokens or self._max_tokens,
                temperature=request.temperature,
                stop=request.stop,
            )
        except openai.AuthenticationError as exc:
            raise AuthError(str(exc), provider=self.provider_name) from exc
        except openai.RateLimitError as exc:
            raise RateLimitedError(
                str(exc), provider=self.provider_name
            ) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise TransientError(
                str(exc), provider=self.provider_name
            ) from exc
        except openai.BadRequestError as exc:
            raise PermanentError(
                str(exc), provider=self.provider_name
            ) from exc
        except openai.APIError as exc:
            # 5xx family falls here; treat as transient.
            raise TransientError(
                str(exc), provider=self.provider_name
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(
                f"OpenAI adapter failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000.0)
        choice = response.choices[0] if response.choices else None
        text = (
            choice.message.content
            if choice and choice.message and choice.message.content
            else ""
        )
        finish = (
            choice.finish_reason if choice and choice.finish_reason else "stop"
        )
        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(
                getattr(usage_obj, "completion_tokens", 0) or 0
            ),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
            cached_tokens=int(
                getattr(
                    getattr(usage_obj, "prompt_tokens_details", None),
                    "cached_tokens",
                    0,
                )
                or 0
            ),
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
        """Close the underlying ``AsyncOpenAI`` client.  Idempotent."""
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.debug("openai close() raised %s", type(exc).__name__)
            self._client = None


__all__ = ["OpenAIProvider"]
