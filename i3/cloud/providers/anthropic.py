"""Anthropic Claude provider adapter.

Wraps the existing :class:`i3.cloud.client.CloudLLMClient` so that the
Anthropic path continues to enjoy all of its hardening (TLS
verification, response-size caps, jittered backoff, Retry-After
clamping, ...)  while presenting the universal :class:`CloudProvider`
surface.

This adapter imports, but does NOT modify, ``CloudLLMClient``.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from typing import Any

from i3.cloud.client import CloudLLMClient
from i3.cloud.providers.base import (
    AuthError,
    CompletionRequest,
    CompletionResult,
    PermanentError,
    ProviderError,
    TokenUsage,
    TransientError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Provider adapter for Anthropic's Messages API.

    Args:
        model: The Claude model id (e.g. ``claude-sonnet-4-5``).
        max_tokens: Default ``max_tokens`` to request upstream.
        timeout: HTTP timeout in seconds.
        fallback_on_error: Forwarded to the underlying
            :class:`CloudLLMClient`; when ``True`` the client returns a
            neutral fallback string instead of raising after retries.
    """

    provider_name: str = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 512,
        timeout: float = 10.0,
        fallback_on_error: bool = False,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._fallback_on_error = fallback_on_error
        # Fabricate a minimal config object with a ``cloud`` namespace
        # matching what :class:`CloudLLMClient` reads.  We keep the
        # shape compatible so future fields (e.g. retry tuning) flow
        # through automatically.
        self._config = SimpleNamespace(
            cloud=SimpleNamespace(
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                fallback_on_error=fallback_on_error,
            )
        )
        self._client: CloudLLMClient | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> CloudLLMClient:
        """Lazily construct the wrapped :class:`CloudLLMClient`."""
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise AuthError(
                    "ANTHROPIC_API_KEY is not set",
                    provider=self.provider_name,
                )
            self._client = CloudLLMClient(self._config)
        return self._client

    @staticmethod
    def _extract_system(request: CompletionRequest) -> str:
        """Pull the system prompt out of ``request`` (field OR message)."""
        if request.system:
            return request.system
        for msg in request.messages:
            if msg.role == "system":
                return msg.content
        return ""

    @staticmethod
    def _extract_history_and_user(
        request: CompletionRequest,
    ) -> tuple[list[dict[str, str]], str]:
        """Split ``messages`` into prior-history + current user turn.

        The Anthropic Messages API does not have a system-role message
        -- ``system`` is a top-level field.  We strip any ``system``
        messages here (they were already consumed by
        :meth:`_extract_system`).  The last ``user`` message becomes
        the "current" turn; everything before it is history.
        """
        non_system = [m for m in request.messages if m.role != "system"]
        if not non_system:
            return [], ""
        last = non_system[-1]
        if last.role == "user":
            history = [
                {"role": m.role, "content": m.content}
                for m in non_system[:-1]
            ]
            return history, last.content
        # No trailing user turn -- send empty user message so the API
        # doesn't reject the request.
        history = [
            {"role": m.role, "content": m.content} for m in non_system
        ]
        return history, ""

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        """Execute ``request`` via the underlying Anthropic client.

        Raises:
            AuthError: if ``ANTHROPIC_API_KEY`` is unset.
            TransientError / PermanentError: wrapping upstream failures.
        """
        client = self._ensure_client()
        system = self._extract_system(request)
        history, user_message = self._extract_history_and_user(request)

        start = time.monotonic()
        try:
            raw: dict[str, Any] = await client.generate(
                user_message=user_message,
                system_prompt=system,
                conversation_history=history,
            )
        except RuntimeError as exc:
            # CloudLLMClient raises RuntimeError on non-retryable
            # upstream failures when fallback is disabled.  Map onto
            # the provider-neutral error tree.
            msg = str(exc)
            if "401" in msg or "403" in msg:
                raise AuthError(
                    msg, provider=self.provider_name
                ) from None
            if "500" in msg or "502" in msg or "503" in msg or "529" in msg:
                raise TransientError(
                    msg, provider=self.provider_name
                ) from None
            raise PermanentError(
                msg, provider=self.provider_name
            ) from None
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(
                f"Anthropic adapter failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000.0)
        # CloudLLMClient computes latency itself; prefer its value when
        # present so tests that inject mocked latencies see them.
        if raw.get("latency_ms"):
            latency_ms = int(raw["latency_ms"])

        usage = TokenUsage(
            prompt_tokens=int(raw.get("input_tokens", 0)),
            completion_tokens=int(raw.get("output_tokens", 0)),
            total_tokens=int(raw.get("input_tokens", 0))
            + int(raw.get("output_tokens", 0)),
            cached_tokens=int(raw.get("cached_tokens", 0)),
        )
        return CompletionResult(
            text=str(raw.get("text", "")),
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason="stop",
        )

    async def close(self) -> None:
        """Close the wrapped HTTP client.  Idempotent."""
        if self._client is not None:
            await self._client.close()
            self._client = None


__all__ = ["AnthropicProvider"]
