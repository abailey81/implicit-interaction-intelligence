"""OpenRouter provider adapter.

OpenRouter is a single-endpoint router in front of 200+ upstream
models.  Model names carry a routing prefix (``anthropic/claude-*``,
``google/gemini-*``, ``meta-llama/llama-*``, ...) and OpenRouter picks
the upstream provider based on that prefix.

Uses raw ``httpx`` against OpenRouter's OpenAI-compatible
``/api/v1/chat/completions`` endpoint -- no SDK required.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

from i3.cloud.providers.base import (
    AuthError,
    CompletionRequest,
    CompletionResult,
    PermanentError,
    RateLimitedError,
    TokenUsage,
    TransientError,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai"
_CHAT_ENDPOINT = "/api/v1/chat/completions"


class OpenRouterProvider:
    """OpenRouter provider adapter.

    Args:
        model: OpenRouter-style model id with provider prefix, e.g.
            ``anthropic/claude-sonnet-4.5``, ``google/gemini-2.5-pro``,
            ``meta-llama/llama-3.3-70b-instruct``.
        max_tokens: Default upper bound on generated tokens.
        timeout: HTTP timeout in seconds.
        referer: Optional HTTP-Referer header value (OpenRouter uses
            it for app attribution and rate-limit tiers).
        x_title: Optional ``X-Title`` header (same attribution use).
    """

    provider_name: str = "openrouter"

    def __init__(
        self,
        *,
        model: str = "anthropic/claude-sonnet-4.5",
        max_tokens: int = 512,
        timeout: float = 60.0,
        referer: str | None = "https://github.com/i3",
        x_title: str | None = "I3",
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._referer = referer
        self._x_title = x_title
        self._client: httpx.AsyncClient | None = None
        # SEC (H-6, 2026-04-23 audit): serialise lazy-init so two
        # concurrent first-hit callers cannot both build a client.
        self._client_init_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        with self._client_init_lock:
            if self._client is not None:
                return self._client
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise AuthError(
                "OPENROUTER_API_KEY is not set",
                provider=self.provider_name,
            )
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._x_title:
            headers["X-Title"] = self._x_title
        # SEC (L-1, 2026-04-23 audit): pin every security-relevant
        # ``httpx.AsyncClient`` kwarg explicitly so a future httpx
        # default-flip cannot silently weaken the outbound path.
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
            verify=True,
            follow_redirects=False,
            limits=httpx.Limits(
                max_keepalive_connections=8,
                max_connections=16,
                keepalive_expiry=60.0,
            ),
        )
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
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_messages(request),
            "max_tokens": request.max_tokens or self._max_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            payload["stop"] = request.stop

        start = time.monotonic()
        try:
            response = await client.post(_CHAT_ENDPOINT, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientError(
                f"OpenRouter timeout: {exc}",
                provider=self.provider_name,
            ) from exc
        except httpx.HTTPError as exc:
            raise TransientError(
                f"OpenRouter HTTP error: {exc}",
                provider=self.provider_name,
            ) from exc
        latency_ms = int((time.monotonic() - start) * 1000.0)

        status = response.status_code
        if status == 401 or status == 403:
            raise AuthError(
                f"OpenRouter HTTP {status}", provider=self.provider_name
            )
        if status == 429:
            retry = response.headers.get("retry-after")
            raise RateLimitedError(
                f"OpenRouter HTTP {status}",
                provider=self.provider_name,
                retry_after=float(retry) if retry else None,
            )
        if status >= 500:
            raise TransientError(
                f"OpenRouter HTTP {status}", provider=self.provider_name
            )
        if status >= 400:
            # SEC (M-4, 2026-04-23 audit): do NOT echo response.text into
            # the exception message — some upstreams reflect headers
            # (including the offending ``Authorization:`` prefix) back in
            # their 4xx bodies.  Log the body to the DEBUG channel (which
            # operators can opt into) while the exception string stays
            # narrow.
            logger.debug(
                "openrouter.4xx_body status=%s body_head=%s",
                status,
                response.text[:200].replace("\n", " "),
            )
            raise PermanentError(
                f"OpenRouter HTTP {status}",
                provider=self.provider_name,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentError(
                f"OpenRouter returned non-JSON body: {exc}",
                provider=self.provider_name,
            ) from exc

        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        text = str(message.get("content", "") or "")
        finish = str(choice.get("finish_reason", "stop") or "stop")
        usage_obj = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_obj.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_obj.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_obj.get("total_tokens", 0) or 0),
        )
        return CompletionResult(
            text=text,
            provider=self.provider_name,
            model=str(data.get("model", self._model)),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish,
        )

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "openrouter close() raised %s", type(exc).__name__
                )
            self._client = None


__all__ = ["OpenRouterProvider"]
