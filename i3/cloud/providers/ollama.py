"""Ollama (local) provider adapter.

Talks to a local Ollama server (default ``http://localhost:11434``)
using raw ``httpx`` -- no API key required.  Default model is
``llama3.3``; any locally pulled model works.

Useful for:

- privacy-first deployments where nothing leaves the device;
- CI / offline test fixtures;
- developer-laptop fallback in the :class:`MultiProviderClient`.
"""

from __future__ import annotations

import logging
import threading
import os
import time
from typing import Any

import httpx

from i3.cloud.providers.base import (
    CompletionRequest,
    CompletionResult,
    PermanentError,
    ProviderError,
    TokenUsage,
    TransientError,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_CHAT_ENDPOINT = "/api/chat"


class OllamaProvider:
    """Local Ollama provider adapter.

    Args:
        model: Local model tag (e.g. ``llama3.3``, ``mistral``,
            ``qwen2.5:32b``).  Must already be pulled.
        base_url: Ollama HTTP endpoint; falls back to the
            ``OLLAMA_BASE_URL`` env var, then :data:`_DEFAULT_BASE_URL`.
        max_tokens: Default upper bound on generated tokens.
        timeout: HTTP timeout in seconds -- local inference can take
            tens of seconds on CPU, so the default is generous.
    """

    provider_name: str = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.3",
        base_url: str | None = None,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # SEC (H-6, 2026-04-23 audit): serialise lazy-init.
        self._client_init_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        with self._client_init_lock:
            if self._client is None:
                # SEC (L-1): pin every security-relevant kwarg explicitly.
                self._client = httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=httpx.Timeout(self._timeout),
                    verify=True,
                    follow_redirects=False,
                    limits=httpx.Limits(
                        max_keepalive_connections=4,
                        max_connections=8,
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
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens or self._max_tokens,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop

        start = time.monotonic()
        try:
            response = await client.post(_CHAT_ENDPOINT, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientError(
                f"Ollama timeout: {exc}", provider=self.provider_name
            ) from exc
        except httpx.HTTPError as exc:
            raise TransientError(
                f"Ollama HTTP error: {exc}", provider=self.provider_name
            ) from exc
        latency_ms = int((time.monotonic() - start) * 1000.0)

        if response.status_code >= 500:
            raise TransientError(
                f"Ollama HTTP {response.status_code}",
                provider=self.provider_name,
            )
        if response.status_code >= 400:
            # SEC (L-2): keep the exception message narrow; body goes to
            # the DEBUG log so operators can still diagnose.
            logger.debug(
                "ollama.4xx_body status=%s body_head=%s",
                response.status_code,
                response.text[:200].replace("\n", " "),
            )
            raise PermanentError(
                f"Ollama HTTP {response.status_code}",
                provider=self.provider_name,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentError(
                f"Ollama returned non-JSON body: {exc}",
                provider=self.provider_name,
            ) from exc

        msg = data.get("message", {}) or {}
        text = str(msg.get("content", "") or "")
        done_reason = str(data.get("done_reason") or "stop")
        p_tokens = int(data.get("prompt_eval_count", 0) or 0)
        c_tokens = int(data.get("eval_count", 0) or 0)
        usage = TokenUsage(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens,
        )
        return CompletionResult(
            text=text,
            provider=self.provider_name,
            model=str(data.get("model", self._model)),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=done_reason,
        )

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover
                logger.debug("ollama close() raised %s", type(exc).__name__)
            self._client = None


__all__ = ["OllamaProvider"]
