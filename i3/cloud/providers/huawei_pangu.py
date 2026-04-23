"""Huawei Cloud PanGu provider adapter.

This is the **highest-strategic-value** provider for the I3 project:
it aligns I3's HarmonyOS-native, Kirin-class design story (see
``docs/huawei/``) with a first-party Huawei Cloud inference endpoint.

Target models
-------------
- **PanGu N-series** (natural language, general-purpose).
- **PanGu Deep Thinking 5.5** -- the 718 B-parameter Mixture-of-Experts
  "Deep Thinking" model unveiled by Huawei Cloud at the Shanghai World
  AI Conference (WAIC 2025, June 2025).  Deep Thinking 5.5 integrates
  a *fast* / *slow* thinking pipeline and is Huawei Cloud's flagship
  reasoning model; it underpins the Huawei Cloud Agentic / Embodied AI
  platform announced alongside it.

Access requirements
-------------------
Huawei Cloud PanGu API access requires:

1. A Huawei Cloud account (https://www.huaweicloud.com/intl/en-us/).
2. The **PanGu Large Models service** enabled for that account in
   the target region (currently ``cn-southwest-2`` Guiyang has the
   broadest model selection, hence the default).
3. An API key provisioned under the PanGu workspace; export it as
   ``HUAWEI_CLOUD_PANGU_APIKEY``.

References
----------
- Huawei Cloud, "PanGu 5.5 family and Deep Thinking model,"
  Shanghai WAIC 2025 announcement, June 2025.
- Huawei Cloud PanGu service docs:
  https://support.huaweicloud.com/intl/en-us/productdesc-pangulargemodels/

This adapter uses raw ``httpx`` -- there is no official Python SDK
with stable public distribution at time of writing, so keeping the
transport explicit avoids an ImportError-at-import-time foot-gun.
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

_DEFAULT_REGION = "cn-southwest-2"
_CHAT_ENDPOINT = "/v1/infers/chat/completions"


def _endpoint_for(region: str) -> str:
    """Return the PanGu HTTPS endpoint for ``region``.

    The PanGu service follows Huawei Cloud's standard regional-DNS
    pattern: ``https://pangu.<region>.myhuaweicloud.com``.
    """
    return f"https://pangu.{region}.myhuaweicloud.com"


class HuaweiPanGuProvider:
    """Huawei Cloud PanGu provider adapter (strategic first-party path).

    Args:
        model: PanGu model id, e.g.
            ``pangu-nlp-n4-32k``, ``pangu-deepthink-5.5``.
        region: Huawei Cloud region slug; defaults to
            ``HUAWEI_CLOUD_PANGU_REGION`` env var, then
            :data:`_DEFAULT_REGION` (``cn-southwest-2`` -- Guiyang).
        max_tokens: Default upper bound on generated tokens.
        timeout: HTTP timeout in seconds.

    Note:
        Calling :meth:`complete` against the real Huawei Cloud endpoint
        requires a provisioned account with PanGu service enabled; the
        adapter surfaces missing-auth as :class:`AuthError` and
        provides a clear error path for CI environments that don't
        have access.
    """

    provider_name: str = "huawei_pangu"

    def __init__(
        self,
        *,
        model: str = "pangu-deepthink-5.5",
        region: str | None = None,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._region = (
            region
            or os.environ.get("HUAWEI_CLOUD_PANGU_REGION")
            or _DEFAULT_REGION
        )
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
            if self._client is not None:
                return self._client
            api_key = os.environ.get("HUAWEI_CLOUD_PANGU_APIKEY", "")
            if not api_key:
                raise AuthError(
                    "HUAWEI_CLOUD_PANGU_APIKEY is not set.  "
                    "Huawei Cloud PanGu API access requires a Huawei Cloud "
                    "account with the PanGu Large Models service enabled in "
                    f"region {self._region}.",
                    provider=self.provider_name,
                )
            base_url = _endpoint_for(self._region)
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-PanGu-Model-Family": self._model.split("-")[0] if "-" in self._model else self._model,
                },
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
        """Flatten the request onto PanGu's OpenAI-compatible shape.

        PanGu's chat-completions endpoint accepts an OpenAI-like
        ``messages: [{role, content}]`` array with a top-level
        ``system`` turn (unlike Anthropic, where ``system`` is a
        separate field).
        """
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
        """Generate via Huawei Cloud PanGu.

        Raises:
            AuthError: missing ``HUAWEI_CLOUD_PANGU_APIKEY``.
            RateLimitedError: 429 from the upstream.
            TransientError: 5xx, timeout, or network-level failure.
            PermanentError: 4xx other than 401/403/429.
        """
        client = self._ensure_client()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_messages(request),
            "max_tokens": request.max_tokens or self._max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = request.stop

        start = time.monotonic()
        try:
            response = await client.post(_CHAT_ENDPOINT, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientError(
                f"PanGu timeout: {exc}", provider=self.provider_name
            ) from exc
        except httpx.HTTPError as exc:
            raise TransientError(
                f"PanGu HTTP error: {exc}", provider=self.provider_name
            ) from exc
        latency_ms = int((time.monotonic() - start) * 1000.0)

        status = response.status_code
        if status in (401, 403):
            raise AuthError(
                f"PanGu HTTP {status}", provider=self.provider_name
            )
        if status == 429:
            retry_hdr = response.headers.get("retry-after")
            raise RateLimitedError(
                f"PanGu HTTP {status}",
                provider=self.provider_name,
                retry_after=float(retry_hdr) if retry_hdr else None,
            )
        if status >= 500:
            raise TransientError(
                f"PanGu HTTP {status}", provider=self.provider_name
            )
        if status >= 400:
            # SEC (L-2, 2026-04-23 audit): body stays in DEBUG log only;
            # exception message carries status + provider identity only.
            logger.debug(
                "pangu.4xx_body status=%s body_head=%s",
                status,
                response.text[:200].replace("\n", " "),
            )
            raise PermanentError(
                f"PanGu HTTP {status}",
                provider=self.provider_name,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise PermanentError(
                f"PanGu returned non-JSON body: {exc}",
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
                    "huawei_pangu close() raised %s", type(exc).__name__
                )
            self._client = None


__all__ = ["HuaweiPanGuProvider"]
