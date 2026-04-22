"""Azure OpenAI provider adapter.

Azure OpenAI speaks the same wire protocol as public OpenAI but via
a per-subscription endpoint with deployment-name routing.  We soft-
import ``openai`` and use its ``AsyncAzureOpenAI`` client.

Env vars:
    AZURE_OPENAI_API_KEY     -- subscription key
    AZURE_OPENAI_ENDPOINT    -- e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_API_VERSION -- optional; defaults to 2024-10-21
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
    "The 'openai' package is required for AzureOpenAIProvider.  "
    "Install it with: pip install openai>=1.40"
)
_DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIProvider:
    """Azure OpenAI provider adapter.

    Args:
        deployment: Azure deployment name (this is NOT the model family;
            it's the per-subscription deployment slug you create in the
            Azure portal).
        model: Logical model id used for cost-tracking attribution
            only (e.g. ``gpt-4.1``).
        max_tokens: Default upper bound on generated tokens.
        timeout: HTTP timeout in seconds.
        api_version: Azure API version; defaults to
            :data:`_DEFAULT_API_VERSION`.
    """

    provider_name: str = "azure"

    def __init__(
        self,
        *,
        deployment: str,
        model: str = "gpt-4.1",
        max_tokens: int = 512,
        timeout: float = 30.0,
        api_version: str | None = None,
    ) -> None:
        self._deployment = deployment
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._api_version = api_version or _DEFAULT_API_VERSION
        self._client: Any | None = None

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

        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not api_key:
            raise AuthError(
                "AZURE_OPENAI_API_KEY is not set",
                provider=self.provider_name,
            )
        if not endpoint:
            raise AuthError(
                "AZURE_OPENAI_ENDPOINT is not set",
                provider=self.provider_name,
            )
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=self._api_version,
            timeout=self._timeout,
        )
        return self._client

    @staticmethod
    def _to_openai_messages(
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
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=self._deployment,  # Azure routes by deployment name.
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
            raise TransientError(
                str(exc), provider=self.provider_name
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(
                f"Azure adapter failed: {type(exc).__name__}",
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
        )
        return CompletionResult(
            text=text or "",
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=str(finish),
        )

    async def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "azure close() raised %s", type(exc).__name__
                    )
            self._client = None


__all__ = ["AzureOpenAIProvider"]
