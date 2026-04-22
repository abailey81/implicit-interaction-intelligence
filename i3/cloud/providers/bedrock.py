"""AWS Bedrock provider adapter.

Bedrock hosts multiple model families behind a single API; body
format varies per underlying model:

- ``anthropic.claude-*``    -- Anthropic Messages-like body.
- ``amazon.titan-*``        -- Titan text-generation body.
- ``meta.llama3-*``         -- Llama 3 prompt body.
- ``mistral.*``             -- Mistral INST body.

This adapter uses :func:`i3.cloud.prompt_translator.bedrock_body` to
dispatch based on the model prefix.  Credentials use the default AWS
credential chain (env vars, shared config, IAM role, ...) -- no
hard-coded keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from i3.cloud.prompt_translator import bedrock_body, parse_bedrock_response
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
    "The 'boto3' package is required for BedrockProvider.  "
    "Install it with: pip install boto3>=1.34"
)


class BedrockProvider:
    """AWS Bedrock provider adapter.

    Args:
        model: Bedrock model id, e.g. ``anthropic.claude-sonnet-4-5``,
            ``amazon.titan-text-premier-v1:0``,
            ``meta.llama3-3-70b-instruct-v1:0``,
            ``mistral.mistral-large-2407-v1:0``.
        region: AWS region.  Falls back to ``AWS_REGION`` env var.
        max_tokens: Default upper bound on generated tokens.
        timeout: Bedrock runtime read timeout (seconds).
    """

    provider_name: str = "bedrock"

    def __init__(
        self,
        *,
        model: str = "anthropic.claude-sonnet-4-5",
        region: str | None = None,
        max_tokens: int = 512,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
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
            import boto3  # type: ignore[import-not-found]
            from botocore.config import Config  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        try:
            cfg = Config(
                read_timeout=self._timeout,
                connect_timeout=min(5.0, self._timeout),
                retries={"max_attempts": 1},
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=cfg,
            )
        except Exception as exc:
            raise AuthError(
                f"Bedrock client init failed: {type(exc).__name__}: {exc}",
                provider=self.provider_name,
            ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        client = self._ensure_client()
        body = bedrock_body(self._model, request, self._max_tokens)
        start = time.monotonic()

        def _invoke() -> dict[str, Any]:
            return client.invoke_model(
                modelId=self._model,
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )

        try:
            raw = await asyncio.to_thread(_invoke)
        except Exception as exc:
            lowered = str(exc).lower()
            if "throttl" in lowered or "429" in lowered:
                raise RateLimitedError(
                    str(exc), provider=self.provider_name
                ) from exc
            if (
                "access" in lowered
                or "credential" in lowered
                or "unauthorized" in lowered
            ):
                raise AuthError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "timeout" in lowered or "network" in lowered:
                raise TransientError(
                    str(exc), provider=self.provider_name
                ) from exc
            if "validation" in lowered:
                raise PermanentError(
                    str(exc), provider=self.provider_name
                ) from exc
            raise ProviderError(
                f"Bedrock invoke failed: {type(exc).__name__}",
                provider=self.provider_name,
            ) from exc

        latency_ms = int((time.monotonic() - start) * 1000.0)
        body_bytes = raw["body"].read() if hasattr(raw.get("body"), "read") else raw.get("body", b"")
        try:
            parsed = json.loads(
                body_bytes.decode("utf-8") if isinstance(body_bytes, bytes) else body_bytes
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise PermanentError(
                f"Bedrock returned non-JSON body: {exc}",
                provider=self.provider_name,
            ) from exc

        text, p_tokens, c_tokens, finish = parse_bedrock_response(
            self._model, parsed
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
        # boto3 clients don't require explicit closure; drop the handle.
        self._client = None


__all__ = ["BedrockProvider"]
