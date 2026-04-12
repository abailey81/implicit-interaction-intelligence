"""Async client for the Anthropic Claude Messages API.

Handles API key management, async message sending with configurable
timeout, retry logic with exponential backoff, graceful fallback on
errors, and cumulative token-usage tracking.

Privacy note:
    The :meth:`generate_session_summary` method accepts only aggregated
    metadata (session counts, topic keywords, engagement scores) -- never
    raw user text.  This aligns with the I3 privacy architecture where
    no verbatim user input is transmitted to the cloud.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import httpx

from i3.config import CloudConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE_URL = "https://api.anthropic.com"
_MESSAGES_ENDPOINT = "/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_RETRIES = 2
_BACKOFF_BASE_SECONDS = 1.0

# Absolute ceiling on the HTTP request timeout.  Even if the config
# requests a longer value we clamp it to prevent a hung upstream from
# tying up the event loop indefinitely.
_MAX_TIMEOUT_SECONDS = 30.0

# Retry-after fallback when the upstream omits the header.
_DEFAULT_RETRY_AFTER_SECONDS = 2.0


def _redact_api_key(key: str) -> str:
    """Return a log-safe rendering of an API key.

    Shows the provider prefix and the last four characters only, e.g.
    ``sk-ant-***abcd``.  Empty strings are rendered as ``<unset>``.
    """
    if not key:
        return "<unset>"
    if len(key) <= 8:
        return "***"
    prefix, tail = key[:7], key[-4:]
    return f"{prefix}***{tail}"


class CloudLLMClient:
    """Async client for Anthropic Claude API.

    Handles:
    - API key management (from ``ANTHROPIC_API_KEY`` environment variable)
    - Async message sending with configurable timeout
    - Retry logic with exponential backoff (up to ``_MAX_RETRIES`` retries)
    - Fallback behaviour on unrecoverable errors
    - Cumulative input / output token usage tracking

    Parameters:
        config: A :class:`~src.config.Config` instance whose ``cloud``
            section provides model name, max tokens, timeout, and fallback
            behaviour.
    """

    def __init__(self, config: Any) -> None:
        cloud: CloudConfig = config.cloud
        self.model: str = cloud.model
        self.max_tokens: int = cloud.max_tokens
        # Clamp timeout to the hard ceiling so a misconfigured config
        # cannot stall the pipeline indefinitely.
        requested_timeout = float(cloud.timeout)
        self.timeout: float = min(requested_timeout, _MAX_TIMEOUT_SECONDS)
        if requested_timeout > _MAX_TIMEOUT_SECONDS:
            logger.warning(
                "Cloud timeout %.1fs exceeds ceiling %.1fs; clamping.",
                requested_timeout,
                _MAX_TIMEOUT_SECONDS,
            )
        self.fallback_on_error: bool = cloud.fallback_on_error

        self._api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None

        # Cumulative token counters
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        if not self._api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set -- cloud LLM calls will fail.  "
                "Set the environment variable or ensure fallback_on_error is True."
            )
        else:
            logger.info(
                "CloudLLMClient ready (model=%s, key=%s)",
                self.model,
                _redact_api_key(self._api_key),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> None:
        """Lazily initialise the shared ``httpx.AsyncClient``."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=_API_BASE_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout, connect=5.0),
            )

    @staticmethod
    def _build_messages(
        user_message: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        """Assemble the ``messages`` array for the API request.

        If *conversation_history* is supplied it is prepended so that
        Claude has prior context.  Each entry must be a dict with
        ``role`` (``"user"`` or ``"assistant"``) and ``content`` keys.
        """
        messages: list[dict[str, str]] = []
        if conversation_history:
            for entry in conversation_history:
                role = entry.get("role", "user")
                content = entry.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> dict[str, Any]:
        """Extract text and token usage from the raw API response body.

        Returns:
            Dict with ``text``, ``input_tokens``, and ``output_tokens``.

        Raises:
            ValueError: If the response structure is unexpected.
        """
        content_blocks = data.get("content")
        if not content_blocks or not isinstance(content_blocks, list):
            raise ValueError(
                f"Unexpected response structure: missing 'content' list.  "
                f"Keys received: {list(data.keys())}"
            )

        # Concatenate all text blocks (there is usually exactly one).
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        if not text_parts:
            raise ValueError("Response contained no text content blocks.")

        usage = data.get("usage", {})
        return {
            "text": "".join(text_parts),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Send a message to Claude and return the response.

        The method implements retry with exponential backoff for transient
        errors (HTTP 429 / 5xx) and returns a fallback response when
        ``fallback_on_error`` is ``True`` and all retries are exhausted.

        Args:
            user_message: The current user message to respond to.
            system_prompt: System-level instructions (built by
                :class:`~src.cloud.prompt_builder.PromptBuilder`).
            conversation_history: Optional prior turns for multi-turn
                context.  Each dict must have ``role`` and ``content``.

        Returns:
            A dict with keys:

            - ``text`` (str): The assistant's reply.
            - ``input_tokens`` (int): Tokens consumed by the prompt.
            - ``output_tokens`` (int): Tokens in the completion.
            - ``latency_ms`` (float): Wall-clock latency in milliseconds.

        Raises:
            RuntimeError: If the API call fails after retries and
                ``fallback_on_error`` is ``False``.
        """
        if not self.is_available:
            return self._fallback_response("API key not configured")

        await self._ensure_client()
        assert self._client is not None  # for type checker

        messages = self._build_messages(user_message, conversation_history)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }

        last_error: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.info(
                    "Retry %d/%d after %.1fs backoff",
                    attempt,
                    _MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)

            start = time.monotonic()
            try:
                response = await self._client.post(
                    _MESSAGES_ENDPOINT, json=payload
                )
                latency_ms = (time.monotonic() - start) * 1000.0

                if response.status_code == 200:
                    parsed = self._parse_response(response.json())
                    self._total_input_tokens += parsed["input_tokens"]
                    self._total_output_tokens += parsed["output_tokens"]
                    logger.debug(
                        "Claude responded in %.0fms  (in=%d, out=%d tokens)",
                        latency_ms,
                        parsed["input_tokens"],
                        parsed["output_tokens"],
                    )
                    return {
                        "text": parsed["text"],
                        "input_tokens": parsed["input_tokens"],
                        "output_tokens": parsed["output_tokens"],
                        "latency_ms": latency_ms,
                    }

                # Decide whether to retry based on status code
                if response.status_code in (429, 500, 502, 503, 529):
                    last_error = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    # Honour Retry-After on 429s (RFC 6585).
                    if response.status_code == 429:
                        retry_after_hdr = response.headers.get("retry-after", "")
                        try:
                            retry_after = float(retry_after_hdr)
                        except ValueError:
                            retry_after = _DEFAULT_RETRY_AFTER_SECONDS
                        retry_after = min(retry_after, _MAX_TIMEOUT_SECONDS)
                        logger.warning(
                            "429 rate-limited by Claude API; sleeping %.1fs",
                            retry_after,
                        )
                        await asyncio.sleep(retry_after)
                    else:
                        logger.warning(
                            "Transient error %d from Claude API (attempt %d/%d)",
                            response.status_code,
                            attempt + 1,
                            _MAX_RETRIES + 1,
                        )
                    continue

                # Non-retryable error (e.g. 400, 401, 403).  We log only
                # the status code and a truncated body without echoing
                # any payload fields that could include user input.
                error_body = response.text[:200]
                logger.error(
                    "Claude API returned HTTP %d (body truncated: %r)",
                    response.status_code,
                    error_body,
                )
                public_msg = (
                    f"Cloud provider returned HTTP {response.status_code}"
                )
                if self.fallback_on_error:
                    return self._fallback_response(public_msg)
                raise RuntimeError(public_msg)

            except httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - start) * 1000.0
                last_error = exc
                logger.warning(
                    "Timeout after %.0fms on attempt %d/%d: %s",
                    latency_ms,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )
                continue

            except httpx.HTTPStatusError:
                # Already handled above; re-raise if we somehow get here.
                raise

            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "HTTP error on attempt %d/%d: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )
                continue

        # All retries exhausted
        error_msg = f"Claude API failed after {_MAX_RETRIES + 1} attempts: {last_error}"
        logger.error(error_msg)
        if self.fallback_on_error:
            return self._fallback_response(error_msg)
        raise RuntimeError(error_msg)

    async def generate_session_summary(
        self, session_metadata: dict[str, Any]
    ) -> str:
        """Generate a privacy-safe session summary from metadata only.

        No raw user text is sent to the cloud.  The input consists solely
        of aggregated metrics and topic keywords extracted on-device.

        Args:
            session_metadata: Dict with keys such as ``session_id``,
                ``turn_count``, ``avg_cognitive_load``, ``topics``,
                ``engagement_trend``, ``duration_minutes``.

        Returns:
            A short natural-language summary suitable for the interaction
            diary, or a fallback string on error.
        """
        system_prompt = (
            "You are an interaction analyst.  Given aggregated session "
            "metrics (no raw user text), write a concise 1-2 sentence "
            "summary of the interaction.  Focus on the user's engagement "
            "pattern and topic areas.  Do not invent specifics beyond "
            "what the metrics suggest."
        )

        # Build a structured but privacy-safe description of the session.
        meta_lines: list[str] = []
        for key, value in session_metadata.items():
            if isinstance(value, list):
                meta_lines.append(f"- {key}: {', '.join(str(v) for v in value)}")
            else:
                meta_lines.append(f"- {key}: {value}")

        user_message = (
            "Summarise this interaction session based on the following "
            "aggregated metrics:\n" + "\n".join(meta_lines)
        )

        result = await self.generate(
            user_message=user_message,
            system_prompt=system_prompt,
        )
        return result["text"]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Return ``True`` if an API key is configured."""
        return bool(self._api_key)

    @property
    def usage_stats(self) -> dict[str, int]:
        """Return cumulative token usage since client construction.

        Returns:
            Dict with ``total_input_tokens`` and ``total_output_tokens``.
        """
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("CloudLLMClient HTTP client closed.")

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_response(reason: str) -> dict[str, Any]:
        """Return a safe fallback response when the cloud is unreachable.

        The fallback text is intentionally short and neutral so that it
        can be displayed to the user without seeming out of place.
        """
        logger.info("Using fallback response: %s", reason)
        return {
            "text": "I'm here if you'd like to chat.",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CloudLLMClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
