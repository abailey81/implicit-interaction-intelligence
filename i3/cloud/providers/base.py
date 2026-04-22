"""Provider-neutral types and protocol for the universal LLM layer.

Defines the :class:`CloudProvider` typing protocol plus the request /
response / usage dataclasses that every concrete adapter speaks.  The
types here deliberately avoid any provider-specific concept (no "system
parameter", no "contents/parts", no "tool_calls") so that the I3
pipeline can treat Anthropic, OpenAI, Google, Huawei PanGu, ... as
interchangeable.

Typing design notes
-------------------
- ``CompletionRequest`` / ``CompletionResult`` / ``TokenUsage`` are
  Pydantic ``BaseModel`` with ``model_config = ConfigDict(frozen=True)``
  so callers cannot mutate them after construction, which keeps the
  request object safe to share across the fallback chain.
- ``ChatMessage.role`` is a ``Literal`` so mypy / pyright catch stray
  roles at type-check time.  The provider-specific translator is
  responsible for mapping "system" into whatever slot the upstream API
  expects.
- Errors are a small closed hierarchy -- adapters MUST map upstream
  status codes onto these four classes so that callers can handle
  transient-vs-permanent without caring which provider failed.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A single provider-neutral chat turn.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``.  Provider
            translators are responsible for mapping these onto the
            upstream API's shape (Anthropic pulls ``system`` out of the
            messages list, OpenAI keeps it inline, ...).
        content: The turn's text content.  Tool-call / function-call
            shapes are intentionally out of scope for this iteration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str


class CompletionRequest(BaseModel):
    """Provider-neutral completion request.

    Attributes:
        system: Optional system prompt.  Use :class:`ChatMessage` with
            ``role="system"`` in ``messages`` OR this field -- not both.
            Adapters prefer ``system`` when both are present.
        messages: Ordered dialogue turns.
        max_tokens: Upper bound on tokens the adapter may request from
            the upstream API.  Adapters MUST clamp this to their
            provider's hard ceiling if necessary.
        temperature: Sampling temperature, 0.0-2.0.
        stop: Optional list of stop strings.  ``None`` means no stop.
        metadata: Free-form dict for routing hints, cost-tracking tags,
            etc.  Never forwarded to upstream APIs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token counters for a single completion.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt / system / history.
        completion_tokens: Tokens in the generated response.
        total_tokens: Convenience sum; defaults to
            ``prompt_tokens + completion_tokens`` when zero.
        cached_tokens: Tokens billed at the cached rate (Anthropic
            prompt caching, OpenAI prefix caching, ...).  ``0`` if the
            provider does not report cache hits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class CompletionResult(BaseModel):
    """Provider-neutral completion response.

    Attributes:
        text: The generated text (concatenated across content blocks if
            the upstream returned multiple).
        provider: The :attr:`CloudProvider.provider_name` that produced
            this result.  Useful for logging and cost attribution.
        model: The actual model id the provider used (may differ from
            the requested one if the provider did automatic routing).
        usage: Token accounting.
        latency_ms: Wall-clock latency in milliseconds for the single
            upstream request that produced this result.
        finish_reason: Provider-normalised finish reason.  Common values:
            ``"stop"``, ``"length"``, ``"content_filter"``, ``"tool_use"``,
            ``"error"``.  Adapters coerce upstream-specific enums onto
            these strings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    provider: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CloudProvider(Protocol):
    """Typing protocol every concrete adapter must implement.

    Implementations are duck-typed: inheriting from ``CloudProvider``
    is optional, but classes MUST expose the same public surface so
    that :func:`isinstance(obj, CloudProvider)` succeeds at runtime
    (thanks to ``@runtime_checkable``).
    """

    provider_name: str

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        """Run ``request`` against the upstream API.

        Adapters MUST raise one of the subclasses of
        :class:`ProviderError` on failure so callers can route by
        error class rather than by upstream status code.
        """
        ...

    async def close(self) -> None:
        """Release any network resources.

        MUST be idempotent: repeated calls are safe and silent.
        """
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class for every provider adapter error.

    Adapters raise subclasses -- never this class directly.
    """

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class AuthError(ProviderError):
    """Credentials missing, invalid, or lacking the required scope.

    Not retryable and not recoverable without operator intervention.
    """


class RateLimitedError(ProviderError):
    """Upstream returned a 429 / equivalent rate-limit signal.

    Callers MAY retry after the ``retry_after`` hint (seconds); the
    fallback chain treats this as a "skip this provider for now"
    signal and tries the next one.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """Temporary failure (5xx, network timeout, transient DNS error).

    The fallback chain will try the next provider; a single-provider
    caller MAY retry with backoff.
    """


class PermanentError(ProviderError):
    """Non-retryable failure (4xx other than 401/403/429, invalid model).

    Neither retrying the same provider nor switching providers is
    guaranteed to help if the error is input-related.
    """


__all__ = [
    "AuthError",
    "ChatMessage",
    "CloudProvider",
    "CompletionRequest",
    "CompletionResult",
    "PermanentError",
    "ProviderError",
    "RateLimitedError",
    "Role",
    "TokenUsage",
    "TransientError",
]
