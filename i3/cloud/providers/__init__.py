"""Universal LLM provider layer for I3.

This package defines a narrow provider protocol so that the existing
:class:`i3.cloud.client.CloudLLMClient` (Anthropic-native) becomes one
of many interchangeable backends. It unlocks:

- vendor-neutral deployments (swap providers via a single config key);
- graceful degradation through :class:`~i3.cloud.multi_provider.MultiProviderClient`;
- research reproducibility across providers;
- a first-class Huawei Cloud PanGu adapter that is strategically aligned
  with the project's HarmonyOS / Kirin-class roadmap.

Public surface
--------------
The symbols re-exported below are the stable contract that downstream
modules should depend on::

    from i3.cloud.providers import (
        CloudProvider,
        CompletionRequest,
        CompletionResult,
        ChatMessage,
        TokenUsage,
        ProviderRegistry,
        MultiProviderClient,
    )

Everything else (individual adapter classes, pricing JSON loader, ...)
is an implementation detail and may change without notice.
"""

from __future__ import annotations

from i3.cloud.providers.base import (
    AuthError,
    ChatMessage,
    CloudProvider,
    CompletionRequest,
    CompletionResult,
    PermanentError,
    ProviderError,
    RateLimitedError,
    TokenUsage,
    TransientError,
)

__all__ = [
    "AuthError",
    "ChatMessage",
    "CloudProvider",
    "CompletionRequest",
    "CompletionResult",
    "MultiProviderClient",
    "PermanentError",
    "ProviderError",
    "ProviderRegistry",
    "RateLimitedError",
    "TokenUsage",
    "TransientError",
]


def __getattr__(name: str):  # pragma: no cover - trivial re-export
    """Lazy re-export to avoid circular imports at package import time.

    :class:`ProviderRegistry` and :class:`MultiProviderClient` live in
    sibling modules that themselves import from this package; deferring
    their import via ``__getattr__`` breaks the cycle.
    """
    if name == "ProviderRegistry":
        from i3.cloud.provider_registry import ProviderRegistry

        return ProviderRegistry
    if name == "MultiProviderClient":
        from i3.cloud.multi_provider import MultiProviderClient

        return MultiProviderClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
