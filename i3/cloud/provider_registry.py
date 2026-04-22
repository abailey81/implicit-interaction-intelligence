"""Factory registry for cloud LLM providers.

:class:`ProviderRegistry` is a dict of ``name -> factory`` callables.
Each factory accepts a free-form ``dict`` of provider-specific options
and returns a :class:`~i3.cloud.providers.base.CloudProvider`.

The 11 first-class adapters ship with auto-registered factories at
module import time.  Downstream callers can register their own (for
OpenAI-compatible proxies, new providers, mocks) via
:meth:`ProviderRegistry.register`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from i3.cloud.providers.base import CloudProvider

logger = logging.getLogger(__name__)


ProviderFactory = Callable[[dict[str, Any]], "CloudProvider"]


class _Registry:
    """Mutable registry singleton.  Not exported -- use the class-level
    methods on :class:`ProviderRegistry` which delegate here."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        name = name.lower().strip()
        if not name:
            raise ValueError("provider name must be non-empty")
        self._factories[name] = factory
        logger.debug("Registered cloud provider factory: %s", name)

    def get(
        self, name: str, options: dict[str, Any] | None = None
    ) -> "CloudProvider":
        key = name.lower().strip()
        if key not in self._factories:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(
                f"Unknown cloud provider {name!r}; registered: {available}"
            )
        factory = self._factories[key]
        return factory(options or {})

    def names(self) -> list[str]:
        return sorted(self._factories)


_REGISTRY = _Registry()


class ProviderRegistry:
    """Class-level facade over the module-global registry.

    Use the ``register`` / ``get`` / ``names`` class methods; they
    delegate to a single process-wide :class:`_Registry` instance.
    """

    @classmethod
    def register(cls, name: str, factory: ProviderFactory) -> None:
        """Register ``factory`` under ``name`` (case-insensitive)."""
        _REGISTRY.register(name, factory)

    @classmethod
    def get(
        cls, name: str, options: dict[str, Any] | None = None
    ) -> "CloudProvider":
        """Instantiate a provider by name with ``options``."""
        return _REGISTRY.get(name, options)

    @classmethod
    def names(cls) -> list[str]:
        """Return the sorted list of registered provider names."""
        return _REGISTRY.names()

    @classmethod
    def from_config(cls, config: Any) -> "CloudProvider":
        """Build a provider from an I3 ``CloudConfig`` (or dict-like).

        Reads ``config.cloud.provider`` (or ``config["cloud"]["provider"]``)
        and forwards the remaining provider-relevant fields as options.
        """
        cloud = _get_cloud_section(config)
        provider_name = _get_attr(cloud, "provider", "anthropic")
        options: dict[str, Any] = {}
        for key in (
            "model",
            "max_tokens",
            "timeout",
            "fallback_on_error",
            "deployment",
            "region",
            "base_url",
            "api_version",
            "referer",
            "x_title",
            "extra_kwargs",
        ):
            val = _get_attr(cloud, key, None)
            if val is not None:
                options[key] = val
        return cls.get(str(provider_name), options)


def _get_attr(obj: Any, name: str, default: Any) -> Any:
    """Read ``name`` from an attr-style or dict-style object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_cloud_section(config: Any) -> Any:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config.get("cloud", {})
    return getattr(config, "cloud", config)


# ---------------------------------------------------------------------------
# Auto-registration of the 11 first-class providers.
# ---------------------------------------------------------------------------


def _register_defaults() -> None:
    """Register the 11 first-class adapters at module import time.

    Each factory is a tiny closure that lazily imports the adapter
    module -- this keeps ``import i3.cloud.provider_registry`` cheap
    and means a missing optional SDK for provider X does not break
    registering provider Y.
    """

    def _anthropic(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=opts.get("model", "claude-sonnet-4-5"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 10.0)),
            fallback_on_error=bool(opts.get("fallback_on_error", False)),
        )

    def _openai(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=opts.get("model", "gpt-4.1"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 30.0)),
            base_url=opts.get("base_url"),
        )

    def _google(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.google import GoogleProvider

        return GoogleProvider(
            model=opts.get("model", "gemini-2.5-pro"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 30.0)),
        )

    def _azure(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.azure import AzureOpenAIProvider

        deployment = opts.get("deployment") or opts.get("model")
        if not deployment:
            raise ValueError(
                "Azure provider requires 'deployment' (or 'model') option"
            )
        return AzureOpenAIProvider(
            deployment=str(deployment),
            model=opts.get("model", str(deployment)),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 30.0)),
            api_version=opts.get("api_version"),
        )

    def _bedrock(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.bedrock import BedrockProvider

        return BedrockProvider(
            model=opts.get("model", "anthropic.claude-sonnet-4-5"),
            region=opts.get("region"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 30.0)),
        )

    def _mistral(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.mistral import MistralProvider

        return MistralProvider(
            model=opts.get("model", "mistral-large-latest"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 30.0)),
        )

    def _cohere(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.cohere import CohereProvider

        return CohereProvider(
            model=opts.get("model", "command-r-plus"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 30.0)),
        )

    def _ollama(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.ollama import OllamaProvider

        return OllamaProvider(
            model=opts.get("model", "llama3.3"),
            base_url=opts.get("base_url"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 120.0)),
        )

    def _openrouter(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(
            model=opts.get("model", "anthropic/claude-sonnet-4.5"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 60.0)),
            referer=opts.get("referer"),
            x_title=opts.get("x_title"),
        )

    def _litellm(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.litellm import LiteLLMProvider

        return LiteLLMProvider(
            model=opts.get("model", "openai/gpt-4.1"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 60.0)),
            extra_kwargs=opts.get("extra_kwargs") or {},
        )

    def _pangu(opts: dict[str, Any]) -> "CloudProvider":
        from i3.cloud.providers.huawei_pangu import HuaweiPanGuProvider

        return HuaweiPanGuProvider(
            model=opts.get("model", "pangu-deepthink-5.5"),
            region=opts.get("region"),
            max_tokens=int(opts.get("max_tokens", 512)),
            timeout=float(opts.get("timeout", 60.0)),
        )

    ProviderRegistry.register("anthropic", _anthropic)
    ProviderRegistry.register("openai", _openai)
    ProviderRegistry.register("google", _google)
    ProviderRegistry.register("azure", _azure)
    ProviderRegistry.register("bedrock", _bedrock)
    ProviderRegistry.register("mistral", _mistral)
    ProviderRegistry.register("cohere", _cohere)
    ProviderRegistry.register("ollama", _ollama)
    ProviderRegistry.register("openrouter", _openrouter)
    ProviderRegistry.register("litellm", _litellm)
    ProviderRegistry.register("huawei_pangu", _pangu)


_register_defaults()


__all__ = ["ProviderFactory", "ProviderRegistry"]
