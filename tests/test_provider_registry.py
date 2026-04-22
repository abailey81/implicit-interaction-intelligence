"""Tests for :mod:`i3.cloud.provider_registry`.

Covers:
    - Registering and retrieving a custom factory.
    - Case-insensitive provider lookup.
    - Unknown provider raises KeyError with a helpful message.
    - Empty names are rejected.
    - Default registration covers all 11 providers.
    - ``from_config`` builds the right adapter for a dict config.
    - ``from_config`` builds the right adapter for a SimpleNamespace config.
    - Options are forwarded to the factory.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from i3.cloud.provider_registry import ProviderRegistry
from i3.cloud.providers.base import (
    CompletionRequest,
    CompletionResult,
    TokenUsage,
)


class _Fake:
    provider_name = "test_fake"

    def __init__(self, opts: dict[str, Any]) -> None:
        self.opts = opts

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        return CompletionResult(
            text="ok",
            provider=self.provider_name,
            model=str(self.opts.get("model", "m")),
            usage=TokenUsage(),
            latency_ms=1,
            finish_reason="stop",
        )

    async def close(self) -> None:
        return None


def test_register_and_get_returns_instance() -> None:
    ProviderRegistry.register("unit_fake_1", lambda opts: _Fake(opts))
    instance = ProviderRegistry.get("unit_fake_1", {"model": "x"})
    assert isinstance(instance, _Fake)
    assert instance.opts == {"model": "x"}


def test_register_is_case_insensitive() -> None:
    ProviderRegistry.register("UNIT_FAKE_CASE", lambda opts: _Fake(opts))
    assert isinstance(ProviderRegistry.get("unit_fake_case"), _Fake)
    assert isinstance(ProviderRegistry.get("Unit_Fake_Case"), _Fake)


def test_unknown_provider_raises_keyerror() -> None:
    with pytest.raises(KeyError) as exc_info:
        ProviderRegistry.get("definitely_not_registered")
    assert "registered" in str(exc_info.value)


def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError):
        ProviderRegistry.register("", lambda opts: _Fake(opts))
    with pytest.raises(ValueError):
        ProviderRegistry.register("   ", lambda opts: _Fake(opts))


def test_default_registration_covers_all_eleven_providers() -> None:
    names = set(ProviderRegistry.names())
    expected = {
        "anthropic",
        "openai",
        "google",
        "azure",
        "bedrock",
        "mistral",
        "cohere",
        "ollama",
        "openrouter",
        "litellm",
        "huawei_pangu",
    }
    missing = expected - names
    assert not missing, f"missing providers: {missing}"


def test_from_config_with_dict() -> None:
    ProviderRegistry.register("unit_fake_dict", lambda opts: _Fake(opts))
    cfg = {
        "cloud": {
            "provider": "unit_fake_dict",
            "model": "mymodel",
            "max_tokens": 42,
        }
    }
    inst = ProviderRegistry.from_config(cfg)
    assert isinstance(inst, _Fake)
    assert inst.opts["model"] == "mymodel"
    assert inst.opts["max_tokens"] == 42


def test_from_config_with_namespace() -> None:
    ProviderRegistry.register("unit_fake_ns", lambda opts: _Fake(opts))
    cfg = SimpleNamespace(
        cloud=SimpleNamespace(
            provider="unit_fake_ns",
            model="m",
            max_tokens=10,
            timeout=5.0,
            fallback_on_error=True,
        )
    )
    inst = ProviderRegistry.from_config(cfg)
    assert isinstance(inst, _Fake)
    assert inst.opts["timeout"] == 5.0


def test_from_config_default_provider_is_anthropic() -> None:
    # When provider is omitted we fall back to anthropic.  Importing
    # the anthropic adapter must not require ANTHROPIC_API_KEY (only
    # calling .complete() does).
    cfg = {"cloud": {"model": "claude-sonnet-4-5"}}
    inst = ProviderRegistry.from_config(cfg)
    assert inst.provider_name == "anthropic"


def test_options_forwarded_to_factory() -> None:
    captured: dict[str, Any] = {}

    def factory(opts: dict[str, Any]) -> _Fake:
        captured.update(opts)
        return _Fake(opts)

    ProviderRegistry.register("unit_fake_fwd", factory)
    ProviderRegistry.get(
        "unit_fake_fwd",
        {"model": "a", "max_tokens": 1, "timeout": 2.0, "region": "r"},
    )
    assert captured == {
        "model": "a",
        "max_tokens": 1,
        "timeout": 2.0,
        "region": "r",
    }
