"""Iter 114 — Cloud-provider error class taxonomy tests."""
from __future__ import annotations

import pytest

from i3.cloud.providers.base import (
    AuthError,
    PermanentError,
    ProviderError,
    RateLimitedError,
    TransientError,
)


def test_auth_error_is_provider_error():
    e = AuthError("bad key", provider="anthropic")
    assert isinstance(e, ProviderError)


def test_rate_limited_is_provider_error():
    e = RateLimitedError("too many", provider="openai")
    assert isinstance(e, ProviderError)


def test_transient_is_provider_error():
    e = TransientError("timeout", provider="openai")
    assert isinstance(e, ProviderError)


def test_permanent_is_provider_error():
    e = PermanentError("bad request", provider="openai")
    assert isinstance(e, ProviderError)


def test_provider_attribute_preserved():
    e = AuthError("x", provider="mistral")
    assert e.provider == "mistral"


def test_provider_optional():
    e = ProviderError("generic")  # no provider kwarg
    # Should not raise; provider is optional
    assert isinstance(e, Exception)


def test_message_str_contains_text():
    e = AuthError("invalid api key", provider="openai")
    assert "invalid api key" in str(e) or "invalid api key" in e.args[0]


def test_distinct_classes_for_isinstance_routing():
    """The chain logic relies on isinstance() to route errors; verify
    the four classes are siblings, not a single class with a 'kind'
    attribute."""
    classes = {AuthError, RateLimitedError, TransientError, PermanentError}
    assert len(classes) == 4
    # None inherits from another (besides ProviderError).
    for cls in classes:
        for other in classes - {cls}:
            assert not issubclass(cls, other) or cls is other
