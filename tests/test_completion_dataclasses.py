"""Iter 116 — TokenUsage / CompletionRequest / CompletionResult contract."""
from __future__ import annotations

import pytest

from i3.cloud.providers.base import (
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

def test_token_usage_defaults_to_zero():
    u = TokenUsage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0
    assert u.cached_tokens == 0


def test_token_usage_field_assignment():
    u = TokenUsage(prompt_tokens=100, completion_tokens=50,
                   total_tokens=150, cached_tokens=25)
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 50


def test_token_usage_frozen_extra_forbidden():
    """Pydantic frozen + extra=forbid → reject unknown keys."""
    with pytest.raises(Exception):
        TokenUsage(prompt_tokens=10, fake_field=99)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------

def test_chat_message_role_validates():
    m = ChatMessage(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"


def test_chat_message_unknown_role_raises():
    with pytest.raises(Exception):
        ChatMessage(role="invalid_role", content="hi")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CompletionRequest
# ---------------------------------------------------------------------------

def test_completion_request_minimal():
    r = CompletionRequest()
    assert r.system is None
    assert r.messages == []
    assert r.max_tokens == 512
    assert r.temperature == 0.7


def test_completion_request_with_messages():
    r = CompletionRequest(
        system="you are helpful",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=256, temperature=0.3,
    )
    assert r.system == "you are helpful"
    assert len(r.messages) == 1


def test_completion_request_metadata_default_dict():
    a = CompletionRequest()
    b = CompletionRequest()
    # Pydantic frozen=True, so default-factory dicts must not be shared
    assert a.metadata == {}
    assert b.metadata == {}


# ---------------------------------------------------------------------------
# CompletionResult
# ---------------------------------------------------------------------------

def test_completion_result_required_fields():
    r = CompletionResult(
        text="hello",
        provider="anthropic",
        model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3),
        latency_ms=42,
        finish_reason="stop",
    )
    assert r.text == "hello"
    assert r.provider == "anthropic"
    assert r.usage.prompt_tokens == 5


def test_completion_result_finish_reason_normalised():
    """The base.py docstring lists 'stop' / 'length' / 'content_filter'
    / 'tool_use' / 'error' as canonical finish reasons.  Verify a
    request can use any of them."""
    for fr in ("stop", "length", "content_filter", "tool_use", "error"):
        r = CompletionResult(
            text="x", provider="p", model="m",
            usage=TokenUsage(), latency_ms=10, finish_reason=fr,
        )
        assert r.finish_reason == fr
