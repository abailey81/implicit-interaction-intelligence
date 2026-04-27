"""Iter 63 — MultiProviderClient fail-fast / fallback / circuit-breaker tests.

Verifies the fallback chain that the iter-52 Gemini cascade arm
relies on as a member of the cloud router.  All tests use stub
providers (no network), so the suite runs in <0.1 s.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from i3.cloud.multi_provider import MultiProviderClient
from i3.cloud.providers.base import (
    CompletionRequest,
    CompletionResult,
    PermanentError,
    TokenUsage,
    TransientError,
    AuthError,
)


class _StubProvider:
    """Minimal CloudProvider stub for the multi-provider chain."""

    def __init__(self, name, *, succeed=True, exc=None, latency_ms=10.0):
        self.provider_name = name
        self.succeed = succeed
        self.exc = exc
        self.latency_ms = latency_ms
        self.calls = 0

    async def complete(self, request):  # noqa: ARG002
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.succeed:
            raise TransientError(f"{self.provider_name} simulated failure",
                                 provider=self.provider_name)
        return CompletionResult(
            text=f"reply from {self.provider_name}",
            provider=self.provider_name,
            model=f"{self.provider_name}-mock",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5,
                             total_tokens=15),
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )

    async def close(self):
        pass


def _req() -> CompletionRequest:
    return CompletionRequest(system="be brief", messages=[])


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_first_provider_success_short_circuits():
    p1 = _StubProvider("first")
    p2 = _StubProvider("second")
    chain = MultiProviderClient([p1, p2])
    res = _run(chain.complete(_req()))
    assert res.text == "reply from first"
    assert p1.calls == 1
    assert p2.calls == 0


def test_falls_back_on_first_failure():
    p1 = _StubProvider("first", succeed=False)
    p2 = _StubProvider("second")
    chain = MultiProviderClient([p1, p2])
    res = _run(chain.complete(_req()))
    assert res.text == "reply from second"
    assert p1.calls == 1
    assert p2.calls == 1


def test_all_failures_raises():
    p1 = _StubProvider("first", succeed=False)
    p2 = _StubProvider("second", succeed=False)
    chain = MultiProviderClient([p1, p2])
    with pytest.raises(Exception):
        _run(chain.complete(_req()))


def test_circuit_breaker_opens_after_threshold():
    p1 = _StubProvider("first", succeed=False)
    p2 = _StubProvider("second")
    chain = MultiProviderClient(
        [p1, p2], failure_threshold=2, cool_down_s=60.0,
    )
    # First two calls open the breaker; subsequent calls should skip p1.
    for _ in range(3):
        _run(chain.complete(_req()))
    # p1 was called for the first 2 attempts, then the breaker opened
    # so it stayed at 2.
    assert p1.calls == 2
    # p2 served all three.
    assert p2.calls == 3
    assert chain.stats["skipped_by_breaker"] >= 1


def test_auth_error_is_treated_as_terminal_for_provider():
    """AuthError should fall through to the next provider, not be retried."""
    p1 = _StubProvider("first", exc=AuthError("bad key", provider="first"))
    p2 = _StubProvider("second")
    chain = MultiProviderClient([p1, p2])
    res = _run(chain.complete(_req()))
    assert res.text == "reply from second"
    # Re-call: the AuthError counted as a failure; second still succeeds
    res2 = _run(chain.complete(_req()))
    assert res2.text == "reply from second"


def test_stats_are_reported():
    p1 = _StubProvider("first")
    chain = MultiProviderClient([p1])
    _run(chain.complete(_req()))
    s = chain.stats
    assert s["attempts"] >= 1
    assert s["successes"] >= 1
    assert isinstance(s["last_errors"], dict)


def test_strategy_validation():
    with pytest.raises(ValueError):
        MultiProviderClient([_StubProvider("p")], strategy="rocket-science")
    with pytest.raises(ValueError):
        MultiProviderClient([])  # empty providers


def test_close_is_idempotent():
    p1 = _StubProvider("first")
    chain = MultiProviderClient([p1])
    _run(chain.close())
    _run(chain.close())  # no-op second close
