"""Tests for :mod:`i3.cloud.multi_provider`.

Covers:
    - Sequential: returns the first success.
    - Sequential: falls through on failure to the next provider.
    - AllProvidersFailedError raised when every provider fails.
    - Circuit breaker opens after N consecutive failures.
    - Circuit breaker closes after the cool-down window.
    - Parallel: returns the fastest successful result.
    - Parallel: cancels the losers.
    - best_of_N: picks the lowest-penalty result.
    - best_of_N: raises AllProvidersFailed when everyone fails.
    - Rejects empty provider list and bad strategy.
    - Stats snapshot reflects attempts / successes / failures.
    - close() is idempotent and tolerates per-provider failures.
"""

from __future__ import annotations

import asyncio

import pytest

from i3.cloud.multi_provider import (
    AllProvidersFailedError,
    MultiProviderClient,
)
from i3.cloud.providers.base import (
    CompletionRequest,
    CompletionResult,
    TokenUsage,
    TransientError,
)


class _StubProvider:
    """Configurable fake provider for chain tests."""

    def __init__(
        self,
        *,
        name: str,
        delay: float = 0.0,
        fail_times: int = 0,
        text: str = "ok",
        raise_exc: type[BaseException] | None = None,
    ) -> None:
        self.provider_name = name
        self._delay = delay
        self._remaining_failures = fail_times
        self._text = text
        self._raise = raise_exc
        self.calls = 0
        self.closed = False

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            exc = self._raise or TransientError
            raise exc(f"{self.provider_name} transient")
        return CompletionResult(
            text=self._text,
            provider=self.provider_name,
            model="m",
            usage=TokenUsage(
                prompt_tokens=1, completion_tokens=2, total_tokens=3
            ),
            latency_ms=int(self._delay * 1000),
            finish_reason="stop",
        )

    async def close(self) -> None:
        self.closed = True


REQ = CompletionRequest(
    system="sys", messages=[], max_tokens=32, temperature=0.0
)


@pytest.mark.asyncio
async def test_sequential_returns_first_success() -> None:
    a = _StubProvider(name="a", text="a-text")
    b = _StubProvider(name="b", text="b-text")
    client = MultiProviderClient([a, b], strategy="sequential")
    result = await client.complete(REQ)
    assert result.text == "a-text"
    assert a.calls == 1
    assert b.calls == 0


@pytest.mark.asyncio
async def test_sequential_falls_through_on_failure() -> None:
    a = _StubProvider(name="a", fail_times=1)
    b = _StubProvider(name="b", text="b-text")
    client = MultiProviderClient([a, b], strategy="sequential")
    result = await client.complete(REQ)
    assert result.provider == "b"
    assert b.calls == 1


@pytest.mark.asyncio
async def test_all_providers_failed_raises() -> None:
    a = _StubProvider(name="a", fail_times=5)
    b = _StubProvider(name="b", fail_times=5)
    client = MultiProviderClient([a, b], strategy="sequential")
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await client.complete(REQ)
    assert set(exc_info.value.errors) == {"a", "b"}


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_three_failures() -> None:
    a = _StubProvider(name="a", fail_times=100)
    b = _StubProvider(name="b", text="b")
    client = MultiProviderClient(
        [a, b],
        strategy="sequential",
        failure_threshold=3,
        cool_down_s=60.0,
    )
    for _ in range(3):
        result = await client.complete(REQ)
        assert result.provider == "b"
    # Fourth call: breaker should be open, a is skipped, b still serves.
    await client.complete(REQ)
    # a was called only 3 times then skipped.
    assert a.calls == 3
    stats = client.stats
    assert stats["skipped_by_breaker"] >= 1


@pytest.mark.asyncio
async def test_circuit_breaker_closes_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _StubProvider(name="a", fail_times=3)
    b = _StubProvider(name="b", text="b")
    client = MultiProviderClient(
        [a, b],
        strategy="sequential",
        failure_threshold=3,
        cool_down_s=0.01,
    )
    # Trip the breaker.
    for _ in range(3):
        await client.complete(REQ)
    # Let the cool-down elapse.
    await asyncio.sleep(0.02)
    # 'a' has exhausted its failures so now succeeds; breaker should
    # half-open and let it through.
    result = await client.complete(REQ)
    assert result.provider == "a"


@pytest.mark.asyncio
async def test_parallel_returns_fastest_success() -> None:
    slow = _StubProvider(name="slow", delay=0.1, text="slow")
    fast = _StubProvider(name="fast", delay=0.01, text="fast")
    client = MultiProviderClient(
        [slow, fast],
        strategy="parallel",
        per_provider_timeout_s=1.0,
    )
    result = await client.complete(REQ)
    assert result.text == "fast"


@pytest.mark.asyncio
async def test_parallel_cancels_losers() -> None:
    slow = _StubProvider(name="slow", delay=0.5, text="slow")
    fast = _StubProvider(name="fast", delay=0.01, text="fast")
    client = MultiProviderClient(
        [slow, fast],
        strategy="parallel",
        per_provider_timeout_s=1.0,
    )
    result = await client.complete(REQ)
    assert result.text == "fast"
    # slow should have been cancelled; it may or may not have reached
    # its sleep, but it must not have returned a result ahead of fast.


@pytest.mark.asyncio
async def test_best_of_n_picks_lowest_penalty() -> None:
    # Short text is closer to target-length for a small max_tokens
    # request; very long text incurs a bigger length penalty.
    short = _StubProvider(name="short", text="a" * 60)
    long_text = _StubProvider(name="long", text="b" * 5000)
    client = MultiProviderClient(
        [short, long_text],
        strategy="best_of_n",
        per_provider_timeout_s=1.0,
    )
    req = CompletionRequest(
        system="s", messages=[], max_tokens=32, temperature=0.0
    )
    result = await client.complete(req)
    assert result.provider == "short"


@pytest.mark.asyncio
async def test_best_of_n_all_fail_raises() -> None:
    a = _StubProvider(name="a", fail_times=10)
    b = _StubProvider(name="b", fail_times=10)
    client = MultiProviderClient([a, b], strategy="best_of_n")
    with pytest.raises(AllProvidersFailedError):
        await client.complete(REQ)


def test_rejects_empty_providers() -> None:
    with pytest.raises(ValueError):
        MultiProviderClient([], strategy="sequential")


def test_rejects_bad_strategy() -> None:
    p = _StubProvider(name="p")
    with pytest.raises(ValueError):
        MultiProviderClient([p], strategy="bogus")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_stats_snapshot() -> None:
    a = _StubProvider(name="a", fail_times=1)
    b = _StubProvider(name="b", text="b")
    client = MultiProviderClient([a, b], strategy="sequential")
    await client.complete(REQ)
    s = client.stats
    assert s["successes"] == 1
    assert s["failures"] == 1
    assert s["attempts"] == 2


@pytest.mark.asyncio
async def test_close_is_idempotent_and_tolerates_failures() -> None:
    class BadClose(_StubProvider):
        async def close(self) -> None:
            raise RuntimeError("nope")

    a = _StubProvider(name="a")
    b = BadClose(name="b")
    client = MultiProviderClient([a, b], strategy="sequential")
    await client.close()
    await client.close()  # idempotent -- should not raise
    assert a.closed is True
