"""Iter 108 — CostTracker cached_tokens pricing correctness."""
from __future__ import annotations

import pytest

from i3.cloud.cost_tracker import (
    CostTracker,
    reset_global_cost_tracker,
)
from i3.cloud.providers.base import TokenUsage


@pytest.fixture(autouse=True)
def _clean():
    reset_global_cost_tracker()
    yield
    reset_global_cost_tracker()


def test_cached_tokens_priced_lower_than_full_input():
    """Anthropic's claude-haiku-4-5 has cached_input < input rate, so
    a fully-cached call should cost less than an uncached call of the
    same prompt size."""
    tr = CostTracker()
    cost_uncached = tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=1_000_000,
                         completion_tokens=0,
                         cached_tokens=0),
        latency_ms=100,
    )
    tr2 = CostTracker()
    cost_cached = tr2.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=1_000_000,
                         completion_tokens=0,
                         cached_tokens=1_000_000),
        latency_ms=100,
    )
    assert cost_cached < cost_uncached, \
        f"cached call cost {cost_cached} not < uncached {cost_uncached}"


def test_partially_cached_in_between():
    tr1 = CostTracker()
    full_uncached = tr1.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=1_000_000,
                         completion_tokens=0, cached_tokens=0),
        latency_ms=100,
    )
    tr2 = CostTracker()
    half_cached = tr2.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=1_000_000,
                         completion_tokens=0, cached_tokens=500_000),
        latency_ms=100,
    )
    tr3 = CostTracker()
    full_cached = tr3.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=1_000_000,
                         completion_tokens=0, cached_tokens=1_000_000),
        latency_ms=100,
    )
    # half-cached should be between the two extremes
    assert full_cached <= half_cached <= full_uncached


def test_zero_tokens_zero_cost():
    tr = CostTracker()
    cost = tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=0, completion_tokens=0,
                         cached_tokens=0),
        latency_ms=10,
    )
    assert cost == 0.0


def test_completion_tokens_priced():
    """Completion tokens should always cost at the higher 'output' rate."""
    tr = CostTracker()
    cost = tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=0, completion_tokens=1_000_000,
                         cached_tokens=0),
        latency_ms=100,
    )
    assert cost > 0
