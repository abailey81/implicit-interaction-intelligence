"""Iter 130 — CostTracker reset semantics edge cases."""
from __future__ import annotations

import pytest

from i3.cloud.cost_tracker import (
    CostTracker,
    get_global_cost_tracker,
    reset_global_cost_tracker,
)
from i3.cloud.providers.base import TokenUsage


@pytest.fixture(autouse=True)
def _clean():
    reset_global_cost_tracker()
    yield
    reset_global_cost_tracker()


def test_reset_idempotent():
    """Calling reset twice in a row should not raise."""
    reset_global_cost_tracker()
    reset_global_cost_tracker()
    reset_global_cost_tracker()


def test_reset_after_many_records():
    tr = get_global_cost_tracker()
    for _ in range(100):
        tr.record(
            provider="anthropic", model="claude-haiku-4-5",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
            latency_ms=1,
        )
    reset_global_cost_tracker()
    fresh = get_global_cost_tracker()
    rep = fresh.report().to_dict()
    assert rep["total_calls"] == 0


def test_per_instance_reset_does_not_affect_global():
    """A standalone CostTracker.reset() shouldn't touch the global
    singleton's state."""
    standalone = CostTracker()
    standalone.record(
        provider="x", model="y",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        latency_ms=1,
    )
    standalone.reset()
    assert standalone.report().to_dict()["total_calls"] == 0
    # The global tracker is untouched (no-op since we never used it)
    glob = get_global_cost_tracker()
    assert glob.report().to_dict()["total_calls"] == 0


def test_global_singleton_recreated_after_reset():
    a = get_global_cost_tracker()
    reset_global_cost_tracker()
    b = get_global_cost_tracker()
    assert a is not b
    assert isinstance(b, CostTracker)


def test_record_after_reset_starts_fresh_aggregate():
    tr = get_global_cost_tracker()
    tr.record(provider="x", model="y",
              usage=TokenUsage(prompt_tokens=999, completion_tokens=99),
              latency_ms=1)
    reset_global_cost_tracker()
    tr2 = get_global_cost_tracker()
    tr2.record(provider="x", model="y",
               usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
               latency_ms=1)
    rep = tr2.report().to_dict()
    assert rep["total_prompt_tokens"] == 10  # not 1009
    assert rep["total_completion_tokens"] == 5
