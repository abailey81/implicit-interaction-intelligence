"""Iter 93 — CostTracker priced-call integration.

Verifies that the CostTracker reads pricing_2026.json correctly:
* a known (provider, model) pair → non-zero cost
* an unknown pair → zero cost + entry in unknown_models
* aggregate cost is the sum of per-call costs
"""
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


def test_known_model_has_nonzero_cost():
    tr = CostTracker()
    cost = tr.record(
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=TokenUsage(prompt_tokens=1_000_000,
                         completion_tokens=200_000),
        latency_ms=1000,
    )
    # 1M prompt @ $3.0 + 0.2M completion @ $15.0 = $3.0 + $3.0 = $6.0
    # (within rounding, depending on cached_tokens default 0)
    assert cost > 0, f"expected non-zero cost, got {cost}"


def test_unknown_model_zero_cost_but_counted():
    tr = CostTracker()
    cost = tr.record(
        provider="totally-fake-provider-xyz",
        model="totally-fake-model-xyz",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        latency_ms=10,
    )
    assert cost == 0.0
    rep = tr.report().to_dict()
    assert rep["total_calls"] == 1
    assert any(u["provider"] == "totally-fake-provider-xyz"
               for u in rep["unknown_models"])


def test_aggregate_cost_is_sum_of_calls():
    tr = CostTracker()
    c1 = tr.record(
        provider="anthropic",
        model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=10_000, completion_tokens=2_000),
        latency_ms=200,
    )
    c2 = tr.record(
        provider="anthropic",
        model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=5_000, completion_tokens=500),
        latency_ms=100,
    )
    rep = tr.report().to_dict()
    assert rep["total_cost_usd"] == pytest.approx(c1 + c2, abs=0.0001)


def test_token_totals_aggregate_correctly():
    tr = CostTracker()
    tr.record(
        provider="openai",
        model="gpt-4o",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        latency_ms=200,
    )
    tr.record(
        provider="openai",
        model="gpt-4o",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=75),
        latency_ms=300,
    )
    rep = tr.report().to_dict()
    assert rep["total_prompt_tokens"] == 300
    assert rep["total_completion_tokens"] == 125


def test_by_provider_breakdown():
    tr = CostTracker()
    tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        latency_ms=10,
    )
    tr.record(
        provider="openai", model="gpt-4o",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=100),
        latency_ms=20,
    )
    rep = tr.report().to_dict()
    assert "anthropic" in rep["by_provider"]
    assert "openai" in rep["by_provider"]


def test_global_reset_zeroes_aggregates():
    tr = get_global_cost_tracker()
    tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        latency_ms=10,
    )
    reset_global_cost_tracker()
    fresh = get_global_cost_tracker()
    rep = fresh.report().to_dict()
    assert rep["total_calls"] == 0
    assert rep["total_cost_usd"] == 0.0
