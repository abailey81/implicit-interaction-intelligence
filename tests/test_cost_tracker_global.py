"""Iter 67 — Global CostTracker singleton + report shape tests."""
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


def test_singleton_returns_same_instance():
    a = get_global_cost_tracker()
    b = get_global_cost_tracker()
    assert a is b
    assert isinstance(a, CostTracker)


def test_reset_clears_singleton():
    a = get_global_cost_tracker()
    reset_global_cost_tracker()
    b = get_global_cost_tracker()
    assert a is not b


def test_record_and_report():
    tr = get_global_cost_tracker()
    tr.record(
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=200),
        latency_ms=850,
    )
    rep = tr.report().to_dict()
    assert rep["total_calls"] == 1
    assert rep["total_prompt_tokens"] == 1000
    assert rep["total_completion_tokens"] == 200
    assert "by_provider" in rep
    assert "anthropic" in rep["by_provider"]


def test_unknown_model_priced_at_zero_but_counted():
    tr = get_global_cost_tracker()
    cost = tr.record(
        provider="some_new_provider",
        model="totally-made-up-model",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        latency_ms=100,
    )
    assert cost == 0.0
    rep = tr.report().to_dict()
    assert rep["total_calls"] == 1
    assert any(u["provider"] == "some_new_provider" for u in rep["unknown_models"])


def test_report_to_dict_shape():
    tr = get_global_cost_tracker()
    rep = tr.report().to_dict()
    for k in ("total_calls", "total_prompt_tokens", "total_completion_tokens",
              "total_cached_tokens", "total_cost_usd", "by_provider",
              "by_model", "unknown_models", "as_of"):
        assert k in rep, f"missing {k!r} in report.to_dict()"
