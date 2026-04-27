"""Iter 123 — CostTracker thread-safety + concurrent-record correctness."""
from __future__ import annotations

import threading

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


def test_concurrent_record_does_not_lose_calls():
    tr = CostTracker()
    n_threads = 8
    calls_per_thread = 50

    def worker():
        for _ in range(calls_per_thread):
            tr.record(
                provider="anthropic", model="claude-haiku-4-5",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                latency_ms=1,
            )

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rep = tr.report().to_dict()
    assert rep["total_calls"] == n_threads * calls_per_thread
    assert rep["total_prompt_tokens"] == n_threads * calls_per_thread * 10
    assert rep["total_completion_tokens"] == n_threads * calls_per_thread * 5


def test_concurrent_global_singleton_safe():
    """Multiple threads racing get_global_cost_tracker() should all
    receive the same singleton (or at least never crash)."""
    instances = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        instances.append(get_global_cost_tracker())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 8
    # All threads should have got the same singleton.
    assert all(inst is instances[0] for inst in instances)


def test_record_is_atomic_per_call():
    """Two calls' total_cost_usd must equal the sum of the two
    individual costs (no torn writes)."""
    tr = CostTracker()
    c1 = tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
        latency_ms=10,
    )
    c2 = tr.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=2000, completion_tokens=1000),
        latency_ms=20,
    )
    rep = tr.report().to_dict()
    assert rep["total_cost_usd"] == pytest.approx(c1 + c2, abs=0.0001)


def test_reset_during_record_does_not_corrupt():
    """Reset between two records starts fresh; no orphan totals."""
    tr1 = get_global_cost_tracker()
    tr1.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        latency_ms=10,
    )
    reset_global_cost_tracker()
    tr2 = get_global_cost_tracker()
    tr2.record(
        provider="anthropic", model="claude-haiku-4-5",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=100),
        latency_ms=20,
    )
    rep = tr2.report().to_dict()
    assert rep["total_calls"] == 1
    assert rep["total_prompt_tokens"] == 200
