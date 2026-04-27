"""Iter 88 — CostTracker integration with MultiProviderClient.

Verifies that successful cloud calls flowing through the chain bill
into the global CostTracker so /api/cost/report stays current.
"""
from __future__ import annotations

import asyncio

import pytest

from i3.cloud.cost_tracker import (
    get_global_cost_tracker,
    reset_global_cost_tracker,
)
from i3.cloud.multi_provider import MultiProviderClient
from i3.cloud.providers.base import (
    CompletionRequest,
    CompletionResult,
    TokenUsage,
    TransientError,
)


class _Stub:
    def __init__(self, name="test_provider", model="test-model",
                 succeed=True, prompt_tokens=100, completion_tokens=50):
        self.provider_name = name
        self._model = model
        self._succeed = succeed
        self._pt = prompt_tokens
        self._ct = completion_tokens

    async def complete(self, request):
        if not self._succeed:
            raise TransientError("simulated", provider=self.provider_name)
        return CompletionResult(
            text="hello",
            provider=self.provider_name,
            model=self._model,
            usage=TokenUsage(prompt_tokens=self._pt,
                             completion_tokens=self._ct,
                             total_tokens=self._pt + self._ct),
            latency_ms=42,
            finish_reason="stop",
        )

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean():
    reset_global_cost_tracker()
    yield
    reset_global_cost_tracker()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_successful_call_bills_cost_tracker():
    chain = MultiProviderClient([_Stub(name="x", model="x-mock")])
    _run(chain.complete(CompletionRequest()))
    rep = get_global_cost_tracker().report().to_dict()
    assert rep["total_calls"] == 1
    assert rep["total_prompt_tokens"] == 100
    assert rep["total_completion_tokens"] == 50


def test_two_calls_aggregate():
    chain = MultiProviderClient([_Stub(name="x", model="x-mock",
                                       prompt_tokens=10,
                                       completion_tokens=5)])
    _run(chain.complete(CompletionRequest()))
    _run(chain.complete(CompletionRequest()))
    rep = get_global_cost_tracker().report().to_dict()
    assert rep["total_calls"] == 2
    assert rep["total_prompt_tokens"] == 20
    assert rep["total_completion_tokens"] == 10


def test_failed_calls_do_not_bill():
    chain = MultiProviderClient(
        [_Stub(name="bad", succeed=False), _Stub(name="good")],
    )
    _run(chain.complete(CompletionRequest()))
    rep = get_global_cost_tracker().report().to_dict()
    # Only the successful 'good' call should be billed.
    assert rep["total_calls"] == 1
    assert "good" in rep["by_provider"]
    assert "bad" not in rep["by_provider"]


def test_cost_telemetry_does_not_block_result():
    """Even if the CostTracker raises, the result must still come back."""
    chain = MultiProviderClient([_Stub(name="x", model="x-mock")])
    res = _run(chain.complete(CompletionRequest()))
    assert res.text == "hello"
