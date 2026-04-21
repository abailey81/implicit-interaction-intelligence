"""Latency benchmarks for the IntelligentRouter (Thompson-sampling bandit).

Two scenarios:

* ``test_router_decision_short`` -- short query, typical chat turn.
* ``test_router_decision_long``  -- long query, stresses the complexity
  estimator and sensitivity detector.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.benchmark(group="router")
def test_router_decision_short(
    benchmark: Any, router: Any, routing_context: Any
) -> None:
    """Routing decision latency for a short query.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        router: Shared :class:`IntelligentRouter`.
        routing_context: Default routing context.
    """
    text = "Hi, how are you today?"

    def _run() -> None:
        router.route(text, ctx=routing_context)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)


@pytest.mark.benchmark(group="router")
def test_router_decision_long(
    benchmark: Any, router: Any, routing_context: Any
) -> None:
    """Routing decision latency for a longer, more complex query.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        router: Shared :class:`IntelligentRouter`.
        routing_context: Default routing context.
    """
    text = (
        "Can you explain the trade-offs between causal dilated convolutions "
        "and self-attention for time-series representation learning, with "
        "special attention to receptive field and parameter efficiency?"
    )

    def _run() -> None:
        router.route(text, ctx=routing_context)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)
