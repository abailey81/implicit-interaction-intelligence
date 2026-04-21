"""Latency benchmarks for the TCN encoder.

Two scenarios:

* ``test_encoder_single`` -- batch size 1, 10-step window (realistic
  online inference path).
* ``test_encoder_batch``  -- batch size 16, same window (bulk path).

SLO targets (see ``benchmarks/slos.yaml``):

* TCN encode, single, P50 <= 5 ms.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.benchmark(group="encoder")
def test_encoder_single(benchmark: Any, encoder: Any, encoder_input: Any) -> None:
    """Single-sample TCN encoding latency.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        encoder: Shared TCN encoder.
        encoder_input: ``[1, 10, 32]`` tensor.
    """

    def _run() -> None:
        with torch.inference_mode():
            encoder(encoder_input)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)


@pytest.mark.benchmark(group="encoder")
def test_encoder_batch(
    benchmark: Any, encoder: Any, encoder_batch_input: Any
) -> None:
    """Batch-of-16 TCN encoding latency.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        encoder: Shared TCN encoder.
        encoder_batch_input: ``[16, 10, 32]`` tensor.
    """

    def _run() -> None:
        with torch.inference_mode():
            encoder(encoder_batch_input)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)
