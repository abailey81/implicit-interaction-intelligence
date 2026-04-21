"""Latency benchmarks for the Adaptive SLM.

Two scenarios:

* ``test_slm_prefill`` -- prompt of length 16, representative of the
  first forward pass on a new turn.
* ``test_slm_decode``  -- append one token to an already-cached
  context.  Without a stable KV cache export we approximate decode by
  calling ``forward`` on a single-token input.

SLO targets (see ``benchmarks/slos.yaml``):

* SLM decode per token, P50 <= 4 ms (CPU reference).
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.benchmark(group="slm")
def test_slm_prefill(benchmark: Any, slm: Any, slm_prefill_input: Any) -> None:
    """Prefill-stage latency for a 16-token prompt.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        slm: Shared Adaptive SLM.
        slm_prefill_input: ``[1, 16]`` int64 prompt tensor.
    """

    def _run() -> None:
        with torch.inference_mode():
            slm(slm_prefill_input)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)


@pytest.mark.benchmark(group="slm")
def test_slm_decode_step(benchmark: Any, slm: Any, slm_decode_input: Any) -> None:
    """Approximate single-token decode latency.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        slm: Shared Adaptive SLM.
        slm_decode_input: ``[1, 1]`` int64 input tensor.
    """

    def _run() -> None:
        with torch.inference_mode():
            slm(slm_decode_input)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)
