"""End-to-end pipeline latency benchmarks.

These benchmarks stitch together the critical components of the I3
request path -- sanitizer, encoder, router -- without requiring the
full async :class:`Pipeline` (which touches the filesystem, SQLite, and
the network).  The goal is to catch regressions in the synchronous
latency budget.

SLO targets (see ``benchmarks/slos.yaml``):

* P50 local pipeline <= 200 ms
* P95 local pipeline <= 260 ms
* P99 local pipeline <= 320 ms
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.benchmark(group="pipeline")
def test_pipeline_local_path(
    benchmark: Any,
    encoder: Any,
    slm: Any,
    router: Any,
    sanitizer: Any,
    routing_context: Any,
) -> None:
    """Synthetic end-to-end local pipeline turn (sanitize + encode + route + prefill).

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        encoder: Shared TCN encoder.
        slm: Shared Adaptive SLM.
        router: Shared :class:`IntelligentRouter`.
        sanitizer: Shared :class:`PrivacySanitizer`.
        routing_context: Default routing context.
    """
    text = "Hi, can you help me summarise this article about transformers?"
    feat = torch.randn(1, 10, 32)
    ids = torch.randint(0, max(slm.vocab_size, 1), (1, 16), dtype=torch.long)

    def _run() -> None:
        sanitizer.sanitize(text)
        with torch.inference_mode():
            encoder(feat)
        router.route(text, ctx=routing_context)
        with torch.inference_mode():
            slm(ids)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)


@pytest.mark.benchmark(group="pipeline")
def test_pipeline_sanitize_then_encode(
    benchmark: Any,
    encoder: Any,
    sanitizer: Any,
) -> None:
    """Tight sanitize+encode loop (no SLM) -- useful for keystroke streams.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        encoder: Shared TCN encoder.
        sanitizer: Shared :class:`PrivacySanitizer`.
    """
    text = "Just checking in about the timeline for the next milestone."
    feat = torch.randn(1, 10, 32)

    def _run() -> None:
        sanitizer.sanitize(text)
        with torch.inference_mode():
            encoder(feat)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)
