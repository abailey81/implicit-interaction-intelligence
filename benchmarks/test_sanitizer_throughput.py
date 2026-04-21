"""Throughput benchmarks for the PrivacySanitizer.

Sanitization is on the critical path of every outgoing message, so its
cost is measured as a raw ops-per-second number via ``pytest-benchmark``.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.benchmark(group="sanitizer")
def test_sanitizer_clean_text(benchmark: Any, sanitizer: Any) -> None:
    """Baseline sanitizer latency on clean text (no matches).

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        sanitizer: Shared :class:`PrivacySanitizer`.
    """
    text = "Hi there, how is your day going? I was thinking about our project."

    def _run() -> None:
        sanitizer.sanitize(text)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)


@pytest.mark.benchmark(group="sanitizer")
def test_sanitizer_pii_heavy(benchmark: Any, sanitizer: Any) -> None:
    """Sanitizer latency on text containing many PII patterns.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        sanitizer: Shared :class:`PrivacySanitizer`.
    """
    text = (
        "Contact Alice at alice@example.com or +1-555-123-4567. "
        "She lives at 123 Main Street and her DOB is 01/02/1990. "
        "Card: 4111 1111 1111 1111. Server IP 192.168.1.10."
    )

    def _run() -> None:
        sanitizer.sanitize(text)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)


@pytest.mark.benchmark(group="sanitizer")
def test_sanitizer_batch(benchmark: Any, sanitizer: Any, sample_texts: list) -> None:
    """Sanitizer throughput over a 20-message batch.

    Args:
        benchmark: ``pytest-benchmark`` fixture.
        sanitizer: Shared :class:`PrivacySanitizer`.
        sample_texts: Canned representative messages.
    """

    def _run() -> None:
        for t in sample_texts:
            sanitizer.sanitize(t)

    benchmark.pedantic(_run, iterations=1, rounds=20, warmup_rounds=3)
