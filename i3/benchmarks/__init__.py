"""Quantitative benchmark suite for I³.

Coordinates the five benchmark families used by the Benchmarks tab in
the demo UI and the ``scripts/run_benchmarks.py`` CLI:

  1. **Latency** (cold + warm, p50/p95/p99 per pipeline stage)
  2. **Perplexity** (eval ppl per category, training-curve series)
  3. **Conversational coherence** (audit hit-rate)
  4. **Adaptation faithfulness** (does response style track the request?)
  5. **Memory + size** (param counts, on-disk, peak RSS)

Output: ``reports/benchmarks/<timestamp>.{json,md}`` plus four
publication-quality SVG plots, plus a ``latest`` symlink/copy that the
Benchmarks UI tab fetches via the REST API.

See :class:`~i3.benchmarks.runner.BenchmarkRunner` for the public API.
"""

from __future__ import annotations

from i3.benchmarks.runner import BenchmarkResult, BenchmarkRunner

__all__ = ["BenchmarkResult", "BenchmarkRunner"]
