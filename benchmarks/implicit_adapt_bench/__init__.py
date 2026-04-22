"""ImplicitAdaptBench — a benchmark for adaptive generation from implicit signals.

This package implements the benchmark described in
``docs/research/implicit_adapt_bench.md``. It measures how well a generator
adapts its response to a behavioural context that has **not** been expressed
as an explicit user profile — filling a gap left by the 2025-2026 wave of
explicit-profile benchmarks (PersonaLens, AlpsBench, PersoBench).

Public modules:

* :mod:`benchmarks.implicit_adapt_bench.data_schema` — Pydantic record types.
* :mod:`benchmarks.implicit_adapt_bench.metrics` — metric implementations.
* :mod:`benchmarks.implicit_adapt_bench.data_generator` — synthetic-data
  generator for the train / dev / test splits.
* :mod:`benchmarks.implicit_adapt_bench.baselines` — three runnable baselines
  (``none``, ``prompt``, ``cross_attention``).
* :mod:`benchmarks.implicit_adapt_bench.scoring` — CLI-usable scoring.

The benchmark measures **responsiveness** of the generator to the implicit
signal, not generation quality — this is intentional so that the benchmark
can be run on a random-init SLM for demonstration and is not confounded by
model scale.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__: str = "0.1.0"
