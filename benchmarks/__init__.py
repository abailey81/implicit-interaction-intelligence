"""I3 micro- and macro-benchmark suite.

The ``benchmarks`` package is intentionally kept outside the test suite
so that ``pytest tests/`` does not pay the cost of heavy benchmark
fixtures.  Run the suite explicitly::

    pytest benchmarks/ --benchmark-only

Fixtures live in :mod:`benchmarks.conftest`.  Each benchmark module
targets a single I3 subsystem (encoder, SLM, router, sanitizer,
end-to-end pipeline).
"""
