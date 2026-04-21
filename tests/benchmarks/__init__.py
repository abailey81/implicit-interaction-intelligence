"""Regression-only benchmark harness.

Real performance benchmarks live in the top-level ``benchmarks/``
directory (owned by the MLOps agent).  This directory holds the
**pytest-style regression gates** that surface speed regressions in
individual components during CI.  If you're looking for end-to-end
latency numbers, see ``/benchmarks/``.
"""
