# I3 Benchmark Tests (regression gates)

This directory holds **regression-only** benchmark tests: tiny,
deterministic perf asserts that a component has not become drastically
slower since the last release.

**For real performance benchmarking, see the top-level `/benchmarks/`
directory** — that is where the MLOps agent keeps the end-to-end
latency harness, the throughput sweeps, and the device-class
compatibility matrix.  The scripts there are out-of-scope for this
pytest suite.

## When to add a regression test

Add a file here when:
- You have measured a stable baseline for a function on the CI runner.
- A regression would be high-impact (e.g. sanitiser latency ballooning
  past the 50 ms/turn budget; TCN forward pass exceeding the 200 ms
  edge budget).
- The test can be made reliable on a shared CI runner — i.e. the
  measurement has generous margin (≥ 2×) over the noise floor.

Do NOT add:
- Wall-clock tests with tight margins (< 2×).  Those belong in
  `/benchmarks/` where they are run on calibrated hardware.
- Throughput / QPS tests that require multiple workers.

## Running

```bash
# All regression gates
poetry run pytest tests/benchmarks/

# The full perf harness (different repo-level dir!)
poetry run python benchmarks/run.py
```

## Relationship to `pytest-benchmark`

We deliberately do **not** depend on `pytest-benchmark` here.  The
regression gates are simple `time.perf_counter()` asserts with wide
margins; richer benchmarking (percentiles, warmup runs, JSON artifacts)
happens in `/benchmarks/`.
