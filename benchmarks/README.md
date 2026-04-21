# I3 Benchmarks

Performance benchmarks for Implicit Interaction Intelligence (I3).

## Layout

```
benchmarks/
    conftest.py                    shared fixtures (models, inputs, timer)
    test_encoder_latency.py        TCN encode single + batch
    test_slm_latency.py            SLM prefill + decode-per-token
    test_router_latency.py         bandit routing decision
    test_sanitizer_throughput.py   PII regex sanitizer
    test_pipeline_latency.py       end-to-end local pipeline latency
    slos.yaml                      Service Level Objectives
    locustfile.py                  Locust load-test scenarios
    k6/load.js                     k6 alternative script
```

## Running locally

```bash
# All benchmarks with pytest-benchmark output
pytest benchmarks/ --benchmark-only

# Just the encoder
pytest benchmarks/test_encoder_latency.py --benchmark-only

# Save a JSON report for tracking regressions over time
pytest benchmarks/ --benchmark-only --benchmark-json=reports/bench.json

# Compare against a baseline
pytest benchmarks/ --benchmark-only --benchmark-compare=reports/bench.json
```

Every benchmark uses:

* `time.perf_counter` as the timer
* 3 warmup rounds, 20 measured rounds
* `min` as the comparison metric

## Checkpoints

Benchmarks construct random-init models by default so the suite is
runnable from a clean checkout.  To enforce trained checkpoints, set:

```bash
export I3_BENCH_REQUIRE_CKPT=1
```

With this flag, benchmarks skip instead of silently running against
random weights.

## SLOs

`benchmarks/slos.yaml` defines the expected P50 / P95 / P99 latency
bounds.  Highlights:

| Component              | P50    | P95    | P99    |
|------------------------|--------|--------|--------|
| Local pipeline         | 200 ms | 260 ms | 320 ms |
| TCN encode (single)    | 5 ms   | 10 ms  | 20 ms  |
| SLM decode/token       | 4 ms   | 8 ms   | 16 ms  |
| Router decision        | 2 ms   | 5 ms   | 10 ms  |

## Load testing

### Locust

```bash
locust -f benchmarks/locustfile.py --host http://localhost:8000
```

Default scenario ramps 10 users over 30 s, steady for 5 min, ramps down
over 30 s.

### k6

```bash
k6 run benchmarks/k6/load.js --env HOST=http://localhost:8000
```

## CI integration

The recommended CI job:

1. Run `pytest benchmarks/ --benchmark-only --benchmark-json=bench.json`
2. Compare `bench.json` against the latest `main` baseline with
   `pytest-benchmark compare`.
3. Fail the job if any SLO in `slos.yaml` is violated.
