# I3 Load / DoS Tests

This directory contains **in-process load tests** that drive the FastAPI
TestClient and the WebSocket testing API to verify:

| Test file                  | What it exercises                                                |
|----------------------------|------------------------------------------------------------------|
| `test_websocket_dos.py`    | 1000 rapid JSON frames; oversized frames -> close code 1009.     |
| `test_rate_limit.py`       | REST rate limiter — verifies 429 kicks in for a burst caller.    |

These tests are pytest-based but tagged `@pytest.mark.load` (and
`@pytest.mark.slow` where appropriate) so they are opt-in:

```bash
# default: skip load tests
poetry run pytest

# run just the load suite
poetry run pytest tests/load/ -m load

# include load tests alongside the rest
poetry run pytest -m "not gpu"
```

## Why in-process?

Running these against a real uvicorn worker would be more realistic but
much harder to reproduce in CI.  FastAPI's `TestClient` shares all of
the middleware with production (CORS, rate-limit, size-limit, security
headers), so in-process tests reliably catch the same regressions with
far lower flakiness.  For end-to-end latency benchmarking see
`benchmarks/` (owned by MLOps).
