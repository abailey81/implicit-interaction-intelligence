# I3 Test Organization

The I3 test suite is organised by **test category**, not by the module
under test.  This lets you run a fast-feedback subset on every save and
escalate to more expensive gates only in CI.

## Layout

```
tests/
├── test_*.py             # Unit tests (hot-path, always run)
├── property/             # Hypothesis property-based tests
├── contract/             # OpenAPI & WebSocket protocol contracts
├── fuzz/                 # atheris coverage-guided fuzzers
├── load/                 # DoS-resistance / rate-limit tests
├── mutation/             # mutmut mutation-testing config
├── chaos/                # Failure-injection / resilience tests
├── snapshot/             # syrupy output-pinning tests
└── benchmarks/           # Regression-only perf gates
```

## Running by category

| Category   | Run command                                                              | Typical duration |
|------------|--------------------------------------------------------------------------|------------------|
| Unit       | `poetry run pytest tests/ -m "not slow and not load and not integration"` | < 30 s |
| Property   | `poetry run pytest tests/property/`                                       | 1–3 min |
| Contract   | `poetry run pytest tests/contract/`                                       | < 30 s |
| Snapshot   | `poetry run pytest tests/snapshot/`                                       | < 10 s |
| Chaos      | `poetry run pytest tests/chaos/`                                          | 1 min |
| Integration| `poetry run pytest tests/ -m integration`                                 | 1–2 min |
| Security   | `poetry run pytest tests/ -m security`                                    | < 30 s |
| Load       | `poetry run pytest tests/load/ -m load`                                   | 1–3 min |
| Mutation   | `poetry run mutmut run` (uses `mutmut-config.toml`)                       | 20–60 min |
| Fuzz       | `python tests/fuzz/fuzz_sanitizer.py -atheris_runs=100000` (see fuzz/README.md) | open-ended |
| Benchmarks | `poetry run pytest tests/benchmarks/`                                     | < 30 s |

## Run all CI gates

```bash
./scripts/run_all_tests.sh
```

Runs lint → type → unit → property → contract → integration → security
in the order the CI pipeline expects, failing fast at the first gate.

## Markers

Declared in `pyproject.toml`:

| Marker        | Meaning                                                          |
|---------------|-------------------------------------------------------------------|
| `slow`        | Wall-clock > 5 s; excluded from the default run.                 |
| `integration` | Hits disk / network / multiple subsystems.                       |
| `security`    | Security-focused (PII, crypto, sandbox).                         |
| `gpu`         | Requires a GPU — skipped on CI runners.                          |
| `load`        | DoS / rate-limit / throughput tests.                             |

To add a new marker, declare it in `pyproject.toml`
(`[tool.pytest.ini_options].markers`) so `--strict-markers` doesn't
complain.

## Dependencies required per category

These are all declared in `pyproject.toml` under `[tool.poetry.group.dev.dependencies]`
(or made optional via `pytest.importorskip` so they degrade gracefully
on a minimal install):

- Property tests → `hypothesis`
- Contract tests → `schemathesis`, `jsonschema`
- Snapshot tests → `syrupy`
- Fuzz tests → `atheris` (Linux/macOS; optional)
- Mutation tests → `mutmut`

## Writing a new test

- Unit / property / contract / chaos / snapshot tests live in the
  corresponding category directory.
- Mark long-running tests with `@pytest.mark.slow` so the hot-path run
  stays fast.
- Never read or write files outside `tmp_path`.
- Never commit credentials, API keys, or real user data into a test
  fixture.

See individual directory READMEs (`fuzz/`, `load/`, `mutation/`,
`benchmarks/`) for category-specific guidance.
