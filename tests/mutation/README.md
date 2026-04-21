# Mutation Testing for I3

We use [mutmut](https://mutmut.readthedocs.io/) to measure how well our
test suite **detects** deliberately introduced bugs ("mutations") in the
security-critical modules:

- `i3/router/`      — routing / bandit / sensitivity
- `i3/adaptation/`  — adaptation controller & types
- `i3/privacy/`     — sanitiser & encryption
- `i3/encoder/`     — TCN encoder

The mutation set is standard Python operator / constant mutations
(`+` → `-`, `True` → `False`, `>` → `>=`, etc.).  See the repo-root
`mutmut-config.toml` for the full configuration.

## Running locally

```bash
# Install mutmut (dev dependency)
poetry add --group dev mutmut

# Run the full mutation suite — takes several minutes
poetry run mutmut run

# Or a quick smoke check on a single module:
poetry run mutmut run --paths-to-mutate i3/adaptation/

# HTML report
poetry run mutmut html
open html/index.html
```

## Interpreting results

- **killed** — at least one test failed on the mutant.  Good.
- **survived** — all tests still passed on the mutant.  Bad: this is an
  under-tested branch.  Add a test that distinguishes the mutant.
- **timeout** — mutant caused an infinite loop or > 10× slowdown.
  Usually indicates a non-termination bug even in the original test
  suite.
- **suspicious** — test failed for reasons unrelated to the mutation.
  Fix the flaky test.

## CI policy

- **PR gate** — surviving mutants in `i3/privacy/` fail the build.  We
  require the privacy layer to be 100% mutation-covered because the
  correctness of PII stripping is a hard dependency.
- **Nightly** — full mutation suite runs on `main`, report is archived.
  The survived-mutant count is tracked over time.

## Test command used

mutmut runs the following pytest invocation for each mutant:

```bash
poetry run pytest tests/ -x -q --no-cov -m 'not slow and not load and not fuzz'
```

We deliberately exclude `slow`, `load`, and `fuzz` markers because
mutation testing launches one process per mutant — any long-running
test would multiply the wall-clock cost prohibitively.

## Why not 100% coverage for everything?

Mutation testing against the TCN encoder is expensive (PyTorch compile
overhead per mutant).  We limit the mutation set to deterministic,
non-ML-heavy modules where a surviving mutant is actionable feedback
rather than compute noise.
