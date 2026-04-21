# I3 Fuzz Targets

This directory contains [atheris](https://github.com/google/atheris) fuzz
harnesses for security-critical entry points:

| Target                     | Function under fuzz                      | Invariant                                    |
|----------------------------|------------------------------------------|----------------------------------------------|
| `fuzz_sanitizer.py`        | `PrivacySanitizer.sanitize(text)`        | Must never raise on arbitrary bytes.         |
| `fuzz_config_loader.py`    | `i3.config.load_config(yaml_bytes)`      | Must never crash on malformed YAML.          |
| `fuzz_tokenizer.py`        | `SimpleTokenizer.encode(text)`           | Must never raise or loop forever.            |

## Installing atheris

atheris is **optional** and not required for day-to-day development.
It pulls libFuzzer from LLVM and therefore takes several minutes to build:

```bash
# Linux / macOS (requires clang 14+)
pip install atheris
```

On Windows native builds are not supported — run the harnesses inside
WSL2 or the project's dev Docker image.

## Running locally

Each harness is a standalone script:

```bash
# 100k iterations against the sanitiser (seconds to minutes on CPU)
python tests/fuzz/fuzz_sanitizer.py -atheris_runs=100000

# Mutate the YAML loader with a 10 MB corpus cap
python tests/fuzz/fuzz_config_loader.py -atheris_runs=50000 -max_len=10000

# Tokenizer: be generous with length so the whitespace regex is exercised
python tests/fuzz/fuzz_tokenizer.py -atheris_runs=100000 -max_len=65536
```

All three harnesses exit **non-zero** only if the target raises an
*unexpected* exception — i.e. anything except the small, explicit
allow-list of expected exceptions (see `_EXPECTED_EXCEPTIONS` in each
file).  An `AssertionError` always fails the fuzz run.

## Running in CI

The CI pipeline runs a short burst of each harness on every PR and a
longer overnight run on `main`:

```yaml
# .github/workflows/fuzz.yml (excerpt)
- run: pip install atheris
- run: python tests/fuzz/fuzz_sanitizer.py -atheris_runs=5000
- run: python tests/fuzz/fuzz_config_loader.py -atheris_runs=5000
- run: python tests/fuzz/fuzz_tokenizer.py -atheris_runs=5000
```

Nightly runs lift the `runs` cap to 1_000_000 and archive the produced
corpus (`crash-*`, `timeout-*`) as CI artifacts so regressions can be
re-replayed locally.

## Investigating a crash

If atheris prints `Uncaught Python exception:` it will write the offending
input to `crash-<hash>`.  Re-run with:

```bash
python tests/fuzz/fuzz_sanitizer.py crash-deadbeef01234567
```

This replays that single input without regenerating new ones.

## Why not `pytest`?

Fuzzing needs **millions** of iterations over untrusted bytes.  pytest's
overhead per iteration (fixtures, teardown, per-test reporting) makes
that impractical.  atheris runs each target in-process with libFuzzer's
coverage-guided mutation, producing several orders of magnitude more
throughput.
