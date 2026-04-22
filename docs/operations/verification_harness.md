# Verification Harness

The I3 **verification harness** is a single-command, ~40-check sweep
across the entire repository. It is the same tool the project uses for
its pre-submit gate (`.github/workflows/verify.yml`) and the one the
G10 *iterative-verification* batch will call repeatedly to confirm that
every fix genuinely closes its failure.

This page explains:

1. What the harness checks.
2. How to run it locally.
3. What `PASS`, `FAIL`, and `SKIP` actually mean.
4. How to extend the harness with a new check.
5. How the G10 iterative loop consumes it.

---

## 1. What the harness checks

Checks are grouped into seven **categories**. Each category is a
separate module under `scripts/verification/checks_*.py`; every
function in those modules registers a single check via the
`@register_check` decorator.

| Category | Example checks |
|---|---|
| `code_integrity` | AST parse, top-level imports, no bare except, soft-import pattern, ruff/mypy, `from __future__ import annotations` |
| `config_data` | YAML/JSON/TOML parse, notebooks valid, env-key documentation, `claude-sonnet-4-5` model-id lock, `mkdocs --strict` build |
| `architecture_runtime` | `create_app()` returns `FastAPI`, route registration, `/api/health`/`/api/live`/`/api/ready`, TCN forward, `AdaptationVector` roundtrip, bandit validity |
| `providers` | `ProviderRegistry` has 11 entries, per-adapter import, `MultiProviderClient` fallback chain, `CostTracker` ledger, `prompt_translator` shapes |
| `infrastructure` | Dockerfile / docker-compose / helm / k8s / Cedar policy parseability |
| `interview_readiness` | Slide count == 15, Q&A pair count == 52, verbatim closing line, honesty slide present, ADR count, CHANGELOG `[Unreleased]` non-empty |
| `security` | PII sanitizer catches 8 categories, PDDL planner refuses `route_cloud` on sensitive topics, `ModelEncryptor` bit-identical roundtrip, no hardcoded secret prefixes |

Each check has a **severity** — one of `blocker`, `high`, `medium`,
`low`, or `info` — which feeds the `--fail-on` CLI filter. Blockers
include the Anthropic model-id lock, the FastAPI smoke test, the PII
sanitizer, and the PDDL safety invariant.

---

## 2. Running it locally

```bash
# Every check, strict mode (any FAIL = non-zero exit).
poetry run python scripts/verify_all.py --strict

# Only interview-readiness + runtime categories.
poetry run python scripts/verify_all.py --categories interview,runtime

# Tolerate medium/low failures; fail only on blocker/high.
poetry run python scripts/verify_all.py --fail-on blocker,high

# Explicit output paths (useful in CI).
poetry run python scripts/verify_all.py \
    --out reports/verification_ci.json \
    --out-md reports/verification_ci.md
```

The CLI writes two reports to `reports/`:

- **`verification_<timestamp>.json`** — machine-readable, same schema as
  the `VerificationReport` Pydantic model. Suitable for diffing between
  runs, feeding into dashboards, or scripted parsing by the G10 loop.
- **`verification_<timestamp>.md`** — human-readable Markdown with an
  executive summary, a per-category table, the full result table,
  inline failure evidence, and an environment block at the end.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | All checks passed, or failures were below the `--fail-on` / `--strict` threshold. |
| `1` | At least one failure crossed the threshold. |
| `2` | The harness itself crashed (Python traceback printed to stderr). |

---

## 3. PASS, FAIL, SKIP — the semantics

- **PASS** — the assertion held.  For smoke-level checks (e.g. the TCN
  forward pass), PASS means the shape matched; for structural checks
  (e.g. `claude-sonnet-4-5` lock) it means the exact expected value was
  read back.

- **FAIL** — the assertion failed *or* the check itself raised an
  exception. A broken check never crashes the harness: the exception is
  trapped in `framework._run_one()`, the traceback becomes the
  `evidence` field, and a `FAIL` is recorded. Timeouts are also
  surfaced as `FAIL` with `message="timeout after Ns"`.

- **SKIP** — the check could not run because an optional dependency is
  missing (e.g. `helm` binary, `cedarpy`, `mkdocs`, `PyYAML`) or a
  prerequisite file is absent (e.g. no `deploy/helm/i3`). **SKIP is
  never an error**; it is *information*. The JSON report lists every
  SKIP with its reason so an operator can decide whether the missing
  prerequisite is acceptable for the current environment (a developer
  workstation without `helm` is fine; a CI environment missing
  `PyYAML` is not).

The `pass_rate` reported by the harness is `passed / (passed + failed)`
— skips are deliberately excluded so a missing optional tool does not
degrade the score.

---

## 4. Adding a new check

1. Pick the right module under `scripts/verification/`. Put new checks
   in an existing `checks_*.py` if they fit one of the seven
   categories; otherwise open a new module and list it in
   `scripts/verify_all.py::_import_all_check_modules`.
2. Write a zero-arg function returning a `CheckResult`:

   ```python
   from scripts.verification.framework import CheckResult, register_check

   @register_check(
       id="runtime.my_new_check",
       name="Short human-readable name",
       category="architecture_runtime",
       severity="medium",
   )
   def check_my_new_thing() -> CheckResult:
       """Docstring describing what PASS means."""
       # ... do work ...
       return CheckResult(
           check_id="runtime.my_new_check",
           status="PASS",
           duration_ms=12,
           message="thing works",
           evidence=None,
       )
   ```

3. Stick to the *five* severities: `blocker`, `high`, `medium`, `low`,
   `info`. The CLI's default `--fail-on blocker,high` assumes the rest
   are tolerated in green-build mode.
4. Optional dependencies are *always* soft-imported: `try: import foo
   except ImportError: return CheckResult(status="SKIP", ...)`. A
   missing tool must never turn into a FAIL.
5. Write a test under `tests/test_verification_framework.py` or a
   sibling file that exercises the new check with the registry
   reset so tests do not contaminate each other.

### Stable IDs

The `id` of a check is a **contract**. It appears in the JSON report,
the Markdown report, the CI artifact history, and the G10 iterative
planner. Rename only in a dedicated refactor, and never silently —
update the downstream consumers in the same change.

---

## 5. How G10 will use the harness

The G10 iterative-verification batch is structured as:

```
repeat until report.failed == 0 or budget exhausted:
    report = run(verify_all.py --strict --out reports/g10.json)
    if report.failed == 0:
        break
    fix = pick_highest_severity_failure(report)
    apply_fix(fix)
```

Two properties of the harness make this loop safe:

1. **Determinism.** Each check is self-contained and uses only
   filesystem inputs + in-process imports. Running the harness twice
   in a row on the same tree produces the same `pass_rate`. This means
   G10 can tell whether an applied fix actually moved the needle.
2. **Isolation.** Because the harness traps every exception and
   enforces a per-check timeout, a single buggy or slow check never
   stops the loop from making progress on the others.

The G10 planner reads `reports/verification_<ts>.json`, groups
failures by `category` and `severity`, and chooses one failure per
iteration to fix — usually starting with `blocker` failures in the
`security` / `config_data` / `interview_readiness` categories because
those protect guarantees the brief treats as non-negotiable (the
`claude-sonnet-4-5` lock, the verbatim closing line, the PII sanitizer,
the PDDL safety invariant).

### Reading the report programmatically

The JSON report is stable against the Pydantic schema
`scripts.verification.framework.VerificationReport`:

```python
import json
from scripts.verification.framework import VerificationReport

data = json.loads(Path("reports/verification_ci.json").read_text())
report = VerificationReport.model_validate(data)
for result in report.results:
    if result.status == "FAIL":
        print(result.check_id, result.message)
```

---

## 6. Troubleshooting

- **"ModuleNotFoundError: scripts.verification"** — run the script from
  the repo root (the CLI prepends the repo root to `sys.path` but only
  when invoked as `python scripts/verify_all.py`; running it via
  `python -m` with a different cwd can still work but `import` resolution
  may need a `PYTHONPATH=.` prefix).
- **"tomllib not available"** — the TOML check silently SKIPs on
  Python 3.10. The CI image uses 3.11+.
- **`fastapi` TestClient errors in `runtime.*`** — the FastAPI lifespan
  tries to load `configs/default.yaml`; make sure you ran the harness
  from the repo root.
- **Too many SKIPs** — run `poetry install --with dev,docs,security,providers,mlops,observability`
  to pull in the optional extras the harness probes.

---

## 7. See also

- `scripts/verify_all.py` — the CLI.
- `scripts/verification/framework.py` — the Pydantic models + registry.
- `.github/workflows/verify.yml` — the CI job that pushes the report
  to `$GITHUB_STEP_SUMMARY` and uploads the artifact.
- `tests/test_verification_framework.py` — unit tests for the
  framework itself.
