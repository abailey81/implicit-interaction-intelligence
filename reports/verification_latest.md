# I3 Verification Report

- Run at: `2026-04-27T11:32:22.977425+00:00`
- Git SHA: `e2404b3124b6b21bc988542233b1e091e88c9464`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Duration: `77.27s`

## Executive Summary

- Total: **44** | PASS: **39** | FAIL: **4** | SKIP: **1** | Pass-rate (excluding skips): **90.7%**

### Per-category summary

| Category | Total | PASS | FAIL | SKIP |
|---|---|---|---|---|
| `architecture_runtime` | 7 | 7 | 0 | 0 |
| `code_integrity` | 10 | 8 | 2 | 0 |
| `config_data` | 7 | 6 | 1 | 0 |
| `infrastructure` | 5 | 4 | 0 | 1 |
| `interview_readiness` | 6 | 6 | 0 | 0 |
| `providers` | 5 | 5 | 0 | 0 |
| `security` | 4 | 3 | 1 | 0 |

## Full Results

| Status | Severity | Check | Duration | Message |
|---|---|---|---|---|
| PASS | `blocker` | `code.ast_parse_all_python` (All .py files parse with ast) | 2437 ms | parsed 566 files cleanly |
| PASS | `blocker` | `code.top_level_imports` (Top-level packages import cleanly) | 3625 ms | all 4 top-level packages imported |
| PASS | `high` | `code.no_bare_except` (No bare except in library code) | 1546 ms | no bare except handlers |
| FAIL | `medium` | `code.no_print_in_library` (No print() in i3/ library code) | 1453 ms | 129 print() call(s) in i3/ |
| PASS | `high` | `code.soft_import_pattern` (Optional SDK imports guarded by try/except ImportError) | 827 ms | all optional-SDK imports guarded |
| PASS | `medium` | `code.ruff_clean` (ruff check (library code)) | 157 ms | ruff non-zero (exit 1) but no critical rules |
| FAIL | `low` | `code.mypy_clean` (mypy (best-effort)) | 60015 ms | timeout after 60s |
| PASS | `info` | `code.from_future_annotations` (from __future__ import annotations in i3/ modules (informational)) | 235 ms | 208/221 modules missing; coverage 5.9% (informational -- never FAIL) |
| PASS | `info` | `code.pep604_union_syntax` (PEP-604 union syntax (X | Y over Optional/Union)) | 156 ms | 11 legacy Optional/Union occurrence(s) (warning only) |
| PASS | `high` | `code.no_todo_personnel_references` (No invalidated personnel references in interviewer-facing docs) | 14 ms | no invalidated personnel references |
| PASS | `high` | `config.yaml_parse_all` (All .yaml/.yml files parse) | 14891 ms | 159 plain-YAML file(s) parsed cleanly (23 templated files skipped) |
| PASS | `high` | `config.json_parse_all` (All .json files parse) | 10952 ms | 1700 JSON file(s) parsed cleanly |
| PASS | `high` | `config.toml_parse_all` (All .toml files parse) | 6453 ms | 4 TOML file(s) parsed cleanly |
| PASS | `medium` | `config.notebooks_valid_nbformat` (All .ipynb files are valid nbformat v4) | 7155 ms | 7 notebook(s) valid |
| PASS | `medium` | `config.env_example_keys_documented` (.env.example keys referenced in python source) | 203 ms | 28 env keys, all referenced (or SDK-chain) |
| FAIL | `blocker` | `config.no_hardcoded_secrets` (No hardcoded secret prefixes outside .env.example) | 40594 ms | 1 potential secret(s) found |
| PASS | `blocker` | `config.claude_model_id_locked` (configs/default.yaml cloud.model == claude-sonnet-4-5) | 0 ms | cloud.model == 'claude-sonnet-4-5' |
| FAIL | `medium` | `config.mkdocs_build_strict` (mkdocs build --strict) | 37484 ms | mkdocs --strict exit 1 |
| PASS | `blocker` | `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance) | 5500 ms | create_app() returned a FastAPI instance |
| PASS | `high` | `runtime.all_routes_registered` (OpenAPI schema registers every expected route) | 46 ms | 11 expected routes registered |
| PASS | `blocker` | `runtime.health_live_endpoints` (GET /api/health and /api/live return 200) | 12969 ms | /api/health and /api/live returned 200 |
| PASS | `high` | `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks) | 10484 ms | /api/ready shape ok (status + checks) |
| PASS | `blocker` | `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories) | 0 ms | sanitizer redacted all tested PII categories |
| PASS | `blocker` | `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics) | 0 ms | safe plan: ('redact_pii', 'route_local') |
| PASS | `high` | `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]) | 62 ms | TCN [1, 10, 32] -> [1, 64] |
| PASS | `high` | `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical) | 0 ms | AdaptationVector roundtrip is bit-identical |
| PASS | `blocker` | `runtime.encryption_roundtrip` (ModelEncryptor encrypt/decrypt ndarray is bit-identical) | 16 ms | roundtrip ok (64 bytes) |
| PASS | `high` | `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index) | 14 ms | bandit selected arm 0 |
| PASS | `blocker` | `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers) | 0 ms | 11 provider(s) registered (>= 11) |
| PASS | `high` | `providers.construct_without_sdk` (Every provider class instantiates without its SDK) | 110 ms | all 11 adapter modules import cleanly |
| PASS | `blocker` | `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure) | 0 ms | fallback chain escalated to second provider as expected |
| PASS | `medium` | `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals) | 0 ms | report.total_calls = 1 |
| PASS | `medium` | `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes) | 0 ms | 4 provider shapes produced cleanly |
| PASS | `high` | `infra.dockerfile_parses` (Every Dockerfile* has FROM and CMD/ENTRYPOINT) | 0 ms | 4 Dockerfile(s) valid |
| PASS | `medium` | `infra.compose_schema` (docker-compose*.yml has services section) | 16 ms | 2 compose file(s) valid |
| SKIP | `low` | `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)) | 16 ms | helm binary not on PATH |
| PASS | `medium` | `infra.kubernetes_manifests` (deploy/k8s/*.yaml have apiVersion/kind/metadata) | 30 ms | 10 k8s manifest(s) valid |
| PASS | `low` | `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy) | 62 ms | 1 cedar policy/ies parsed |
| PASS | `high` | `interview.slide_count` (docs/slides/presentation.md has exactly 15 slides (frontmatter-aware)) | 0 ms | 16 '---' separator(s); expected 16 (Marp frontmatter=True) |
| PASS | `high` | `interview.qa_pair_count` (docs/slides/qa_prep.md has exactly 52 Q&A pairs) | 0 ms | 52 '### ' Q&A heading(s), expected 52 |
| PASS | `blocker` | `interview.closing_line_verbatim` (closing_lines.md contains the verbatim closing line) | 0 ms | verbatim closing line present |
| PASS | `high` | `interview.honesty_slide_title_case` (presentation.md contains the 'What This Prototype Is Not' slide) | 0 ms | honesty slide title present |
| PASS | `medium` | `interview.adr_count` (docs/adr has >= 10 numbered ADRs) | 0 ms | 10 numbered ADR(s) (>=10 required) |
| PASS | `low` | `interview.changelog_latest_release_nonempty` (CHANGELOG.md latest release section > 500 chars) | 0 ms | latest release section is 14271 chars (>500 required) |

## Failures

### `code.no_print_in_library` - No print() in i3/ library code (`medium`)

- Message: 129 print() call(s) in i3/

```
i3\biometric\keystroke_auth.py:937
i3\biometric\keystroke_auth.py:954
i3\biometric\keystroke_auth.py:970
i3\biometric\keystroke_auth.py:988
i3\biometric\keystroke_auth.py:989
i3\biometric\keystroke_auth.py:991
i3\biometric\keystroke_auth.py:949
i3\biometric\keystroke_auth.py:965
i3\biometric\keystroke_auth.py:982
i3\critique\critic.py:705
i3\critique\critic.py:710
i3\critique\critic.py:707
i3\critique\critic.py:709
i3\dialogue\coref.py:1550
i3\dialogue\coref.py:1551
i3\dialogue\coref.py:1552
i3\dialogue\coref.py:1563
i3\dialogue\coref.py:1575
i3\dialogue\coref.py:1576
i3\dialogue\coref.py:1577
```

### `code.mypy_clean` - mypy (best-effort) (`low`)

- Message: timeout after 60s

### `config.no_hardcoded_secrets` - No hardcoded secret prefixes outside .env.example (`blocker`)

- Message: 1 potential secret(s) found

```
training\train_intent_gemini.py: AIzaSy*
```

### `config.mkdocs_build_strict` - mkdocs build --strict (`medium`)

- Message: mkdocs --strict exit 1

```
nd
no library called "cairo" was found
no library called "libcairo-2" was found
cannot load library 'libcairo.so.2': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo.so.2'
cannot load library 'libcairo.2.dylib': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo.2.dylib'
cannot load library 'libcairo-2.dll': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo-2.dll'

--> Check out the troubleshooting guide: https://t.ly/MfX6u
WARNING -  "cairosvg" Python module is installed, but it crashed with:
no library called "cairo-2" was found
no library called "cairo" was found
no library called "libcairo-2" was found
cannot load library 'libcairo.so.2': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo.so.2'
cannot load library 'libcairo.2.dylib': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo.2.dylib'
cannot load library 'libcairo-2.dll': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo-2.dll'

--> Check out the troubleshooting guide: https://t.ly/MfX6u
WARNING -  "cairosvg" Python module is installed, but it crashed with:
no library called "cairo-2" was found
no library called "cairo" was found
no library called "libcairo-2" was found
cannot load library 'libcairo.so.2': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo.so.2'
cannot load library 'libcairo.2.dylib': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo.2.dylib'
cannot load library 'libcairo-2.dll': error 0x7e.  Additionally, ctypes.util.find_library() did not manage to locate a library called 'libcairo-2.dll'

--> Check out the troubleshooting guide: https://t.ly/MfX6u

```

## Skipped

- `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)): helm binary not on PATH

## Environment

- Git SHA: `e2404b3124b6b21bc988542233b1e091e88c9464`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Run at: `2026-04-27T11:32:22.977425+00:00`

