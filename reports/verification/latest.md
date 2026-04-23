# I3 Verification Report

- Run at: `2026-04-23T15:22:36.342013+00:00`
- Git SHA: `96b18de2d33a0ffd2d5f1772208dcff2b104bf5e`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Duration: `7.45s`

## Executive Summary

- Total: **44** | PASS: **28** | FAIL: **0** | SKIP: **16** | Pass-rate (excluding skips): **100.0%**

### Per-category summary

| Category | Total | PASS | FAIL | SKIP |
|---|---|---|---|---|
| `architecture_runtime` | 7 | 0 | 0 | 7 |
| `code_integrity` | 10 | 7 | 0 | 3 |
| `config_data` | 7 | 6 | 0 | 1 |
| `infrastructure` | 5 | 3 | 0 | 2 |
| `interview_readiness` | 6 | 6 | 0 | 0 |
| `providers` | 5 | 4 | 0 | 1 |
| `security` | 4 | 2 | 0 | 2 |

## Full Results

| Status | Severity | Check | Duration | Message |
|---|---|---|---|---|
| PASS | `blocker` | `code.ast_parse_all_python` (All .py files parse with ast) | 1952 ms | parsed 420 files cleanly |
| SKIP | `blocker` | `code.top_level_imports` (Top-level packages import cleanly) | 1750 ms | 1 package(s) blocked by missing runtime deps |
| PASS | `high` | `code.no_bare_except` (No bare except in library code) | 1031 ms | no bare except handlers |
| PASS | `medium` | `code.no_print_in_library` (No print() in i3/ library code) | 967 ms | no print() calls in i3/ |
| PASS | `high` | `code.soft_import_pattern` (Optional SDK imports guarded by try/except ImportError) | 1157 ms | all optional-SDK imports guarded |
| SKIP | `medium` | `code.ruff_clean` (ruff check (library code)) | 93 ms | ruff binary not on PATH |
| SKIP | `low` | `code.mypy_clean` (mypy (best-effort)) | 125 ms | mypy binary not on PATH |
| PASS | `info` | `code.from_future_annotations` (from __future__ import annotations in i3/ modules (informational)) | 967 ms | 177/189 modules missing; coverage 6.3% (informational -- never FAIL) |
| PASS | `info` | `code.pep604_union_syntax` (PEP-604 union syntax (X | Y over Optional/Union)) | 452 ms | 113 legacy Optional/Union occurrence(s) (warning only) |
| PASS | `high` | `code.no_todo_personnel_references` (No invalidated personnel references in interviewer-facing docs) | 47 ms | no invalidated personnel references |
| PASS | `high` | `config.yaml_parse_all` (All .yaml/.yml files parse) | 1608 ms | 88 plain-YAML file(s) parsed cleanly (23 templated files skipped) |
| PASS | `high` | `config.json_parse_all` (All .json files parse) | 250 ms | 21 JSON file(s) parsed cleanly |
| PASS | `high` | `config.toml_parse_all` (All .toml files parse) | 219 ms | 4 TOML file(s) parsed cleanly |
| PASS | `medium` | `config.notebooks_valid_nbformat` (All .ipynb files are valid nbformat v4) | 204 ms | 7 notebook(s) valid |
| PASS | `medium` | `config.env_example_keys_documented` (.env.example keys referenced in python source) | 797 ms | 28 env keys, all referenced (or SDK-chain) |
| PASS | `blocker` | `config.no_hardcoded_secrets` (No hardcoded secret prefixes outside .env.example) | 1717 ms | no hardcoded secret prefixes found |
| PASS | `blocker` | `config.claude_model_id_locked` (configs/default.yaml cloud.model == claude-sonnet-4-5) | 15 ms | cloud.model == 'claude-sonnet-4-5' |
| SKIP | `medium` | `config.mkdocs_build_strict` (mkdocs build --strict) | 15 ms | mkdocs binary not on PATH |
| SKIP | `blocker` | `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance) | 719 ms | fastapi or server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.all_routes_registered` (OpenAPI schema registers every expected route) | 437 ms | server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `blocker` | `runtime.health_live_endpoints` (GET /api/health and /api/live return 200) | 3780 ms | TestClient unavailable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks) | 3765 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| SKIP | `blocker` | `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories) | 452 ms | PrivacySanitizer import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `blocker` | `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics) | 484 ms | PDDL planner import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]) | 484 ms | torch or TCN not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical) | 906 ms | AdaptationVector not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| PASS | `blocker` | `runtime.encryption_roundtrip` (ModelEncryptor encrypt/decrypt ndarray is bit-identical) | 1047 ms | roundtrip ok (64 bytes) |
| SKIP | `high` | `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index) | 625 ms | bandit not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `blocker` | `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers) | 452 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| PASS | `high` | `providers.construct_without_sdk` (Every provider class instantiates without its SDK) | 875 ms | all 11 adapter modules import cleanly |
| PASS | `blocker` | `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure) | 422 ms | fallback chain escalated to second provider as expected |
| PASS | `medium` | `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals) | 16 ms | report.total_calls = 1 |
| PASS | `medium` | `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes) | 0 ms | 4 provider shapes produced cleanly |
| PASS | `high` | `infra.dockerfile_parses` (Every Dockerfile* has FROM and CMD/ENTRYPOINT) | 32 ms | 4 Dockerfile(s) valid |
| PASS | `medium` | `infra.compose_schema` (docker-compose*.yml has services section) | 16 ms | 2 compose file(s) valid |
| SKIP | `low` | `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)) | 47 ms | helm binary not on PATH |
| PASS | `medium` | `infra.kubernetes_manifests` (deploy/k8s/*.yaml have apiVersion/kind/metadata) | 32 ms | 10 k8s manifest(s) valid |
| SKIP | `low` | `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy) | 15 ms | cedarpy not available |
| PASS | `high` | `interview.slide_count` (docs/slides/presentation.md has exactly 15 slides (frontmatter-aware)) | 0 ms | 16 '---' separator(s); expected 16 (Marp frontmatter=True) |
| PASS | `high` | `interview.qa_pair_count` (docs/slides/qa_prep.md has exactly 52 Q&A pairs) | 0 ms | 52 '### ' Q&A heading(s), expected 52 |
| PASS | `blocker` | `interview.closing_line_verbatim` (closing_lines.md contains the verbatim closing line) | 0 ms | verbatim closing line present |
| PASS | `high` | `interview.honesty_slide_title_case` (presentation.md contains the 'What This Prototype Is Not' slide) | 0 ms | honesty slide title present |
| PASS | `medium` | `interview.adr_count` (docs/adr has >= 10 numbered ADRs) | 0 ms | 10 numbered ADR(s) (>=10 required) |
| PASS | `low` | `interview.changelog_latest_release_nonempty` (CHANGELOG.md latest release section > 500 chars) | 0 ms | latest release section is 14271 chars (>500 required) |

## Skipped

- `code.top_level_imports` (Top-level packages import cleanly): 1 package(s) blocked by missing runtime deps
- `code.ruff_clean` (ruff check (library code)): ruff binary not on PATH
- `code.mypy_clean` (mypy (best-effort)): mypy binary not on PATH
- `config.mkdocs_build_strict` (mkdocs build --strict): mkdocs binary not on PATH
- `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance): fastapi or server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.all_routes_registered` (OpenAPI schema registers every expected route): server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.health_live_endpoints` (GET /api/health and /api/live return 200): TestClient unavailable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories): PrivacySanitizer import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics): PDDL planner import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]): torch or TCN not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical): AdaptationVector not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index): bandit not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)): helm binary not on PATH
- `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy): cedarpy not available

## Environment

- Git SHA: `96b18de2d33a0ffd2d5f1772208dcff2b104bf5e`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Run at: `2026-04-23T15:22:36.342013+00:00`

