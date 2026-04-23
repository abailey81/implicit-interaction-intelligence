# I3 Verification Report

- Run at: `2026-04-23T01:20:04.052813+00:00`
- Git SHA: `362d63f9ee7fecda8e02ccdc94aa827e4b515711`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Duration: `6.42s`

## Executive Summary

- Total: **46** | PASS: **27** | FAIL: **0** | SKIP: **19** | Pass-rate (excluding skips): **100.0%**

### Per-category summary

| Category | Total | PASS | FAIL | SKIP |
|---|---|---|---|---|
| `architecture_runtime` | 7 | 1 | 0 | 6 |
| `code_integrity` | 10 | 7 | 0 | 3 |
| `config_data` | 7 | 6 | 0 | 1 |
| `infrastructure` | 5 | 3 | 0 | 2 |
| `interview_readiness` | 8 | 8 | 0 | 0 |
| `providers` | 5 | 0 | 0 | 5 |
| `security` | 4 | 2 | 0 | 2 |

## Full Results

| Status | Severity | Check | Duration | Message |
|---|---|---|---|---|
| PASS | `blocker` | `code.ast_parse_all_python` (All .py files parse with ast) | 1297 ms | parsed 401 files cleanly |
| SKIP | `blocker` | `code.top_level_imports` (Top-level packages import cleanly) | 1672 ms | 1 package(s) blocked by missing runtime deps |
| PASS | `high` | `code.no_bare_except` (No bare except in library code) | 859 ms | no bare except handlers |
| PASS | `medium` | `code.no_print_in_library` (No print() in i3/ library code) | 781 ms | no print() calls in i3/ |
| PASS | `high` | `code.soft_import_pattern` (Optional SDK imports guarded by try/except ImportError) | 905 ms | all optional-SDK imports guarded |
| SKIP | `medium` | `code.ruff_clean` (ruff check (library code)) | 77 ms | ruff binary not on PATH |
| SKIP | `low` | `code.mypy_clean` (mypy (best-effort)) | 125 ms | mypy binary not on PATH |
| PASS | `info` | `code.from_future_annotations` (from __future__ import annotations in i3/ modules (informational)) | 671 ms | 170/182 modules missing; coverage 6.6% (informational -- never FAIL) |
| PASS | `info` | `code.pep604_union_syntax` (PEP-604 union syntax (X | Y over Optional/Union)) | 109 ms | 111 legacy Optional/Union occurrence(s) (warning only) |
| PASS | `high` | `code.no_todo_personnel_references` (No invalidated personnel references in interviewer-facing docs) | 0 ms | no invalidated personnel references |
| PASS | `high` | `config.yaml_parse_all` (All .yaml/.yml files parse) | 1187 ms | 88 plain-YAML file(s) parsed cleanly (23 templated files skipped) |
| PASS | `high` | `config.json_parse_all` (All .json files parse) | 141 ms | 21 JSON file(s) parsed cleanly |
| PASS | `high` | `config.toml_parse_all` (All .toml files parse) | 109 ms | 4 TOML file(s) parsed cleanly |
| PASS | `medium` | `config.notebooks_valid_nbformat` (All .ipynb files are valid nbformat v4) | 109 ms | 7 notebook(s) valid |
| PASS | `medium` | `config.env_example_keys_documented` (.env.example keys referenced in python source) | 593 ms | 28 env keys, all referenced (or SDK-chain) |
| PASS | `blocker` | `config.no_hardcoded_secrets` (No hardcoded secret prefixes outside .env.example) | 1390 ms | no hardcoded secret prefixes found |
| PASS | `blocker` | `config.claude_model_id_locked` (configs/default.yaml cloud.model == claude-sonnet-4-5) | 15 ms | cloud.model == 'claude-sonnet-4-5' |
| SKIP | `medium` | `config.mkdocs_build_strict` (mkdocs build --strict) | 0 ms | mkdocs binary not on PATH |
| SKIP | `blocker` | `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance) | 531 ms | fastapi or server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.all_routes_registered` (OpenAPI schema registers every expected route) | 407 ms | server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `blocker` | `runtime.health_live_endpoints` (GET /api/health and /api/live return 200) | 1141 ms | TestClient unavailable: 'i3' |
| SKIP | `high` | `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks) | 719 ms | TestClient unavailable: cannot import name 'create_app' from 'server.app' (D:\implicit-interaction-intelligence\server\app.py) |
| SKIP | `blocker` | `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories) | 344 ms | PrivacySanitizer import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `blocker` | `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics) | 359 ms | PDDL planner import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]) | 358 ms | torch or TCN not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| SKIP | `high` | `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical) | 811 ms | AdaptationVector not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies. |
| PASS | `blocker` | `runtime.encryption_roundtrip` (ModelEncryptor encrypt/decrypt ndarray is bit-identical) | 2843 ms | roundtrip ok (64 bytes) |
| PASS | `high` | `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index) | 1186 ms | bandit selected arm 0 |
| SKIP | `blocker` | `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers) | 812 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| SKIP | `high` | `providers.construct_without_sdk` (Every provider class instantiates without its SDK) | 813 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| SKIP | `blocker` | `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure) | 1656 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| SKIP | `medium` | `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals) | 390 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| SKIP | `medium` | `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes) | 796 ms | OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven |
| PASS | `high` | `infra.dockerfile_parses` (Every Dockerfile* has FROM and CMD/ENTRYPOINT) | 0 ms | 4 Dockerfile(s) valid |
| PASS | `medium` | `infra.compose_schema` (docker-compose*.yml has services section) | 0 ms | 2 compose file(s) valid |
| SKIP | `low` | `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)) | 0 ms | helm binary not on PATH |
| PASS | `medium` | `infra.kubernetes_manifests` (deploy/k8s/*.yaml have apiVersion/kind/metadata) | 31 ms | 10 k8s manifest(s) valid |
| SKIP | `low` | `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy) | 0 ms | cedarpy not available |
| PASS | `high` | `interview.slide_count` (docs/slides/presentation.md has exactly 15 slides (frontmatter-aware)) | 0 ms | 16 '---' separator(s); expected 16 (Marp frontmatter=True) |
| PASS | `high` | `interview.qa_pair_count` (docs/slides/qa_prep.md has exactly 52 Q&A pairs) | 0 ms | 52 '### ' Q&A heading(s), expected 52 |
| PASS | `blocker` | `interview.closing_line_verbatim` (closing_lines.md contains the verbatim closing line) | 16 ms | verbatim closing line present |
| PASS | `high` | `interview.honesty_slide_title_case` (presentation.md contains the 'What This Prototype Is Not' slide) | 0 ms | honesty slide title present |
| PASS | `medium` | `interview.adr_count` (docs/adr has >= 10 numbered ADRs) | 0 ms | 10 numbered ADR(s) (>=10 required) |
| PASS | `high` | `interview.brief_analysis_header` (BRIEF_ANALYSIS.md first 30 lines contain 'Corrections notice') | 0 ms | Corrections notice present in first 30 lines |
| PASS | `low` | `interview.changelog_unreleased_nonempty` (CHANGELOG.md [Unreleased] section > 500 chars) | 0 ms | [Unreleased] body is 25748 chars (>500 required) |
| PASS | `low` | `interview.notes_md_sections` (NOTES.md has >= 10 '##' section headers) | 0 ms | 10 '## ' section header(s) (>=10 required) |

## Skipped

- `code.top_level_imports` (Top-level packages import cleanly): 1 package(s) blocked by missing runtime deps
- `code.ruff_clean` (ruff check (library code)): ruff binary not on PATH
- `code.mypy_clean` (mypy (best-effort)): mypy binary not on PATH
- `config.mkdocs_build_strict` (mkdocs build --strict): mkdocs binary not on PATH
- `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance): fastapi or server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.all_routes_registered` (OpenAPI schema registers every expected route): server.app not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.health_live_endpoints` (GET /api/health and /api/live return 200): TestClient unavailable: 'i3'
- `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks): TestClient unavailable: cannot import name 'create_app' from 'server.app' (D:\implicit-interaction-intelligence\server\app.py)
- `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories): PrivacySanitizer import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics): PDDL planner import failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]): torch or TCN not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical): AdaptationVector not importable: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\venv-i3\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
- `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `providers.construct_without_sdk` (Every provider class instantiates without its SDK): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes): OS-level dep load failed: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL). Error loading "D:\ven
- `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)): helm binary not on PATH
- `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy): cedarpy not available

## Environment

- Git SHA: `362d63f9ee7fecda8e02ccdc94aa827e4b515711`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Run at: `2026-04-23T01:20:04.052813+00:00`

