# I3 Verification Report

- Run at: `2026-04-23T00:02:11.657567+00:00`
- Git SHA: `56a447f488bfccb3627c63812348dffbed757d24`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Duration: `2.66s`

## Executive Summary

- Total: **46** | PASS: **22** | FAIL: **3** | SKIP: **21** | Pass-rate (excluding skips): **88.0%**

### Per-category summary

| Category | Total | PASS | FAIL | SKIP |
|---|---|---|---|---|
| `architecture_runtime` | 7 | 0 | 0 | 7 |
| `code_integrity` | 10 | 6 | 1 | 3 |
| `config_data` | 7 | 5 | 1 | 1 |
| `infrastructure` | 5 | 3 | 0 | 2 |
| `interview_readiness` | 8 | 8 | 0 | 0 |
| `providers` | 5 | 0 | 0 | 5 |
| `security` | 4 | 0 | 1 | 3 |

## Full Results

| Status | Severity | Check | Duration | Message |
|---|---|---|---|---|
| PASS | `blocker` | `code.ast_parse_all_python` (All .py files parse with ast) | 1812 ms | parsed 399 files cleanly |
| SKIP | `blocker` | `code.top_level_imports` (Top-level packages import cleanly) | 16 ms | 1 package(s) blocked by missing runtime deps |
| PASS | `high` | `code.no_bare_except` (No bare except in library code) | 1484 ms | no bare except handlers |
| PASS | `medium` | `code.no_print_in_library` (No print() in i3/ library code) | 1343 ms | no print() calls in i3/ |
| PASS | `high` | `code.soft_import_pattern` (Optional SDK imports guarded by try/except ImportError) | 1530 ms | all optional-SDK imports guarded |
| SKIP | `medium` | `code.ruff_clean` (ruff check (library code)) | 202 ms | ruff binary not on PATH |
| SKIP | `low` | `code.mypy_clean` (mypy (best-effort)) | 108 ms | mypy binary not on PATH |
| FAIL | `low` | `code.from_future_annotations` (from __future__ import annotations in i3/ modules) | 328 ms | 170/182 modules missing; coverage 6.6% (threshold 30%) |
| PASS | `info` | `code.pep604_union_syntax` (PEP-604 union syntax (X | Y over Optional/Union)) | 328 ms | 111 legacy Optional/Union occurrence(s) (warning only) |
| PASS | `high` | `code.no_todo_personnel_references` (No invalidated personnel references in interviewer-facing docs) | 16 ms | no invalidated personnel references |
| PASS | `high` | `config.yaml_parse_all` (All .yaml/.yml files parse) | 890 ms | 88 plain-YAML file(s) parsed cleanly (23 templated files skipped) |
| PASS | `high` | `config.json_parse_all` (All .json files parse) | 188 ms | 16 JSON file(s) parsed cleanly |
| PASS | `high` | `config.toml_parse_all` (All .toml files parse) | 171 ms | 4 TOML file(s) parsed cleanly |
| PASS | `medium` | `config.notebooks_valid_nbformat` (All .ipynb files are valid nbformat v4) | 187 ms | 7 notebook(s) valid |
| FAIL | `medium` | `config.env_example_keys_documented` (.env.example keys referenced in python source) | 530 ms | 4/28 env keys not referenced anywhere |
| FAIL | `blocker` | `config.no_hardcoded_secrets` (No hardcoded secret prefixes outside .env.example) | 594 ms | 2 potential secret(s) found |
| PASS | `blocker` | `config.claude_model_id_locked` (configs/default.yaml cloud.model == claude-sonnet-4-5) | 0 ms | cloud.model == 'claude-sonnet-4-5' |
| SKIP | `medium` | `config.mkdocs_build_strict` (mkdocs build --strict) | 15 ms | mkdocs binary not on PATH |
| SKIP | `blocker` | `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance) | 0 ms | fastapi or server.app not importable: No module named 'fastapi' |
| SKIP | `high` | `runtime.all_routes_registered` (OpenAPI schema registers every expected route) | 0 ms | server.app not importable: No module named 'fastapi' |
| SKIP | `blocker` | `runtime.health_live_endpoints` (GET /api/health and /api/live return 200) | 0 ms | TestClient unavailable: No module named 'fastapi' |
| SKIP | `high` | `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks) | 0 ms | TestClient unavailable: No module named 'fastapi' |
| SKIP | `blocker` | `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories) | 16 ms | PrivacySanitizer import failed: No module named 'numpy' |
| SKIP | `blocker` | `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics) | 0 ms | PDDL planner import failed: No module named 'numpy' |
| SKIP | `high` | `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]) | 0 ms | torch or TCN not importable: No module named 'torch' |
| SKIP | `high` | `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical) | 15 ms | AdaptationVector not importable: No module named 'numpy' |
| SKIP | `blocker` | `runtime.encryption_roundtrip` (ModelEncryptor encrypt/decrypt ndarray is bit-identical) | 0 ms | ModelEncryptor not importable: No module named 'numpy' |
| SKIP | `high` | `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index) | 0 ms | bandit not importable: No module named 'numpy' |
| SKIP | `blocker` | `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers) | 0 ms | runtime dep not installed: numpy |
| SKIP | `high` | `providers.construct_without_sdk` (Every provider class instantiates without its SDK) | 16 ms | runtime dep not installed: numpy |
| SKIP | `blocker` | `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure) | 0 ms | runtime dep not installed: numpy |
| SKIP | `medium` | `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals) | 0 ms | runtime dep not installed: numpy |
| SKIP | `medium` | `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes) | 14 ms | runtime dep not installed: numpy |
| PASS | `high` | `infra.dockerfile_parses` (Every Dockerfile* has FROM and CMD/ENTRYPOINT) | 0 ms | 4 Dockerfile(s) valid |
| PASS | `medium` | `infra.compose_schema` (docker-compose*.yml has services section) | 16 ms | 2 compose file(s) valid |
| SKIP | `low` | `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)) | 16 ms | helm binary not on PATH |
| PASS | `medium` | `infra.kubernetes_manifests` (deploy/k8s/*.yaml have apiVersion/kind/metadata) | 31 ms | 10 k8s manifest(s) valid |
| SKIP | `low` | `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy) | 0 ms | cedarpy not available |
| PASS | `high` | `interview.slide_count` (docs/slides/presentation.md has exactly 15 slides (frontmatter-aware)) | 0 ms | 16 '---' separator(s); expected 16 (Marp frontmatter=True) |
| PASS | `high` | `interview.qa_pair_count` (docs/slides/qa_prep.md has exactly 52 Q&A pairs) | 0 ms | 52 '### ' Q&A heading(s), expected 52 |
| PASS | `blocker` | `interview.closing_line_verbatim` (closing_lines.md contains the verbatim closing line) | 0 ms | verbatim closing line present |
| PASS | `high` | `interview.honesty_slide_title_case` (presentation.md contains the 'What This Prototype Is Not' slide) | 0 ms | honesty slide title present |
| PASS | `medium` | `interview.adr_count` (docs/adr has >= 10 numbered ADRs) | 0 ms | 10 numbered ADR(s) (>=10 required) |
| PASS | `high` | `interview.brief_analysis_header` (the brief analysis first 30 lines contain 'Corrections notice') | 0 ms | Corrections notice present in first 30 lines |
| PASS | `low` | `interview.changelog_unreleased_nonempty` (CHANGELOG.md [Unreleased] section > 500 chars) | 16 ms | [Unreleased] body is 19671 chars (>500 required) |
| PASS | `low` | `interview.notes_md_sections` (engineering notes has >= 10 '##' section headers) | 0 ms | 10 '## ' section header(s) (>=10 required) |

## Failures

### `code.from_future_annotations` - from __future__ import annotations in i3/ modules (`low`)

- Message: 170/182 modules missing; coverage 6.6% (threshold 30%)

```
i3\config.py
i3\adaptation\ablation.py
i3\adaptation\controller.py
i3\adaptation\dimensions.py
i3\adaptation\types.py
i3\adaptation\uncertainty.py
i3\analytics\arrow_interop.py
i3\analytics\duckdb_engine.py
i3\analytics\ibis_queries.py
i3\analytics\lance_vector.py
i3\analytics\polars_features.py
i3\authz\cedar_adapter.py
i3\biometric\continuous_auth.py
i3\biometric\keystroke_id.py
i3\cloud\client.py
i3\cloud\cost_tracker.py
i3\cloud\dspy_adapter.py
i3\cloud\guarded_client.py
i3\cloud\guardrails.py
i3\cloud\guardrails_nemo.py
i3\cloud\instructor_adapter.py
i3\cloud\multi_provider.py
i3\cloud\outlines_constrained.py
i3\cloud\postprocess.py
i3\cloud\prompt_builder.py
i3\cloud\prompt_translator.py
i3\cloud\provider_registry.py
i3\cloud\pydantic_ai_adapter.py
i3\continual\drift_detector.py
i3\continual\ewc.py
i3\continual\ewc_user_model.py
i3\continual\replay_buffer.py
i3\crossdevice\ai_glasses_arm.py
i3\crossdevice\hmos_ddm_sync.py
i3\diary\logger.py
i3\diary\store.py
i3\diary\summarizer.py
i3\edge\coreml_export.py
i3\edge\executorch_export.py
i3\edge\iree_export.py
```

### `config.env_example_keys_documented` - .env.example keys referenced in python source (`medium`)

- Message: 4/28 env keys not referenced anywhere

```
I3_ENCODER_CHECKPOINT
I3_OFFLINE_ONLY
I3_SLM_CHECKPOINT
I3_TELEMETRY_DEBUG
```

### `config.no_hardcoded_secrets` - No hardcoded secret prefixes outside .env.example (`blocker`)

- Message: 2 potential secret(s) found

```
reports\verification_pass1.json: ghp_*
reports\verification_pass1.md: ghp_*
```

## Skipped

- `code.top_level_imports` (Top-level packages import cleanly): 1 package(s) blocked by missing runtime deps
- `code.ruff_clean` (ruff check (library code)): ruff binary not on PATH
- `code.mypy_clean` (mypy (best-effort)): mypy binary not on PATH
- `config.mkdocs_build_strict` (mkdocs build --strict): mkdocs binary not on PATH
- `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance): fastapi or server.app not importable: No module named 'fastapi'
- `runtime.all_routes_registered` (OpenAPI schema registers every expected route): server.app not importable: No module named 'fastapi'
- `runtime.health_live_endpoints` (GET /api/health and /api/live return 200): TestClient unavailable: No module named 'fastapi'
- `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks): TestClient unavailable: No module named 'fastapi'
- `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories): PrivacySanitizer import failed: No module named 'numpy'
- `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics): PDDL planner import failed: No module named 'numpy'
- `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]): torch or TCN not importable: No module named 'torch'
- `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical): AdaptationVector not importable: No module named 'numpy'
- `runtime.encryption_roundtrip` (ModelEncryptor encrypt/decrypt ndarray is bit-identical): ModelEncryptor not importable: No module named 'numpy'
- `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index): bandit not importable: No module named 'numpy'
- `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers): runtime dep not installed: numpy
- `providers.construct_without_sdk` (Every provider class instantiates without its SDK): runtime dep not installed: numpy
- `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure): runtime dep not installed: numpy
- `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals): runtime dep not installed: numpy
- `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes): runtime dep not installed: numpy
- `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)): helm binary not on PATH
- `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy): cedarpy not available

## Environment

- Git SHA: `56a447f488bfccb3627c63812348dffbed757d24`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Run at: `2026-04-23T00:02:11.657567+00:00`

