# I3 Verification Report

- Run at: `2026-04-22T23:54:39.990517+00:00`
- Git SHA: `56a447f488bfccb3627c63812348dffbed757d24`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Duration: `1.86s`

## Executive Summary

- Total: **46** | PASS: **17** | FAIL: **14** | SKIP: **15** | Pass-rate (excluding skips): **54.8%**

### Per-category summary

| Category | Total | PASS | FAIL | SKIP |
|---|---|---|---|---|
| `architecture_runtime` | 7 | 0 | 0 | 7 |
| `code_integrity` | 10 | 4 | 4 | 2 |
| `config_data` | 7 | 4 | 2 | 1 |
| `infrastructure` | 5 | 3 | 0 | 2 |
| `interview_readiness` | 8 | 6 | 2 | 0 |
| `providers` | 5 | 0 | 5 | 0 |
| `security` | 4 | 0 | 1 | 3 |

## Full Results

| Status | Severity | Check | Duration | Message |
|---|---|---|---|---|
| PASS | `blocker` | `code.ast_parse_all_python` (All .py files parse with ast) | 1313 ms | parsed 399 files cleanly |
| FAIL | `blocker` | `code.top_level_imports` (Top-level packages import cleanly) | 16 ms | 1 import failures |
| PASS | `high` | `code.no_bare_except` (No bare except in library code) | 1063 ms | no bare except handlers |
| FAIL | `medium` | `code.no_print_in_library` (No print() in i3/ library code) | 1000 ms | 2 print() call(s) in i3/ |
| FAIL | `high` | `code.soft_import_pattern` (Optional SDK imports guarded by try/except ImportError) | 1077 ms | 2 unguarded soft-import(s) |
| SKIP | `medium` | `code.ruff_clean` (ruff check (library code)) | 93 ms | ruff binary not on PATH |
| SKIP | `low` | `code.mypy_clean` (mypy (best-effort)) | 46 ms | mypy binary not on PATH |
| FAIL | `low` | `code.from_future_annotations` (from __future__ import annotations in i3/ modules) | 188 ms | 170 module(s) without __future__ annotations (threshold 20) |
| PASS | `info` | `code.pep604_union_syntax` (PEP-604 union syntax (X | Y over Optional/Union)) | 233 ms | 111 legacy Optional/Union occurrence(s) (warning only) |
| PASS | `high` | `code.no_todo_personnel_references` (No invalidated personnel references in interviewer-facing docs) | 31 ms | no invalidated personnel references |
| FAIL | `high` | `config.yaml_parse_all` (All .yaml/.yml files parse) | 608 ms | 11 YAML parse failure(s) |
| PASS | `high` | `config.json_parse_all` (All .json files parse) | 125 ms | 15 JSON file(s) parsed cleanly |
| PASS | `high` | `config.toml_parse_all` (All .toml files parse) | 108 ms | 4 TOML file(s) parsed cleanly |
| PASS | `medium` | `config.notebooks_valid_nbformat` (All .ipynb files are valid nbformat v4) | 94 ms | 7 notebook(s) valid |
| FAIL | `medium` | `config.env_example_keys_documented` (.env.example keys referenced in python source) | 125 ms | 9/28 env keys not referenced in source |
| FAIL | `blocker` | `config.no_hardcoded_secrets` (No hardcoded secret prefixes outside .env.example) | 437 ms | 1 potential secret(s) found |
| PASS | `blocker` | `config.claude_model_id_locked` (configs/default.yaml cloud.model == claude-sonnet-4-5) | 0 ms | cloud.model == 'claude-sonnet-4-5' |
| SKIP | `medium` | `config.mkdocs_build_strict` (mkdocs build --strict) | 16 ms | mkdocs binary not on PATH |
| SKIP | `blocker` | `runtime.fastapi_app_creation` (create_app() returns a FastAPI instance) | 0 ms | fastapi or server.app not importable: No module named 'fastapi' |
| SKIP | `high` | `runtime.all_routes_registered` (OpenAPI schema registers every expected route) | 0 ms | server.app not importable: No module named 'fastapi' |
| SKIP | `blocker` | `runtime.health_live_endpoints` (GET /api/health and /api/live return 200) | 0 ms | TestClient unavailable: No module named 'fastapi' |
| SKIP | `high` | `runtime.ready_endpoint_shape` (/api/ready JSON includes status + checks) | 0 ms | TestClient unavailable: No module named 'fastapi' |
| SKIP | `blocker` | `runtime.privacy_sanitizer_patterns` (PrivacySanitizer redacts 10 PII categories) | 14 ms | PrivacySanitizer import failed: No module named 'numpy' |
| SKIP | `blocker` | `runtime.pddl_refuses_sensitive_cloud` (PDDL planner refuses cloud route on sensitive topics) | 0 ms | PDDL planner import failed: No module named 'numpy' |
| SKIP | `high` | `runtime.tcn_forward_pass` (TCN forward pass returns [1, 64]) | 0 ms | torch or TCN not importable: No module named 'torch' |
| SKIP | `high` | `runtime.adaptation_vector_roundtrip` (AdaptationVector to_dict/from_dict roundtrip is bit-identical) | 0 ms | AdaptationVector not importable: No module named 'numpy' |
| SKIP | `blocker` | `runtime.encryption_roundtrip` (ModelEncryptor encrypt/decrypt ndarray is bit-identical) | 0 ms | ModelEncryptor not importable: No module named 'numpy' |
| SKIP | `high` | `runtime.bandit_thompson_sample_validity` (Thompson bandit returns a valid arm index) | 0 ms | bandit not importable: No module named 'numpy' |
| FAIL | `blocker` | `providers.all_registered` (ProviderRegistry contains >= 11 first-class providers) | 16 ms | ProviderRegistry not importable: No module named 'numpy' |
| FAIL | `high` | `providers.construct_without_sdk` (Every provider class instantiates without its SDK) | 0 ms | exception: ModuleNotFoundError: No module named 'numpy' |
| FAIL | `blocker` | `providers.multi_provider_fallback` (MultiProviderClient falls through to second provider on first failure) | 0 ms | multi_provider not importable: No module named 'numpy' |
| FAIL | `medium` | `providers.cost_tracker_basic` (CostTracker.record then .report yields non-zero totals) | 0 ms | cost_tracker not importable: No module named 'numpy' |
| FAIL | `medium` | `providers.prompt_translator_shapes` (prompt_translator produces provider-specific shapes) | 0 ms | prompt_translator import failed: No module named 'numpy' |
| PASS | `high` | `infra.dockerfile_parses` (Every Dockerfile* has FROM and CMD/ENTRYPOINT) | 16 ms | 4 Dockerfile(s) valid |
| PASS | `medium` | `infra.compose_schema` (docker-compose*.yml has services section) | 0 ms | 2 compose file(s) valid |
| SKIP | `low` | `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)) | 15 ms | helm binary not on PATH |
| PASS | `medium` | `infra.kubernetes_manifests` (deploy/k8s/*.yaml have apiVersion/kind/metadata) | 16 ms | 10 k8s manifest(s) valid |
| SKIP | `low` | `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy) | 0 ms | cedarpy not available |
| FAIL | `high` | `interview.slide_count` (docs/slides/presentation.md has exactly 15 '---' separators) | 0 ms | 16 slide separator(s), expected 15 |
| PASS | `high` | `interview.qa_pair_count` (docs/slides/qa_prep.md has exactly 52 Q&A pairs) | 0 ms | 52 '### ' Q&A heading(s), expected 52 |
| FAIL | `blocker` | `interview.closing_line_verbatim` (closing_lines.md contains the verbatim closing line) | 0 ms | verbatim closing line NOT found |
| PASS | `high` | `interview.honesty_slide_title_case` (presentation.md contains the 'What This Prototype Is Not' slide) | 0 ms | honesty slide title present |
| PASS | `medium` | `interview.adr_count` (docs/adr has >= 10 numbered ADRs) | 0 ms | 10 numbered ADR(s) (>=10 required) |
| PASS | `high` | `interview.brief_analysis_header` (BRIEF_ANALYSIS.md first 30 lines contain 'Corrections notice') | 0 ms | Corrections notice present in first 30 lines |
| PASS | `low` | `interview.changelog_unreleased_nonempty` (CHANGELOG.md [Unreleased] section > 500 chars) | 0 ms | [Unreleased] body is 19671 chars (>500 required) |
| PASS | `low` | `interview.notes_md_sections` (NOTES.md has >= 10 '##' section headers) | 0 ms | 10 '## ' section header(s) (>=10 required) |

## Failures

### `code.top_level_imports` - Top-level packages import cleanly (`blocker`)

- Message: 1 import failures

```
i3: ModuleNotFoundError: No module named 'numpy'
```

### `code.no_print_in_library` - No print() in i3/ library code (`medium`)

- Message: 2 print() call(s) in i3/

```
i3\encoder\onnx_export.py:202
i3\slm\onnx_export.py:303
```

### `code.soft_import_pattern` - Optional SDK imports guarded by try/except ImportError (`high`)

- Message: 2 unguarded soft-import(s)

```
i3\cloud\dspy_adapter.py:306: dspy
i3\cloud\dspy_adapter.py:315: dspy
```

### `code.from_future_annotations` - from __future__ import annotations in i3/ modules (`low`)

- Message: 170 module(s) without __future__ annotations (threshold 20)

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

### `config.yaml_parse_all` - All .yaml/.yml files parse (`high`)

- Message: 11 YAML parse failure(s)

```
deploy\helm\i3\templates\configmap.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 7, column 7:
        {{- include "i3.labels" . | ninden ... 
          ^
deploy\helm\i3\templates\deployment.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 7, column 7:
        {{- include "i3.labels" . | ninden ... 
          ^
deploy\helm\i3\templates\hpa.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 1, column 3:
    {{- if .Values.autoscaling.enabled -}}
      ^
deploy\helm\i3\templates\ingress.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 1, column 3:
    {{- if .Values.ingress.enabled -}}
      ^
deploy\helm\i3\templates\networkpolicy.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 1, column 3:
    {{- if .Values.networkPolicy.enabl ... 
      ^
deploy\helm\i3\templates\pdb.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 1, column 3:
    {{- if .Values.podDisruptionBudget ... 
      ^
deploy\helm\i3\templates\service.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 7, column 7:
        {{- include "i3.labels" . | ninden ... 
          ^
deploy\helm\i3\templates\serviceaccount.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 1, column 3:
    {{- if .Values.serviceAccount.crea ... 
      ^
deploy\helm\i3\templates\servicemonitor.yaml: while parsing a flow node
expected the node content, but found '-'
  in "<unicode string>", line 1, column 3:
    {{- if .Values.serviceMonitor.enab ... 
      ^
deploy\helm\i3\templates\tests\test-connection.yaml: while parsing a block mapping
  in "<unicode string>", line 4, column 3:
      name: "{{ include "i3.fullname"  ... 
      ^
expected <block end>, but found '<scalar>'
  in "<unicode string>", line 4, column 22:
      name: "{{ include "i3.fullname" . }}-test-connection"
                         ^
mkdocs.yml: could not determine a constructor for the tag 'tag:yaml.org,2002:python/name:material.extensions.emoji.twemoji'
  in "<unicode string>", line 145, column 20:
          emoji_index: !!python/name:material.extension ... 
                       ^
```

### `config.env_example_keys_documented` - .env.example keys referenced in python source (`medium`)

- Message: 9/28 env keys not referenced in source

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
I3_DB_PATH
I3_ENCODER_CHECKPOINT
I3_LLM_PROVIDER
I3_OFFLINE_ONLY
I3_SLM_CHECKPOINT
I3_TELEMETRY_DEBUG
I3_TEST_LIVE_PROVIDERS
```

### `config.no_hardcoded_secrets` - No hardcoded secret prefixes outside .env.example (`blocker`)

- Message: 1 potential secret(s) found

```
i3\cloud\guardrails.py: ghp_*
```

### `providers.all_registered` - ProviderRegistry contains >= 11 first-class providers (`blocker`)

- Message: ProviderRegistry not importable: No module named 'numpy'

### `providers.construct_without_sdk` - Every provider class instantiates without its SDK (`high`)

- Message: exception: ModuleNotFoundError: No module named 'numpy'

```
Traceback (most recent call last):
  File "D:\implicit-interaction-intelligence\scripts\verification\framework.py", line 291, in _run_one
    result = future.result(timeout=timeout_s)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\User\AppData\Local\Programs\Python\Python312\Lib\concurrent\futures\_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\User\AppData\Local\Programs\Python\Python312\Lib\concurrent\futures\_base.py", line 401, in __get_result
    raise self._exception
  File "C:\Users\User\AppData\Local\Programs\Python\Python312\Lib\concurrent\futures\thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\implicit-interaction-intelligence\scripts\verification\checks_providers.py", line 80, in check_provider_construct_without_sdk
    from i3.cloud import providers as _pkg  # noqa: F401 - ensures package
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\implicit-interaction-intelligence\i3\__init__.py", line 51, in <module>
    from i3.config import Config, load_config
  File "D:\implicit-interaction-intelligence\i3\config.py", line 29, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'

```

### `providers.multi_provider_fallback` - MultiProviderClient falls through to second provider on first failure (`blocker`)

- Message: multi_provider not importable: No module named 'numpy'

### `providers.cost_tracker_basic` - CostTracker.record then .report yields non-zero totals (`medium`)

- Message: cost_tracker not importable: No module named 'numpy'

### `providers.prompt_translator_shapes` - prompt_translator produces provider-specific shapes (`medium`)

- Message: prompt_translator import failed: No module named 'numpy'

### `interview.slide_count` - docs/slides/presentation.md has exactly 15 '---' separators (`high`)

- Message: 16 slide separator(s), expected 15

### `interview.closing_line_verbatim` - closing_lines.md contains the verbatim closing line (`blocker`)

- Message: verbatim closing line NOT found

## Skipped

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
- `infra.helm_lint` (helm lint deploy/helm/i3 (if helm available)): helm binary not on PATH
- `infra.cedar_policy_parses` (deploy/policy/cedar/*.cedar parses via cedarpy): cedarpy not available

## Environment

- Git SHA: `56a447f488bfccb3627c63812348dffbed757d24`
- Python: `3.12.10`
- Platform: `Windows-11-10.0.22621-SP0`
- Run at: `2026-04-22T23:54:39.990517+00:00`

