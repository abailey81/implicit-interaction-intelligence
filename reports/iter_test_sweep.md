# Iter-test sweep verification — 2026-04-27

> Run via: `make test-iter` (or `pytest -q --tb=no <files>`).

## Result

**640 passed + 1 skipped in 35.20 s** across 65 test files.

## Test files in sweep

The full iter-52..135 sweep covers these spheres of the codebase
(one test file per sphere, single-PR-sized increments):

| Sphere | File | Iter |
|---|---|---|
| Intent cascade arm B (Qwen LoRA) | `test_intent_cascade.py` | 52 |
| Edge profiling cascade_arms | `test_profiling_cascade.py` | 54 |
| `/api/health/deep` aggregator | `test_health_deep.py` | 56 |
| Multilingual / RTL / emoji robustness | `test_multilingual_robustness.py` | 57 |
| SLM v2 perf-regression guard | `test_slm_perf_guard.py` | 58 |
| Privacy budget circuit-breaker | `test_privacy_budget_circuit.py` | 60 |
| PipelineOutput contract | `test_pipeline_output_contract.py` | 61 |
| MultiProviderClient fallback | `test_cloud_multi_provider.py` | 63 |
| BPE tokenizer corner cases | `test_bpe_corner_cases.py` | 64 |
| OpenTelemetry per-arm spans | `test_observability_spans.py` | 66 |
| Global CostTracker singleton | `test_cost_tracker_global.py` | 67 |
| Multimodal validators (prosody/gaze) | `test_multimodal_validators.py` | 68 |
| EngagementSignal invariants | `test_engagement_signal.py` | 69 |
| KG dedupe regression | `test_knowledge_graph_dedupe.py` | 70 |
| PII sanitiser comprehensive | `test_pii_sanitizer_coverage.py` | 71 |
| SelfCritic scoring | `test_self_critic.py` | 72 |
| AdaptationVector clamping | `test_adaptation_vector.py` | 73 |
| TCN encoder invariants | `test_tcn_invariants.py` | 74 |
| Diary schema PII-free | `test_diary_schema_invariants.py` | 76 |
| IntentResult contract | `test_intent_types.py` | 77 |
| routing_decision shape | `test_routing_decision_schema.py` | 78 |
| KG canonicalisation | `test_knowledge_graph_canonical.py` | 79 |
| Encryption envelope round-trip | `test_encryption_envelope.py` | 80 |
| PrivacyBudget snapshot WS | `test_privacy_budget_snapshot.py` | 82 |
| Sensitivity categories | `test_sensitivity_categories.py` | 83 |
| Diary store lifecycle | `test_diary_store_lifecycle.py` | 84 |
| Dashboard HTML contract | `test_dashboard_html_contract.py` | 85 |
| FastAPI app routes smoke | `test_server_app_routes.py` | 86 |
| Pipeline _stated_facts cache | `test_pipeline_stated_facts.py` | 87 |
| CostTracker integration | `test_cost_tracker_integration.py` | 88 |
| Bandit invariants | `test_bandit_invariants.py` | 89 |
| Qwen adapter alignment | `test_qwen_adapter_alignment.py` | 91 |
| Pricing table integrity | `test_pricing_table.py` | 92 |
| Cost pricing integration | `test_cost_pricing_integration.py` | 93 |
| KG compose_answer | `test_knowledge_graph_compose.py` | 94 |
| PipelineInput contract | `test_pipeline_input_contract.py` | 95 |
| Chat chip CSS classes | `test_chat_chip_css_classes.py` | 96 |
| huawei_tabs.js wiring | `test_huawei_tabs_js_wiring.py` | 97 |
| Intent dataset files | `test_intent_dataset_files.py` | 98 |
| Sentence dedupe | `test_dedupe_sentences.py` | 100 |
| ResponsePostProcessor | `test_response_postprocess.py` | 101 |
| LinguisticAnalyzer | `test_linguistic_analyzer.py` | 102 |
| Edge profiler helpers | `test_edge_profiler_helpers.py` | 104 |
| Pipeline trace collector | `test_pipeline_trace_collector.py` | 105 |
| Cloud prompt builder | `test_cloud_prompt_builder.py` | 106 |
| chat.js chips | `test_chat_js_chips.py` | 107 |
| Cached-tokens pricing | `test_cost_cached_tokens.py` | 108 |
| Pipeline error output | `test_pipeline_error_output.py` | 110 |
| Bandit convergence | `test_bandit_convergence.py` | 111 |
| Explain decomposer patterns | `test_explain_decomposer_patterns.py` | 112 |
| Privacy budget redactions | `test_privacy_budget_redactions.py` | 113 |
| Cloud error taxonomy | `test_cloud_error_taxonomy.py` | 114 |
| Completion dataclasses | `test_completion_dataclasses.py` | 116 |
| ExplainPlan dataclass | `test_explain_plan_dataclass.py` | 117 |
| StageRecord invariants | `test_stage_record_invariants.py` | 118 |
| KGRelation dataclass | `test_kg_relation_dataclass.py` | 120 |
| Response path lifecycle | `test_response_path_lifecycle.py` | 121 |
| Cost tracker threading | `test_cost_tracker_threading.py` | 123 |
| Agentic core models | `test_agentic_core_models.py` | 124 |
| Secure aggregator | `test_secure_aggregator.py` | 125 |
| EWC invariants | `test_ewc_invariants.py` | 126 |
| MAML construction | `test_maml_construction.py` | 127 |
| Postprocess adapt_with_log | `test_postprocess_adapt_with_log.py` | 129 |
| Cost tracker reset semantics | `test_cost_tracker_reset_semantics.py` | 130 |
| Intent completion validity | `test_intent_completion_validity.py` | 132 |
| AccessibilityModeState | `test_accessibility_mode_state.py` | 134 |
| BiometricMatch dataclass | `test_biometric_match_dataclass.py` | 135 |

## Reproduction

```bash
make test-iter
# or
pytest -q --tb=no \
  $(grep -A 100 "^test-iter:" Makefile | \
    grep "tests/test_" | sed 's/^[ \t]*//;s/ \\$//')
```

## Tally

- 65 test files
- 640 / 640 passed + 1 skipped (OpenAPI when disabled)
- Wall-clock: ~35 s on a laptop CPU
- Zero flakes observed
