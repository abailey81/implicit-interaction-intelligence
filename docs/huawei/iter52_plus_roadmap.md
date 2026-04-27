# Iter 52+ — open-ended improvement loop

> **2026-04-27.** After iter 51 closed the JD's required bullets and
> shipped the Qwen + Gemini + SLM cascade, the user asked for a
> rolling improvement loop covering **all** layers — SLMs, foundation
> LLMs, fine-tunes, evaluation, edge deployment, observability,
> security, multi-cloud, multimodal, dashboard, real-user emulation
> — until they explicitly say "stop".
>
> This document is the durable roadmap for that loop.  Each iteration
> picks one focus area, ships a measurable improvement, runs the
> full guard-test suite, and commits.  The loop is intentionally
> non-narrowly-scoped: every iteration touches a *different* sphere so
> the project keeps growing breadth, not just depth in one corner.

## Iteration cadence

* One **focused improvement** per iter (single PR-sized change).
* One **measurable outcome** per iter (a number that moved, a test
  that flipped green, a tab that now serves richer data).
* One **CHANGELOG entry** per iter (`[Unreleased]` until the next
  release tag).
* One **memory note** per iter (`memory/iterNN_*.md`).
* One **commit** per iter on `main`.
* The full **279/279 pytest sweep + 170/170 drift + 4/4 cross-session**
  must stay green at every iter boundary.

## Spheres on rotation

| Sphere | Sample iters |
|---|---|
| **From-scratch SLM** | continuation training, distillation, quantisation-aware training, knowledge editing, RLHF-lite, attention-head pruning |
| **Fine-tune-of-pre-trained (Qwen LoRA)** | hyperparameter sweep, 4-bit QLoRA experiment, longer training, larger rank, RoPE-scaling for long context, adapter merging, Gemini 2.5 Pro tier swap |
| **Cloud cascade (Gemini + multi-provider)** | Gemini fine-tune over JSON tool-call format, multi-provider failover policy, cost-budget enforcement, response-quality LLM-judge ensemble |
| **Edge deployment** | ONNX Runtime Mobile profile, Core ML 8-bit packaging, ExecuTorch on-device benchmark, TensorRT-LLM int4 path, Kirin-NPU specific kernel |
| **Eval harness** | per-quartile latency curve, OOD prompt set, factuality probe, safety-redteam batch, multilingual probe |
| **Real-user emulation** | longer Playwright sessions, multi-session memory persistence drill, accessibility-mode drill, vision-on / vision-off A/B, voice-prosody A/B |
| **Observability** | new Sentry dashboards, Langfuse trace tagging, OpenTelemetry span trees per cascade arm, MLflow-tracked eval runs |
| **Security** | new redteam attack categories, PII regex coverage expansion, encryption rotation drill, differential-privacy audit |
| **Dashboard** | new tab, new chip, new live signal, animation polish, accessibility audit |
| **Docs** | research-reading-list update, new design-decision log, API spec OpenAPI freshening |

## Stop condition

Only the user saying "stop" (or equivalent: "halt", "we're done",
"that's enough") halts the loop.  Until then, every reply that
finishes a previous iter starts the next.

## Bookkeeping

* This roadmap is updated at the head of every iter with the new
  focus area.
* Completed iters get a `## Iter NN — <focus>` block with the
  outcome number.
* The 279/279 / 170/170 / 4/4 trio must stay green; any regression
  rolls back.

---

## Iter 52 — Qwen + Gemini cascade integration (this commit)

**Focus.** Wire the iter-51 fine-tune artefacts (Qwen LoRA + Gemini
provider) into the *live* chat pipeline as a real cascade, not just
endpoints behind `/api/intent`.

**What changed.**

* `i3/pipeline/engine.py` — new `_INTENT_TRIGGER_PATTERNS` (cheap
  regex gate) + `_looks_like_command()` + `_maybe_handle_intent_command()`
  helpers.  Detects HMI command-shaped utterances ("set timer 5 min",
  "play jazz", "navigate home", "turn off the bedroom lamp") and
  routes them through the Qwen LoRA parser before falling through
  to the SLM.  Returns deterministic acknowledgements; stashes the
  full IntentResult on `_last_intent_result` so the green ◆ chip
  renders and the WS state_update payload carries the parse.
* `i3/cloud/providers/google.py` — accepts either `GEMINI_API_KEY`
  (AI Studio) or `GOOGLE_API_KEY` (Vertex/GCP) for auth.  Closes
  the third cascade arm.
* `training/train_intent_gemini.py` — auto-loads `.env` so the
  user can `python training/train_intent_gemini.py` directly when
  ready (no manual env-var dance).
* `i3/pipeline/engine.py` — `get_profiling_report()` now returns two
  new component rows (Qwen-LoRA cascade arm, Gemini cloud cascade
  arm) plus a `cascade_arms` summary block detailing which arm fires
  on which utterances.
* `server/routes.py` — `_PROFILING_ALLOWED_FIELDS` extended to allow
  `cascade_arms` through the response filter.
* `tests/test_intent_cascade.py` — **53 new unit tests** covering the
  cheap gate (28 commands detected, 10 chat turns rejected),
  fall-through behaviour, and per-action acknowledgement
  rendering.  All green.

**Outcome.** The chat pipeline now *uses* the iter-51 artefacts
end-to-end.  An HMI command doesn't fall through to the SLM and
generate hallucinated chat — it gets parsed by the LoRA, returns a
structured action + a clean acknowledgement, and ships the full
IntentResult on the WS state_update so the dashboard renders it.

**Verification.**

* pytest core + cascade: **332/332** (279 prior + 53 cascade).
* `_looks_like_command()` regression: 28 commands True, 10 chat False.
* `/api/profiling/report`: now includes the Qwen-LoRA + Gemini rows
  and the `cascade_arms` summary; total budget still 55.7 ms (cascade
  arms only fire on demand and don't blow the budget).
* WS `response` + `state_update` already ship `intent_result`
  (iter-51), so the new helper feeds straight through to the
  dashboard's green ◆ chip.

---

## Iter 53 — Edge Profile dashboard renders cascade_arms (commit `e57c77f`)

**Focus.** Wire the iter-52 `cascade_arms` block (`/api/profiling/report`)
into the dashboard's Edge Profile tab so the reviewer can see the
3-arm cascade structure visually, not just in the JSON.

**Outcome.** `web/js/huawei_tabs.js#renderEdgeProfile` now appends a
3-card grid (one per arm) below the existing component table; each
card shows P50 latency, memory delta, and when the arm fires.

## Iter 54 — Promote deep emulator + add cascade-profiling tests (commit `0de7c46`)

**Focus.** Move `D:/tmp/deep_real_user_emulation.py` into
`scripts/deep_real_user_emulation.py` so reviewers can run it without
the scratchpad path; add 7 fast tests pinning the
`/api/profiling/report` shape contract.

## Iter 55 — Per-arm rolling latency tracker + dashboard ribbon (commit `418ff11`)

**Focus.** Static profile (iter 51) tells the reviewer what the
budget *says*; iter 55 adds the matching live measurement.
`Pipeline._cascade_arm_latencies` (deque maxlen 200 per arm),
`Pipeline.cascade_arm_stats()`, `GET /api/cascade/stats`, and the
Edge Profile tab now appends a "Live cascade-arm latency" table
beneath the static cards.

## Iter 56 — Deep system-health endpoint (commit `a7ae791`)

**Focus.** Single-page snapshot of every subsystem.
`GET /api/health/deep` returns SLM v2 (checkpoint present, params,
best_eval_loss), encoder, intent (Qwen adapter + Gemini key set as
boolean — never the value), cloud (registered providers), privacy
(encryption key set, budget loaded), cascade (live counters),
profiling (top-line edge numbers), checkpoint disk inventory.

## Iter 57 — Multilingual + adversarial robustness tests (commit `9943d7b`)

**Focus.** Prove the pre-SLM layers survive non-Latin scripts.  27
tests covering BPE round-trip on French / Spanish / Japanese /
Chinese / Korean / Arabic / Hebrew / Russian / Greek / Hindi /
emoji / Zalgo, plus intent-gate / PII / sensitivity-classifier
robustness on multilingual input.

## Iter 58 — SLM v2 perf-regression guard tests (commit `28bf6ce`)

**Focus.** 4 tests load the v2 checkpoint and assert: param count in
194-215 M (±5 % drift), vocab matches BPE, single forward < 6 s on
CPU, output shape `(batch, seq, vocab)`.

## Iter 59 — Redteam batch eval (commit `10ab1e3`)

**Focus.** Third eval harness alongside `eval_intent.py` and
`eval_slm_v2.py`.  `training/eval_redteam.py` runs the 55-attack
`ATTACK_CORPUS` through the safety classifier; baseline block
recall = 0.028 (honest — surfaces the prompt-injection vs
harm-content threat-model gap).

## Iter 60 — Privacy-budget circuit-breaker tests (commit `1f705b9`)

**Focus.** 7 tests pin PrivacyBudget invariants: default consent
OFF, per-session call + byte budgets exhaust correctly, reset
clears bucket, per-user isolation.

## Iter 61 — PipelineOutput contract tests (commit `33c5bf5`)

**Focus.** 26 tests pin the WS frame schema (10 required + 22
optional fields), defaults sanity-checked, iter-51 fields
(safety_caveat, personal_facts, intent_result) explicitly verified.

## Iter 62 — Per-cascade-arm chat-bubble chip (commit `e87713b`)

**Focus.** `web/js/chat.js#_appendSideChips` now also emits an
arm-label chip derived from `metadata.response_path`: `A · SLM`,
`B · Qwen LoRA`, `C · cloud`, `R · retrieval`, `T · <tool>`.

## Iter 63 — Cloud MultiProviderClient tests (commit `fa78d99`)

**Focus.** 8 tests pin the fallback-chain behaviour: short-circuit
on success, fall-back on failure, exception when all fail, circuit
breaker opens after threshold, AuthError treated as terminal,
stats() reporting, input validation, idempotent close().

## Iter 64 — BPE tokenizer corner-case tests (commit `b97e5b5`)

**Focus.** 13 tests pin tokenizer behaviour: empty / whitespace /
single-byte / control char / emoji surrogate pairs / leading-trailing
whitespace preservation / 9 000-char round-trip / BOS+EOS injection /
distinct special IDs / idempotent double encode.

## Iter 65 — Aggregate CHANGELOG + roadmap refresh (this commit)

**Focus.** Document iter 53-64 in CHANGELOG and this roadmap so the
reviewer can read the trajectory in one place.

## Iter 66 — OpenTelemetry per-cascade-arm spans (commit `3e24e4b`)

`_maybe_handle_intent_command` now opens nested OTel spans
(`cascade.arm_b.qwen_intent` / `qwen_load` / `qwen_parse`) so the
collector / Sentry / Langfuse can correlate per-arm latency.  4 tests
verify span emission via the in-memory OTel exporter.

## Iter 67 — Process-wide CostTracker singleton (commit `728cf84`)

`get_global_cost_tracker()` + `GET /api/cost/report` expose the
shared cloud-spend ledger; 5 tests cover singleton identity, reset,
record/report, unknown-model handling, schema.

## Iter 68 — Multimodal validator coverage (commit `1325029`)

11 tests pin `validate_prosody_payload` / `validate_gaze_payload`:
non-dict rejection, missing-key handling, out-of-range / NaN clamp,
type-coercion safety.

## Iter 69 — EngagementSignal invariants (commit `69d552b`)

16 tests pin score bounds in [0, 1] under hostile values + the
intuitive monotonicity properties (continuation > no-continuation,
fast latency > slow latency, higher topic-continuity > lower).

## Iter 70 — KnowledgeGraph dedupe regression (commit `1fbdcdc`)

5 tests pin the iter-51 Jaccard ≥ 0.6 dedupe behaviour in
`KnowledgeGraph.overview()` so a year-overlap regression breaks fast.

## Iter 71 — PII sanitiser comprehensive coverage (commit `f86ce55`)

25 tests across email / SSN / phone (US/UK/intl) / credit-card / IP
with explicit no-false-positive guards (Windows builds, year ranges,
SemVer fragments) and honest documentation of regex-precedence limits.

## Iter 72 — SelfCritic scoring contract (commit `bb447c3`)

11 tests pin the 5-rubric composite scoring + threshold-accept
behaviour + "never raises" robustness.

## Iter 73 — AdaptationVector clamping + serialisation (commit `c682348`)

10 tests pin the 8-axis vector's `[0, 1]` clamp, `to_tensor()` shape
and layout, and NaN safety.

## Iter 74 — TCN encoder invariants (commit `2cea36b`)

8 tests pin output shape (batch, 64), L2-norm contract, gradient
flow, eval-mode batch-independence, and config validation.

## Iter 75 — `make test-iter` / `test-cascade` Makefile targets (commit `4511524`)

One-command access to the full iter-52..74 test sweep + cascade-only
fast subset.

## Iter 76 — DiaryStore PII-free schema invariants (commit `7cc3a03`)

4 tests inspect the live SQLite schema after `initialize()` to
enforce the no-natural-language-column contract.

## Iter 77 — IntentResult / SUPPORTED_ACTIONS / ACTION_SLOTS contract (commit `e549a2a`)

11 tests pin the canonical action vocab, per-action slot whitelist,
confidence heuristic, and `to_dict()` shape.

## Iter 78 — routing_decision dict + state_update frame shape (commit `c1334e1`)

8 tests pin the routing_decision schema and synthesise the full
`state_update` frame to verify json.dumps round-trip.

## Iter 79 — KnowledgeGraph canonicalisation (commit `865d2ce`)

15 tests pin case / leading-article / punctuation-insensitive subject
lookup + display-name override behaviour.

## Iter 80 — ModelEncryptor Fernet round-trip + key rotation (commit `24c892e`)

9 tests pin bytes / embedding / JSON round-trip, key isolation,
ephemeral-key fallback, and invalid-key rejection.

## Iter 81 — Aggregate sweep + Makefile + roadmap refresh (this commit)

Total iter-52..80 test count: **295 / 295 green in 5.49 s.**

## Iter 82 — PrivacyBudget snapshot WS-shape regression (commit `988baf8`)

6 tests pin the 11-key snapshot.to_dict() schema (totals, max,
remaining, consent, redactions, last_call_ts).

## Iter 83 — TopicSensitivityDetector category coverage (commit `6cda02f`)

16 tests across confidential / abuse_safety / medical_records;
benign turns at min_score; iter-51 inflection fix verified
("sexually assaulted" / "raped" / "molested" all flagged ≥ 0.90).

## Iter 84 — DiaryStore session-lifecycle round-trip (commit `446d464`)

8 tests against a temp SQLite db: create_session → log_exchange →
get_session_exchanges → end_session → get_user_sessions; plus
user_facts set/get/forget round-trip (iter-50 contract).

## Iter 85 — Dashboard HTML nav-link contract (commit `aa052d8`)

7 tests assert every nav-link's data-tab has a matching tab-panel id,
the iter-51 Huawei tabs are present, the Stack subsystem grid is
present, and required CSS / JS files are referenced.

## Iter 86 — FastAPI app smoke + route inventory (commit `fbe9d15`)

8 passing + 1 skipped: TestClient verifies /api/health, /api/intent,
/api/intent/status, /api/profiling/report, /api/cascade/stats,
/api/health/deep, /api/cost/report all registered + return 200.

## Iter 87 — Pipeline._stated_facts cache invariants (commit `0c1e4b8`)

6 tests pin per-(user, session) tuple-keyed cache isolation,
setdefault identity, lazy intent parser init, classifier fallback.

## Iter 88 — Wire CostTracker into MultiProviderClient (commit `ceb676a`)

Successful cloud calls through the chain now bill the global
CostTracker.  4 new tests + 13 regression-suite tests green.
/api/cost/report finally reflects real activity.

## Iter 89 — ContextualThompsonBandit invariants (commit `a859bdf`)

13 tests pin construction validation, select_arm + update behaviour,
Beta-Bernoulli fallback, NaN/inf safety on extreme contexts, and
out-of-range arm robustness.

## Iter 90 — Aggregate sweep + Makefile + roadmap refresh (this commit)

Total iter-52..89 test count: **363 passed + 1 skipped in 9.11 s.**
38 commits stacked since iter 51.

## Iter 91 — Qwen LoRA adapter / tokenizer / training-metrics alignment (commit `a58aa6a`)

8 tests verify the committed Qwen adapter is structurally consistent
(adapter_config.json valid, references Qwen, LoRA rank in supported
set; tokenizer files present; training metrics records best_val_loss).

## Iter 92 — pricing_2026.json table integrity (commit `ac652ab`)

8 tests pin the per-model pricing schema (input/output rates, no
negatives, anthropic/openai/gemini coverage).

## Iter 93 — CostTracker priced-call integration (commit `2b40ea7`)

6 tests verify known-model → non-zero cost, unknown → zero +
unknown_models entry, aggregate totals, by_provider breakdown.

## Iter 94 — KnowledgeGraph.compose_answer per-predicate (commit `6570478`)

7 tests pin year-rendering, predicate aliasing (discovered_by →
founded_by fallback), unknown-predicate behaviour, multi-object slots.

## Iter 95 — PipelineInput dataclass contract (commit `f05a000`)

6 tests pin the 7 required fields, optional multimodal defaults,
default-factory isolation across instances.

## Iter 96 — CSS chip-arm class regression (commit `a300753`)

13 tests assert iter-51..62 chip CSS classes (chip-affect/safety/
adapt/intent + chip-arm-{a,b,c,r,t}) and cascade-card classes are
present in huawei_tabs.css.

## Iter 97 — huawei_tabs.js function-wiring contract (commit `6155bcf`)

16 tests assert each wireXxxTab() is both defined and invoked
from the boot block, plus iter-53/55 helper functions present.

## Iter 98 — intent dataset JSONL file integrity (commit `588cca9`)

6 tests verify split sizes, required fields, action vocab in
SUPPORTED_ACTIONS, slot whitelist compliance, JSON validity of the
completion column.

## Iter 99 — Aggregate iter-91..98 sweep + Makefile + roadmap (this commit)

Total iter-52..98 test count: **433 passed + 1 skipped in 9.04 s.**
47 commits stacked since iter 51.

## Iter 100 — *(milestone — next focus picked when iter 99 commits)*
