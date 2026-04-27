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

## Iter 66 — *(next focus, picked when iter 65 commits)*
