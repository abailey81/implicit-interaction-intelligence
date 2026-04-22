# From a Documentation Mock to a Runnable HMAF Agent

> **Thesis.** Batch D part 2 turns three stubs into three runnable
> features -- a PDDL-grounded privacy-safety planner, an end-to-end
> HMAF agentic runtime, and an AdaptationVector-conditioned real-time
> translation endpoint -- each mapped one-to-one onto a 2026 Huawei
> product direction. The output is three concrete artefacts an
> interviewer can execute against and trace through, plus a connective
> narrative that shows the three are not separate demos but three
> layers of the same "on-device agent" story.

This document is the unifying writeup. It supersedes nothing; it
*connects* features documented at a sibling level in
`harmony_hmaf_integration.md` and `harmonyos6_ai_glasses_alignment.md`.

---

## 1. Why these three? Why now?

Three public 2025-2026 Huawei commitments frame the work:

1. **HarmonyOS 6 + Harmony Multi-Agent Framework (HMAF)** -- launched
   October 2025 with 50+ agent plugins and 80+ agents in the Xiaoyi
   Intelligent Agent Hub. The architectural brief introduces the four
   pillars (interactions, protocol, development, trust) that every
   HMAF-native agent must satisfy.
2. **AI Glasses** -- launched 20 April 2026 (9 days before the
   interview), 35.5 g, HarmonyOS, dual-engine AI, 12-hour battery, and
   real-time translation across Chinese + 20 languages as a headline
   feature.
3. **Agentic safety** -- a Huawei public paper grounds risky-operation
   interception in the Planning Domain Definition Language (PDDL,
   Ghallab et al., 1998; McDermott, 1998) and reports **99.9 %
   interception** of the highest-risk class of agent operations.

Each matching I³ feature in this batch lifts an *existing* mock into an
*executable* artefact:

| Huawei commitment   | I³ Batch D part 2 artefact                                                   |
|---------------------|------------------------------------------------------------------------------|
| Agentic safety PDDL | `i3/safety/pddl_planner.py` + `i3/safety/certificates.py`                    |
| HMAF agentic core   | `i3/huawei/agentic_core_runtime.py` (`HMAFAgentRuntime`)                     |
| AI Glasses xlation  | `server/routes_translate.py` (`POST /api/translate`)                         |

---

## 2. Feature 1 -- PDDL-grounded safety planner

### 2.1 What the existing code already did

Before this batch, I³'s privacy behaviour was policy-shaped, not
proof-shaped. `TopicSensitivityDetector` (in `i3/router/sensitivity.py`)
matched a regex battery; if the score exceeded `sensitivity_threshold`,
`IntelligentRouter.route(...)` forced the local arm. Correct behaviour,
but only as verifiable as the regex list. Reviewers had to trust the
regex order and a `>` comparison.

### 2.2 What the planner adds

`PrivacySafetyPlanner` declares the privacy-safety domain in-code:

- Six **predicates**: `sensitive_topic`, `network_available`,
  `authenticated_user`, `encryption_key_loaded`, `rate_limited`,
  `contains_pii` (plus three derived goal predicates -- `routed_local`,
  `routed_cloud`, `denied` -- used in the effects).
- Four **actions**: `route_local`, `route_cloud`, `redact_pii`,
  `deny_request`, each with a declared precondition set, a forbidden
  set, add/del effects, and a one-line rationale.
- A tiny forward-chaining planner that produces the *shortest*
  ordered plan for a given grounded context.

The safety *invariant* is encoded at the domain level: `route_cloud`
has `sensitive_topic` in its forbidden set. The planner cannot produce
a plan that fires `route_cloud` while `sensitive_topic` holds, and if
a hand-rolled plan tries to, `certify(...)` raises
`SafetyPlannerError` with the exact step index that violated the
invariant.

### 2.3 The certificate

`certify(plan)` returns a Pydantic v2 `SafetyCertificate` that
re-verifies every step:
- preconditions are a subset of the pre-state;
- the forbidden set is disjoint from the pre-state;
- the post-state equals `apply(action, pre-state)`;
- no step violates a listed invariant.

The certificate is YAML-round-trippable (`certificate_to_yaml` /
`certificate_from_yaml`), making it a natural audit artefact for
`Auditor` runs and for SRE post-mortems. The CLI demo
(`scripts/run_safety_planner_demo.py`) emits this YAML on stdout for
any grounded context.

### 2.4 Why this matters to the interviewer

Huawei publishes 99.9 % risky-op interception grounded in PDDL. I³'s
planner is the same architectural pattern at a much smaller scale:
declarative domain + machine-checkable certificate. Two consequences:

- The privacy override is no longer an untestable "policy". It is a
  proof schema that can be reasoned about by anyone, not just the
  author.
- Dropping the planner into an HMAF Pillar-4 audit pipeline requires
  nothing more than persisting the YAML certificate alongside each
  telemetry event.

---

## 3. Feature 2 -- Runnable HMAF agentic runtime

### 3.1 From adapter to runtime

`i3/huawei/hmaf_adapter.py` defined the capability / plan / execute /
telemetry shape, but there was no loop to drive it. Reviewers reading
only the adapter had to imagine the bus.

`HMAFAgentRuntime` closes that gap. It wraps the adapter and provides:

1. **A simulated event bus** -- an `asyncio.Queue[HMAFIntent]`
   bounded at 64 items. Real HMAF deployments replace this with their
   own cross-device bus; the producer/consumer contract is identical.
2. **A lifecycle**: `start()`, `receive_intent(intent)`,
   `plan_and_execute(intent)`, `stop()`. `start` is idempotent.
   `stop` drains pending intents via a sentinel and cancels the
   consumer task after a 5 s timeout.
3. **A soft-imported safety planner integration.** When
   `i3.safety` is available, every intent passes through the PDDL
   planner; the terminal action becomes the response's
   `terminal_action` field. When the safety package is absent the
   runtime reports `"no_safety_planner"` and continues -- the demo
   stays runnable in minimal environments.
4. **A soft-imported pipeline.** The default constructor uses an
   in-package `_MockPipeline` that conforms to
   `I3PipelineProtocol` and returns deterministic, text-free scalar
   payloads; tests and the CLI demo use this mock. Real deployments
   pass in the full `PipelineEngine`.

### 3.2 Intent surface

Five canned intents wired to handlers cover the agentic surface the
Darwin HMI reviewer is most likely to ask about:

- `get_user_adaptation` -- returns a text-free AdaptationVector
  summary (the capability HMAF siblings *actually* consume).
- `summarise_session` -- returns scalar rollups (turns, engagement),
  never raw diary text.
- `translate` -- acknowledges translation intents; the real
  translation pipeline is served by the HTTP endpoint in feature 3.
- `route_recommendation` -- produces a recommended route terminal
  action from the `prefer_cloud` parameter.
- `explain_adaptation` -- returns a structured (non-textual)
  description of which adaptation dimensions are active.

### 3.3 Privacy guard

Two forbidden intent names (`dump_raw_diary`, `exfiltrate_text`) are
rejected before any plan is even produced. The rejection emits an
`intent.refused` telemetry event so HMAF's cross-agent observability
dashboard can flag the attempt.

### 3.4 Telemetry

Every step of the loop emits a `HMAFAgentAdapter`-validated telemetry
event -- and the adapter's text-free invariant (no `text`/`prompt`/
`response`/`body`/`content`/`raw` keys) is enforced at emission. The
test `test_telemetry_events_are_text_free` exhaustively verifies this
on every emitted event across the runtime lifecycle.

### 3.5 The CLI demo

`scripts/run_hmaf_runtime_demo.py` boots the runtime, feeds the five
canned intents, prints each structured response as JSON, and stops the
runtime cleanly. It requires no network, no checkpoint, no secrets --
only Python 3.10+ and the I³ source tree. This is the
"hello-world-HMAF" of the repository.

---

## 4. Feature 3 -- Real-time translation endpoint

### 4.1 Why translation, why AdaptationVector

Huawei's AI Glasses launch positioned translation as the headline
consumer feature. Every other shipping competitor also translates. I³'s
differentiator is *style-conditioned* translation: the same French
sentence lands differently for a user whose AdaptationVector has high
`formality` + low `emotional_tone` vs one with the inverse. This is
where user-modelling stops being a side-stream and starts being the
product.

### 4.2 The endpoint

`POST /api/translate` accepts:

```
{"user_id": "alice",
 "text": "...",
 "target_language": "fr",
 "source_language": "en"      // optional
}
```

and returns the translated text plus the AdaptationVector that was
*actually* applied, the source/target languages, the end-to-end latency,
a `fallback_mode` boolean, and the count of PII fragments redacted
before the cloud call.

### 4.3 Layered defence

Four defensive layers are applied in order:

1. **Body-size cap (413)** -- raw body must be <= 4 KiB. The check
   runs before Pydantic so the worst-case materialised Python string
   is bounded. Translation requests are short by design; the cap is
   generous.
2. **Pydantic validation (422)** -- `LanguageCode` enum rejects any
   language not explicitly supported, the user_id pattern is the same
   anchored regex used everywhere else, text is non-empty after strip.
3. **PII sanitisation** -- `PrivacySanitizer` strips the ten-pattern
   battery (emails, phones, IBANs, SSNs, credit cards, URLs, DoBs,
   addresses, passports, IPs) before anything goes to the cloud. The
   number of redactions is exposed in the response.
4. **Fallback mode** -- when the `CloudLLMClient` is unavailable (no
   API key, no pipeline, or transient failure), the endpoint returns
   a deterministic pseudo-translation with `fallback_mode=true`. The
   demo stays runnable in a laptop dev loop without secrets.

### 4.4 AdaptationVector wiring

The Anthropic system prompt embeds the user's current AdaptationVector
in plain text (only scalars, never raw diary content), e.g.:

```
- cognitive_load: 0.62
- formality: 0.48
- verbosity: 0.71
- emotionality: 0.33
- directness: 0.55
- emotional_tone: 0.62
- accessibility: 0.18
```

The cloud model is instructed to match the user's communication style
along those dimensions, and to emit only the translated text. This is
the same cross-boundary contract the rest of I³ uses: structured
scalars across the boundary, never raw text.

### 4.5 Tests

`tests/test_routes_translate.py` exercises the happy path,
fallback-mode, cloud-absent-with-pipeline, oversized body (413),
unsupported language (422), empty text (422), path-traversal user id
(422), PII redaction, and the `/languages` enumeration endpoint.

---

## 4.6 What a reviewer should pay attention to

The endpoint is not meant to replace a dedicated translation model. The
interesting contribution is not the translation itself, which any
competitor ships, but the **conditioning shape**: the user's
personalisation profile is the thing that changes the output, not the
prompt. The AdaptationVector is computed from implicit signals
(keystroke rhythm, revision rate, temporal deltas, lexical complexity)
without ever asking the user a preference question. Two users sending
the same `text=""` to the same `target_language` receive two
stylistically distinct translations -- and the delta is reproducible
from their vectors alone. This is exactly the claim the JD's
"user-modelling" keyword promises; the translate endpoint is the
cleanest place to demonstrate it because the cloud model's output is
measurably different between vectors, in ways a reviewer can eyeball
side by side.

The CLI equivalent `POST /api/translate` is one `curl` away. In the
live demo the interviewer can toggle the AdaptationVector through the
what-if endpoint (`server/routes_whatif.py::whatif_respond`) and then
re-issue the translate call; the difference is immediate and
defensible.

---

## 5. Cross-feature invariants

The three features share three load-bearing invariants, and the batch's
tests pin each one:

1. **No raw text crosses a boundary.** HMAF telemetry events are
   validated text-free at emission; the translate endpoint sanitises
   before the cloud call; the safety planner operates only over
   predicate valuations, not prose.
2. **Every decision is reconstructible.** The safety certificate is
   a verbatim replay of the planner's reasoning trace. HMAF telemetry
   events carry correlation ids. The translation response carries the
   exact AdaptationVector it used.
3. **Graceful degradation, never silent.** If `i3.safety` is missing,
   the runtime advertises `no_safety_planner`. If the cloud client is
   missing, the translate endpoint advertises `fallback_mode=true`. No
   branch silently synthesises an answer when a subsystem is absent.

---

## 6. Interview-ready demo script

1. `python scripts/run_safety_planner_demo.py --sensitive --network
   --auth --keyed --pii` -- the reviewer sees a redacted + routed-local
   plan with a YAML certificate.
2. `python scripts/run_hmaf_runtime_demo.py` -- five structured
   responses from five canned HMAF intents, plus telemetry on stdout.
3. `uvicorn server.app:app` in one terminal, `curl -X POST
   http://127.0.0.1:8000/api/translate -H 'Content-Type:
   application/json' -d '{"user_id":"demo","text":"Good morning",
   "target_language":"ja"}'` in another -- the reviewer sees fallback
   mode return a deterministic echo; pointing `ANTHROPIC_API_KEY` at a
   real key flips `fallback_mode` to `false` and yields the
   style-conditioned Claude translation.

Three runnable features, each a direct analogue of a public Huawei
initiative, each grounded in the existing I³ architecture, each with
tests and an audit story. Together they demonstrate that user-modelling
from implicit signals is not a demo detour: it is the first layer of a
runnable agent.

---

## 6.1 What to NOT say in the demo

Three common pitfalls if an interviewer asks the obvious questions:

- "Is the PDDL planner a real planner?" -- No. It is a forward-chaining
  dispatcher over a tiny domain. A real PDDL planner (Fast-Downward,
  LAMA, POPF) would add no value because the privacy-safety domain is
  small enough that a six-rule forward pass is both provably correct
  and drastically cheaper. The *shape* of the artefact -- declarative
  domain + machine-checkable certificate -- is what matches Huawei's
  published work; the search strategy is a scale-dependent choice.
- "Does the HMAF runtime actually talk to HarmonyOS?" -- No. It
  simulates the bus in-process. Every call-site is a protocol shape
  (`HMAFIntent`, capability ids, telemetry events) that maps one-to-one
  onto HMAF's SDK; replacing the queue with an actual HMAF transport
  is a pluggable change. The runtime is the *agentic core* framework
  on an `asyncio.Queue`, not an HMAF client.
- "Does the translate endpoint do better than a vanilla cloud call?"
  -- On translation-quality metrics, no. The differentiator is style
  fidelity across repeated translations for the same user, which is
  how the AdaptationVector earns its place in the pipeline. If the
  reviewer pushes, explain the experiment: (1) freeze a text corpus,
  (2) translate with uniform AdaptationVectors, (3) translate with
  user-specific vectors, (4) score style similarity to the user's own
  historical messages in the target language. The signal is the
  measurable increase in style-similarity when the vector is
  user-specific.

These limits are stated up front. They are also the reasons each
feature is self-contained and reviewable in a one-hour interview, not
a three-month integration.

---

## 7. References

- Ghallab, M., Howe, A., Knoblock, C., McDermott, D., Ram, A., Veloso,
  M., Weld, D., Wilkins, D. (1998). *PDDL -- The Planning Domain
  Definition Language*. AIPS-98.
- McDermott, D. (1998). *PDDL 1.2*. Yale Center for Computational
  Vision and Control, Technical Report.
- Huawei Developer Conference, Late 2025 -- HarmonyOS 6 & HMAF launch
  materials ("Four Pillars of HMAF" brief).
- Huawei Product Launch, 20 Apr 2026 -- AI Glasses press kit.
- Huawei public agentic-safety PDDL paper (99.9 % risky-operation
  interception).
- Nosek, B. A., et al. (2018). *The preregistration revolution*. PNAS
  115(11):2600-2606. (For the safety-certificate-as-audit-artefact
  argument.)
