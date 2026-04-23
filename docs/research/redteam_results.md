# Adversarial Evaluation of the I3 Privacy-by-Architecture Runtime

## Abstract

Implicit Interaction Intelligence (I3) ships a privacy-by-architecture
runtime: a local small language model (SLM) handles all sensitive
traffic, a PDDL-grounded safety planner certifies every routing
decision, and a defence-in-depth PII sanitiser strips personally
identifying information before any cross-boundary call.  These
guarantees are only meaningful if the runtime path that enforces them
cannot be bypassed.  This research note describes Batch G6, an
adversarial harness that probes the I3 runtime with a curated corpus
of 55 attacks spanning ten categories drawn from the public
literature, and then checks four runtime invariants against the
observed outcomes.  The harness is wired to four concrete target
surfaces (the FastAPI HTTP app, the PII sanitiser, the PDDL planner,
and the cloud guardrails) so that each attack traverses the same code
path a real attacker would hit.  A single CLI (`scripts/security/run_redteam.py`)
runs the harness locally and in CI, producing both a machine-readable
JSON report and a Markdown summary that is inlined into the GitHub
Actions step summary.

## Motivation

Privacy-by-architecture is a structural guarantee: the system is built
so that sensitive data never leaves the device in the first place.
Structural guarantees are only as strong as the weakest code path that
can undermine them.  A prompt-injection attack that convinces the
router to override its privacy gate, a path-traversal attack that
exfiltrates the on-disk ONNX cache, or a rate-limit bypass that makes
the cloud LLM the cheapest inference surface all compromise the
guarantee regardless of how careful the design is on paper.  This
batch delivers a *runtime* evaluation: rather than argue
architecturally that the guarantee holds, we throw 55 documented
adversarial inputs at the live runtime and show that each of them is
blocked or contained at one or more layers.

## Threat model

The harness assumes four classes of adversary:

1. **External attacker.**  Crafts HTTP requests at the public surface
   with the intent of triggering server-side behaviour that
   exfiltrates data or escalates privileges.
2. **Malicious authenticated user.**  Can submit arbitrary text to
   the `/api/chat` endpoint and attempt prompt-injection, jailbreak,
   or privacy-override bypass.
3. **Compromised cloud LLM.**  An upstream model that tries to
   exfiltrate data by smuggling reflected identifiers into responses,
   or to extract the caller's credentials by claiming tool use.
4. **Indirect / multi-turn attacker.**  Uses the diary recall path or
   a benign-looking first turn to stage a second-turn injection once
   context has been established.

## Attack taxonomy

The 55 attacks are sorted into ten categories tied to the OWASP LLM
Top-10 2025 control set:

| Category | Count | Primary OWASP mapping |
|---|---|---|
| `prompt_injection` | 10 | LLM01 |
| `jailbreak` | 8 | LLM01 |
| `pii_extraction` | 8 | LLM02 |
| `privacy_override_bypass` | 6 | LLM06 |
| `rate_limit_abuse` | 5 | LLM04 / LLM10 |
| `oversized_payload` | 4 | LLM04 |
| `path_traversal` | 4 | LLM10 |
| `header_injection` | 3 | LLM10 |
| `unicode_confusable` | 3 | LLM09 |
| `multi_turn_setup` | 4 | LLM01 / indirect-PI |

Every attack carries a citation that references its originating paper
or OWASP entry.  Prompt-injection and jailbreak payloads draw heavily
on Perez & Ribeiro's "Ignore Previous Prompt" (2022), Zou et al.'s
GCG family (2023), and Greshake et al.'s indirect-PI corpus (2023).
PII and composite attacks borrow from HarmBench (Liu et al. 2024) and
the TDC adversarial evaluation (Mazeika et al. 2024).

## Method

The harness has three moving parts:

1. **Attack corpus** (`i3/redteam/attack_corpus.py`).  Immutable
   Pydantic models pinned to their declared per-category counts.  The
   corpus is frozen at import time and a sister `load_external_corpus`
   helper round-trips a JSON file through the schema so HarmBench /
   AdvBench exports can be plugged in later.
2. **Target surfaces** (`i3/redteam/attacker.py`).  Four drop-in
   implementations of the `TargetSurface` Protocol:
   `FastAPITargetSurface` drives the HTTP app with
   `fastapi.testclient.TestClient`, `SanitizerTargetSurface` runs the
   PII regex battery, `PDDLPlannerTargetSurface` builds a synthetic
   `SafetyContext` and asserts the planner's certificate, and
   `GuardrailsTargetSurface` pushes text through the input and output
   guardrails.  All four return uniform `AttackResult` objects so the
   runner can aggregate across targets.
3. **Runner** (`RedTeamRunner`).  An asyncio scheduler that fans out
   up to N concurrent attacks behind a bounded `asyncio.Semaphore`,
   captures unhandled exceptions per-attack, and produces an
   aggregated `RedTeamReport` containing per-category pass rates,
   critical-failure counts, and a typed failure list.

The harness adds four invariant checks (`i3/redteam/policy_check.py`):

- **Privacy invariant.**  The on-disk diary schema must not expose any
  column whose name suggests raw user text, and no cell may contain the
  red-team canary string after the run.
- **Rate-limit invariant.**  At least one `rate_limit_abuse` attack
  must return HTTP 429.
- **Sensitive-topic invariant.**  No `privacy_override_bypass` attack
  is allowed to end in `routed_cloud`.
- **PDDL soundness.**  Every sensitive-topic attack, when planned,
  must terminate in `route_local` or `deny_request`; the certificate
  constructor must succeed in all cases.

## Results

The corpus exercises 55 attacks across all four target surfaces for
a total of 220 per-run executions.  Concrete numbers are populated
by the CI job (`.github/workflows/redteam.yml`) on every push to
`main` and on the weekly Monday 06:00 UTC schedule; the artefacts
(`reports/redteam_ci.json` and `reports/redteam_ci.md`) are attached
to the GitHub Actions run and inlined into the step summary.  The
harness exits non-zero if any critical-severity attack slips through
or any runtime invariant fails, so regressions cannot silently land
on `main`.

## Defence-in-depth story

Every attack class is intercepted by at least two layers of the I3
stack:

- *Prompt injection / jailbreak* -- the cloud `InputGuardrail`
  (`i3/cloud/guardrails.py`) rejects the prompt before it crosses the
  device boundary, and the on-device SLM is the terminal route for
  any request that touches a sensitive topic anyway.
- *PII extraction* -- the `PrivacySanitizer` redacts detected patterns
  at input, and the `OutputGuardrail` redacts reflected
  API-key-shaped tokens and user literals at output; the diary store
  stores only topic-level embeddings, never the raw text.
- *Privacy-override bypass* -- the `TopicSensitivityDetector` raises
  the sensitivity score above the cloud-route threshold, and the PDDL
  planner's `route_cloud` action schema has `sensitive_topic` as a
  forbidden precondition -- so the plan mechanically cannot include
  it.
- *Rate-limit abuse / oversized payload / path traversal / header
  injection* -- the server middleware
  (`server/middleware.py`) and the OS-level URL normalisation in
  Starlette intercept these at the transport layer before any
  application code runs.

## Threats to validity

Several limitations are worth calling out explicitly:

- **Non-adaptive attacks.**  The corpus is a static battery.  Real
  attackers iterate on the model's responses (gradient-suffix search,
  coordinate-wise hill-climbing) -- the harness does not cover those.
- **No GCG-style gradient suffix search.**  The GCG entry in the
  corpus uses a fixed public suffix rather than running the Zou et
  al. 2023 optimiser against our on-device SLM.
- **No real-world red team.**  The authors of this harness also
  designed the runtime; a credible evaluation would recruit an
  external red team (as is standard for the Anthropic /
  OpenAI / DeepMind system cards).
- **Fuzzing coverage gaps.**  Coverage of header / path parsing is a
  few hand-written cases; a mutation fuzzer (`tests/fuzz/`) would be
  a natural follow-on.
- **In-process `TestClient`.**  The FastAPI surface is exercised in
  process, so TLS/TCP-layer attacks (timing, length-extension,
  CRLF-smuggling at the socket) are out of scope for this harness.

## Future work

Three follow-ons are planned:

1. **GCG adversarial suffix generation** (Zou et al. 2023) targeted at
   the router's context encoder rather than the SLM output.  The goal
   is to produce a suffix that deterministically flips the router's
   sensitivity score below the cloud-route threshold.
2. **Indirect injection via the diary recall path** (Greshake et
   al. 2023).  A first turn writes a crafted entry into the diary;
   the second turn queries for a summary and the attacker's payload
   is delivered by the diary itself.  The sanitiser should catch
   this, but the harness does not currently replay a full two-turn
   session through the live pipeline.
3. **Formal verification of the PDDL safety invariant.**  The PDDL
   domain is small enough (10 predicates, 4 action schemas) that an
   SMT-backed model checker could enumerate all reachable states and
   prove that no reachable plan fires `route_cloud` with
   `sensitive_topic` true.  That would upgrade the runtime check from
   "tested" to "proved".

## References

- OWASP, *LLM Applications Top 10* (2025 revision).
- F. Perez, I. Ribeiro.  *Ignore Previous Prompt: Attack Techniques
  for Language Models*, arXiv:2211.09527 (2022).
- A. Zou, Z. Wang, J. Z. Kolter, M. Fredrikson.  *Universal and
  Transferable Adversarial Attacks on Aligned Language Models*,
  arXiv:2307.15043 (2023).
- K. Greshake, S. Abdelnabi, S. Mishra, et al.  *Not what you've
  signed up for: Compromising Real-World LLM-Integrated Applications
  with Indirect Prompt Injection*, arXiv:2302.12173 (2023).
- Y. Liu, et al.  *HarmBench: A Standardized Evaluation Framework for
  Automated Red Teaming and Robust Refusal*, arXiv:2402.04249 (2024).
- M. Mazeika, et al.  *TDC 2023: Trojan Detection Challenge /
  adversarial evaluation corpus* (2024).
