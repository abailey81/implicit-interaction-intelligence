# I³ — open problems (the issue tracker I'd hand to a teammate)

> Six PR-shaped open problems with clear constraints, acceptance
> criteria, and rough effort estimates.  This is what a github-issues
> board would look like if I'd been working in a team.  Each one is
> framed as something an intern + reviewer could tackle in a single
> two-week iteration.

## #1 — Kirin watch deployment of the encoder

**Background.**  The TCN encoder is INT8-quantised (162 KB ONNX) and
runs in-browser today (`web/models/encoder_int8.onnx`,
`web/js/browser_inference.js`).  We have NOT shipped to a real
wearable.

**Acceptance criteria.**
- [ ] Encoder running on a HarmonyOS / Kirin A2 dev kit via the on-device ONNX runtime.
- [ ] p50 inference latency measured under 50 ms, peak resident memory under 8 MB.
- [ ] Field-test 30 turns of typing on the watch + a paired phone for response generation (cascade arm B / C running on the phone).

**Effort.** 1 week, blocked on access to a Kirin A2 dev kit.

**Why now.**  The JD asks specifically about wearable deployment.  Closing this gap takes I³ from "infrastructure ready" to "shipped".

---

## #2 — Full-corpus SLM v2 retrain

**Background.**  Current `checkpoints/slm_v2/best_model.pt` was trained
on a 300 k-pair subset of the 974 k full corpus
(`data/processed/dialogue/triples.json`).  Two perplexity numbers
are recorded:
- **Training-time held-out (response-only, same-300 k-subset distribution): ≈ 147**
  (`best_eval_loss = 4.987`, persisted in the checkpoint blob).
- **Full-corpus stress-test (response + history tokens, broader sample): ≈ 1725**
  (`reports/slm_v2_eval.md`, run via `training/eval_slm_v2.py`).
The 12 × spread between the two is **distribution shift + history-loss inclusion**,
not over-fitting; the architecture is data-bound at this size.  Iter 51
phase 6 demonstrated `--resume` works (warm-restart confirmed end-to-end);
the warm-restart attempt didn't beat baseline because the issue
isn't epochs, it's data.

**Acceptance criteria.**
- [ ] 4-epoch training run on the full 974 k corpus (3.2× more data).
- [ ] Training-time held-out perplexity below **80** (target: 1.8 ×
      improvement on a 3.2 × data scale-up).
- [ ] Full-corpus stress-test perplexity below **600**.
- [ ] Standalone-SLM regeneration of the demo prompts produces
      coherent multi-clause responses (currently fragmentary).
- [ ] Updated CHANGELOG entry + new artefact under `checkpoints/slm_v2_full/`.

**Effort.**  ~30 hours of GPU wall time on a single A100; ~4 days on
the laptop's RTX 4050.  Multi-day, not multi-hour.

**Why this matters.**  Drops the cascade's reliance on Gemini for
"world-chat" queries that the SLM could in principle handle.

---

## #3 — A/B harness for the cascade routing chip

**Background.**  Every reply ships a `route_decision` dict containing
arm, model, query class, reason, threshold, and per-arm scores.  The
chat UI renders this as a chip with a reasoning tooltip.  We claim
this builds *calibrated trust* (Lee & See 2004), but we have no
evidence.

**Acceptance criteria.**
- [ ] Within-subjects study, n ≈ 20.
- [ ] Half the cohort sees the chip + tooltip; half sees just the
      response text.
- [ ] Measure trust-calibration via the Madsen-Gregor scale; measure
      task completion rate on a fixed 8-turn HMI script.
- [ ] Pre-registered analysis plan; report Cohen's d effect size
      and 95 % CI.

**Effort.**  2 weeks; needs IRB approval, recruitment, and a
session-replay harness.  Already have the engine instrumentation.

---

## #4 — Multilingual cascade

**Background.**  The corpus is English-only; the BPE tokenizer is
byte-level so Cyrillic / CJK won't hard-break, but accuracy isn't
measured.  The Huawei deployment context is global.

**Acceptance criteria.**
- [ ] Three-language smoke (en, zh, es) with ≥ 90 % action-accuracy
      on a 50-turn intent eval per language.
- [ ] Round-trip latency budget held (intent ≤ 1.5 s, chat ≤ 800 ms).
- [ ] Language detector on the front of the cascade (cheap regex /
      n-gram model — no extra ML round-trip).

**Effort.**  3 days; majority is curating the per-language eval set.

---

## #5 — User-state validation study

**Background.**  The 8 adaptation axes (cognitive load, formality,
verbosity, …) are derived from typing biometrics + linguistic
features.  We assert correspondence to user state but never
validated.

**Acceptance criteria.**
- [ ] n = 20 users, within-subjects (rested / tired / under
      time-pressure conditions, randomised order).
- [ ] Self-report on a 7-point Likert per axis after each session.
- [ ] Pearson correlation between live adaptation vector and
      self-report; report which axes are recoverable.
- [ ] Drop axes that don't validate; tighten thresholds on the ones
      that do.

**Effort.**  3 weeks.  IRB-blocked.

---

## #6 — Replace the warm-restart "retrain pending" doc with a real run

**Background.**  `reports/slm_v2_eval.md` honestly notes the
warm-restart attempt didn't beat baseline (5.10 vs 4.99 at the
post-resume eval).  The doc concludes the v2 architecture is
*data-bound*, not epoch-bound.  That conclusion is correct but
the report should *also* show what a real full-corpus run would
look like (#2 above).

**Acceptance criteria.**
- [ ] After #2 lands, replace the "warm-restart experiment notes"
      section in `reports/slm_v2_eval.md` with the real
      full-corpus result.
- [ ] Add the data-scaling curve (held-out PPL vs corpus size) so
      future readers can extrapolate to a 5 M / 10 M corpus.

**Effort.**  4 hours after #2 is done.

---

## How this list reads to a recruiter

I built the project solo.  I could pretend everything is finished.
Instead this is the punch list **I would hand a teammate on day one**:
the gaps I know about, the constraints they're under, what done looks
like, and rough effort.  That's the shape of how I'd work in an HMI
Lab — scope tight, constraints explicit, validation criteria
pre-registered.
