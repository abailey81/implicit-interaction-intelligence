# Onboarding a teammate to I³ in one day

> **Iter 51 (2026-04-27).**  Closes the JD's *"Strong communication
> skills with the ability to collaborate across design and technical
> disciplines"* required bullet by simulating the document I would
> hand to a teammate joining the project.

---

## Audience

You're a new hire / collaborator joining I³ (the project, or — by
extension — the team that picks up I³ at HMI Lab).  This document
gets you from `git clone` to *"I understand the codebase, can run
the demo, can find the code that does X, and have a sense of the
design rationale"* in one workday.

---

## 0900 – 0930 — what you're looking at (30 min)

Read these two files in order:

1. [`HUAWEI_PITCH.md`](../../HUAWEI_PITCH.md) — the elevator pitch +
   how the project maps to the four recruiter screening questions.
2. [`docs/huawei/design_brief.md`](design_brief.md) — the persona and
   interaction principle.  *Why does this project exist?*

If you only had time to read one thing, read the design brief.

---

## 0930 – 1030 — bring up the stack (1 hr)

```bash
git clone <repo>
cd implicit-interaction-intelligence
poetry install                          # ~5 min
python scripts/run_everything.py        # ~10 min
                                        # builds tokenizer, downloads
                                        # synthetic data, trains
                                        # encoder, builds retrieval
                                        # index, exports ONNX, runs
                                        # benchmarks, starts the
                                        # FastAPI server on :8000
```

Open http://127.0.0.1:8000 — the chat UI loads.  Try:

* *"hi"*
* *"what is python"*
* *"give me an analogy"*
* *"my name is YOUR_NAME"*
* *"what's my name"*  (cross-session memory; works across server
  restarts because of the encrypted `user_facts` table)

Click through the dashboard tabs (Chat / State / Trace / Intent /
Edge Profile / Personal Facts / Multimodal / Fine-tune Comparison
/ Research Notes / JD Map).  You're now familiar with the surface.

---

## 1030 – 1130 — the architecture (1 hr)

Read [`docs/architecture/full-reference.md`](../architecture/full-reference.md).

The 90-second mental model:

```
user types
  → KeystrokeFeatureExtractor (32-d per turn)
  → TCN encoder (64-d user state, dilated temporal conv, from scratch)
  → AdaptationController (8-axis adaptation vector)
  → LinUCB router (local SLM vs cloud LLM)
  → AdaptiveTransformerV2 SLM (cross-attention conditioning at every layer)
  → SafetyClassifier
  → DialogueCoref / EntityTracker
  → Postprocess (adaptation rewrites)
  → DiaryStore (encrypted; no raw text)
```

Files in load-bearing order:

* [`server/app.py`](../../server/app.py) → FastAPI + WS entry
* [`i3/pipeline/engine.py`](../../i3/pipeline/engine.py) → 14-stage
  orchestration (the brain)
* [`i3/encoder/tcn.py`](../../i3/encoder/tcn.py) → user-state encoder
* [`i3/slm/model.py`](../../i3/slm/model.py) → custom transformer
* [`i3/router/bandit.py`](../../i3/router/bandit.py) → LinUCB router
* [`i3/safety/classifier.py`](../../i3/safety/classifier.py)
* [`i3/diary/store.py`](../../i3/diary/store.py) → encrypted persistence

---

## 1130 – 1230 — the test infrastructure (1 hr)

Run the tests:

```bash
pytest                                  # 90+ unit tests, ~30 sec
python D:/tmp/context_drift_test.py     # 36 multi-turn scenarios; expect 170/170
python D:/tmp/cross_session_test.py     # cross-session memory; expect 4/4
python training/eval_intent.py          # intent-parser eval; expect ~95% action acc
```

If any of those fails, **stop and read** the relevant memory file
in [`memory/`](../../memory/) before touching code.  The drift test
in particular has 51 iterations of work behind it and a regression
is almost always a mis-understanding of the existing logic.

---

## 1230 – 1330 — lunch / read [`docs/huawei/iteration_log.md`](iteration_log.md)

The 51-iteration trajectory.  You'll see how the system evolved and
what failure modes drive new iterations.

---

## 1330 – 1430 — the recruiter-facing docs (1 hr)

Skim:

* [`docs/huawei/jd_to_repo_map.md`](jd_to_repo_map.md) — every JD
  bullet → file:line
* [`docs/huawei/feature_matrix.md`](feature_matrix.md) — I³ vs Apple,
  Pixel, Galaxy, Pangu, Phi, Qwen, DeepSeek, Gemma, Kimi
* [`docs/huawei/finetune_artefact.md`](finetune_artefact.md) —
  Qwen3-1.7B + LoRA vs Gemini 2.5 Flash side-by-side
* [`docs/huawei/forward_roadmap.md`](forward_roadmap.md) — what's
  next, sequenced for a 6-month internship

These are the documents you'd hand to a recruiter who has 5 minutes
to scan.  They also encode 90 % of the strategic argument for why
this project matters at HMI Lab.

---

## 1430 – 1530 — adopt one issue (1 hr)

By now you should have a sense of where you can contribute.  Pick
one of:

* A failing test (none currently — but add a new scenario to
  `D:/tmp/context_drift_test.py` and fix until 100 %).
* A "what I would build next" item from
  [`docs/huawei/forward_roadmap.md`](forward_roadmap.md).
* A research direction from
  [`docs/huawei/research_reading_list.md`](research_reading_list.md)
  — many are tagged "F-N" / "G-N" in
  [`/.internal/ADVANCEMENT_PLAN.md`](../.internal/ADVANCEMENT_PLAN.md)
  and are sequenced so you can pick one without conflict.

---

## 1530 – 1630 — collaboration norms (1 hr)

Read these in order:

* [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — branch / commit / PR
  conventions
* [`SECURITY.md`](../../SECURITY.md) — vulnerability disclosure
* [`CODE_OF_CONDUCT.md`](../../CODE_OF_CONDUCT.md) — community norms
* [`docs/adr/`](../adr/) — architecture decision records, 10
  numbered docs (`0001-custom-slm-over-huggingface.md`, …)

When you write a PR:

1. Branch off `main` with a name like `feat/intent-vision-modality`.
2. Each commit message starts with a Conventional Commit type
   (`feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`).
3. Update `CHANGELOG.md` `[Unreleased]` section with one bullet per
   substantive change — the project values the audit trail.
4. If your change touches the load-bearing pipeline (engine.py,
   coref.py, retrieval.py), re-run `D:/tmp/context_drift_test.py`
   and link the 170/170 result in the PR description.
5. If your change adds a new memory rule that future iterations
   should respect, write it to `memory/` per the auto-memory schema
   (see existing files for the pattern).

---

## End-of-day checklist

- [ ] Stack runs locally (chat UI loads)
- [ ] Tests pass (`pytest` 90+/90+)
- [ ] Drift test 170/170
- [ ] You've read the design brief, iteration log, and JD map
- [ ] You know which file holds: TCN encoder / SLM / router / safety
      classifier / diary store
- [ ] You know where the 51-iteration trajectory of work is
      documented (this doc, `CHANGELOG.md`, and
      `memory/project_pipeline_quality_guards.md`)
- [ ] You have a draft of your first PR

If any of those is unchecked at end-of-day, page someone (in the
single-user case: file a GitHub issue against yourself with the
specific question — the answer probably already lives in the docs
above).

---

## "What I'd want to know on day 2 onwards"

The questions a real teammate would have, with the doc that answers
them:

| Question | Doc |
|---|---|
| Why is the SLM custom rather than a fine-tuned base? | [`docs/adr/0001-custom-slm-over-huggingface.md`](../adr/0001-custom-slm-over-huggingface.md) |
| Why TCN over LSTM/transformer for the encoder? | [`docs/adr/0002-tcn-over-lstm-transformer.md`](../adr/0002-tcn-over-lstm-transformer.md) |
| Why Thompson sampling over UCB for the router? | [`docs/adr/0003-thompson-sampling-over-ucb.md`](../adr/0003-thompson-sampling-over-ucb.md) |
| What's the privacy contract? | [`docs/adr/0004-privacy-by-architecture.md`](../adr/0004-privacy-by-architecture.md) |
| Why FastAPI? | [`docs/adr/0005-fastapi-over-flask.md`](../adr/0005-fastapi-over-flask.md) |
| Why Poetry? | [`docs/adr/0006-poetry-over-pip-tools.md`](../adr/0006-poetry-over-pip-tools.md) |
| OpenTelemetry plumbing? | [`docs/adr/0007-opentelemetry-for-observability.md`](../adr/0007-opentelemetry-for-observability.md) |
| Fernet key choice? | [`docs/adr/0008-fernet-over-custom-crypto.md`](../adr/0008-fernet-over-custom-crypto.md) |
| Why SQLite (not Redis)? | [`docs/adr/0009-sqlite-over-redis.md`](../adr/0009-sqlite-over-redis.md) |
| Pydantic v2 config rationale? | [`docs/adr/0010-pydantic-v2-config.md`](../adr/0010-pydantic-v2-config.md) |
