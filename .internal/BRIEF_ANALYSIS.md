# THE_COMPLETE_BRIEF — Structured Analysis

> **Corrections notice (added after external verification).** The brief
> contained several unverified claims about the hiring manager and related
> public figures. Where public evidence (LinkedIn, job posting, company
> records) disagrees with the brief, the public record is used below. The
> original as-summarised content is preserved in later sections for
> historical context; see §0.1 for the corrected facts.
>
> See `ADVANCEMENT_PLAN.md` §1 for the full discrepancy audit.

## 0.1 Verified facts (April 2026 research)

- **Hiring manager's role.** Matthew Riches is at Huawei Technologies R&D
  (UK) Ltd, Greater London. His public LinkedIn describes him as
  specialising in "design with a skew towards future technologies" — i.e.
  a **senior HMI designer**, not an ML engineer. He will weight concept
  clarity, design storytelling, UX prototype quality, and cross-
  disciplinary communication over kernel-level ML micro-optimisation.
- **VocalIQ connection.** His LinkedIn shows he created **VocalIQ branding
  in February 2015**, before Apple's October 2015 acquisition. He was not
  an Apple staff employee; the "VocalIQ → Siri" lineage is correct as
  industry context but must not be framed as his personal Apple product
  history.
- **TextSpaced.** TextSpaced MMO was created by **Celina Riches** (podcast
  interview record confirms this), not Matthew. The surname-match led to
  an earlier misattribution. Do **not** reference TextSpaced as "his"
  project in the interview.
- **Fall Detection, Crash Detection, Hearing Loss Prevention, Apple
  Intelligence, Siri, visionOS.** No public evidence ties Matthew Riches to
  these as an employee; drop from any interviewer-facing material.
- **Actual job description** (verified on huaweiuk.teamtailor.com):
  "desired" skills include **user modeling, human-AI interaction, HCI
  principles, design thinking, concept-driven prototyping, AI product
  development or academic research.** Up to 2-year internship. London
  Research Center.

Use `user modeling` as the primary framing phrase — it's the single most
load-bearing word in the JD for this project's positioning.

## 1. Context & Goals

- **Project:** Implicit Interaction Intelligence (I³). An AI companion system that builds a rich, evolving user model from **implicit behavioural signals** (keystroke dynamics, pauses, edits, linguistic complexity, session dynamics) and adapts responses across four dimensions simultaneously: cognitive load, communication style, emotional tone, accessibility needs.
- **Audience:** Matthew Riches (hiring manager, 10-year Apple AI/UX designer, joined Huawei 2025 to set up the London HMI Lab) and possibly one additional technical interviewer. Vicky Li / Mingwai Li are TA/logistics contacts. Slides due to `matthew.riches@huawei.com` on April 28. Interview 12:00–13:00, April 29, 2026, Meeting Room MR1, 5th floor, Gridiron Building, 1 Pancras Square.
- **Role:** AI/ML Specialist — Human-Machine Interaction (internship, up to 2 years) in the newly-established Huawei London HMI Laboratory. Concept-to-prototype unit (3–5 people), peer to designers and researchers, outputs are prototypes/patents/papers — not shipped features.
- **What the build must prove:** "Yes" to all four screening questions, **simultaneously and in one integrated live demo**:
  1. Traditional ML from scratch (raw PyTorch, no framework)
  2. SLM built/modified without heavy open-source frameworks (no HuggingFace Transformers)
  3. Orchestration pipeline from architectural blueprint
  4. Edge deployment on memory/power-constrained hardware
- **Additionally shows:** Human-AI interaction / user modelling / intelligent UX (the "desired" skill Tamer's CV lacks), privacy-by-design, portability across modalities.
- **Core differentiator:** "The conversation is just the demo vehicle." Device-agnostic behavioural signal extraction, three-timescale user model, edge-cloud orchestration. Explicitly aligns with Smart Hanhan, AI Glasses, HarmonyOS 6, Edinburgh Joint Lab ("sparse or implicit signals" talk, Prof. Malvina Nissim, March 10, 2026).

## 2. Explicit Requirements

### Functional
- **Web demo** at `localhost:8000` via FastAPI backend + web frontend with WebSocket bidirectional protocol.
- **Interaction Monitor:** captures keystroke timing (inter-key intervals, bursts, pauses, backspace ratio, composition time), message-level features, session-level sliding-window features, server-side linguistic features (TTR, Flesch-Kincaid, formality, emoji density, lexicon-based sentiment).
- **32-dim InteractionFeatureVector** in 4 groups of 8 (keystroke dynamics / message content / session dynamics / deviation metrics).
- **TCN user-state encoder** from scratch in raw PyTorch: 4 causal conv blocks, dilations [1,2,4,8], residual + LayerNorm + GELU, global average pooling → 64-dim L2-normalised embedding. Trained with **NT-Xent** contrastive loss on synthetic interaction data.
- **Three-timescale user model:** Instant state, session EMA, long-term EMA. Baseline established after 5 messages. Deviation metrics computed as z-scores. Persists to SQLite (embeddings only, never raw text).
- **Adaptation Controller:** maps user state → `AdaptationVector(cognitive_load, StyleVector(formality, verbosity, emotionality, directness), emotional_tone, accessibility)`. Flattened to 8-dim.
- **Custom SLM** (no HuggingFace): ~8–15M params, Pre-LN transformer, 4 blocks, d_model=256, 4 heads, vocab_size=8000, max_seq_len=256, tied weights, multi-head self-attention + **cross-attention conditioning** (4 conditioning tokens projected from UserStateEmbedding + AdaptationVector), sinusoidal positional encoding, top-k/top-p generation, INT8 quantisation.
- **Custom SimpleTokenizer** word-level with special tokens.
- **Contextual Thompson sampling bandit router** (2 arms: local_slm / cloud_llm) with Bayesian logistic regression + Laplace approximation, 12-dim context (user state summary, query complexity, topic sensitivity, user patience, session progress, baseline established, previous route, previous engagement, time-of-day, message count, cloud latency est, SLM confidence). Reward = composite engagement signal (continuation, sentiment, latency, topic continuity).
- **Privacy layer:** PII regex sanitiser (email/phone/address/date/card), Fernet encryption for embeddings at rest, sensitive-topic classifier (regex with weighted patterns for health/financial/relationship/mental/secrets/family/work) that **forces local processing**.
- **Cloud LLM integration:** Anthropic Claude (`claude-sonnet-4-5`), dynamic system prompt built from AdaptationVector, response post-processing, fallback on error.
- **Interaction diary:** SQL schema with sessions, exchanges (embedding BLOB, adaptation JSON, route, latency, engagement, topic keywords — NO raw text), user_profiles. Session summaries. TF-IDF topic extraction.
- **Edge profiler:** param count, fp32/int8 size, latency (mean/std/p95), peak memory, device compatibility against 4 targets (Kirin 9000 512MB, Kirin 820 256MB, Kirin A2 128MB, Smart Hanhan 64MB).
- **Dashboards:** 4-panel — chat, 2D animated embedding projection with past-state trail, adaptation gauges (4 axes + style subdimensions), routing confidence, engagement score, expandable diary panel.

### Non-functional / Architectural
- Type hints (PEP 604 `X | Y`, `list[int]`), dataclasses/Pydantic, specific exceptions, structured logging (`structlog`), modular + testable, reproducible seeds, checkpoints with metadata (architecture hash, config, metrics, git SHA, wall-clock, hardware), README that gets reviewer to demo in 5 numbered steps.
- **Latency:** TCN <5ms laptop / <20ms Kirin; SLM <100ms/20-token laptop / ~350–450ms Kirin; full pipeline <200ms local / <1500ms cloud.
- **Memory:** SLM ~32 MB INT8, TCN <1 MB INT8, total footprint <50 MB.

### Presentational / Submissions
- 15-slide deck (PDF + PPTX) sent April 28 — subject "Technical Presentation — Implicit Interaction Intelligence (I³)".
- 30-min presentation with live demo at minute ~12, 10-min technical Q&A, 10-min behavioural, 5-min lab overview, 5-min candidate Q&A.
- **Backup demo video** (5 min MP4, two USB drives).
- Printed slides (6-per-page), HDMI+USB-C adapters, USB with slides PDF, photo ID.
- Closing line verbatim: *"I build intelligent systems that adapt to people. I'd like to do that in your lab."*

## 3. Explicit Non-Requirements / Anti-Goals

- **No HuggingFace Transformers** / heavy frameworks for the SLM — raw PyTorch only (Question 2 explicitly excludes this).
- **No pre-trained models as the main SLM** (prototype fallback allowed — see §12.4 options A/B/C; but default is from-scratch).
- **No LoRA adapter** over a large base — defeats on-device deployment and implies a HuggingFace-style base.
- **No raw user text persisted** — architectural, not optional. Diary stores only embeddings + metadata.
- **No running on Raspberry Pi for the demo** — deliberately reversed; laptop-only for demo reliability. Edge credibility preserved via profiling report only.
- **No salary/progression questions** in candidate Q&A.
- **No generic chatbot framing.** Must reframe as "conversation is the demo vehicle for three transferable capabilities."
- **No ML-jargon-lead** in the pitch; lead with experience words.
- **No overselling synthetic data** — acknowledge honestly.
- **No bluffing** under technical probe — structured admission of uncertainty is respected.
- **No pretending to know Matthew's Apple work intimately.**
- **No empty commits;** "Every commit message should read as if Matthew will read it."
- **No full suit;** no t-shirt; smart business casual.
- **Don't start the presentation with architecture diagrams** — start with the experience story.

## 4. Timeline / Phases

Three-phase 17-day build (April 11 → April 27), plus April 28 final prep, April 29 interview.

- **Phase 1 Core ML (Days 1–7):** foundation/features → synthetic data + TCN arch → TCN training → user model + adaptation → tokenizer + dialogue prep → attention/transformer blocks → full SLM + training setup.
- **Phase 2 Integration (Days 8–14):** bandit router → cloud + privacy → pipeline → FastAPI backend → frontend core → frontend dashboards → polish + backup video.
- **Phase 3 Polish (Days 15–17):** edge profiling → slides → rehearsal + slides submission (Day 17 = April 27).
- Slip strategy documented (fewer sessions, fewer epochs, simpler dashboards, fewer slides — never skip rehearsal).

## 5. Demo Expectations — 4-Phase (+ Diary)

- **Phase 1 Cold Start (2 min):** fresh user; 2–3 normal messages; defaults; router favours cloud; after 5 messages baseline establishes.
- **Phase 2 Energetic (1 min):** fast typing, long messages, rich vocab; cognitive load rises, style mirror shifts, complex responses.
- **Phase 3 Fatigue (2 min) — KEY MOMENT:** slow down, shorter messages, simpler words; embedding dot migrates visibly; cognitive load drops; responses become shorter and warmer; router shifts to SLM because latency matters more.
- **Phase 4 Accessibility (2 min) — MOST IMPRESSIVE MOMENT:** very slow typing, many backspaces, short fragments; accessibility gauge rises; system shifts to yes/no questions, simpler vocab, shorter responses. "No settings menu. No toggle. It just adapts." — this is Matthew's values moment.
- **Phase 5 Diary (30 sec):** expand panel, show session summaries; "Raw text never stored."

Expected visible widgets: 2D embedding projection with trail, adaptation gauges (cognitive_load, style formality/verbosity/emotionality, emotional_tone, accessibility), edge vs cloud routing confidence, engagement score, deviation-from-baseline, baseline-established flag.

## 6. Success Criteria

- All four demo scenarios run live, cleanly, no manual intervention.
- TCN: silhouette ≥ 0.5, KNN top-1 ≥ 0.80 on archetype labels; visible PCA clustering.
- SLM: forward/backward/optimiser step clean; loss decreasing epochs 1–2; perplexity <40; conditioning measurably shapes output (auxiliary loss + adaptation-fidelity metrics).
- Bandit: offline replay regret sublinear vs oracle.
- Pipeline: <300 ms local, <1500 ms cloud latency.
- Backup video exists on two USB drives.
- Slides submitted on time; rehearsal under time budget.
- Profiling report proves fits in 50% memory budget of Smart Hanhan-class device after INT8.

## 7. Constraints

- **Privacy:** never persist raw text; encrypt embeddings (Fernet, `I3_ENCRYPTION_KEY`); sensitive topics forced local; PII stripped before any cloud call; logging has no PII; TrustZone noted as production target.
- **Edge budget:** target devices (Kirin 9000 / 820 / A2 / Smart Hanhan 64 MB / 1800 mAh).
- **Latency tolerance:** sub-second for companion interaction.
- **17-day calendar** alongside MSc coursework.
- **No HuggingFace** for SLM.
- **No pretrained** as primary (fallback only if SLM output is very bad by Day 12/14).
- **Laptop-only demo** for reliability.
- **Synthetic training data** for TCN (8 archetypes from Epp 2011, Vizer 2009, Zimmermann 2014).

## 8. Technologies, Frameworks, People to Honour

- PyTorch 2.x, FastAPI, Uvicorn, SQLite, Pydantic, `cryptography.Fernet`, `structlog`, `pytest`, numpy/scipy, Anthropic SDK (`claude-sonnet-4-5`).
- Datasets: **DailyDialog**, **EmpatheticDialogues**, **NRC Emotion Lexicon** (~500 pos + ~500 neg for lexicon sentiment).
- Papers to cite in code + slides: Chen et al. 2020 SimCLR (NT-Xent), Bai/Kolter/Koltun 2018 (TCN), Xiong et al. 2020 (Pre-LN), Vaswani 2017, Russo et al. 2018 (Thompson sampling tutorial), Chapelle & Li 2011, Rabiner 1989 (HMM).
- HCI references: Epp et al. 2011, Vizer 2009, Zimmermann 2014 (keystroke dynamics).
- People/orgs: **Matthew Riches** (former Apple — Fall Detection / Crash Detection / Hearing Loss Prevention / Apple Intelligence / Siri / visionOS / VocalIQ; TextSpaced MMO with >50% blind/partially-sighted players, ARIA compliance). **Prof. Malvina Nissim** (Edinburgh Joint Lab talk cited verbatim). **Eric Xu** ("experience, not computing power"). **Vicky Li, Mingwai Li** (logistics).
- Huawei products: **Smart Hanhan** (Nov 2025, 399 RMB, 1800 mAh, XiaoYi, interaction diary), **AI Glasses** (launched April 21, 2026 — 8 days pre-interview; 30 g; offloads to phone), **HarmonyOS 6 + HMAF + Agentic Core Framework** (large model + small model + inference framework — explicitly mirrors I³'s three tiers), **XiaoYi/Celia**, **PanGu family** (E/P/U/S), **MindSpore Lite** (conversion target — "this would convert to MindSpore Lite and run on the Kirin NPU").
- **L1–L5 Intelligence Framework** (Huawei + Tsinghua IAIR).

## 9. Advanced / Nice-to-Have / Stretch Goals

- Attention visualisations of the cross-attention conditioning.
- Interpretability panel showing which features drove current adaptation.
- Ablation-mode toggle in UI (encoder on/off).
- "What-if" mode showing alternative responses from different adaptations.
- Per-feature contribution heatmaps.
- Auxiliary conditioning-consistency loss.
- Multi-modal extension (voice pace, touch pressure, gaze duration, accelerometer).
- Cross-device federated averaging of long-term profile (HarmonyOS Distributed Data Management).
- Keystroke-biometric user identification for multi-user devices.
- Per-subgroup fairness evaluation surfacing confidence.

## 10. Stylistic / Presentation Requirements

- **README voice:** designer + engineer; "how you say things" framing; cite the role and screening questions; explicit from-first-principles claims.
- **Design language:** dark theme. Palette: `#1a1a2e` bg, `#16213e` panels, `#0f3460` accents, `#e94560` highlights, `#a0a0b0` muted text, `#f0f0f0` active text. Mono for data, sans for content. Smooth animations, easing on embedding dot, progressive gauge fills.
- **Slide structure:** 15 slides, 30 minutes, emotional arc (Hook → Tension → Context → Promise → Architecture → Live demo → Edge → Implications → Honesty → Close). Must include a "What This Prototype Is Not" honesty slide.
- **Vocabulary control:** lead with experience words (adapt, feel, understand, notice, companion, implicit), descend to architecture only when pressed.
- **Tone:** direct, non-redundant, honest about limitations, calibrated claims. Matthew "recognises calibrated thinking and recognises BS at 20 paces."

## 11. Future Work / L1–L5 / Multi-Modal / Federated / Cross-Device

- Prototype "hits ~L2–L3"; future work targets L4–L5.
- Multi-modal port: TCN is modality-agnostic — keystroke → touch pressure → gaze duration → accelerometer.
- Federated long-term profile updates (federated averaging; MindSpore Federated).
- Cross-device HarmonyOS Distributed Data Management sync of 64-dim embedding.
- AI Glasses arm extension: "route to paired smartphone" as a third arm — config change, not arch change.
- XiaoYi integration: I³ = understanding layer, XiaoYi = speaking layer (adaptation vector as side-channel to XiaoYi's prompt).
- Internationalisation: TCN portable, SLM + linguistic features per-language.

## 12. Huawei-Specific Factors

- **Three-tier architecture deliberately mirrors HarmonyOS 6.**
- **INT8 footprint targets Smart Hanhan's 64 MB / 1800 mAh.**
- **AI Glasses mentioned as portability example** (launched 8 days pre-interview — Matthew will be acutely aware).
- **Edinburgh Joint Lab / Nissim** cited explicitly in presentation.
- **Eric Xu quote memorised.**
- **MindSpore Lite** is the stated conversion path (candidate need not have used it — just needs models that *could* convert).
- **Federated learning** as privacy architecture, not bolt-on.
- **Accessibility** is Matthew's core value (TextSpaced) — the accessibility phase is "the single most important moment in the live demo."

## 13. Testing / Benchmarking / Reproducibility / Scientific Rigour

- The 4 demo scenarios as end-to-end tests.
- 30-minute continuous-use WebSocket soak test.
- Every Part-17 fallback tested (kill Anthropic API, kill SLM).
- Unit tests for numerical components (features, attention, bandit Laplace update, tokenizer round-trip).
- Shape tests for every NN component; causality test for TCN.
- Property-based tests noted.
- Reproducible seeds everywhere.
- Checkpoint metadata (architecture hash, config, metrics, git SHA, wall-clock, hardware).
- Benchmark iterations = 100 for latency; warmup = 5.
- Held-out splits: synthetic 80/10/10 for TCN; dialogue 80/20 + 50-example hand-annotated adaptation fidelity set; bandit offline replay.
- Quantisation quality validation (<5% divergence on 100 samples).
- Two-axis SLM eval: perplexity + adaptation fidelity (length/formality/vocab/sentiment match).

## 14. Interviewer-Facing Materials

- 15-slide PDF + PPTX submitted Apr 28.
- Speaker notes per slide (PART 14.1).
- Live demo script (§13.6, PART 17 recovery drills).
- 5-min backup demo video, two USB copies.
- Edge profiling report (Markdown + charts) at `docs/edge_profiling_report.md`.
- `NOTES.md` at repo root documenting deviations from spec (repeatedly required throughout Parts 10–12, 19).
- Prepared Q&A (52 questions across strategic, architecture, privacy, data, deployment, behavioural, depth-probing). Closing line verbatim. 3 candidate questions selected from 6 in PART 6.8.
- README walks reviewer from `git clone` to demo in 5 steps.

---

## 15. Gap Analysis vs Repo

### Already Implemented (skip)
- Top-level repo hygiene: `pyproject.toml`, `Makefile`, `Dockerfile` + dev + prod, `docker-compose*`, `.pre-commit-config.yaml`, `.editorconfig`, `.github/`, `mkdocs.yml`, extensive `README.md`, `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`, `.env.example`, `.devcontainer/`.
- Full package `i3/` with submodules: `interaction/` (monitor, features, linguistic, types), `encoder/` (tcn, blocks, train, inference), `user_model/` (model, deviation, store, types), `adaptation/` (controller, dimensions, types), `router/` (bandit, complexity, sensitivity, router, types), `slm/` (tokenizer, embeddings, attention, cross_attention, transformer, model, generate, train, quantize), `cloud/` (client, prompt_builder, postprocess), `diary/` (logger, summarizer, store), `pipeline/` (engine, types), `privacy/` (sanitizer, encryption), `profiling/` (memory, latency, report), plus `config.py`, `mlops/`, `observability/`.
- Training scripts: `training/` has `generate_synthetic.py`, `prepare_dialogue.py`, `train_encoder.py`, `train_slm.py`, `evaluate.py`.
- Server: `server/` has `app.py`, `websocket.py`, `routes.py`, `routes_health.py`, `middleware.py`.
- Web frontend: `web/index.html`, `web/css/style.css`, `web/js/{app,chat,dashboard,embedding_viz,websocket}.js`.
- Tests: `tests/test_{tcn,slm,bandit,user_model,pipeline,integration,security,property_based}.py`, `conftest.py`.
- Configs: `configs/default.yaml`, `demo.yaml`, `observability.yaml`.
- Demo utilities: `demo/{seed_data,scenarios,profiles}.py`.
- Docs scaffolding: `docs/ARCHITECTURE.md`, `docs/DEMO_SCRIPT.md` (exists with pre-flight checklist and 4-phase narrative), plus `adr/`, `api/`, `architecture/`, `assets/`, `contributing/`, `getting-started/`, `index.md`, `javascripts/`, `operations/`, `overrides/`, `research/`, `security/`, `stylesheets/`.
- Scripts: `scripts/generate_encryption_key.py`, `run_demo.sh`, `setup.sh`.
- Deploy + docker infrastructure (goes beyond brief).

### Missing / Partial

1. **`NOTES.md` at repo root — MISSING.** Brief mandates this repeatedly ("Tamer reads every commit and every note"; deviations from spec must be documented). Not present.
2. **Dedicated `loss.py` for NT-Xent in `i3/encoder/` — MISSING.** Spec listed `encoder/loss.py`. NT-Xent appears inside `encoder/train.py` and `__init__.py` only. Consider extracting for reuse/test.
3. **Sentiment lexicon assets — NOT FOUND as data files.** Brief specifies ~500 pos + ~500 neg words (NRC Emotion Lexicon). Likely inlined in `interaction/linguistic.py` but no `sentiment.py` module, no lexicon `.txt`/`.json` under `data/`. Spec had `interaction/sentiment.py`.
4. **Trained model checkpoints — MISSING.** `checkpoints/encoder/` and `checkpoints/slm/` are **empty** directories. No `best.pt`, `encoder_int8.pt`, `final.pt`, `slm_int8.pt`, or tokenizer JSON. Synthetic dataset not yet generated (`data/synthetic/` empty). Training pipeline exists but has not been executed/committed yet.
5. **Edge profiling report — MISSING.** Spec requires `docs/edge_profiling_report.md` with device-comparison table; `docs/operations/` is empty.
6. **Backup demo video — MISSING.** 5-minute MP4; no video artifact in repo.
7. **15-slide deck (PDF + PPTX) — MISSING.** No `.pdf`/`.pptx` at repo root or in docs.
8. **Sensitive-topic classifier module — PARTIAL.** `router/sensitivity.py` exists but spec also calls for topic sensitivity at pipeline/privacy boundary; verify short-circuit masking in router matches spec (router code has `sensitivity.py` — likely fine, worth confirming).
9. **Pipeline admin endpoints:** spec calls for `POST /admin/reset`, `POST /admin/profiling` — verify presence in `server/routes.py`.
10. **Embedding visualisation 2D projection:** `web/js/embedding_viz.js` present; verify state-cluster colouring matches spec's 8-archetype palette.
11. **Device-class extrapolation methodology** in profiling report (needs `docs/edge_profiling_report.md`).
12. **Accessibility-augmented dialogue data** (paired normal/simplified examples) — presence depends on `prepare_dialogue.py`; verify.
13. **Slide speaker notes** — spec requires per-slide notes; deck missing.
14. **`scripts/record_backup_demo.py` or equivalent** — no record-demo artifact.
15. **Possibly a `src/` vs `i3/` naming deviation** — brief's directory layout (`src/...`) vs actual (`i3/...`). Not a gap per se — allowed deviation but should be noted in a NOTES.md (which is itself missing).

### Hidden / Subtle Requirements (easy to overlook)

- **"Every commit message should read as if Matthew will read it."** (§19.7) — ongoing discipline.
- **Reproducibility seeds** in every stochastic function (§19.2). Worth auditing.
- **Checkpoint metadata must include git commit SHA + hardware profile + wall-clock**, not just weights (§19.2).
- **Causality unit test for TCN** — output at t must not depend on input at t+1 (§18.2 Day 2).
- **Cross-attention verification test** — different conditioning produces different outputs (§18.2 Day 6).
- **Sensitive-topic short-circuit must be enforced at code level, not just policy** — router masks cloud arm probability to zero when classifier flags sensitive (§11.5 / §15.3 Q20).
- **Numerical stability:** use large negatives not `-inf` in attention mask for fp16 compatibility (§18.2 Day 6).
- **No-raw-text enforcement at storage layer**, not switchable (§15.3 Q17).
- **Auxiliary conditioning-consistency loss** recommended (§18.2 Day 7, §15.7 Q52) to ensure the conditioning actually shapes output — otherwise the "from scratch SLM" looks like a chatbot and the cross-attention is invisible.
- **Warmup message count = 5 before baseline** — demo Phase 1 depends on this visibly flipping.
- **Tokenizer must round-trip losslessly on test cases** (§18.2 Day 5).
- **Demo pre-seed:** `demo_user` with ~20 previous sessions, 5–10 diary entries "spanning last week" loaded before interview (§13.6). Present as `demo/seed_data.py` — verify content matches.
- **Battery-only operation test** — 60+ minutes unplugged (§18.4 Day 18).
- **WebSocket auto-reconnect with exponential backoff** on frontend (§17.2).
- **Visible "reset session" button** for demo control (§17.2, §18.3 Day 12).
- **Honesty slide titled exactly "What This Prototype Is Not"** (§8.2 Move 5) — single most important signal of maturity.
- **52 prepared Q&A answers** to rehearse (Part 15).
- **Cite papers in both code comments and slides** (§19.6). Worth auditing docstrings.
- **Anthropic model ID `claude-sonnet-4-5`** (config) — verify not silently replaced with a placeholder.

---

## 16. Tensions / Conflicts in the Brief

1. **From-scratch purity vs SLM quality.** A ~8 M parameter SLM trained from scratch on DailyDialog/EmpatheticDialogues is likely to produce weak text. §12.4 gives Fallbacks A/B/C but the brief also warns against overselling — there is a live trade-off between "honour the screening question literally" and "ship a crisp demo." Answer: ship from-scratch SLM + keep cloud as fallback path; avoid claiming parity.
2. **Edge credibility vs laptop-only demo.** Brief explicitly chooses laptop and compensates with a profiling report. The extrapolated Kirin numbers are "honest caveat" territory — any claim must flag extrapolation.
3. **Privacy architecture vs embedding information content.** §15.3 Q16 concedes embeddings still encode identity signal. Messaging must stress "lossy abstract representation," not "zero-information."
4. **Accessibility detection vs accessibility harm.** §15.3 Q18 admits keystroke-only detection misses screen-reader / voice-control users — adaptation must be opt-out and not replace explicit settings. Tension with the "it just adapts, no toggle" demo narrative.
5. **17-day ceiling vs ambition ceiling.** §19.8 says 17 days is a constraint, not a ceiling on ambition — but §19.3 says cuts must be deliberate and lists what's hardest to drop. Requires judgement.
6. **Three-tier architectural mirror of HarmonyOS 6 vs "this is not accidental" framing.** Must state it deliberately mirrors without sounding like copy-paste.
7. **"Experience-level words" vs technical depth.** Vocabulary control says lead with feel/adapt; but Matthew has 10 years at Apple and will descend the stack hard. Prep for pivots.
8. **Full `from-scratch` narrative vs `cryptography.Fernet` dependency.** Using Fernet is fine (and spec allows) but must be framed as placeholder for TrustZone in production (§15.3 Q16).
9. **Directory structure `src/` (brief) vs `i3/` (repo).** Acceptable deviation but requires a `NOTES.md` entry that does not currently exist.

---

**Bottom line:** The repo is architecturally very close to the specified design — every named submodule exists, tests and infrastructure are rich — but the **training artefacts (checkpoints, synthetic data, tokenizer), the slide deck, the backup video, the edge profiling report, and `NOTES.md` are missing.** Those are the deliverables that convert "a clean codebase" into "a working live demo with a story." The single highest-value next actions: (1) run the training pipeline and commit checkpoints + synthetic data; (2) generate the edge profiling report; (3) build the 15-slide deck with honesty slide and verbatim closing; (4) record the backup demo video; (5) create `NOTES.md` documenting the `src/` → `i3/` rename and any other deviations.
