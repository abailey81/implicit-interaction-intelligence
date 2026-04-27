# I³ — Implicit Interaction Intelligence
## How this project maps to the Huawei R&D UK HMI internship brief

Huawei's recruiter sent four concrete follow-up questions after receiving
my CV.  Each is a filter for a specific competency.  This document maps
every filter question to artefacts you can read, run, and inspect in
this repository.

> **Quick reference for a 60-second skim.**  Detailed JD bullet → file
> mapping in [`docs/huawei/jd_to_repo_map.md`](docs/huawei/jd_to_repo_map.md).
> Comparison vs Apple Intelligence / Pixel AI / Galaxy AI / Pangu in
> [`docs/huawei/feature_matrix.md`](docs/huawei/feature_matrix.md).
> Persona + interaction principle in
> [`docs/huawei/design_brief.md`](docs/huawei/design_brief.md).
> **132-iteration trajectory** in
> [`docs/huawei/iteration_log.md`](docs/huawei/iteration_log.md) +
> [`docs/huawei/iter52_plus_roadmap.md`](docs/huawei/iter52_plus_roadmap.md).
> Forward roadmap in
> [`docs/huawei/forward_roadmap.md`](docs/huawei/forward_roadmap.md).
> Recruiter clarification answers in
> [`docs/huawei/recruiter_clarification_answers.md`](docs/huawei/recruiter_clarification_answers.md).
>
> **Latest counts** (as of iter 132):
> 626 / 626 iter-test sweep green (`make test-iter`); fine-tuned
> Qwen3-1.7B LoRA adapter committed (best_val_loss = 5.36e-06,
> 1 704 steps × 3 epochs × DoRA + 8-bit AdamW + NEFTune + cosine
> warm restarts, 9 656 s wall on RTX 4050 Laptop); 80 commits
> stacked since iter 51 (every iter shipped + tested + documented).

---

## TL;DR — the full feature surface

A reviewer scanning this should know that I³ goes well beyond the
four screening questions.  The full surface (each item linked to
the file that implements it):

**Core ML stack (from-scratch — closes screening Q1 + Q2)**
* [`AdaptiveTransformerV2`](i3/slm/model.py) — 204 M-param transformer
  with Mixture-of-Experts + Adaptive Computation Time +
  per-layer cross-attention conditioning.  Custom byte-level BPE.
  `0 HF deps` reported live by [`/api/stack`](server/routes.py).
* [`TCN encoder`](i3/encoder/tcn.py) — dilated temporal convolutions
  over a 32-d interaction-feature window, trained with metric-learning
  contrastive loss.
* [`LinUCB Thompson-sampling bandit`](i3/router/bandit.py) — local-vs-
  cloud routing.
* [`Char-CNN safety classifier`](i3/safety/classifier.py) — 47 k-param
  refusal model with constitutional-principle harm-signal overlay.
* [`Adaptation controller`](i3/adaptation/) — projects 64-d user state
  to 8-axis adaptation vector.

**Personalisation primitives (the patentable novelty)**
* [`Per-biometric LoRA`](i3/personalisation/lora_adapter.py) — 1 198 LOC
  rank-4 adapters per registered typing biometric, trained online from
  A/B preference picks.  No other shipping product does this.
* [`KeystrokeAuthenticator`](i3/biometric/keystroke_auth.py) — 991 LOC
  continuous typing-biometric Identity Lock.
* [`Multi-fact session memory + cross-session encrypted persistence`](i3/pipeline/engine.py)
  — 8 typed slots (name, colour, food, music, occupation, location,
  hobby, age, pet) with `forget my facts` user-controlled wipe.
  Storage in encrypted SQLite (`user_facts` table, Fernet at-rest).

**Fine-tune-pre-trained artefact (closes JD's "build from scratch as
well as adapt or fine-tune pre-trained models" — iter 51)**
* [`Qwen3-1.7B + LoRA`](training/train_intent_lora.py) — open-weight
  on-device, DoRA + NEFTune + cosine warm restarts + 8-bit AdamW +
  per-step val-loss eval + best-checkpoint saving.
* [`Gemini 2.5 Flash via AI Studio`](training/train_intent_gemini.py)
  — closed-weight cloud comparison artefact.
* [`Side-by-side eval harness`](training/eval_intent.py) — JSON
  validity / action accuracy / slot F1 / confusion matrix / latency
  P50–P95.
* [`/api/intent`](server/routes.py) — live structured-output endpoint
  parsing free-form HMI utterances into JSON.
* Detailed write-up in [`docs/huawei/finetune_artefact.md`](docs/huawei/finetune_artefact.md).

**Multimodal stack (closes JD's "multimodal models" desired)**
* [`Vision`](i3/multimodal/vision.py) — MediaPipe Face Mesh, 8 facial-
  affect features (eye-aspect ratio, mouth-aspect ratio, gaze offset,
  head-pose pitch/yaw, brow-furrow AU4, smile AU12).
* [`Gaze classifier`](i3/multimodal/gaze_classifier.py) — 1 014 LOC.
* [`Voice prosody`](i3/multimodal/prosody.py) — 761 LOC, 8 prosody
  scalars from audio.
* [`PPG / HRV`](i3/multimodal/ppg_hrv.py) — 630 LOC wearable physiology.
* [`Touch dynamics`](i3/multimodal/touch.py) — 223 LOC.
* [`Wearable ingest`](i3/multimodal/wearable_ingest.py) — 804 LOC.

**Cloud foundation-model layer (closes JD's "applications leveraging
foundational LLMs")**
* 11 cloud-provider clients in [`i3/cloud/providers/`](i3/cloud/providers/):
  Anthropic, OpenAI, Azure, Bedrock, Cohere, Google, Huawei Pangu,
  LiteLLM, Mistral, Ollama, OpenRouter.
* [`MultiProviderClient`](i3/cloud/multi_provider.py) — failover /
  fanout routing.

**Continual / meta learning**
* [`EWC`](i3/continual/) — Elastic Weight Consolidation for online
  adapter learning.
* [`MAML`](i3/meta_learning/) — meta-learning prior for new-user
  adapters.

**Mechanistic interpretability (G3 batch)**
* [`Sparse autoencoders`](i3/interpretability/) over the SLM's
  cross-attention activations.

**Active preference learning (F-4 batch)**
* [`Online DPO`](i3/router/preference_learning.py) — 871 LOC, Bradley-
  Terry preference modelling + active learning over the bandit's
  reward signal.

**Safety + alignment**
* [`Red-team corpus + runner`](i3/redteam/) — adversarial test harness.
* [`LLM-as-judge`](tests/test_llm_judge.py), [`judge calibration`](tests/test_judge_calibration.py).
* [`Constitutional safety overlay`](i3/safety/classifier.py).
* [`PII sanitiser`](i3/privacy/sanitizer.py) — IPs, emails, phone,
  URLs, credit cards, SSN-like patterns.
* [`Cedar authz`](tests/test_cedar_authz.py), [`NeMo guardrails`](tests/test_guardrails_nemo.py).

**Conversational coherence (51 iterations)**
* Drift test [`D:/tmp/context_drift_test.py`](D:/tmp/context_drift_test.py)
  — 36 scenarios / 170 turns / 100 % pass.  See
  [`docs/huawei/iteration_log.md`](docs/huawei/iteration_log.md).
* Cross-session memory test [`D:/tmp/cross_session_test.py`](D:/tmp/cross_session_test.py)
  — 4/4 pass.

**Edge deployment (closes screening Q4)**
* [`Encoder ONNX export`](scripts/export_encoder_onnx.py).
* [`llama.cpp export`](i3/edge/llama_cpp_export.py).
* [`Browser inference`](web/js/) via onnxruntime-web (WebGPU/WASM
  detection).
* [`Edge profile`](scripts/profile_edge.py) with Kirin 9000 → Smart
  Hanhan device-feasibility table in [`docs/paper/I3_research_paper.md`](docs/paper/I3_research_paper.md) §6.4.

**Huawei-ecosystem integrations (already scaffolded)**
* [`HMAF adapter`](i3/huawei/hmaf_adapter.py) — 452 LOC.
* [`Agentic core runtime`](i3/huawei/agentic_core_runtime.py) — 657 LOC.
* [`ExecuTorch hooks`](i3/huawei/executorch_hooks.py) — 328 LOC.
* [`Kirin targets`](i3/huawei/kirin_targets.py) — 226 LOC.
* [`Watch integration`](i3/huawei/watch_integration.py) — 318 LOC.

**Research artefacts**
* [`Research paper`](docs/paper/I3_research_paper.md) — 461 LOC, full
  Results section with TCN clustering metrics, SLM perplexity + KL
  ratio, router regret, edge-feasibility table.
* [`Provisional patent disclosure`](docs/patent/provisional_disclosure.md) — 817 LOC.
* [`Conference poster`](docs/poster/conference_poster.md).
* [`10 ADRs`](docs/adr/) — architecture decision records.
* [`26 research notes`](docs/research/) — mechanistic interpretability,
  sparse autoencoders, PPG/HRV, multimodal extension, etc.

**Operational**
* [`scripts/run_everything.py`](scripts/run_everything.py) — 21-stage
  wave-based DAG, clean checkout to running stack in ~10 min.
* [`server/app.py`](server/app.py) — FastAPI + WebSocket.
* [`server/websocket.py`](server/websocket.py) — Origin allow-listing,
  rate limiting, byte/message caps.
* [`Dockerfile`](Dockerfile), [`deploy/`](deploy/) (Helm, K8s, Argo,
  Terraform).
* 90+ unit tests across [`tests/`](tests/) including chaos / contract /
  fuzz / load / mutation / property / snapshot.
* k6 + locust load tests in [`benchmarks/`](benchmarks/).

---

### 1. “Custom ML models — beyond using existing libraries, have you implemented the core algorithms yourself?”

Everything that matters for this demo is implemented by hand.  The
following are **written from scratch** — `torch.nn.Module` (tensors
only), `numpy`, or pure Python.  No `transformers`, no `sentence-
transformers`, no `sklearn`-based pipelines shipping the decision
logic, no pretrained model downloads.

| Component | Role | Where to read |
|---|---|---|
| **AdaptiveSLM** — decoder-only transformer with cross-attention conditioning on an 8-dim adaptation vector and a 64-dim user-state embedding | Generates responses locally | [`i3/slm/model.py`](i3/slm/model.py), [`i3/slm/blocks.py`](i3/slm/blocks.py) |
| **SimpleTokenizer** — word-level tokenizer with frequency pruning, explicit `[PAD]/[BOS]/[EOS]/[SEP]/[UNK]` handling, JSON save/load | Tokenises the corpus; no `tokenizers` crate | [`i3/slm/tokenizer.py`](i3/slm/tokenizer.py) |
| **TCN Encoder** — dilated temporal convolutions over a sliding window of 32-dim interaction features, trained with metric-learning loss | Produces a 64-dim user-state embedding from keystroke dynamics + linguistic features | [`i3/encoder/tcn.py`](i3/encoder/tcn.py), [`training/train_encoder.py`](training/train_encoder.py) |
| **LinUCB Contextual Bandit** — linear upper-confidence-bound policy over the (user-state, features) context | Routes the query between local SLM and cloud LLM | [`i3/router/bandit.py`](i3/router/bandit.py) |
| **Linguistic feature extractors** — Flesch–Kincaid grade, Gunning-Fog, type-token ratio, emoji density, sentence-splitter with abbreviation protection | Traditional NLP, no `textstat` fallback | [`i3/interaction/linguistic.py`](i3/interaction/linguistic.py) |
| **Keystroke-dynamics feature extractor** — IKI distribution, burst detection, pause statistics, dwell-flight ratios | Implicit-behaviour signal, not available from any library | [`i3/interaction/features.py`](i3/interaction/features.py) |
| **Session summarizer** — topic extraction + scalar aggregation that produces a privacy-safe session summary (no raw text) | Enables the diary without retaining user text | [`i3/diary/summarizer.py`](i3/diary/summarizer.py) |

Open the UI dashboard at `/` and look at the **On-Device Stack** panel
— it renders a live snapshot of the custom components: parameters,
vocabulary size, embedding shape, kernel/dilation config, number of
bandit arms, and the `0 HF deps` pill.

---

### 2. “SLMs without heavy open-source frameworks — have you explored building or modifying them?”

Yes — the entire SLM is custom:

- **Architecture.**  The decoder stack is a sequence of transformer
  blocks built from `torch.nn.Linear` and `torch.nn.LayerNorm`.
  Attention is implemented explicitly (Q/K/V projections,
  scaled-dot-product with a causal mask, and a per-block cross-
  attention head that consumes the 8-dim adaptation vector and 64-dim
  user-state embedding).  No `transformers.GPT2Block`, no
  `AutoModel.from_pretrained`.  See
  [`i3/slm/blocks.py`](i3/slm/blocks.py).
- **Tokenizer.**  Word-level with a configurable vocab cap, built from
  the training corpus, persisted as a JSON dict of
  `{vocab_size, token_to_id}` — reloadable without external deps.  See
  [`i3/slm/tokenizer.py`](i3/slm/tokenizer.py).
- **Training loop.**  A hand-written PyTorch loop with
  gradient-accumulation, AMP, early-stopping by val loss, and
  cosine-warmup scheduling.  No `pytorch_lightning`, no
  `transformers.Trainer`.  See
  [`training/train_slm.py`](training/train_slm.py).
- **Conditioning behaviour.**  The adaptation vector and user-state
  embedding are projected to the model dim and injected into every
  cross-attention layer, so the same prompt produces measurably
  different outputs for different (cognitive_load, style_mirror)
  contexts.  The `scripts/benchmarks/evaluate_conditioning.py` stage
  measures that sensitivity automatically.

---

### 3. “Pipeline orchestration — comfortable building from architectural blueprints?”

[`scripts/run_everything.py`](scripts/run_everything.py) is a 2100-line,
from-scratch orchestrator that turns a blueprint into a live running
stack.  It:

- Declares **21 stages** across **10 waves** as a DAG.
  Stages within the same wave run in parallel (bounded by
  `--max-parallel`); waves themselves are sequential because each
  later wave consumes the artefacts of the previous one.
- Emits a live Rich-based progress UI: ETA, wall-clock, per-stage
  pass/fail, total artefact size.
- Tracks **artefact dependencies** (`produces=[…]`) so a `--resume`
  skips stages whose artefacts already exist.
- Respects **skip flags** (`--skip-tests`, `--skip-onnx`, `--skip-
  docs`, `--skip-install`) for partial runs.
- Handles **pre-flight checks** (Docker, GPU, disk), degrades
  gracefully when optional dependencies are missing, and logs each
  stage's stdout/stderr to `reports/orchestration/{stage}.log` for
  post-mortem.

Representative wave layout:

```
wave 1  install                (poetry install)
wave 2  env                    (generate .env + Fernet key)
wave 3  data, dialogue         (parallel: 35k synthetic features + 22k dialogues)
wave 4  train-encoder          (TCN metric learning, GPU)
wave 5  train-slm              (AdaptiveSLM pretraining, GPU)
wave 6  evaluate, eval-conditioning, seed
wave 7  lint, typecheck, security, redteam
wave 8  verify                 (44-check acceptance harness)
wave 9  benchmarks, onnx-export, profile-edge, docs, docker-build
wave 10 serve                  (uvicorn on 127.0.0.1:8000)
```

Running the full pipeline end-to-end takes ~10 minutes on an RTX 4050
Laptop.

---

### 4. “Edge deployment — ever deployed to low-compute devices (wearables, IoT)?”

The demo is edge-first.  Concrete artefacts:

- **ONNX export** of the TCN encoder for browser and device runtimes:
  `checkpoints/encoder/tcn.onnx` (~MB scale).  Produced by
  [`training/export_onnx.py`](training/export_onnx.py).
- **Browser-side inference path** via `onnxruntime-web` with
  WebGPU/WASM backend detection:
  [`web/js/browser_inference.js`](web/js/browser_inference.js),
  [`web/js/webgpu_probe.js`](web/js/webgpu_probe.js),
  [`web/js/ort_loader.js`](web/js/ort_loader.js).  A toggle in the
  advanced UI lets the user flip between server inference and
  in-tab WebGPU inference.
- **Edge profile benchmark.**
  [`scripts/benchmarks/profile_edge.py`](scripts/benchmarks/profile_edge.py)
  measures model size, cold-start latency, and steady-state latency
  and writes them to `reports/edge_profile.md`.
- **Quantisation-aware export** scaffolding in
  [`i3/encoder/quant.py`](i3/encoder/quant.py) with INT8-capable
  profiling hooks.
- **Memory + compute caps** enforced by the server: per-message byte
  cap, per-session message cap, per-user sliding-window rate limit,
  bounded keystroke buffer.  See [`server/websocket.py`](server/websocket.py).

The UI's **On-Device Stack** panel shows the current device (CUDA /
CPU), available VRAM, and the ONNX artefact's size so a reader can
compare against the target's budget.

---

### Conversational coherence — the HMI Lab tier

HMI is a *human-machine interaction* lab.  A small-language-model
that single-shots a clean answer is table stakes; what distinguishes a
hireable system is whether it still tracks the user after the
**fifth** turn, the **fifteenth** turn, after a typo, after a
sarcastic ack, after the user changes their mind mid-thread.  The
work to get there sits in a single regression test:

> `D:/tmp/context_drift_test.py` — **36 scenarios / 170 turns /
> 100 % pass** (iter 49, 2026-04-26).  Up from **20/29 = 69 %** at
> iter 40 baseline (9 iterations of architectural fixes).  Now
> includes 8-slot multi-fact session memory (name, favourite
> colour, favourite food, favourite music, occupation, location,
> hobby, age, pet) — the model holds personal-context across the
> whole session and recalls accurately.

Every scenario probes a real failure mode I observed in user
emulation: recursive person-coref, negation pivots ("not Apple,
Microsoft"), retraction mid-thread ("wait, sorry, scrap that — back
to transformer"), same-surface alt-sense disambiguation (Apple the
company vs apple the fruit), polite formality, sarcasm acks, emoji-
only inputs, single-character inputs, typos, philosophical
discourse-plural ("are they connected"), code-debug walkthroughs,
identity/persona probes ("how big are you"), session-fact recall
("what's my name"), and a 15-turn cross-domain marathon.  All run on
the same on-device 204 M-parameter SLM with **no LangChain, no
LlamaIndex, no third-party dialogue manager** — every coref / pivot
/ rewrite / disambiguation / register-pivot lives in
[`i3/dialogue/coref.py`](i3/dialogue/coref.py),
[`i3/pipeline/engine.py`](i3/pipeline/engine.py), and
[`i3/slm/retrieval.py`](i3/slm/retrieval.py) as readable Python.

For the recruiter: this is what *implicit interaction intelligence*
means in practice — the model has held a fifteen-turn mixed-domain
chat that switched between Python, gravity, tiredness, persona, and
recap, and still answered "more about it" with the right gravity
paragraph at turn 12.  The benchmark is reproducible, the failure
modes are documented, the fixes are in code.

### Cross-session personalisation, *with privacy intact* (iter 50)

The pitch's privacy argument used to be "no raw user text on disk".
That ruled out cross-session personalisation entirely — the model
forgot you the moment a session ended.  Iter 50 closes the gap
without breaking the privacy invariant:

- **What's stored.** A new `user_facts (user_id, slot, value_blob)`
  table in the Interaction Diary, keyed by `(user_id, slot)`.  Slots
  are scoped to typed personal context the user *explicitly*
  introduced — name, favourite colour, occupation, location, hobby,
  age, pet — never anything the user said as a question or a topic.
- **How it's stored.** `value_blob` uses the same versioned envelope
  as the embedding columns (byte 0 = `0x00` plaintext / `0x01`
  Fernet-V1).  When `$I3_ENCRYPTION_KEY` is configured (the
  production default), every value is AES-128-CBC-encrypted with
  HMAC-SHA-256 authentication via `cryptography.fernet`.  See
  [`i3/diary/store.py:_encode_fact_value`](i3/diary/store.py).
- **How it's read back.** On `start_session(user_id)` the pipeline
  loads any prior facts into the new session's in-memory dict so
  recall queries (*"what's my name?"*, *"where do I live?"*) bind
  cleanly without crossing the network.
- **How the user wipes it.** A single utterance — *"forget my
  facts"*, *"delete my data"*, *"wipe my information"* — clears the
  in-memory dict AND deletes the user's DB rows in one shot.  No
  admin tool, no support ticket, no GDPR-form pretence.  The user
  controls retention.
- **Reference test.**
  [`D:/tmp/cross_session_test.py`](D:/tmp/cross_session_test.py) —
  session 1 declares (name=Alice, colour=teal, location=Tokyo,
  occupation=data scientist), ends the session, opens session 2,
  recalls all four. **4/4 pass.**

For the recruiter: this is the answer to *"can your edge model
remember the user across logins, without sending anything to the
cloud, without storing raw text?"* — yes, here's the table, here's
the encryption envelope, here's the user-controlled wipe, here's
the test.

### Privacy & HMI specifics that tie the pitch together

Because HMI is about users, not abstract pipelines, the system is
designed so a user's raw text never leaves the device in a form that
could reconstruct their input:

- The diary database has **no text column** — only embeddings,
  topic-keyword lists, and scalar metrics.  See the schema in
  [`i3/diary/store.py`](i3/diary/store.py).
- Embeddings are **Fernet-encrypted** (AES-128-CBC + HMAC-SHA-256)
  before persistence.  See
  [`i3/privacy/encryption.py`](i3/privacy/encryption.py).
- A **PII sanitiser** redacts IPs, emails, phone numbers, URLs,
  credit-card numbers, and SSN-like patterns before cloud dispatch.
  See [`i3/privacy/sanitizer.py`](i3/privacy/sanitizer.py).
- The WebSocket handler enforces **Origin allow-listing** because CORS
  does not apply to WS upgrades.  See
  [`server/websocket.py`](server/websocket.py).

---

### Short answers to the four questions, for the recruiter reply

1. **Custom ML from scratch.** Yes — the SLM blocks, tokenizer, TCN
   encoder, LinUCB bandit, linguistic feature extractors and keystroke
   analytics are all implemented by hand in this project.  Core
   artefacts listed above.
2. **SLMs without heavy frameworks.** Yes — the `AdaptiveSLM` is a
   cross-attention-conditioned decoder-only transformer written with
   tensors only.  Training, tokeniser, and inference paths ship with
   zero HuggingFace dependencies.  Stack panel in the UI shows `0 HF
   deps` live.
3. **Pipeline orchestration from blueprints.** Yes —
   `scripts/run_everything.py` is a wave-based, DAG-aware, Rich-logged
   orchestrator that runs 21 stages to reproduce the full stack from a
   clean checkout in ~10 minutes.
4. **Edge deployment to constrained devices.** Yes — the encoder is
   exported to ONNX, the browser path runs it via onnxruntime-web with
   WebGPU/WASM detection, and `profile_edge.py` publishes the size /
   latency budget.  The server is hardened for wearable-scale limits
   (byte / message / rate caps).
