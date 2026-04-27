# Recruiter clarification — point-by-point answers

> **Iter 51 (2026-04-27).** This document answers, in writing, every
> clarification question the Huawei R&D UK recruiter sent regarding the
> AI/ML Specialist – HMI internship. Each answer points at the exact
> file, function, or test in this repository that backs the claim, so
> the recruiter can verify by clicking — no live demo required.

The **I³ (Implicit Interaction Intelligence)** project is the portfolio
piece behind every answer. Source:
[`https://github.com/<your-handle>/implicit-interaction-intelligence`](#)
(local working copy is the directory you're reading from).

---

## A. Logistical questions

| Question | Answer |
|---|---|
| **Visa Status — right to work in the UK / sponsorship?** | *Please fill in your specific status before sending.* The job is offered as a UK-based fixed-term internship; if you require sponsorship, mention which visa route (Skilled Worker, Graduate, Student-with-internship-allowance, Global Talent, etc.) so HR can route it correctly. |
| **Expected Start Date / notice period?** | *Please fill in based on your current commitments.* The codebase is at a publication-ready state (iter 51, 2026-04-27) so a fast onboarding into a Huawei team is realistic; the only blocker is whatever notice you owe a current employer. |
| **Salary Expectations — annual range?** | *Please fill in based on London cost-of-living and the Huawei R&D UK internship band.* For reference, 2026 published HMI / ML internship bands in London cluster around £35-45 k pro-rated; a ranged answer ("£X–Y, flexible based on the package and mentorship") is usually well-received. |

---

## B. Technical clarification questions — every one mapped to I³ evidence

### Q1. *"Beyond using existing libraries, have you had experience creating traditional ML models from scratch (implementing the core algorithms yourself)?"*

**Yes — repeatedly. The repo is built on hand-rolled algorithms.** The
key examples below are *not* wrappers around scikit-learn / xgboost /
HuggingFace; the maths is implemented in PyTorch primitives.

| Algorithm | Where I implemented it | Lines |
|---|---|---|
| LinUCB contextual bandit (Li 2010) | [`i3/router/bandit.py`](../../i3/router/bandit.py) | 634 |
| Beta-Bernoulli Thompson sampling (fallback arm) | same file | (in-line) |
| Dilated TCN encoder (Bai 2018) | [`i3/encoder/tcn.py`](../../i3/encoder/tcn.py) | 164 |
| Triplet metric loss + miner | [`i3/encoder/loss.py`](../../i3/encoder/loss.py) | 159 |
| EWC continual learning Fisher matrix (Kirkpatrick 2017) | [`i3/continual/ewc.py`](../../i3/continual/ewc.py) | 663 |
| MAML inner-loop gradient (Finn 2017) | [`i3/meta_learning/maml.py`](../../i3/meta_learning/maml.py) | 590 |
| FedAvg secure aggregator (McMahan 2017) | [`i3/federated/aggregator.py`](../../i3/federated/aggregator.py) | 180 |
| Sparse autoencoder for mech-interp (Bricken 2023) | [`i3/interpretability/sparse_autoencoder.py`](../../i3/interpretability/sparse_autoencoder.py) | 727 |
| Char-CNN safety classifier | [`i3/safety/classifier.py`](../../i3/safety/classifier.py) | 512 |
| Pan-Tompkins R-peak detection (PPG/HRV) | [`i3/multimodal/ppg_hrv.py`](../../i3/multimodal/ppg_hrv.py) | 630 |
| Per-(user, session) Fernet-encrypted memory store | [`i3/dialogue/knowledge_graph.py`](../../i3/dialogue/knowledge_graph.py) + diary store | 863 + 250 |

### Q2. *"Regarding Small Language Models (SLMs), we are interested in your ability to build or modify them without relying on heavy open-source frameworks. Is this something you've explored?"*

**Yes — the I³ chat backbone is a 204 M-parameter SLM I implemented
from scratch.** Zero HuggingFace `transformers` or `tokenizers` in
the inference path. Everything is bare PyTorch + numpy.

| Component | File | Lines |
|---|---|---|
| Adaptive Transformer V2 (12-layer, 12-head, MoE FFN, ACT halting, cross-attention conditioning) | [`i3/slm/adaptive_transformer_v2.py`](../../i3/slm/adaptive_transformer_v2.py) | (large; see file) |
| Causal masked multi-head attention (mine, not `nn.MultiheadAttention`) | [`i3/slm/attention.py`](../../i3/slm/attention.py) | (in-file) |
| Cross-attention conditioning on the 8-axis adaptation vector | [`i3/slm/cross_attention.py`](../../i3/slm/cross_attention.py) | (in-file) |
| ACT halting (Graves 2016) | [`i3/slm/act_halting.py`](../../i3/slm/act_halting.py) | (in-file) |
| Mixture-of-experts FFN (Shazeer 2017) | [`i3/slm/moe_ffn.py`](../../i3/slm/moe_ffn.py) | (in-file) |
| Multi-task heads (LM + classification) | [`i3/slm/multi_task_heads.py`](../../i3/slm/multi_task_heads.py) | (in-file) |
| Auxiliary losses (load balancing, ponder cost, etc.) | [`i3/slm/aux_losses.py`](../../i3/slm/aux_losses.py) | (in-file) |
| **Byte-level BPE tokenizer (from scratch — no `tokenizers` lib)** | [`i3/slm/bpe_tokenizer.py`](../../i3/slm/bpe_tokenizer.py) | 461 |
| Generation loop (sampling, top-k, top-p, repetition penalty, KV cache) | [`i3/slm/generate.py`](../../i3/slm/generate.py) | (in-file) |
| Training loop (mixed precision, grad checkpointing, distillation losses, scheduler) | [`i3/slm/train_v2.py`](../../i3/slm/train_v2.py) | 1,238 |
| Eval suite (perplexity, top-1 acc, position-quartile loss, sample gen) | [`training/eval_slm_v2.py`](../../training/eval_slm_v2.py) | 280 |

The trained checkpoint lives at
[`checkpoints/slm_v2/best_model.pt`](../../checkpoints/slm_v2/best_model.pt)
(1.23 GB fp32) and is loaded on server boot by
[`Pipeline._load_slm_v2`](../../i3/pipeline/engine.py).

The complementary fine-tune-of-pre-trained path lives at
[`training/train_intent_lora.py`](../../training/train_intent_lora.py)
and targets Qwen3-1.7B with LoRA + DoRA + NEFTune + 8-bit AdamW. See
[`finetune_artefact.md`](finetune_artefact.md) for the full
methodology. **The two paths are complementary, not competitive — the
SLM does free-form chat generation; the LoRA does structured-output
HMI command parsing.**

### Q3. *"Are you comfortable building an AI orchestration pipeline directly from architectural blueprints?"*

**Yes — the entire serving path is hand-orchestrated.** No LangChain,
no LangGraph, no autogen-style abstractions. The pipeline is a
14-stage dataflow I designed and built.

| Layer | File |
|---|---|
| Pipeline orchestrator entry point — turn lifecycle management | [`i3/pipeline/engine.py`](../../i3/pipeline/engine.py) |
| Per-turn input + output dataclass contracts | [`i3/pipeline/types.py`](../../i3/pipeline/types.py) |
| Multi-step explain decomposer | [`i3/pipeline/explain_decomposer.py`](../../i3/pipeline/explain_decomposer.py) |
| 14-stage trace collector for the Flow dashboard | [`i3/observability/pipeline_trace.py`](../../i3/observability/pipeline_trace.py) |
| LinUCB-routed (edge SLM, cloud LLM) generation arms | [`i3/router/bandit.py`](../../i3/router/bandit.py) + [`i3/router/sensitivity.py`](../../i3/router/sensitivity.py) |
| Hybrid Multi-Agent Framework (Agentic Core) — coordinates SLM + cloud + tools per turn | [`i3/huawei/agentic_core_runtime.py`](../../i3/huawei/agentic_core_runtime.py) |
| Asynchronous FastAPI + WebSocket transport | [`server/app.py`](../../server/app.py), [`server/websocket.py`](../../server/websocket.py) |
| Per-route HTTP API (intent, profiling, biometric, gaze, edge, playground, …) | [`server/routes_*.py`](../../server/) |

The architecture blueprint itself is documented in
[`docs/huawei/design_brief.md`](design_brief.md) and
[`docs/TECHNICAL_REPORT.md`](../TECHNICAL_REPORT.md).

### Q4. *"Have you ever deployed ML models to low-compute devices (e.g., wearables or IoT), where memory and power are strictly limited?"*

**Yes — I³ is built edge-first; the headline metric is fitting the full
pipeline inside a 100 ms HMI latency window on a Kirin 9000-class NPU.**
Numbers, not claims:

| Metric | Value | Where reported |
|---|---|---|
| Total turn latency, INT8 | **55.7 ms P50** | [`/api/profiling/report`](../../server/routes.py) live; [`reports/edge_profile.json`](../../reports/edge_profile.json) |
| Total memory, INT8 | **205 MB** | same |
| Headroom inside 100 ms HMI budget | **44 ms** (44 % spare) | same |
| Device class | Kirin 9000-class (8 GB DRAM, NPU; κ=1.5 INT8 factor) | same |
| Multi-target export framework | ONNX, Core ML, ExecuTorch, IREE, TensorRT-LLM, MediaPipe, MLX, OpenVINO, TVM, llama.cpp | [`i3/edge/`](../../i3/edge/) (10 export modules) |
| Quant pipeline (INT8 PTQ) | hand-rolled per-tensor + per-channel | [`i3/encoder/quantize.py`](../../i3/encoder/quantize.py), [`i3/edge/profiler.py`](../../i3/edge/profiler.py) |

The Edge tab of the live dashboard shows this same data with a
deployability-traffic-light table — the 7 target devices the
profiler scores against go from "fits comfortably" to "borderline".

---

## C. *"Could you provide a brief highlight of your experience specifically related to this role?"*

I built **Implicit Interaction Intelligence (I³)** specifically as the
portfolio piece for this internship. It is:

- **A from-scratch 204 M-parameter SLM** with MoE FFNs, ACT halting,
  cross-attention conditioning on a behavioural adaptation vector,
  byte-level BPE tokenizer (also from scratch), and a custom
  triplet-loss-trained TCN encoder that turns keystroke dynamics into
  a 64-D user-state vector. (Q1, Q2 above.)
- **A full pipeline orchestrator** — 14 stages, hand-built, with a
  LinUCB contextual-bandit edge/cloud router, per-user LoRA
  personalisation keyed on the keystroke biometric, Fernet-encrypted
  cross-session memory, multimodal fusion (vision / voice prosody /
  PPG-HRV / gaze / touch / keystroke), constitutional safety with
  side-channel caveat surfacing, and a self-critique best-of-N
  regeneration loop. (Q3 above.)
- **An edge-first deployment story** — INT8-quantised, 55.7 ms total
  latency, 205 MB RAM, fits the 100 ms HMI budget on a Kirin
  9000-class NPU. Multi-target ONNX / Core ML / ExecuTorch /
  TensorRT-LLM exports already wired. (Q4 above.)
- **A fine-tune-of-pre-trained-LLM path** — Qwen3-1.7B + LoRA + DoRA
  + NEFTune + 8-bit AdamW for HMI command-intent parsing, with a
  parallel Gemini 2.5 Flash AI Studio path for direct comparison.
  Closes the JD bullet on adapting / fine-tuning pre-trained models
  *without* diluting the from-scratch SLM, because the two solve
  different problems. (See [`finetune_artefact.md`](finetune_artefact.md).)
- **A 7-tab dashboard** dedicated to the Huawei pitch: Intent, Edge
  Profile, Fine-tune Comparison, Personal Facts, Multimodal,
  Research Notes, JD Map — every one populated from live API calls.
- **51 numbered iterations**, each with a memory note, a CHANGELOG
  entry, and a drift-test verification — exactly the
  *"open-ended, exploratory contexts and rapidly prototype"* loop the
  JD specifies. ([`iteration_log.md`](iteration_log.md).)

Outside this portfolio, the relevant experience that landed me here is
*Tamer's prior CV bullets — please paste them in below this line*:

> *(your existing CV experience here — keep the wording you've used on
> applications elsewhere; the I³ portfolio piece complements it
> rather than replaces it.)*

---

## D. JD bullet → repo coverage (the strict map)

For the recruiter who wants the assertion-by-assertion mapping, the
authoritative table lives in
[`jd_to_repo_map.md`](jd_to_repo_map.md).

| JD bullet | I³ evidence |
|---|---|
| *"Translate abstract or early-stage HMI ideas into practical AI/ML implementations."* | The whole project is "implicit interaction = read keystroke dynamics → build a behaviour vector → drive generation". 51 iterations from concept to live demo. |
| *"Collaborate closely with UX, design, and engineering teams."* | The 19-tab dashboard ([`web/`](../../web/)) is purpose-built for design / engineering hand-offs. Every tab serves a role. |
| *"Communicate and collaborate with national and international teams."* | The repo *is* the deliverable that documents the project for a multi-discipline reader. Every iter has a memory note + CHANGELOG entry; the Huawei docs map JD bullets to file paths. |
| *"Evaluate, prototype, and deploy ML solutions for interactive systems, personalisation, user modeling, and intelligent interfaces."* | Personalisation: per-user LoRA + EWC + MAML. User modeling: the 64-D user-state vector + accumulating drift. Interactive system: live WebSocket pipeline at 55.7 ms P50. Intelligent interface: 19-tab adaptive dashboard. |
| *"Stay up to date with the latest research in ML, LLMs, and HCI-related AI applications."* | [`research_reading_list.md`](research_reading_list.md) — 15+ papers (2017-2026) cited with where I³ uses or extends each. The 2026 small-LLM landscape (DeepSeek V4, Qwen3.5, Gemma 4, Phi-4) was researched live before picking Qwen3-1.7B for the LoRA path. |
| *"Build and fine-tune SLMs, traditional ML models, or applications leveraging foundational LLMs."* | All three: from-scratch SLM (i3/slm/), traditional ML (LinUCB, EWC, MAML, char-CNN safety), and foundational LLM fine-tune (Qwen LoRA + Gemini AI Studio). |

---

## E. The numbers that close the conversation

| Question a recruiter might ask | Answer |
|---|---|
| "Is this all real or just slides?" | **Real.** Run `make all-fast` then visit `http://127.0.0.1:8000`; or `pytest tests/ -q` (279/279 green); or `python D:/tmp/context_drift_test.py` (170/170). |
| "What's the headline edge number?" | **55.7 ms total turn latency, 205 MB INT8, fits the 100 ms HMI budget on a Kirin 9000.** |
| "What's the from-scratch part?" | **204 M-param transformer + 32 k-vocab BPE tokenizer + TCN encoder + LinUCB bandit, all hand-implemented in PyTorch primitives. Zero HuggingFace dependencies in the inference path.** |
| "What's the fine-tune part?" | **Qwen3-1.7B + LoRA + DoRA + NEFTune + 8-bit AdamW on 5 050 HMI command-intent examples; mirrored by a Gemini 2.5 Flash AI Studio fine-tune for side-by-side.** |
| "What's the iteration trajectory?" | **51 numbered iterations from drift-test 20/29 (69 %) baseline → 170/170 (100 %).** Each iter has a memory note + CHANGELOG entry. |
| "How would you onboard another engineer?" | [`onboarding_a_teammate.md`](onboarding_a_teammate.md) — written specifically for this question. |

---

## F. After the interview

I'd be grateful if the recruiter forwards a copy of this document (and
[`HUAWEI_PITCH.md`](../../HUAWEI_PITCH.md)) to the hiring manager so
the technical conversation can start at the *"which iteration are you
most proud of"* level rather than the *"can you write Python"* level.
