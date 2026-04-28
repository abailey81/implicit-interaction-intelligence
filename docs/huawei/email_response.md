# I³ — direct response to Huawei R&D UK's pre-screen questions

> The recruiter asked five concrete questions in the pre-screen.
> Each one is answered here with evidence from the repository so a
> reviewer can verify the claim without taking my word for it.

## 1.  *"Beyond using existing libraries, have you had experience creating traditional ML models from scratch (implementing the core algorithms yourself)?"*

**Yes — five from-scratch implementations live in this repo:**

| Component | Where | Lines | Algorithm |
|---|---|---:|---|
| **AdaptiveTransformerV2 SLM** | [`i3/slm/adaptive_transformer_v2.py`](../../i3/slm/adaptive_transformer_v2.py) | ~ 900 | Pre-LN decoder, MoE FFN (Shazeer 2017), ACT halting (Graves 2016), per-layer cross-attention conditioning. No `transformers` import. |
| **Byte-level BPE tokeniser** | [`i3/slm/bpe_tokenizer.py`](../../i3/slm/bpe_tokenizer.py) | ~ 460 | Sennrich 2015 BPE with byte-level fallback. No `tokenizers` library. 32 k vocab trained on 977 k pairs. |
| **TCN user-state encoder** | [`i3/encoder/blocks.py`](../../i3/encoder/blocks.py) + [`loss.py`](../../i3/encoder/loss.py) | ~ 320 | 4-layer dilated causal TCN, NT-Xent contrastive loss on augmented session pairs. |
| **LinUCB contextual bandit** | [`i3/router/bandit.py`](../../i3/router/bandit.py) | ~ 280 | Bayesian logistic regression per arm, Laplace-approximated posterior, Newton-Raphson MAP refit. |
| **Char-CNN safety classifier** | [`i3/safety/classifier.py`](../../i3/safety/classifier.py) | ~ 180 | Constitutional-AI-shaped char-level CNN; Bai et al. 2022. |

To verify any of these, grep the file for `from transformers` / `from sklearn`
/ `import tokenizers` — none of them appear.

## 2.  *"Regarding Small Language Models (SLMs), we are interested in your ability to build or modify them without relying on heavy open-source frameworks. Is this something you've explored?"*

**Yes — the SLM is end-to-end from scratch in pure PyTorch.**

- **204 M parameters**, `d_model=768`, 12 layers × 12 heads, MoE-2 +
  ACT-halting + per-layer cross-attention.  Source: 
  [`i3/slm/adaptive_transformer_v2.py`](../../i3/slm/adaptive_transformer_v2.py).
- **Training loop** is in [`i3/slm/train_v2.py`](../../i3/slm/train_v2.py) —
  cosine LR with warmup, 8-bit AdamW (bitsandbytes), bf16 mixed
  precision, gradient checkpointing.  No `accelerate` /
  `pytorch-lightning` / `transformers.Trainer`.
- **Tokeniser** is the BPE in §1 above.  Trained from the corpus
  with `training/build_intent_dataset.py`-style data prep scripts.
- **Inference** is `model.forward(input_ids, conditioning=...)` —
  raw PyTorch, no `model.generate(...)` wrapper.  See
  `Pipeline._generate_response_inner` in
  [`i3/pipeline/engine.py`](../../i3/pipeline/engine.py).

**Honest caveat.**  HuggingFace's `transformers` *is* used for one
arm of the cascade — the **Qwen 1.7 B + LoRA intent parser**.  That
arm exists *because* the JD also asks about fine-tuning pretrained
models.  The from-scratch SLM and the LoRA-fine-tuned Qwen are two
separate arms in the cascade, deliberately, with different roles.

## 3.  *"Are you comfortable building an AI orchestration pipeline directly from architectural blueprints?"*

**Yes — I built one.  Eight commits this week alone tightened it.**

[`i3/pipeline/engine.py`](../../i3/pipeline/engine.py) (≈ 8 000 LOC)
implements a **14-stage hand-orchestrated cascade**:

```
1. Intake / sanitise → 2. Coref / topic anchor → 3. Encode (TCN)
   → 4. Adapt (8-axis vector) → 5. Smart Router (multi-signal scorer)
   → 6. Command gate → 7. Qwen LoRA intent → 8. Gemini intent backup
   → 9. Retrieval (KG + topic-consistency gate)
   → 10. SLM forward + on-topic critic → 11. Cloud chat fallback
   → 12. Tool branches (diary, math, refusal, …)
   → 13. Adaptation rewrite → 14. Side-effect dispatcher (actuators)
```

**Per-turn `route_decision` is a structured dict** containing:
`{arm, model, query_class, reason, threshold, score, arms_used,
smart_scores}` — visible in the chat chip + tooltip.  The recruiter
watching the demo sees the routing math live.

**The smart router is a multi-signal scorer**, not a regex.  Six
deterministic signals (greeting / cascade-meta / system-intro /
question-shape / KG-anchor / system-topic) feed weighted
combination rules; highest-scoring class wins; all scores exposed
on the chip.  See `_smart_score_arms` in `engine.py`.

**Verification:** 22 / 22 routing classifications match expectation
on `D:/tmp/precision_smoke.py`; live `tool:intent` round-trips
end-to-end with **real actuator side-effects** — `set timer for 30
seconds` schedules an asyncio task, the timer fires 30 s later, and
a gold-pulse banner drops into the chat.

## 4.  *"Have you ever deployed ML models to low-compute devices (e.g., wearables or IoT), where memory and power are strictly limited?"*

**Partial — the encoder ships to the edge today; the SLM does not yet.**

**What's deployed today (live, demonstrable in the demo):**
- **Encoder INT8 ONNX**: 162 KB on disk, **runs in the user's browser
  tab** via ONNX Runtime Web (WASM / WebGPU).  See
  [`web/models/encoder_int8.onnx`](../../web/models/encoder_int8.onnx),
  [`web/js/browser_inference.js`](../../web/js/browser_inference.js).
- **Toggle is in the State tab** under "Edge inference · Run on this
  device."  Flip it ON, open Chrome DevTools → Network panel, type
  a message — **zero `/api/encode` requests fire**.  The 32-d
  feature vector hit the encoder client-side.  Keystrokes never
  left the page.
- **Profile** ([`reports/edge_profile_2026-04-28.md`](../../reports/edge_profile_2026-04-28.md)):
  p50 460 µs, p95 637 µs, 2 176 encodes/sec.  12.5 × under the
  Kirin A2 watch's 2 MB encoder RAM budget.  Parity vs FP32: MAE
  0.00055.

**What's not deployed yet:**
- **The 204 M SLM has not run on a Kirin watch / phone.**  ONNX
  export plumbing exists ([`i3/slm/onnx_export.py`](../../i3/slm/onnx_export.py))
  and an INT8 quantisation spec is in
  [`reports/edge_profile_2026-04-28.md`](../../reports/edge_profile_2026-04-28.md),
  but I haven't run on-device latency / power profiling.  This is
  the first item in
  [`docs/huawei/open_problems.md#1`](open_problems.md) — what I'd
  close in week 1 of the internship if given a Kirin dev kit.
- **No real-world wearable battery / thermal data.**  All numbers
  above are laptop-CPU bounds.

I'd rather be honest about this gap than oversell.  The
infrastructure is real; the field deployment is the next step.

## 5.  *"Could you provide a brief highlight of your experience specifically related to this role?"*

**The full project is the answer; here's what it demonstrates against the JD:**

| JD bullet | Evidence |
|---|---|
| Build models from scratch | §1 — five hand-written implementations |
| Fine-tune pre-trained | §2 — Qwen 1.7B + LoRA, val_loss 5.4 × 10⁻⁶ |
| Pipeline orchestration | §3 — 14-stage cascade, `route_decision` per turn |
| Edge deployment | §4 — INT8 encoder live in-browser, watch budget held |
| User modeling | TCN encoder + 8-axis adaptation vector, [HCI design brief](hci_design_brief.md) |
| Context-aware systems | coref-aware cascade + topic-consistency gate + conversation-history-aware cloud arm |
| HCI principles | [HCI design brief](hci_design_brief.md) — Strayer / Wobbrock / Lee references |
| Concept-driven prototyping | 88 commits, iter docs ([`docs/huawei/iter51_summary.md`](iter51_summary.md)), [open problems list](open_problems.md) |
| Communication / cross-discipline | This document, [presenter cheat sheet](PRESENTER_CHEAT_SHEET.md), [About tab Q&A](http://127.0.0.1:8000/#about) |

## Logistics

- **Visa:** [Tamer's individual answer]
- **Earliest start:** [Tamer's individual answer]
- **Salary expectations:** [Tamer's individual answer]
