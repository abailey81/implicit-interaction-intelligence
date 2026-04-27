# I³ — 30-min interview cheat sheet

> One page. The only thing you need open during the live demo.
> Last refreshed: iter 51 phase 7 (2026-04-27).

## The 12-minute live demo (memorise the flow)

Open browser to `http://127.0.0.1:8000/` — nav defaults to **Simple** (5 tabs).

| Time | Step | What you say | What you click / type |
|---:|---|---|---|
| 0:00 | Open Chat tab | "I³ is implicit interaction intelligence — an HMI prototype where a from-scratch SLM conditions on *how* you type, not just *what*. Three-arm cascade: from-scratch SLM, Qwen 1.7B + LoRA intent parser, optional Gemini cloud." | (Chat tab is default) |
| 0:30 | Plain chat | "First arm: a 204 M custom transformer trained from scratch — no HuggingFace, no Llama, no Phi. Watch it answer." | Type **"hi how are you"** → Send |
| 2:00 | In-domain command | "Second arm. The same input lands on Qwen + LoRA — a deterministic JSON intent parser. Notice the green chip: backend `qwen-lora`, action `set_timer`, params `{duration_seconds: 300}`. Confidence 1.0." | Type **"set timer for 5 minutes"** → Send. Point at the intent chip on the message. |
| 4:00 | OOD command (the killer) | "Now the same intent in a phrasing the LoRA was *not* trained on. The cascade routes to the third arm — Gemini — which returns a parse, then a slot-normaliser maps it back to the canonical schema. Same `duration_seconds: 300`, same `set_timer`. **OOD recovery without retraining.**" | Type **"could you start a five minute timer please"** → Send. Point at chip again. |
| 6:00 | Personal facts (privacy) | "Long-term memory is on-device, encrypted at rest with Fernet, gated by a Cedar policy that denies cloud calls on sensitive topics. I'll teach it a fact." | Type **"my name is Tamer"** → Send. |
| 7:00 | Recall | "Same key, new session would still recall it. Survives restart." | Type **"what is my name"** → Send. |
| 8:00 | Switch to **Stack** tab | "14-stage hand-orchestrated pipeline. Each box maps to a module in `i3/`. The cascade I just showed is stages 8 and 9." | Click **Stack** tab. Point at the pipeline diagram. |
| 9:30 | Switch to **State** tab | "This is the from-scratch SLM's *own* attention — 12 layers × 12 heads, real-time. Token-level. Not a static screenshot." | Click **State** tab. Point at the heatmap. |
| 10:30 | Switch to **Adaptation** tab | "And this is the 8-dim user-state vector that conditions the SLM via cross-attention. It updates from typing biometrics — keystroke timings, pause patterns. *That's* the implicit-interaction part." | Click **Adaptation** tab. |
| 11:30 | Close demo | "Five tabs you've seen. There are 16 more behind the Advanced toggle if you want subsystem detail — KG canonicalisation, Cedar policy explorer, edge-profile, fine-tune comparison, JD map." | (Optionally click **Advanced** toggle in nav-trailing.) |

## The three numbers (memorise these)

1. **204 M parameters** in the from-scratch SLM (`d_model=768, n_layers=12, n_heads=12, vocab=32k`, MoE + ACT).
2. **5.4 × 10⁻⁶ val loss** on the Qwen LoRA intent parser (DoRA + NEFTune + 8-bit AdamW, 4 545 train / 252 val examples, 1 704 steps × 3 epochs, **9 656 s wall**).
3. **6 / 6 OOD-command salvage** in the offline cascade smoke (Phase-4 Gemini-backup + Phase-5 slot-normaliser, against an always-fail Qwen stub).

If you can only quote one: **#3** — it's what the live demo just showed.

## Four JD bullets, one paragraph each

> The Huawei R&D HMI Lab JD asked four questions; here are the one-paragraph answers.

**Q1 — Domain-specific HMI fine-tuning?**
Yes, twice. (a) A from-scratch 204 M custom transformer trained on 974 k synthetic dialogue triples with cross-attention over an 8-dim user-state vector — no pretrained backbone. (b) A Qwen 1.7B + LoRA intent parser supervised-fine-tuned on 4 545 HMI command examples; DoRA rank 16, NEFTune α=5, 8-bit AdamW, cosine warm restarts. Final val_loss 5.4 × 10⁻⁶, intent_eval@100 % action accuracy. Both adapters live under `checkpoints/`.

**Q2 — On-device inference + privacy?**
Server defaults to local: Qwen LoRA preloaded at startup (`I3_PRELOAD_QWEN=1`), SLM v2 forward + attention extraction on CPU/CUDA, ONNX export under `checkpoints/encoder/tcn.onnx`. Cloud is opt-in: `GEMINI_API_KEY` gates the Gemini-backup leg of the cascade, and Cedar policies (`deploy/policy/cedar/i3.cedar`) refuse cloud routes on sensitive topics. Diary store is SQLite + Fernet at-rest encryption; `forget my facts` wipes it.

**Q3 — Multi-modal / multi-signal interaction?**
Implicit signals already wired: typing biometrics (Monrose & Rubin 1997, Killourhy & Maxion 2009 — see the Identity Lock pill), pause/edit patterns feeding a TCN encoder → 64-dim embedding → 8-dim user-state. Optional `mediapipe`-backed facial-affect path under `i3/multimodal/vision.py` (off by default). Voice input wired via `i3/audio/whisper_adapter.py`.

**Q4 — How do you know it works?**
467 / 468 unit tests green across 18 subsystem files; 39/44 verification harness checks pass; live WebSocket smoke 8/8 turns green including OOD command + fact recall; cascade Phase-4/5 offline experiment salvaged 6/6 OOD commands via Gemini-backup. Reports under `reports/verification_latest.md` and `reports/slm_v2_eval.md`.

## "What this prototype is *not*" (the honesty paragraph)

- **Not chat-quality competitive** with GPT-4 / Claude. SLM v2 perplexity ≈ 1725 on held-out — the architecture is *data-bound* at this size (300 k subset of 974 k corpus), not epoch-bound. The from-scratch path is the differentiator, not the chat polish.
- **Not edge-deployed yet.** The ONNX exports exist, the edge profiler emits realistic budgets, but I haven't shipped to a Huawei watch / phone runtime — that's the next iteration.
- **Not a single trained model.** Three arms; the demo is the *cascade*, not any one model in isolation.
- **Not LoRA-on-top-of-Llama.** The "from-scratch" SLM is genuinely from scratch; the Qwen LoRA is a separate intent-parsing arm, deliberate.

## Eight likely questions + crisp answers

| Q | A |
|---|---|
| Why three models, isn't that complexity? | Each arm has a *different role*: SLM = chat (the differentiator), LoRA = deterministic JSON for commands, Gemini = OOD safety net. Mixing roles in one model would either over-budget the SLM or sacrifice the from-scratch claim. |
| Why is SLM perplexity so high? | 204 M from-scratch on 300 k synthetic dialogue. Data-bound. Production scales data, not architecture — the next run is the full 974 k corpus. The cross-attention + ACT + MoE deliver the *implicit-interaction* premise even at this checkpoint. |
| Why Qwen if you're pitching from-scratch? | Two arms, two roles. The chat is from-scratch. The intent parser is Qwen+LoRA because production needs deterministic, audited JSON output for vehicle commands — not free-form text. |
| Wouldn't a single GPT-4 call beat this? | On chat quality, yes, today. On *latency, privacy, cost, and provable on-device behaviour for HMI* — no. The cascade hits Gemini only when the local arms can't, and only when the user opted in. |
| How do you avoid hallucinated commands? | Schema validation: the Qwen output is rejected unless `action ∈ SUPPORTED_ACTIONS` and `params.keys ⊆ ACTION_SLOTS[action]`. Same gate runs on the Gemini backup output, after slot normalisation. |
| Show me the cascade fallback in code. | `i3/pipeline/engine.py` `_maybe_handle_intent_command` — searches for `primary_failed`. The `gemini-backup` branch fires only when Qwen returns `unsupported` / `valid_action=False` / `valid_slots=False` and `GEMINI_API_KEY` is set. |
| What about non-English? | The corpus is English-only today. The BPE tokenizer (`checkpoints/slm/tokenizer_bpe.json`) is byte-pair, so Cyrillic / CJK won't hard-break, but accuracy isn't measured. Multilingual is a roadmap item. |
| Why no streaming? | The chat surface is request/response WebSocket today; tokens stream in the SLM forward but the API joins them. Streaming-to-UI is one short refactor in `server/websocket.py` — happy to walk through. |

## Pre-demo checklist (run T-30 minutes before the meeting)

```bash
# 1. Make sure .env has the keys
grep -E "GEMINI_API_KEY=|I3_ENCRYPTION_KEY=" .env   # both lines present

# 2. Start the server with Qwen pre-warmed (eats ~13 s of cold-start once)
I3_PRELOAD_QWEN=1 .venv/Scripts/python -m uvicorn server.app:app \
    --host 127.0.0.1 --port 8000 > /tmp/server.log 2>&1 &

# 3. Wait for "Application startup complete" in /tmp/server.log

# 4. Open http://127.0.0.1:8000 — should default to Simple nav (5 tabs)

# 5. Send one warm-up turn yourself ("hi") so the next call is hot

# 6. Sanity: real-user emulation
.venv/Scripts/python D:/tmp/ws_smoke.py    # all 8 turns must show tool:intent or retrieval, no errors
```

If GEMINI fails / no internet: the cascade falls through to Qwen-only on chat. Demo still works for the first 5 steps; OOD step (4:00) will return a softer ack but won't crash. Mention "I'm offline so the Gemini-backup arm is dormant — let me show you the wired path in `engine.py`" and switch to the code.

## Closing line (verbatim — in `docs/closing_lines.md`)

> "I³ is the smallest end-to-end stack I could build that *actually* implements implicit interaction — a from-scratch language model that conditions on how you type, end-to-end privacy, and a cascade that degrades gracefully. Whatever happens with this internship, this is the project I'd keep building."

(Pause. Smile. Stop talking.)
