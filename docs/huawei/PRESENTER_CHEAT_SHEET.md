# I³ — 30-min interview cheat sheet

> One page. The only thing you need open during the live demo.
> Last refreshed: iter 51 phase 18 (2026-04-27).

## The 10-minute live demo (5 chip + 1 actuator)

Open browser to `http://127.0.0.1:8000/` — nav defaults to **Simple** (5 tabs).
Five **suggestion chips** under the hero — click them in order; each one
exercises a different cascade arm and lights up a different routing chip.

| Time | Click this chip | Routing | What you say |
|---:|---|---|---|
| 0:00 | (intro) | — | "I³ is a three-arm cascade. From-scratch 204 M SLM + retrieval first, Qwen LoRA for HMI commands, Gemini cloud only when the local arm can't ground the query. The chip under each reply tells you which arm answered and why." |
| 0:45 | **How do you adapt to me?** | `SLM·0.85   Qwen   Gemini   used: SLM + retrieval` | "First arm: from-scratch SLM with curated retrieval. The system answers about its own architecture from a hand-curated knowledge graph. Look at the chip — SLM lit, Qwen and Gemini idle." |
| 2:30 | **Set timer for 30 seconds** | `SLM   Qwen·1.0   Gemini   used: Qwen LoRA` | "Second arm: Qwen 1.7B + LoRA, our deterministic JSON intent parser. Action `set_timer`, params `{duration_seconds: 30}`, confidence 1.0. **And it actually fires** — watch the gold pulse banner that drops in here in 30 seconds." (Continue with other chips while you wait.) |
| 4:00 | **Tell me about Uzbekistan** | `SLM   Qwen   Gemini·0.92   used: Gemini` | "Third arm: Gemini, but only as a last resort. The local KG only has 31 subjects; Uzbekistan isn't in it, so the topic-grounding gate demotes retrieval and Gemini tags in. Hover the chip — `world_chat 0.92, default_chat 0.05` — that's the routing math." |
| 5:30 | **What is photosynthesis?** | `SLM·0.85   Qwen   Gemini   used: SLM + retrieval` | "Back to the local arm. Photosynthesis IS in the KG, so the cascade prefers it over the cloud. Curated, deterministic, on-device." |
| 6:30 | **Navigate to Trafalgar Square** | `SLM   Qwen·1.0   Gemini   used: Qwen LoRA` | "Another HMI command — Qwen LoRA again. Action `navigate`, slot `location: trafalgar square`. The blue banner is the actuator state — in production this would hand off to the navigation system." |
| ~8:00 | **(timer fires)** | actuator_event banner pulses | "And there it is — the timer I set 30 seconds ago. Real asyncio task, not a faked animation. The cascade isn't just acks, it actually does things." |
| 8:30 | Switch to **Stack** tab | — | "Eight subsystems shown by default; click 'Show all 22' for the full map. Each card cites the paper or standard it implements." |
| 9:30 | Switch to **State** tab | — | "12-layer × 12-head attention from the from-scratch SLM, live, token-level. The 64-d typing-biometric embedding is the cross-attention conditioning vector." |
| 10:30 | Close | — | "Five visible tabs; sixteen more behind the Advanced toggle if you want subsystem detail. The cascade is the differentiator; the from-scratch SLM is the on-device anchor." |

## The five numbers (memorise these)

1. **204 M params** in the from-scratch SLM (`d_model=768, 12 layers × 12 heads, vocab=32k, MoE + ACT`).
2. **5.4 × 10⁻⁶ val loss** on the Qwen LoRA intent parser (DoRA r16 + NEFTune α=5 + 8-bit AdamW + cosine warm restarts, 4 545 / 252 split, 1 704 steps × 3 epochs, 9 656 s wall).
3. **22 / 22 routing classifications correct** on the precision smoke. Six deterministic signals (greeting · cascade-meta · system-intro · question-shape · KG-anchor · system-topic) feed a multi-signal scorer; the highest-scoring route class wins, all scores visible in the chip tooltip.
4. **162 KB INT8 encoder, 460 µs p50 inference, runs in-browser** via ONNX Runtime Web. 12.5 × under the 2 MB Kirin watch RAM budget. Toggle is in the State tab; DevTools → Network proves it's client-side.
5. **Timer-actually-fires latency: 30 s exact**. End-to-end verified — `set_timer` → asyncio task → `actuator_event` frame → gold pulse banner in the chat.

If you can only quote one: **#4** — it's the JD's edge-deployment question answered with a live demo, not a slide.

## The edge-inference power move (do this when probed about wearables)

Add this 30-second move during chip 1 ("How do you adapt to me?"):

> *"While the SLM is generating — quick aside. Open DevTools → Network panel. I'll click the Edge inference toggle in the State tab and re-send. Watch what happens."*
>
> *(switch to State tab → flip "Run inference in browser" toggle ON → switch back to Chat → send another turn)*
>
> *"No `/api/encode` request. The 32-d feature vector hit the encoder in your browser tab via ONNX Runtime Web — 162 KB INT8 model, 460 µs inference. The Kirin A2 watch budget is 2 MB encoder + 8 MB peak resident; we're 12.5 × under. Keystrokes never left this page. Privacy-by-architecture, enforced by the network boundary."*

This answers the email's *"have you ever deployed ML models to low-compute devices"* question with proof, not infrastructure-only.

## The cascade in one diagram

```
                                ┌───────────────┐
   user message ───────────────►│ smart router  │
                                │ (6 signals →  │
                                │  scored)      │
                                └──┬───┬───┬────┘
              ┌────────────────┬──┴┐  │   │
              ▼                ▼   ▼  ▼   ▼
          greeting       cascade_  …  command  default / world / system
          (regex)        meta              (regex
              │                │               gate)         │
              ▼                ▼               ▼              ▼
         hand-written     C: Gemini       B: Qwen LoRA    A: SLM + retrieval
         local reply      (only arm        primary,        topic-consistency
         (no LLM)         that can         Gemini          gate; if demoted
                          describe         backup if       → C: Gemini
                          the cascade)     LoRA flunks     "tags in" with
                                                           conversation history
```

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

- **Not chat-quality competitive** with GPT-4 / Claude. SLM v2 training-time held-out perplexity is **≈ 147** (eval_loss 4.99, response-token-only, same-300 k-subset distribution — comparable to other small from-scratch SLMs). A more conservative full-corpus stress test that scores history-tokens too lands at **≈ 1725** — captures the distribution-shift gap from training on 300 k of the 974 k corpus.  Architecture is *data-bound* at this size, not epoch-bound; the from-scratch path is the differentiator, not the chat polish.
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
