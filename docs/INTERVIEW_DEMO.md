# I³ Interview Demo Script — 5 minutes

A precise, minute-by-minute live-demo script for the Huawei R&D UK
HMI Lab AI/ML Specialist Internship interview. Every step is
**executable** against the running system; if you follow this script
in order, it works.

---

## Pre-flight checklist

Run through this in the 60 seconds before the interview starts.

- [ ] Server is up: `make all-fast` or
      `poetry run uvicorn server.app:app --host 127.0.0.1 --port 8000`.
- [ ] Browser open at <http://127.0.0.1:8000>, **hard-refreshed**
      (Ctrl + Shift + R) to bust the SPA cache.
- [ ] Identity Lock template **reset** so the demo starts cold:
      ```bash
      curl -X POST http://127.0.0.1:8000/api/biometric/demo/reset
      ```
- [ ] Diary cleared so the recency stack starts empty (optional):
      ```bash
      rm -f data/diary.db data/diary.db-shm data/diary.db-wal
      ```
- [ ] Cloud-route consent is **OFF** (the default; the cloud pill in
      the nav header should read `cloud · off`).
- [ ] Microphone + camera permissions granted at the browser level
      (the prompts will fire when you click the toggles in Minute 4 —
      better to dismiss them once now in advance).
- [ ] Notes window minimised; only the demo browser visible.

If any of the above fails, fall back to:
`make all-fast` rebuilds the seed state in ~30 seconds.

---

## Minute 1 — "Type, and watch yourself being read"

**The story:** the assistant is reading me as I type.

1. **Open the Chat tab.** Lead with one sentence:
   > "I built an HMI assistant that reads *how* you type and adapts.
   > Watch the badges in the nav as I start."

2. **Type:** `hello`
   - Point at the **state badge** (top-right of nav) — it classifies
     to `calm`.
   - Point at the **Identity Lock pill** — it reads
     `learning · 1/5`.
   - Reply lands. Click **"Why this response?"** (small chip below
     the AI bubble) — a 5-paragraph plain-English narration unfolds:
     "I read your keystroke timing as steady (IKI variance = 64 ms,
     within baseline). Affect classifier landed on calm. Adaptation
     vector set cognitive_load = 0.18, brevity = 0.4. The bandit
     picked the local SLM with confidence 0.91…"

3. **Type four more calm messages** (anything — `tell me about you`,
   `what can you do`, `how does this work`, `nice`).
   - The Identity Lock fills: `learning · 2/5` → `3/5` → `4/5` → and
     on the fifth turn → **`you · 0.94 ✓`** with a brief green flash.
   - The state badge stays on `calm`.

4. **Re-open the "Why this response?" panel** on the latest reply.
   - Now the narration includes: *"You're recognised as the
     registered user (Mahalanobis 0.91, threshold 0.7). Your personal
     LoRA adapter is patched into the SLM."*

**Time check:** ~60 seconds. Land on:
> "Five turns in, and the model knows it's me. Without ever asking
> me to log in. Now watch what happens when I type *differently*."

---

## Minute 2 — "Now type differently and watch the system notice"

**The story:** affect-shift detection + adaptation in one turn.

1. **Type the next message slowly and with deliberate backspaces.**
   Hit backspace at least four times mid-message:
   `argh i kep` → backspace twice → `keep` → `t typing this wro` →
   backspace twice → `wrong`.

   The full message in the box should read:
   `argh i keep typing this wrong`.

2. **Before you hit Send, point at the affect-shift chip** that has
   already lit up in the nav: **`affect-shift · rising_load · 1.6σ`**.

3. **Send.** The reply lands and *ends with a pivot sentence* like:
   > "Your typing pattern shifted — want me to break this into
   > shorter pieces?"

4. **Point at the Identity Lock.** It may **shake** (drift detected
   relative to your registered template). The pill colour shifts to
   amber. Narrate:
   > "The drift is intentional — I'm typing with deliberate noise.
   > The system noticed the pattern doesn't match my registered
   > template and re-evaluated."

5. **Click the Profile tab.** Point at:
   - The **personalisation tile** showing accumulating drift (a small
     line plot of Mahalanobis distance over the last 10 turns).
   - The **recency stack** (entities seen) — currently empty
     because we haven't named anyone yet.

**Time check:** ~120 seconds total.

---

## Minute 3 — "Multi-turn understanding ChatGPT can't do"

**The story:** rule-based coreference + entity tools in the
retrieval path.

1. **Click back to Chat.** Type:
   `tell me about huawei`
   - Reply: a curated factoid (founded 1987, HQ Shenzhen, …). The
     entity `huawei` is pushed onto the recency stack.

2. **Type:** `where are they located?`
   - Coref resolves: `they → huawei`. The entity tool fires; the
     reply is `Shenzhen, China`. Point at the **`tool:entity` chip**
     under the bubble.

3. **Type:** `tell me about apple`
   - Reply: a curated factoid. `apple` is pushed onto the stack.

4. **Type:** `which is bigger?`
   - The comparison tool fires across both entities still on the
     stack. Reply names which is larger by revenue / market cap.
   - Point at the **`tool:compare` chip** and the **Profile tab's
     recency stack** — `[apple, huawei]` now visible.

**Talking point during the responses:**
> "ChatGPT does this with a foundation-model context window. I do it
> with a deterministic recency stack and a rule-based coref resolver.
> 1 ms latency, fully debuggable, runs in 100 KB of state."

**Time check:** ~180 seconds total.

---

## Minute 4 — "Multimodal + edge"

**The story:** voice + gaze + the actual edge measurements.

1. **Click the microphone toggle** in the chat composer.
   - Browser permission prompt fires (already granted in pre-flight).
   - The **live prosody bar** in the composer animates with pitch,
     energy, jitter. Narrate:
   > "Audio never leaves the browser. The prosody features —
   > pitch envelope, jitter, shimmer, speaking rate — are extracted
   > client-side and only the small derived feature vector posts
   > over the WebSocket."

2. **Click the camera toggle** in the chat composer.
   - Calibration kicks in: 4 dots appear at the screen corners,
     each fixated for ~1 second.
   - Once calibrated, narrate:
   > "Now the model also knows where I'm looking. MobileNetV3
   > backbone, frozen, with a small head trained per session in-tab.
   > Camera frames never leave the browser either."

3. **Switch to the Edge tab.** Point at the live numbers:
   - **Parameters: 53,307,392** (v1 currently loaded)
   - **int8 size: 110.2 MB**
   - **p50 greedy decode: 612.8 ms**
   - **TCN encoder: 0.4 MB int8 / 3.7 ms p50**
   - **Peak RSS: 1.31 GB**

   Narrate:
   > "These are real measurements from `scripts/measure_edge.py`,
   > not estimates. Mid-range phone deployable today; v2 (204 M)
   > training overnight."

4. **Switch to the Flow tab.** Type a quick message in the chat (use
   the inline input the Flow tab exposes, or hop back to Chat for
   one turn) — point at the **animated 14-stage pipeline** firing.
   Each stage shows real timings and data shapes.

**Time check:** ~240 seconds total.

---

## Minute 5 — "I built both routes; here's the cloud fallback story"

**The story:** privacy budget, PII sanitisation, and the bandit's
cloud-routing path.

1. **Switch to the Privacy tab.** Point at:
   - **Privacy budget panel:** `cloud calls used: 0 · redactions: 0`
   - **Cloud consent state:** `OFF (default)`

2. **Click the cloud-consent toggle in the nav.** It flips to
   `cloud · on · budget 8/8`. Narrate:
   > "Default off. Even with consent on, the bandit only routes to
   > cloud when complexity exceeds 0.65 *and* the privacy budget
   > allows another call. Two hard gates."

3. **Switch to Chat. Type a high-complexity prompt:**
   `explain the key differences between Bayesian and frequentist
   statistics with examples`

4. **As the reply streams**, switch to the **Routing tab** and point
   at:
   - The new dot in the scatter plot at high complexity (X) +
     reasonable retrieval (Y) — coloured cloud-blue.
   - The decision banner: `cloud · 0.84 · complexity 0.71 > 0.65
     threshold`.

5. **Switch back to Chat. Test the PII sanitiser** with:
   `my email is test@example.com`
   - Open "Why this response?" → narration shows the redaction:
     *"PII sanitiser redacted 1 token (email) before cloud
     dispatch."*
   - Switch back to Privacy tab — the redaction counter incremented.

6. **Open the Flow tab one final time.** The cloud branch of the
   pipeline is now lit; narrate the end-to-end:
   > "Sanitiser → bandit → cloud client → response → post-processor
   > → critique. Same closed loop as the local path, just with the
   > cloud LLM as the generator. The post-processor and critique
   > are the *user-side* quality gate; they're not the cloud
   > provider's responsibility."

**Time check:** ~300 seconds total. Land on:
> "Filter Q4 said 'edge deployment' — I built it. Filter Q3 said
> 'pipeline orchestration' — you can watch it. Filter Q2 said 'SLMs
> without heavy frameworks' — that's the v2 SLM training overnight.
> Filter Q1 said 'custom ML' — that's the entire `i3/` directory,
> hand-written from the original papers."

---

## Talking points (cheat sheet)

Memorise these one-liners; pick whichever the conversation invites.

- **Filter Q1 (custom ML from scratch):**
  TCN encoder, byte-level BPE tokenizer, MoE-FFN, ACT halting,
  LinUCB bandit, BM25 reranker, biometric authenticator, LoRA
  adapter, per-layer cross-attention head — all hand-implemented from
  the original papers in `torch.nn.Module`.

- **Filter Q2 (SLM without heavy frameworks):**
  Pure PyTorch `nn.Module`, zero HuggingFace dependencies in the
  inference path. v2 SLM is 204 M params: 12 layers × 12 heads,
  `d_model=768`, `d_ff=3072`, MoE × 2 experts, ACT halting. Trained
  with bf16 + 8-bit AdamW + grad-checkpoint on a 6 GB laptop GPU.
  BPE tokenizer is 217 LOC.

- **Filter Q3 (pipeline orchestration):**
  Two orchestrators. The runtime is a 14-stage closed-loop you can
  *watch fire* in the Flow tab. The build is `scripts/run_everything.py`,
  21 stages across 10 waves with parallel-within-wave + DAG resume +
  Rich live UI; ~10 minutes from clean checkout.

- **Filter Q4 (edge deployment):**
  110 MB int8 on disk, p50 612 ms CPU greedy decode (v1, real
  measurements). ONNX export of TCN encoder. Browser inference path
  via `onnxruntime-web` with WebGPU + WASM detection. Edge tab shows
  live measurements.

---

## Likely questions + 30-second answers

### *"Why a custom transformer instead of fine-tuning a small LLM like TinyLlama or Phi-3?"*

Three reasons. (1) The JD literally asks for SLM development without
heavy frameworks — fine-tuning Phi-3 is the *opposite* of that.
(2) The v1 53 M / v2 204 M shapes are deliberately chosen to be
edge-deployable; even Phi-3's 3.8 B doesn't fit my latency budget.
(3) The from-scratch path forces understanding — I can tell you why
each architectural decision was made because I made it. Fine-tuners
inherit those decisions opaquely.

### *"How does the personalisation differ from federated learning?"*

The current implementation is **purely on-device LoRA**. Per-user
adapters keyed by SHA-256 of the biometric template, never
aggregated, never uploaded. Federated learning would *aggregate* the
LoRA deltas across devices via FedAvg with DP noise. The skeleton is
in `i3/federated/`; the missing piece is a server-side aggregator
and a key-rotation policy. So today: personalisation is local;
tomorrow: federated.

### *"What's the privacy story?"*

Five layers. (1) Diary persists no raw text — only encrypted
embeddings + scalar metrics. (2) PII sanitiser at every cloud
boundary, with redaction counts surfaced live. (3) Per-session
privacy budget gates cloud routing. (4) Microphone + camera capture
happens client-side; only derived features post over the WebSocket.
(5) Biometric template stored as hashed feature vector, never raw
key events.

### *"How would you scale this to a 1 B-parameter production model?"*

The architecture doesn't change — only the shape and the corpus.
On a single A100 the architecturally-ideal shape (`d_model=960,
n_layers=16, d_ff=3840, 2 experts, seq=1024`, ~400 M) trains in
2-3 days; 1 B would take ~5-7 days at full corpus × 3 epochs. The
limiting resource is corpus scale and quality, not compute. Going
1 B also breaks the edge-deployment story for the laptop tier;
production would shift the SLM to the cloud and use distilled-50 M
for edge. The pipeline (router, post-processor, critique) is the
same — only the generator changes.

### *"Why a rule-based coreference resolver?"*

Three reasons. (1) Latency: <1 ms vs ~50 ms for a small neural
coref. (2) Debuggability: I can show you which rule fired. (3) The
demo's dialogue horizon is short (2-3 entities, 5-10 turns); a
neural model is overkill at that horizon. For long-horizon, I'd
swap in a learned model behind the same `i3/dialogue/coref.py`
interface.

### *"What's the real bottleneck on the edge profile?"*

The vocabulary embedding matrix. At 32 k vocab × 768 = 25 M
parameters in just the embedding, weight-tied to the output
projection. To hit a 50 MB int8 wearable budget, the path is
distillation to a 10-20 M student with a smaller vocab (8 k via
forced-alignment vocab pruning). The ACT halting controller already
gives a runtime speedup; combining ACT-aware width pruning with
LoRA-targeted distillation should land us at the wearable target.

### *"How did you train it on a 6 GB GPU?"*

Three tricks. (1) bf16 mixed precision halves activation memory.
(2) `bnb.optim.AdamW8bit` quantises the optimiser state to 8-bit,
saving another ~3× on the optimiser footprint. (3) Gradient
checkpointing trades ~30 % wall-clock for ~50 % activation memory.
Combined, the 204 M model trains at 3.15 GB peak, well under the
~5 GB usable budget on the RTX 4050 Laptop's 6.4 GB.

### *"Why MoE with only 2 experts? Isn't that token gating overkill?"*

It's the most experts that fit. Each expert is its own d_ff = 3072
projection, so 2 experts ≈ 50 M extra params; 4 would push the
total over my GPU's training budget. With 2 experts the gating still
works — the load-balance auxiliary loss keeps both experts active,
and the conditioning gate lets the adaptation vector pick which
expert to weight. Specialisation is mild but measurable. With an
A100, I'd run 4-8 experts.

---

## Failure modes + recovery

If the live demo glitches:

| Symptom | Recovery |
|---|---|
| Identity Lock stuck on `unregistered` | `curl -X POST http://127.0.0.1:8000/api/biometric/demo/reset` then refresh |
| State badge stays on `—` | First few keystrokes haven't filled the 10-event window — type 3-4 short messages and it resolves |
| Cloud toggle won't flip | Privacy budget exhausted — restart server: `make all-fast` |
| Flow tab shows only the previous turn | Send one new message; the trace is collected per-turn |
| WebSocket disconnects | Page refresh (Ctrl+Shift+R); state is reconstructed from the diary |
| Microphone/camera permissions denied | Talk through the UX without enabling — the rest of the demo still works |

---

*Repository: <https://github.com/abailey81/implicit-interaction-intelligence>
· License: MIT*
