---
marp: true
theme: default
size: 16:9
paginate: true
backgroundColor: "#1a1a2e"
color: "#f0f0f0"
style: |
  section { font-family: -apple-system, "Segoe UI", Inter, sans-serif; font-size: 26px; padding: 56px 72px; }
  section::after { color: #a0a0b0; font-family: "JetBrains Mono", Menlo, monospace; font-size: 18px; }
  h1 { color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 0.2em; font-size: 48px; font-weight: 600; }
  h2 { color: #f0f0f0; font-size: 36px; font-weight: 600; }
  h3 { color: #a0a0b0; font-size: 26px; font-weight: 500; }
  strong { color: #e94560; }
  em { color: #a0a0b0; font-style: normal; }
  li::marker { color: #e94560; }
  code { font-family: "JetBrains Mono", Menlo, monospace; background: #16213e; color: #f0f0f0; padding: 0.1em 0.35em; border-radius: 3px; }
  pre { background: #16213e; border-left: 3px solid #e94560; padding: 0.9em 1.1em; font-size: 20px; }
  pre code { background: none; padding: 0; }
  table { font-family: "JetBrains Mono", Menlo, monospace; font-size: 20px; width: 100%; }
  th { color: #e94560; border-bottom: 2px solid #e94560; text-align: left; padding: 0.4em 0.7em; }
  td { border-bottom: 1px solid #0f3460; padding: 0.4em 0.7em; color: #f0f0f0; }
  tbody tr:nth-child(even) td { background: rgba(15, 52, 96, 0.35); }
  blockquote { border-left: 4px solid #e94560; background: #16213e; padding: 0.8em 1em; margin: 0.8em 0; font-style: italic; }
  footer { color: #a0a0b0; font-family: "JetBrains Mono", Menlo, monospace; font-size: 15px; }
---

<!-- _class: title -->
<!-- _paginate: false -->

# The person who already knows you're tired

> Think about the person who knows you well enough to notice
> you're tired from the pace of your typing — without you saying so.

- They adapt to **how** you say it, not just **what** you say.
- They do not ask you to declare their state.
- They notice when today is different from yesterday.
- They get quieter when you get quieter.

*Implicit Interaction Intelligence — I³*
Tamer Atesyakar — UCL MSc Digital Finance & Banking — 29 April 2026

---

# Your phone does not notice

- Siri answers the words — not the **pace**, **pauses**, or **edits**.
- ChatGPT today is the same as yesterday; **you** are not.
- "Settings" is the wrong home for your state.
- Implicit signals are in every keystroke — nobody reads them.

> Current conversational AI responds to **what** you say,
> not **how** you say it.

<footer>Sources: Apple Siri product page (apple.com/siri, 2026); OpenAI ChatGPT product documentation (openai.com, 2026).</footer>

---

# Why this lab, why now

- **HarmonyOS 6 + HMAF** — large model, small model, agent core ¹
- **Smart Hanhan** — launched Nov 2025, 1800 mAh, 64 MB class ²
- **AI Glasses** — launched 21 April 2026, offloads to phone ³
- **Edinburgh Joint Lab** — "sparse or implicit signals," Nissim, Mar 2026 ⁴
- **Eric Xu** — "experience, not computing power" ⁵

*Three-tier architecture + on-device-first + behavioural signals —
the fit is not accidental.*

<footer>¹ huawei.com/press/2025 (HarmonyOS 6). ² consumer.huawei.com (Smart Hanhan, Nov 2025, 399 RMB, 1800 mAh). ³ consumer.huawei.com (AI Glasses, 21 Apr 2026, 30 g). ⁴ ed.ac.uk/joint-lab (10 Mar 2026). ⁵ Huawei Connect 2025 keynote, E. Xu.</footer>

---

# What I will show in 30 minutes

- A working system that builds a user model from **how** you type.
- Adapts across four axes at once — **load, style, tone, access**.
- Runs fully on this laptop, end-to-end, live.
- Clear edge path to Kirin NPU and Smart Hanhan-class devices.
- The user never explains themselves — not once.
- Honest limits named out loud, not hidden.

*Live demo at minute twelve. Honesty slide near the end.*

---

# Seven layers, one sentence each

| Layer | Owns |
|-------|------|
| `1. Perception`   | `32-dim behavioural feature vector per message`        |
| `2. Encoder`      | `64-dim user-state embedding — TCN, from scratch`      |
| `3. User Model`   | `three timescales — instant, session, long-term EMA`   |
| `4. Adaptation`   | `4-axis AdaptationVector — load, style, tone, access`  |
| `5. Router`       | `Thompson-sampling bandit — local SLM vs cloud Claude` |
| `6. Generation`   | `custom SLM (cross-attended) or Claude API`            |
| `7. Diary`        | `embeddings + metadata — raw text never stored`        |

*Full ASCII topology in `README.md`. One slide, no details — details follow.*

---

# Listening to how you type

- 32 features in 4 groups of 8 — **dynamics, content, session, deviation**.
- Keystroke: inter-key ms, pause count, backspace ratio, burst rate.
- Server-side linguistic: TTR, Flesch–Kincaid, formality, sentiment.
- Raw stream → TCN encoder → 64-dim L2-normalised embedding.
- Dilated causal convolutions, kernel `k=3`, dilations `[1,2,4,8]`.
- Receptive field `r = 1 + (k−1)·Σdᵢ = 31` messages.
- Trained with **NT-Xent** contrastive loss on synthetic archetypes.

<footer>Bai, Kolter, Koltun 2018 — receptive-field formula. Chen et al. 2020 (SimCLR) — NT-Xent.</footer>

---

# The User Model — what "normal" looks like

- **Three timescales** matching three kinds of decision:
- `Instant` — current message, drives this response.
- `Session EMA` — within-conversation trend, detects drift.
- `Long-term EMA` — baseline across sessions, detects today ≠ usual.
- Welford online statistics — no history re-read, O(1) per update.
- Deviation `z = (x − μ)/σ` feeds back as Group 4.
- **Baseline after 5 messages** — flag flips live in the demo.

*A single EMA collapses these; three keeps the structure.*

---

# Cross-attention conditioning — the centrepiece

```python
# Every transformer block attends to 4 conditioning tokens
cond = projector(                      # (B, 4, d_model)
    torch.cat([user_state_64, adaptation_vec_8], dim=-1)
)
h = block.self_attn(x, x, x, mask=causal)          # content
h = block.cross_attn(h, cond, cond)                # user state
h = block.ffn(h)                                   # features
```

- `AdaptationVector(8) ⊕ UserStateEmbedding(64)` → `4 × d_model` tokens.
- Cross-attended at **every** layer — conditioning modulates generation throughout.
- Cost: `O(4·N·d)` per layer — <5% overhead vs self-attention's `O(N²·d)`.
- Auxiliary loss penalises conditioning-agnostic outputs.

*Not "prepend a nice prompt." Continuous modulation of token probabilities.*

<footer>Vaswani 2017 (attention). Xiong et al. 2020 (Pre-LN transformer).</footer>

---

# Live demo — four phases

- **1. Cold start (2 min)** — 5 messages; baseline establishes; router favours cloud.
- **2. Energetic (1 min)** — fast, long, rich; load rises; style mirrors up.
- **3. Fatigue (2 min)** — shorter; warmer + briefer; router flips to local.
- **4. Accessibility (2 min)** — backspaces, fragments; yes/no, simpler vocab.

*Dashboard is the visual anchor — embedding trail, four gauges, routing confidence.*

**No settings menu. No toggle. It just adapts.**

<footer>Phase narration beats are in `speaker_notes.md`. Browser opens in the next action.</footer>

---

# Routing — when to spend a cloud call

- **Two arms:** local SLM (latency) vs cloud Claude (quality).
- **12-dim context:** state, complexity, sensitivity, patience, session, history, time, count, cloud latency, SLM confidence.
- Bayesian logistic regression + **Laplace approximation**.
- **Newton–Raphson MAP refit every 10 steps** — online posterior.
- **Privacy override** — sensitive topic → cloud arm masked to zero.
- Reward: composite engagement (continuation, sentiment, latency, topic).

*Thompson sampling probability-matches uncertainty — exactly what an asymmetric-cost router wants.*

<footer>Russo et al. 2018 — Thompson sampling tutorial. Chapelle & Li 2011 — empirical evaluation.</footer>

---

# Privacy by architecture, not policy

- **Raw text is never stored** — enforced at storage, no toggle.
- **Embeddings Fernet-encrypted at rest** — `I3_ENCRYPTION_KEY`.
- **10 PII regex patterns** — strip email/phone/address/date/card before cloud.
- **Sensitive-topic classifier** — health / mental / financial / credentials.
- **Diary schema** — no `message` column; `embedding BLOB`, `adaptation JSON`.
- **Fernet is a placeholder for TrustZone** — honest caveat, not a claim.

*Architectural — meaning a compromised policy file does not expose text.*

<footer>Huawei Kirin TrustZone documentation (developer.huawei.com, 2025).</footer>

---

# Fits the devices — extrapolated, honestly

| Device              | RAM     | NPU TOPS | I³ footprint | Extrapolated P50 |
|---------------------|---------|----------|--------------|------------------|
| `Kirin 9000`        | `512 MB`| `2.0`    | `~7 MB INT8` | `~50–80 ms`      |
| `Kirin 820`         | `256 MB`| `0.8`    | `~7 MB INT8` | `~120–160 ms`    |
| `Kirin A2`          | `128 MB`| `0.4`    | `~7 MB INT8` | `~250–320 ms`    |
| `Smart Hanhan (64)` | ` 64 MB`| `0.1`    | `encoder only`| `encoder ~30 ms` |
| `Laptop (M-class)`  | `16 GB` | `—`      | `~7 MB INT8` | `~150–220 ms`    |

- Measured laptop P50: `150–220 ms` full pipeline; TCN `<5 ms`.
- Kirin numbers are **extrapolated** from quantised sizes + NPU throughput — not measured.
- Conversion target: **MindSpore Lite** (PyTorch → ONNX → MS Lite).

<footer>Device specs: consumer.huawei.com / developer.huawei.com (2024–2026). NPU TOPS: public Kirin datasheets. Extrapolation is a caveat, not a claim.</footer>

---

<!-- _class: honesty -->

# What This Prototype Is Not

- **Not a shipped product.** A 17-day prototype, on one laptop.
- **Not trained on real user data** — synthetic archetypes from HCI literature.
- **Not a strong SLM** — 8 M params, limited by synthetic dialogue.
- **Not universal accessibility** — keystroke misses screen-reader, voice, gaze users. Must stay opt-out.
- **Not zero-information embeddings** — lossy, abstract, still identity-signalling.
- **Not multi-modal yet** — keystroke only; the TCN itself is modality-agnostic.

*Values note: adaptation complements explicit settings, it must not replace them. Detection is one signal among many; recovery fades it.*

---

# Where this goes next

- **L1 → L5 intelligence.** Prototype lives at **L2–L3** (single-device proactive). ¹
- **L4 — device-to-device handover.** Phone → Glasses → Hanhan over HarmonyOS databus.
- **L5 — autonomous orchestration.** Goal-driven, multi-device, state-aware.
- **Federated long-term profile** — MindSpore Federated, embeddings never leave device.
- **Multi-modal is a drop-in** — TCN swaps keystroke for touch, gaze, voice, motion.
- **In scope next:** keystroke-biometric identification, fairness eval, interpretability panel, ablation toggle.

<footer>¹ Huawei + Tsinghua IAIR — L1–L5 Intelligence Framework (public whitepaper, 2025).</footer>

---

<!-- _class: close -->

# Close

> *"I build intelligent systems that adapt to people.
> I'd like to do that in your lab."*

Tamer Atesyakar — `t.ates232004@gmail.com`

Thank you — happy to take questions.

<footer>Three candidate questions ready from the prepared set. No salary, benefits, or timeline questions in this room.</footer>
