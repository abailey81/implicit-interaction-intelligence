---
marp: true
theme: default
size: 16:9
paginate: false
backgroundColor: '#0E1116'
color: '#E6EDF3'
style: |
  @import url('https://rsms.me/inter/inter.css');
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #0E1116;
    --bg-elevated: #171B22;
    --bg-subtle: #1F252E;
    --border: #2A3441;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --text-muted: #6E7681;
    --accent: #F2C25B;
    --arm-slm: #7EE787;
    --arm-qwen: #F0C870;
    --arm-gemini: #A5B4FC;
    --space-1: 8px;
    --space-2: 16px;
    --space-3: 24px;
    --space-4: 32px;
    --space-5: 48px;
    --space-6: 64px;
    --space-7: 96px;
    --space-8: 128px;
  }

  section {
    font-family: 'Inter', 'Helvetica Neue', system-ui, sans-serif;
    font-feature-settings: "tnum", "ss01";
    font-size: 22px;
    font-weight: 400;
    line-height: 1.5;
    letter-spacing: 0;
    color: var(--text-primary);
    background: var(--bg);
    padding: var(--space-6);
  }

  h1 {
    font-size: 56px;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.025em;
    margin: 0 0 var(--space-5) 0;
    color: var(--text-primary);
  }

  h2 {
    font-size: 28px;
    font-weight: 500;
    line-height: 1.3;
    letter-spacing: -0.01em;
    margin: 0 0 var(--space-3) 0;
    color: var(--text-secondary);
  }

  p, li { max-width: 60ch; }
  strong { font-weight: 600; }

  code {
    font-family: 'JetBrains Mono', Menlo, Consolas, monospace;
    font-size: 0.9em;
    font-weight: 500;
    color: var(--text-primary);
    background: transparent;
    padding: 0;
  }

  pre {
    font-family: 'JetBrains Mono', Menlo, Consolas, monospace;
    font-size: 18px;
    line-height: 1.55;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: var(--space-3);
  }
  pre code { font-size: inherit; }

  table {
    border-collapse: collapse;
    margin: var(--space-3) 0;
    width: 100%;
    font-feature-settings: "tnum";
  }
  table th {
    text-align: left;
    font-size: 16px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding: var(--space-1) var(--space-2);
  }
  table td {
    padding: var(--space-1) var(--space-2);
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  table tr:last-child td { border-bottom: none; }

  /* cover */
  section.cover {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-left: var(--space-7);
  }
  section.cover h1 {
    font-size: 96px;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--accent);
    margin-bottom: var(--space-3);
  }
  section.cover h2 {
    font-size: 56px;
    color: var(--text-primary);
    font-weight: 500;
    margin-bottom: var(--space-7);
    max-width: 24ch;
  }
  section.cover .meta {
    font-size: 16px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* hero-metric */
  section.hero-metric {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-left: var(--space-7);
  }
  section.hero-metric .metric {
    font-size: 128px;
    font-weight: 600;
    letter-spacing: -0.04em;
    color: var(--accent);
    line-height: 1;
    margin-bottom: var(--space-3);
  }
  section.hero-metric .label {
    font-size: 28px;
    color: var(--text-primary);
    margin-bottom: var(--space-5);
    max-width: 50ch;
    font-weight: 500;
    line-height: 1.3;
  }
  section.hero-metric .footer {
    font-size: 16px;
    color: var(--text-secondary);
    border-top: 1px solid var(--border);
    padding-top: var(--space-2);
    max-width: 80ch;
    letter-spacing: 0.02em;
  }

  /* divider */
  section.divider {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-left: var(--space-7);
  }
  section.divider .number {
    font-size: 16px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: var(--space-3);
  }
  section.divider .label {
    font-size: 56px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
  }

  /* inverted */
  section.inverted {
    background: #F4F2EE;
    color: #1A1A1A;
  }
  section.inverted h1 { color: #1A1A1A; }
  section.inverted h2 { color: #4D4D4D; }
  section.inverted table th { color: #6E6E6E; border-color: #D9D6CF; }
  section.inverted table td { border-color: #D9D6CF; }
  section.inverted em { color: #4D4D4D; }

  /* split-3 */
  section.split-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: var(--space-4);
    padding: var(--space-6);
  }
  section.split-3 h1 {
    grid-column: 1 / -1;
    margin-bottom: var(--space-4);
  }
  section.split-3 .col-slm h2 { color: var(--arm-slm); }
  section.split-3 .col-qwen h2 { color: var(--arm-qwen); }
  section.split-3 .col-gemini h2 { color: var(--arm-gemini); }
  section.split-3 .col-slm,
  section.split-3 .col-qwen,
  section.split-3 .col-gemini {
    border-top: 1px solid var(--border);
    padding-top: var(--space-3);
  }
  section.split-3 .arm-row {
    margin-bottom: var(--space-2);
  }
  section.split-3 .arm-label {
    font-size: 14px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
  }

  /* split-2 */
  section.split-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-5);
    padding: var(--space-6);
  }
  section.split-2 h1 { grid-column: 1 / -1; }

  /* diagram-anchor */
  section.diagram-anchor {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: var(--space-6) var(--space-7);
  }
  section.diagram-anchor .caption {
    font-size: 28px;
    color: var(--text-secondary);
    margin-bottom: var(--space-4);
    max-width: 60ch;
  }
  section.diagram-anchor .takeaway {
    font-size: 20px;
    font-style: italic;
    color: var(--text-secondary);
    margin-top: var(--space-4);
    max-width: 70ch;
  }

  /* callout */
  .callout {
    border-left: 4px solid var(--accent);
    background: var(--bg-elevated);
    padding: var(--space-3);
    margin: var(--space-4) 0;
    max-width: 70ch;
  }
  .callout .label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin-bottom: var(--space-1);
  }

  /* metric cards row */
  .metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: var(--space-3);
    margin-bottom: var(--space-4);
  }
  .metric-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: var(--space-3);
  }
  .metric-card .big {
    font-size: 64px;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: var(--accent);
    line-height: 1;
    font-feature-settings: "tnum";
  }
  .metric-card .keyline {
    width: 32px;
    height: 1px;
    background: var(--border);
    margin: var(--space-2) 0;
  }
  .metric-card .cap {
    font-size: 16px;
    color: var(--text-secondary);
    line-height: 1.4;
  }

  /* source line */
  .source {
    font-size: 13px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: var(--space-4);
    font-family: 'JetBrains Mono', monospace;
  }

  /* data-table emphasis */
  .data-table .section-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-top: var(--space-3);
    margin-bottom: var(--space-1);
    border-bottom: 1px solid var(--border);
    padding-bottom: var(--space-1);
  }

  /* perplexity bar viz */
  .ppl-bar-row { display: flex; align-items: center; margin: var(--space-2) 0; gap: var(--space-2); }
  .ppl-bar-label { width: 22ch; font-size: 18px; color: var(--text-primary); }
  .ppl-bar { height: 28px; border-radius: 2px; }
  .ppl-bar-headline { background: var(--arm-slm); }
  .ppl-bar-stress { background: var(--text-secondary); }
  .ppl-bar-value { font-size: 18px; font-weight: 600; color: var(--text-primary); margin-left: var(--space-2); font-feature-settings: "tnum"; }

  /* closing */
  section.closing {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding-left: var(--space-7);
  }
  section.closing h1 {
    font-size: 56px;
    color: var(--text-primary);
    margin-bottom: var(--space-5);
  }
  section.closing .repo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    color: var(--text-secondary);
  }
---

<!-- _class: cover -->

# I³

## Implicit Interaction Intelligence

<div class="meta">Tamer Atesyakar · Huawei R&D UK · 29 April 2026</div>

---

# What if the assistant adapted to *how* you typed?

Most chat assistants treat each prompt as the only signal of intent. But typing rhythm, edit patterns, prosody, and gaze already encode cognitive load, affect, and identity.

I³ is a working prototype of an assistant that reads those signals **on-device** and adapts its generation accordingly.

<div class="source">SOURCE — README.md · docs/huawei/hci_design_brief.md</div>

---

# Project at a glance

<div class="metric-row">
  <div class="metric-card">
    <div class="big">204 M</div>
    <div class="keyline"></div>
    <div class="cap">parameters · from-scratch transformer<br/>MoE + ACT, BPE 32 k vocab</div>
  </div>
  <div class="metric-card">
    <div class="big">5.36e⁻⁶</div>
    <div class="keyline"></div>
    <div class="cap">Qwen3-1.7B + LoRA val loss<br/>100 % test accuracy on 253 examples</div>
  </div>
  <div class="metric-card">
    <div class="big">162 KB</div>
    <div class="keyline"></div>
    <div class="cap">INT8 ONNX encoder<br/>runs in the user's browser tab</div>
  </div>
</div>

One repository. Three language-model arms. Stitched together by a multi-signal smart router.

<div class="source">SOURCE — scripts/verify_numbers.py · 22 / 22 PASS · 2026-04-28</div>

---

<!-- _class: divider -->

<div class="number">01</div>
<div class="label">The cascade</div>

---

<!-- _class: split-3 -->

# Three arms, one cascade

<div class="col-slm">
<div class="arm-label">Arm A — mint</div>
<h2>From-scratch SLM</h2>
<div class="arm-row"><strong>What.</strong> Custom 204 M decoder transformer.</div>
<div class="arm-row"><strong>Role.</strong> Open-ended chat from on-device weights.</div>
<div class="arm-row"><strong>Fires when.</strong> Default for every chat turn.</div>
</div>

<div class="col-qwen">
<div class="arm-label">Arm B — amber</div>
<h2>Qwen3-1.7B + LoRA</h2>
<div class="arm-row"><strong>What.</strong> LoRA-fine-tuned intent parser.</div>
<div class="arm-row"><strong>Role.</strong> Deterministic JSON for HMI commands.</div>
<div class="arm-row"><strong>Fires when.</strong> Command regex matches.</div>
</div>

<div class="col-gemini">
<div class="arm-label">Arm C — lavender</div>
<h2>Gemini 2.5 Flash</h2>
<div class="arm-row"><strong>What.</strong> Cloud foundation-model fallback.</div>
<div class="arm-row"><strong>Role.</strong> Out-of-distribution safety net.</div>
<div class="arm-row"><strong>Fires when.</strong> Local arms can't ground.</div>
</div>

---

# `route_decision` ships on every reply

```json
{
  "arm":          "slm+retrieval",
  "model":        "AdaptiveTransformerV2",
  "query_class":  "default_chat",
  "score":        0.91,
  "smart_scores": {
    "greeting":      0.02,
    "cascade_meta":  0.05,
    "system_intro":  0.03,
    "world_chat":    0.04,
    "default_chat":  0.91
  }
}
```

Six deterministic signals → five route classes → highest-scoring class wins. Surfaced to the user as a routing chip with hover tooltip.

<div class="source">SOURCE — i3/pipeline/engine.py · _smart_score_arms</div>

---

<!-- _class: diagram-anchor -->

<div class="caption">One turn, end-to-end:</div>

```
intake  →  coref  →  encode (TCN)  →  adapt (8-axis)
   →  smart router  →  command gate  →  Qwen LoRA
   →  Gemini backup  →  retrieve  →  SLM forward + critic
   →  cloud fallback  →  tools  →  adapt rewrite
   →  side-effect dispatcher (timers fire here)
```

<div class="takeaway">Every reply animates this 14-stage pipeline live in the Flow tab.</div>

---

<!-- _class: divider -->

<div class="number">02</div>
<div class="label">From-scratch and fine-tuned, side by side</div>

---

# AdaptiveTransformerV2 — 204 M params, hand-written

<div class="data-table">
<div class="section-label">Architecture</div>

| Property | Value |
|---|---|
| Parameters (unique, tied weights) | **204.4 M** |
| Layers · heads · `d_model` | 12 · 12 · 768 |
| FFN | MoE, 2 experts, top-1 routing |
| Halting | Adaptive Computation Time (Graves 2016) |
| Conditioning | Per-layer cross-attention onto 8-axis adaptation vector + 64-dim TCN user state |
| Tokenizer | Byte-level BPE, 32 k vocab, written from scratch — `0` HuggingFace deps |
| Training corpus | 977 332 dialogue pairs (300 k subset trained on) |

<div class="section-label">Result</div>

| Metric | Value |
|---|---|
| Best `eval_loss` | **4.987** at step 18 000 |
| Headline perplexity (training-eval, response-only) | **≈ 147** |
| Stress-test perplexity (n=500, full corpus, all-token loss) | ≈ 1 725 |

</div>

<div class="source">SOURCE — checkpoints/slm_v2/best_model.pt · reports/slm_v2_eval.md</div>

---

# Qwen3-1.7B + LoRA — both/and, not either/or

<!-- _class: split-2 -->

<div>
<div class="section-label" style="font-size:13px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:var(--space-2);border-bottom:1px solid var(--border);padding-bottom:var(--space-1);">Recipe</div>

| | |
|---|---|
| Base | Qwen3-1.7B |
| Rank · alpha | 16 · 32 |
| DoRA | enabled |
| NEFTune α | 5.0 |
| 8-bit AdamW | enabled |
| Schedule | cosine warm restarts |
| Effective batch | 8 (2 × grad-accum 4) |
| Epochs · steps | 3 · 1 704 |
| Train · val · test | 4 545 · 252 · 253 |

</div>

<div>
<div class="section-label" style="font-size:13px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:var(--space-2);border-bottom:1px solid var(--border);padding-bottom:var(--space-1);">Result</div>

| | |
|---|---|
| Best val_loss | **5.36 × 10⁻⁶** |
| Test action accuracy | **100 %** |
| Test full-match | **100 %** |
| Test macro F1 | **1.000** |
| Wall time | 9 656 s ≈ 2.68 h |
| Hardware | RTX 4050 Laptop, 6.4 GB VRAM |

<div class="callout" style="margin-top:var(--space-4);">
<div class="label">JD framing</div>
"Build models from scratch <strong>and</strong> adapt or fine-tune pre-trained models." — both/and.
</div>

</div>

---

<!-- _class: divider -->

<div class="number">03</div>
<div class="label">The edge</div>

---

<!-- _class: hero-metric -->

<div class="metric">162 KB</div>
<div class="label">INT8-quantised TCN encoder, running in the user's browser tab</div>

<div class="footer">441.4 KB → 162.2 KB · −63.25 % size · MAE 0.000548 vs FP32 · p50 460 µs · 2 176 enc/s · 12.5 × under the Kirin A2 watch's 2 MB encoder budget · DevTools shows zero /api/encode requests when the toggle is on</div>

---

<!-- _class: divider -->

<div class="number">04</div>
<div class="label">The HCI argument</div>

---

# Implicit > explicit, when cognitive bandwidth is scarce

<!-- _class: split-2 -->

<div>

Drivers, wearable users, accessibility users — none of them have spare cognitive bandwidth for self-reflective preference elicitation.

Reading state from typing rhythm costs **zero** additional work; the system infers context from the interaction itself.

<div class="callout">
<div class="label">Honest gap</div>
No real user study yet. n = 20 within-subjects validation is open problem #5.
</div>

</div>

<div>

<div class="section-label" style="font-size:13px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:var(--space-2);border-bottom:1px solid var(--border);padding-bottom:var(--space-1);">Three references</div>

**Strayer & Cooper (2017)**<br/>
<span style="color:var(--text-secondary);font-size:18px">35 % reaction-time drop during in-vehicle interaction.</span>

**Wobbrock et al. (2011)**<br/>
<span style="color:var(--text-secondary);font-size:18px">Ability-based design: tap-input fails for tremor / wrist-screens.</span>

**Lee & See (2004)**<br/>
<span style="color:var(--text-secondary);font-size:18px">Calibrated trust requires reasoning visible to the user — the routing chip's job.</span>

</div>

---

<!-- _class: inverted -->

# The punch list I'd hand a teammate

| | | |
|---|---|---|
| **#1** | Kirin watch deployment of the encoder | 1 week, blocked on dev kit |
| **#2** | Full-corpus SLM v2 retrain | 30 h GPU, target ppl < 80 |
| **#3** | A/B harness for the route_decision chip | 2 weeks, IRB-blocked |
| **#4** | Multilingual cascade | 3 days |
| **#5** | User-state validation study | 3 weeks, IRB-blocked |
| **#6** | Replace warm-restart notes with real run | 4 h after #2 |

*Solo project. Honest list. This is how I'd work in HMI Lab — scope tight, constraints explicit, validation criteria pre-registered.*

---

<!-- _class: closing -->

# Questions?

<div class="repo">github.com/abailey81/implicit-interaction-intelligence</div>
