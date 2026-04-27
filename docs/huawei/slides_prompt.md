# Prompt for an LLM with GitHub access — generate the I³ interview slide deck

> **For the user:** copy everything between the two `═══` rules below
> into a fresh Claude conversation that has GitHub repository access
> for `https://github.com/abailey81/implicit-interaction-intelligence`.
> Ask Claude to follow the instructions exactly.  Do not edit the
> prompt; the discipline rules are tuned to prevent the kinds of
> errors recruiters notice.

═══════════════════════════════════════════════════════════════════════

# Task — produce the interview slide deck for the I³ project

You are an expert technical-presentation designer creating a
**30-minute interview slide deck** for an AI/ML internship at Huawei
R&D UK's Human-Machine Interaction Lab.  The candidate is **Tamer
Bailey**.  The project is **I³ — Implicit Interaction Intelligence**.
Source repo:
<https://github.com/abailey81/implicit-interaction-intelligence>.

You will produce **one Markdown file**, formatted for **Marp**
(<https://marp.app/>), which renders to PowerPoint / PDF / HTML.
Output filename: `docs/slides/i3_interview_deck.md`.

## 0.  STOP — read the repository before drafting any slide content

You must not paraphrase from your own knowledge.  Every claim on every
slide must come from a file in the repo.  Before writing any slide:

1. **Read these files in full** (you have GitHub access — fetch them):
   - `README.md`
   - `HUAWEI_PITCH.md`
   - `CHANGELOG.md` (the most recent two `[2026-04-2X]` sections)
   - `docs/huawei/PRESENTER_CHEAT_SHEET.md`
   - `docs/huawei/email_response.md`
   - `docs/huawei/hci_design_brief.md`
   - `docs/huawei/open_problems.md`
   - `docs/huawei/iter51_summary.md`
   - `docs/huawei/jd_to_repo_map.md`
   - `reports/edge_profile_2026-04-28.md`
   - `reports/slm_v2_eval.md`
   - `scripts/verify_numbers.py`
2. **Skim the directory tree** under `i3/` to confirm the components
   exist where the docs say they do.

If a number, file path, or claim doesn't appear in those files,
**don't invent it.  Ask the user.**

## 1.  Audience and tone

- Audience: 2-4 interviewers from Huawei R&D UK HMI Lab.  Mix of
  ML researchers (probably 50 %), HCI / UX people (probably 30 %),
  and a hiring manager (20 %).  Assume PhD-level ML literacy.
- Tone: technical, honest, calm.  Not marketing.  Not boastful.
  Acknowledge limitations explicitly.
- This is an *internship* application.  The bar isn't "shipped
  product"; the bar is "credible technical depth + clear thinking
  about what's next".

## 2.  Time budget — what the deck must cover

Total slot: **30 minutes**.  Suggested split:

| Block | Time | What's in it |
|---|---:|---|
| Opening | 2 min | Slides 1-3.  Title, the HMI problem, the thesis. |
| Architecture overview | 4 min | Slides 4-6.  One cascade diagram + the three arms named. |
| Live demo | 12 min | Slides 7-8 are demo guides; the candidate drives the actual UI on `localhost:8000`. |
| Deep-dive on ONE arm (probably edge) | 4 min | Slides 9-10.  Numbers + DevTools-as-proof story. |
| Honesty + JD mapping | 4 min | Slides 11-12. |
| Closing | 2 min | Slide 13.  The closing line + thanks. |
| Q&A buffer | 2 min | None of your slides; reactive. |

This is a ceiling of **13 content slides**.  Do not exceed it.
**Less is better** — the candidate has been told repeatedly that
"overloaded" was the problem in earlier versions.

## 3.  Slide-by-slide outline (use this as your spine)

### Slide 1 — Title

```
I³  ·  Implicit Interaction Intelligence
A from-scratch HMI assistant that reads how you type

Tamer Bailey  ·  AI/ML Specialist (HMI) — Huawei R&D UK
2026
```

No sub-bullets.  One sentence of subtitle.  Repo link in tiny text
at the bottom.

### Slide 2 — The HMI problem (no project content yet)

Frame: "Why implicit-first matters in HMI".  Three bullets, one
literature reference each:

- **Cognitive load.**  Strayer & Cooper 2017 measured a 35 % drop in
  reaction time during in-vehicle infotainment interaction.
  Preference-elicitation prompts directly compound that.
- **Motor accessibility.**  Wobbrock et al. 2011 frame this as the
  *ability-based design* gap.  Fine sliders / 8 toggle controls fail
  on a 1.4-inch wrist screen.
- **Recurring overhead.**  Telling the assistant "be concise" every
  turn is friction; saying it once becomes stale state.

Source: `docs/huawei/hci_design_brief.md` §"The HMI problem".

### Slide 3 — The thesis (the project's one big idea)

One sentence at the top:

> Read **how** the user types, not **what** they declare.  The
> 32-dim feature vector our TCN encoder consumes is already produced
> by the act of typing — cost to the user is zero additional work.

Then a 2-column layout:
- **Left:** signals (keystroke inter-arrival, edit count,
  composition cadence, linguistic complexity).
- **Right:** what those signals condition (cognitive-load axis,
  formality axis, …).

Source: `README.md` lead + `docs/huawei/hci_design_brief.md` §"The
design choice".

### Slide 4 — The cascade in one diagram

The single most important architectural slide.  Do **not** use a
cluttered "every component" diagram.  Render this exact ASCII /
Mermaid flow:

```
                       ┌────────────────────────┐
   user message ──────►│   smart router         │
                       │   (5 classes, scored)  │
                       └─┬──────┬─────┬─────┬───┘
                  ┌──────┘      │     │     └──────┐
                  ▼             ▼     ▼            ▼
              greeting     command   chat /     cascade-
              (regex)      (regex    KG-hit     meta /
              local        gate)                world-chat
              hand-                                 ▼
              written       ▼          ▼       ┌────────┐
                            B          A       │   C    │
                       Qwen 1.7B   from-scratch│ Gemini │
                       + LoRA       SLM 204 M  │ cloud  │
                                    + retrieval│ (last  │
                                               │ resort)│
                                               └────────┘
```

Caption: *"Cheapest arm that can give a confident, schema-valid,
on-topic answer wins.  Per-arm confidence shown on every reply."*

Source: `HUAWEI_PITCH.md` TL;DR.

### Slide 5 — Arm A: the from-scratch SLM

| Property | Value |
|---|---|
| Architecture | 12 layers × 12 heads, d_model 768, MoE-2 + ACT halting |
| Vocab | 32 k byte-level BPE (from-scratch tokeniser) |
| Parameters | **204 M** (unique, tied weights) |
| HF dependencies in generation path | **0** |
| Training data | 974 k synthetic dialogue pairs, 300 k subset trained on |
| Training-time held-out perplexity | **≈ 147** (`exp(4.987)`, response-only) |

**Be precise about the perplexity:** quote 147 (training-time, response-only,
same-distribution).  There is *also* a stress-test number of 1725
(broader sample + history-token loss) — mention only if asked.

Source: `reports/slm_v2_eval.md` (the "Read this before quoting any
number" preamble).

### Slide 6 — Arm B: Qwen + LoRA intent parser

Headline: **"Fine-tuned a 1.7 B foundation model for a specific HMI task."**

| Property | Value |
|---|---|
| Base | Qwen3-1.7B |
| Adapter | LoRA rank 16, alpha 32, **DoRA** + **NEFTune** + **8-bit AdamW** |
| Schedule | Cosine warm restarts, 3 epochs, 1 704 optim steps |
| Data | 4 545 train / 252 val (synthetic HMI commands) |
| Best val_loss | **5.36 × 10⁻⁶** |
| Wall time | 9 656 s (2.68 h) on a 6 GB laptop GPU |

Source: `checkpoints/intent_lora/qwen3.5-2b/training_metrics.json`,
verifiable via `scripts/verify_numbers.py`.

### Slide 7 — Arm C: Gemini, only as last resort

One sentence: *"Cloud arm fires only when the local arm can't ground
the query.  Conversation-history-aware via the last 4
(user, assistant) pairs."*

Then a tiny 4-row table showing the routing decision per query
class:

| Query class | Arm | Why |
|---|---|---|
| `greeting` | local hand-written | no LLM call needed |
| `command` (regex gate matches) | Qwen LoRA | deterministic JSON |
| `default_chat` (KG / system anchor) | SLM + retrieval | curated content |
| `world_chat` (no anchor) | Gemini | local KG can't ground |
| `cascade_meta` (asking about the system) | Gemini | only arm with the I³ persona prompt |

Source: `docs/huawei/jd_to_repo_map.md` "Smart cascade" table.

### Slide 8 — LIVE DEMO (this slide is just a header during the demo)

Single full-screen line:

> **LIVE DEMO**

Speaker notes (visible to candidate via Marp's presenter mode):

- The 5-chip flow from `docs/huawei/PRESENTER_CHEAT_SHEET.md` §"The 10-minute live demo (5 chip + 1 actuator)".
- Pre-flight: server running with `I3_PRELOAD_QWEN=1`; tab hard-reloaded; DevTools Network tab open.
- Order: chip 1 (SLM intro) → chip 2 (Qwen + 30-sec timer arms) → chip 3 (Gemini Uzbekistan; timer fires here) → chip 4 (KG photosynthesis) → chip 5 (Qwen navigate).
- Edge-power-move (during chip 1 idle): switch to State tab, flip "Run inference in browser" toggle ON, point at Network panel showing zero `/api/encode` requests.

### Slide 9 — Edge deployment, demonstrable live

Headline: **"The encoder runs in your browser tab.  Verifiable in DevTools."**

| Metric | Value |
|---|---|
| Encoder INT8 ONNX size | **162 KB** |
| Reduction vs FP32 | **63 %** (442 → 162 KB) |
| Parity vs FP32 | **MAE 0.0006**, max abs err 0.0018 |
| p50 inference (x86 CPU, WASM) | **460 µs** |
| Throughput | 2 176 encodes/sec |
| Kirin A2 watch RAM budget | 2 MB |
| **Headroom** | **12.5 ×** |

One-line caption: *"Demonstrable proof in DevTools — zero `/api/encode`
requests when the toggle is ON.  Keystrokes never leave the page."*

Source: `reports/edge_profile_2026-04-28.md` §"Encoder (TCN, 64-d
user-state embedding)".

### Slide 10 — Real side-effects (not just acks)

Headline: **"Commands actually do things."**

Three rows:

- `set timer for 30 seconds` → asyncio task scheduled → **gold-pulse banner fires after 30 s**: *"⏰ Your 30 sec timer is up."*
- `navigate to trafalgar square` → blue actuator banner: *"➤ Navigating to · trafalgar square"*
- `cancel` → tears down all pending tasks for the user

Source: `server/websocket.py` `_fire_actuator_side_effects` +
`web/js/chat.js` `appendActuatorEvent`.

### Slide 11 — What this is NOT (the honesty slide)

Title literally: **"What this prototype is NOT"**

Five bullets, each one sentence:

- **Not chat-quality competitive with GPT-4 / Claude.** SLM standalone
  is fragmentary; cascade content quality comes from retrieval.
- **Not deployed to a Kirin device.** Browser is a stand-in for a
  watch; the actual on-device deployment is open problem #1.
- **Not validated by users.**  The 8 adaptation axes have HCI-
  literature rationale but no user study.
- **Not a single trained model.**  Three arms with different roles;
  the cascade is the differentiator.
- **Not collaborative work.**  Solo project — `open_problems.md`
  shows how I'd scope work for a teammate.

Source: `docs/huawei/PRESENTER_CHEAT_SHEET.md` §"What this prototype
is *not*" + `docs/huawei/hci_design_brief.md` §"What this brief is
*not*".

### Slide 12 — Mapping to the JD

Two-column table.  Left: JD bullet.  Right: evidence in the repo.

| JD requirement | Evidence |
|---|---|
| Build models from scratch | 5 hand-written components — `email_response.md §1` |
| Fine-tune pre-trained | Qwen3-1.7B + LoRA, val_loss 5.36×10⁻⁶ |
| Pipeline orchestration from blueprints | 14-stage cascade, `route_decision` per turn |
| Edge deployment to wearables | INT8 encoder running in-browser; SLM is open problem #1 |
| User modeling | TCN encoder + 8-axis adaptation vector |
| Context-aware systems | coref-aware cascade + topic-consistency gate |
| HCI principles | Strayer / Wobbrock / Lee references in `hci_design_brief.md` |
| Concept-driven prototyping | 100+ iter docs, 23 commits this week, `open_problems.md` |

Source: `docs/huawei/jd_to_repo_map.md`.

### Slide 13 — Closing

Single sentence, large type, centered:

> *"I³ is the smallest end-to-end stack I could build that actually
> implements implicit interaction — a from-scratch language model that
> conditions on how you type, end-to-end privacy, and a cascade that
> degrades gracefully.  Whatever happens with this internship, this is
> the project I'd keep building."*

Then a smaller line:

> *Thank you.  Happy to take questions.*

No bullet points.  No logo.  Just the line.

## 4.  Marp specifics

The output **must** start with this front-matter so it renders cleanly:

```markdown
---
marp: true
theme: default
size: 16:9
paginate: true
header: "I³ — Implicit Interaction Intelligence · Tamer Bailey · Huawei R&D UK HMI Lab"
footer: "github.com/abailey81/implicit-interaction-intelligence"
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0e1116;
    color: #e6edf3;
    padding: 60px 80px;
  }
  h1 { font-size: 40px; font-weight: 600; color: #f0f6fc; }
  h2 { font-size: 28px; font-weight: 500; color: #c8d3e0; }
  table { font-size: 18px; border-collapse: collapse; }
  th, td { padding: 6px 12px; border-bottom: 1px solid #30363d; text-align: left; }
  th { color: #8b949e; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; font-size: 12px; }
  code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
  blockquote { border-left: 3px solid #58a6ff; padding-left: 16px; color: #c8d3e0; }
  strong { color: #f0f6fc; }
  ul, ol { line-height: 1.5; }
---
```

Slide separator: a line containing only `---`.  Use Marp's `<!-- _class: -->`
directive only if you genuinely need a different layout for one slide
(e.g. the closing slide may want `<!-- _class: lead -->` for centred text).

## 5.  Anti-patterns — do NOT do any of these

- ❌ Do **not** put more than 6 lines of body text on a slide.
- ❌ Do **not** use bullet points more than 3 levels deep.
- ❌ Do **not** put screenshots on slides.  The live demo IS the screenshot.
- ❌ Do **not** invent metrics or round numbers in a way that drifts
  from the source.  Every number must trace to a file.
- ❌ Do **not** quote 1725 perplexity as the headline number.  Use 147
  (training-time, response-only) and qualify it.  1725 is the
  conservative stress-test number; mention it only if asked.
- ❌ Do **not** claim the SLM has been deployed to a watch.  It hasn't.
- ❌ Do **not** add a "Future work" or "Roadmap" slide that goes on
  for paragraphs.  Open problems live in `open_problems.md`; the
  honesty slide (#11) is enough.
- ❌ Do **not** include emoji as decoration.  One discreet icon per
  cascade arm in slide 4 is the limit.
- ❌ Do **not** translate any of the content.  Keep British English
  spelling throughout (organisation, recognise, prioritise) — match
  the existing repo style.
- ❌ Do **not** use clip-art or stock photos.

## 6.  Verification discipline (do this BEFORE writing the file)

Before producing the deck:

1. **Run `scripts/verify_numbers.py`** (or read its source).  Confirm
   every claim in §3 above traces to the artefact:
   - SLM 204 M (unique), eval_loss 4.987 → ppl ≈ 147
   - Qwen LoRA val_loss 5.36×10⁻⁶, 1 704 steps × 3 epochs, 9 656 s wall
   - Encoder INT8 162 KB, FP32 442 KB (-63 %), parity MAE 0.00055
   - 31 KG subjects, 974 k corpus
2. **Cross-check perplexity** with the slm_v2_eval.md preamble.
   The 147 is from `best_eval_loss = 4.987`; the 1725 is the
   stress-test.  Quote the right one.
3. **Cross-check the cascade route classes** by grepping
   `i3/pipeline/engine.py` for `_smart_score_arms` — there should be
   **5 classes**: `greeting`, `cascade_meta`, `system_intro`,
   `world_chat`, `default_chat`.
4. **Cross-check edge timings** with `reports/edge_profile_2026-04-28.md`
   — p50 460 µs is the real number; throughput is 2 176 enc/sec; size
   is 162.2 KB.

If anything fails to verify, **stop and report it** — do not produce
slides that drift from the artefacts.

## 7.  Output discipline

- **One file**: `docs/slides/i3_interview_deck.md`.
- **Marp-renderable** out of the box (`npx @marp-team/marp-cli@latest
  docs/slides/i3_interview_deck.md -o deck.pdf` should produce a clean
  PDF without warnings).
- **No companion files** unless absolutely required.  No image
  attachments.
- **Speaker notes**: include them where they help (especially slide 8,
  the live-demo slide).  Use `<!-- ... -->` HTML comments inside each
  slide section for Marp speaker notes.
- **Word count discipline**: the entire deck body should fit in
  ~1 200 words of slide content (excluding speaker notes).

## 8.  Self-review checklist (run after drafting, before returning)

Before returning the file to the user, walk this checklist
mentally:

- [ ] Every number on every slide can be traced to a specific file in
      the repo.
- [ ] Slide count is **≤ 13**.
- [ ] No slide has more than 6 visible body lines.
- [ ] Slide 11 (honesty) is present and honest — not softened.
- [ ] Slide 4 (cascade diagram) is the visual centerpiece — not
      buried among other diagrams.
- [ ] The closing line on slide 13 is verbatim from
      `docs/huawei/PRESENTER_CHEAT_SHEET.md`.
- [ ] British English spelling throughout.
- [ ] No emoji used as decoration; only the discreet arm icons in
      slide 4 if at all.
- [ ] Marp front-matter is at the top; theme renders correctly.
- [ ] Speaker notes on slide 8 (LIVE DEMO) include the chip-by-chip
      flow + the edge-inference power move.

## 9.  How to deliver

Return:
1. The complete contents of `docs/slides/i3_interview_deck.md`.
2. A short summary (≤ 10 lines) of any choice points you made (e.g.
   "I combined arms B and C onto a single slide because…") and any
   verification outcomes (e.g. "verify_numbers.py: 22/22 PASS").
3. **A list of any number you couldn't trace to a file**, with the
   exact phrasing from this prompt and an explanation of why you
   couldn't verify it.  Don't invent.

That's the whole task.  No additional artefacts, no companion docs,
no PowerPoint — Marp markdown only.

═══════════════════════════════════════════════════════════════════════

## How to use this prompt (for the user)

1. Open a fresh Claude conversation that has GitHub repo access (Claude.ai with the GitHub connector enabled, or Claude Code with the repo cloned locally).
2. Paste the entire block between the two `═══` rules above as a single message.
3. Wait for Claude to produce `docs/slides/i3_interview_deck.md`.
4. Save the file into the repo at that path.
5. Render it:
   ```powershell
   npx @marp-team/marp-cli@latest docs/slides/i3_interview_deck.md -o deck.pdf
   # or for PowerPoint:
   npx @marp-team/marp-cli@latest docs/slides/i3_interview_deck.md -o deck.pptx
   ```
   (Needs Node.js installed.  If you don't have Node, use the **Marp for VS Code** extension instead — install it, open the .md file, click "Export slide deck" in the bottom-right.)
6. Walk through the deck once, comparing every number to `scripts/verify_numbers.py` output.  If anything looks off, regenerate the slide with a follow-up prompt: *"Slide N has X — but the artefact says Y.  Fix it."*
7. The deck is your visual layer; the live demo is the substance.  **Don't read your slides during the talk** — they're a backdrop.
