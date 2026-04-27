# Design brief — Implicit Interaction Intelligence (I³)

> **Iter 51 (2026-04-27).**  One-page persona + interaction principle +
> A/B vs status quo.  Closes the JD's "HCI principles, design thinking,
> concept-driven prototyping" desired bullet.

---

## The hypothesis in one sentence

> *On-device personalisation is most useful when it adapts to **how** a
> user interacts, not **what** they say about themselves.*

The status-quo personalisation pattern (ChatGPT memory, Gemini Personal
Context, Apple Personal Knowledge Graph) requires the user to **declare**
preferences, then the model **remembers** them.  This trades effort for
recall.  I³ inverts the trade: the model learns from the **rhythm** of
typing, the **richness** of vocabulary, and the **drift** of session
state — signals the user generates whether they want to or not.  No
declaration; no recall failure.

---

## Persona

**Name.**  Maya Chen.  29.  Product designer at a Cambridge wearable
startup.

**Why she'd use I³.**  Maya iterates on her own prompts and her team's
prompts every day.  She's tired of:
1. **Re-introducing herself** to every assistant.  ("I'm a designer,
   I think visually, please be concise.")
2. **Having to ask for "simpler" / "shorter" / "more formal"** explicitly
   on every turn.
3. **Watching her partner accidentally trigger her ChatGPT history**
   when they share a laptop.

**What changes for her with I³.**
* Her keystroke rhythm registers her at the keyboard within 5 turns
  (Identity Lock).  Her partner's rhythm doesn't unlock her per-user
  LoRA adapter.
* The model picks up her preference for short, direct, low-jargon
  answers from how she types — without her having to say so.
* Her facts (name, role, location) persist across sessions, encrypted
  at rest, and one utterance ("forget my facts") wipes them.

**What this constrains.**
* Maya is one user.  We don't build for the everyone-on-Earth user; we
  build for *the user with a keyboard pattern long enough to converge
  the encoder*.  Smartphone-only users are a degraded experience until
  the touch / prosody encoders mature.
* Maya values privacy enough to forgive the slightly-smaller model
  (204 M vs 3 B) and the absence of cloud knowledge.  She's the
  user who'd choose Signal over WhatsApp.

---

## Interaction principle

**Implicit > explicit.**  Every adaptation signal in I³ comes from
*behaviour the user already produces*, not from *settings they
configure*:

| Signal | Source | What it tells the model |
|---|---|---|
| Inter-keystroke interval (IKI) | Native key events | Tiredness, cognitive load, mood drift |
| Backspace burst rate | Native key events | Uncertainty, drafting style |
| Word-length distribution | Composed text | Vocabulary register, intended formality |
| Sentence-length distribution | Composed text | Verbosity preference |
| Pause-before-send | Native key events | Confidence, second-guessing |
| Voice prosody | Mic (opt-in) | Affect state |
| Facial action units | Camera (opt-in) | Engagement, surprise, frustration |
| PPG / HRV | Wearable (opt-in) | Stress, recovery state |
| Touch dynamics | Touchscreen | Frustration, deliberation |
| Session topic graph | Tracked entities | What's currently in the user's head |

**Translated to the architecture**: each signal feeds the TCN encoder's
input channel ([`i3/encoder/tcn.py`](../../i3/encoder/tcn.py)).  The
encoder produces a 64-d user-state embedding.  The
:class:`AdaptationController` ([`i3/adaptation/`](../../i3/adaptation/))
projects to an 8-axis adaptation vector (cognitive_load, verbosity,
formality, directness, emotionality, simplification, accessibility_mode,
emotional_tone).  The 8-d vector + 64-d user state cross-attend into
every layer of the SLM during generation.  No prompt prefix; no
preference text; no remembered settings.

---

## What I3 looks like for a user vs the status quo

| Workflow | ChatGPT (status quo) | I³ |
|---|---|---|
| First three turns | Verbose, generic, formal-by-default | Slightly-verbose baseline; encoder warming up |
| Turn 5+ once Maya's pattern is recognised | Same as turn 1 unless she explicitly steers | Adapts: shorter, more direct, lighter jargon |
| Cross-laptop continuity | Cloud account; same answers regardless of keyboard | Identity Lock detects the new typist; per-user LoRA only fires when her rhythm matches |
| Privacy contract | Cloud-stored history; manual deletion | On-device storage; one utterance clears every fact |
| Tiredness / accessibility | User must say "give me the short version" | Inferred from typing dynamics; auto-engages accessibility mode (smaller words, shorter sentences) |
| Topic carryover | Cloud-stored history; usually reliable but occasionally surprises | On-device entity tracker with explicit pivot/anchor model — verifiable in [`i3/dialogue/coref.py`](../../i3/dialogue/coref.py) |

---

## Three design decisions, with rationale

### Decision 1: Cross-attention conditioning on every block, not a prompt prefix

Prompt-prefix conditioning ("you are a helpful assistant talking to a
designer who prefers short answers") is what every other personalisation
stack does.  It's brittle: the model can ignore the prefix; the prefix
inflates the context window; it must be re-emitted on every call.

Cross-attention conditioning makes the personalisation
**architectural** — the user-state vector is consumed by the same
mechanism that consumes content, with the same gradient flow.  The
model can no more "ignore" it than it can ignore its own input.

Trade-off accepted: this requires custom transformer code (the SLM is
not a HuggingFace base).  The from-scratch SLM is the price of
architectural personalisation.

### Decision 2: Contextual bandit for routing, not RLHF

Local SLM vs cloud LLM is a routing decision per-turn.  Full RLHF is
overkill — we have one routing axis, sparse rewards, and a strong
prior (default to local).  A LinUCB / Thompson-sampling bandit
([`i3/router/bandit.py`](../../i3/router/bandit.py)) converges in
~10 turns per user; full RLHF would need orders of magnitude more
data and a critic model.

Trade-off accepted: the bandit is myopic per-turn; multi-turn
optimisation is left to manual rules + the explicit user "switch
to cloud" override.  Iter 49+'s preference-learning module (DPO over
the bandit's reward signal) closes this gap.

### Decision 3: Encrypted at-rest fact memory, not in-RAM only

The simplest version of cross-session memory is a Python dict.  That
loses everything on restart.  The next-simplest is an SQLite table
with raw text.  That violates the "no raw text persisted" privacy
invariant.

We picked: **typed slots** (name, location, occupation, etc.)
encrypted via the same Fernet envelope as the embedding columns.
The user controls retention via "forget my facts."  This is the
only piece of personal text in the entire DB and it's both opt-in
(only stored when the user says "my name is …") and reversible.

Trade-off accepted: free-form facts ("I love hiking on the weekends
when the weather is nice") aren't captured — only the typed slots.
Free-form fact extraction is left to a future iteration with a real
NER model.

---

## What HMI Lab would extend this to

If I joined HMI Lab, the obvious next steps:

1. **Multi-device profile sync** via HarmonyOS Distributed Data
   Management — the encrypted facts table is already shaped for
   syncing.
2. **Kirin NPU quantisation** beyond INT8 — INT4 for the smallest
   wearables (Smart Hanhan band).  See [`docs/huawei/kirin_deployment.md`](kirin_deployment.md).
3. **On-device TTS conditioned on the same adaptation vector** — the
   voice should speed up / slow down / soften as the model does.
4. **Real user study** — n ≥ 50 over 4 weeks, currently scaffolded
   only with synthetic personas in [`tests/test_simulation_personas.py`](../../tests/test_simulation_personas.py).
5. **Hardware-backed key storage** — TrustZone / SecureEnclave for
   the Fernet key.  Currently uses an env var.

Forward roadmap detailed in
[`docs/huawei/forward_roadmap.md`](forward_roadmap.md).
