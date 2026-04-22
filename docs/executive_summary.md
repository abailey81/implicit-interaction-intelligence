# Implicit Interaction Intelligence (I³) — Executive Summary

**Audience:** product managers, HMI research leads, and non-technical
reviewers who want to understand *what this system does, what makes
it different, and why it matters* — without being asked to read a
research paper.

**Two-sentence story.** I³ is an on-device AI companion that notices
how you are typing — not what you type — and quietly adjusts the way
it responds: shorter sentences when you are tired, warmer tone when
you are down, simpler language when you are struggling. Unlike most
personalisation systems, it does not ask you to fill in a profile, it
does not keep the words you wrote, and it works on a phone-class
device without sending anything to the cloud.

---

## What it actually does

Imagine a close friend at dinner. They notice, without being told,
that you are tired before you do. They shorten their sentences a
little. They ask yes-or-no questions instead of open-ended ones.
They drop the work jargon. They do not say "I notice you are
tired." They just adapt.

I³ is the same idea, built into software. It watches *how* the user
is typing — the speed, the pauses, the corrections, the rhythm —
and builds a quiet, private picture of the user's current state.
Then, when the user sends a message, the system uses that picture
to shape its reply. Not by adding instructions to a prompt, but by
feeding the picture directly into the brain of the small language
model that writes the reply.

The better analogy might be an automatic adjustment of a glasses
prescription. You do not tell the lens how strong to be. The lens
just is, and what you see is clearer for it.

### Four axes of adaptation

I³ adapts along four independent axes at the same time:

- **Cognitive load** — if you seem tired or distracted, responses
  become shorter and the vocabulary simpler.
- **Style mirroring** — the system sounds roughly the way you
  sound: formal or informal, wordy or terse, warm or neutral,
  direct or circumspect.
- **Emotional tone** — if recent messages have been low-affect or
  sad, the system becomes warmer without becoming performative.
- **Accessibility** — if the typing pattern suggests motor or
  cognitive difficulty, the system switches to short sentences, no
  jargon, explicit step-by-step language. There is no settings
  menu. It just adapts.

---

## Why most "personalised" AI is not actually personal

Most consumer AI products that advertise personalisation do one of
two things. They ask you to describe yourself (the "tell us about
yourself" onboarding flow), or they put a description of you into
the system prompt of a giant cloud model ("you are talking to a
user who prefers…"). Both have problems:

- The first is a chore. Most users do not fill it in, and the
  description goes stale.
- The second only works for giant models. Small models that run on
  a phone often ignore subtle instructions. And every instruction
  you add eats into the space available for the actual
  conversation.

I³ fixes this by not treating personalisation as text. The
implicit-signal picture is a small numerical tensor, not a paragraph.
It goes into a purpose-built slot in the model's architecture.
Every layer of the model is conditioned on it at every step. The
model cannot ignore it — it was trained with the tensor as a first-
class input.

---

## Architecture at a glance

```
  +----------------------+
  |  WHAT YOU TYPE       |   (never saved to disk)
  +----------+-----------+
             |
             v
  +----------+-----------+
  |  HOW YOU TYPED IT    |   (32 features: speed, pauses, edits,
  |                      |    sentence length, vocabulary, ...)
  +----------+-----------+
             |
             v
  +----------+-----------+
  |  YOUR STATE NOW      |   (a small embedding; stored encrypted)
  +----------+-----------+
             |
             v
  +----------+-----------+     +---------------------+
  |  ADAPTATION DECISION |---->| YOUR DEVICE'S AI    |
  |  (4 axes, silent)    |     | (small model, fast) |
  +----------+-----------+     +---------------------+
             |
             v
  +----------+-----------+
  |  WHAT THE AI SAYS    |
  |  BACK TO YOU         |
  +----------------------+
```

---

## What it does today vs what's next

| Area                              | Today (prototype)                                                       | Next                                                                      |
| :-------------------------------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| Input signals                     | Keystroke timing, message content, session rhythm (text only).          | Voice pace, touch pressure, gaze dwell, wearable heart rate.              |
| Device footprint                  | 7 MB quantised, fits on a flagship phone comfortably.                   | 1.3 MB at INT4 — fits on a 64 MB companion device.                        |
| Number of devices                 | Single device.                                                          | Phone + watch + home hub share a synchronised understanding of the user.  |
| Handover between devices          | Not supported.                                                          | Start a conversation on the phone, continue on the tablet, no break.      |
| Training data for user model      | Eight synthetic user archetypes from the literature.                    | Real anonymised longitudinal traces, under a federated-learning setup.    |
| Privacy level                     | Raw text never saved; everything else encrypted on-disk.                | Forward-secret key rotation; hardware root-of-trust on supported devices. |
| Languages                         | English.                                                                | Additional languages; feature extractor is already language-agnostic.     |
| Accessibility detection           | Keystroke-only (misses screen-reader and voice-control users).          | Multi-modal; user-visible state indicator; one-tap opt-out.               |

---

> ## Why this matters for Huawei
>
> The HMI Lab's stated outputs are **prototypes, patents, and
> papers.** I³ contributes to all three in a single project. The
> prototype runs end-to-end in a browser against a laptop backend.
> The architectural cross-attention conditioning mechanism is
> drafted as a provisional patent. The research paper submits to a
> CHI- or ACL-style venue.
>
> The architecture also aligns with three active Huawei directions
> without having been retrofitted to them: the three-tier
> local / phone / cloud structure mirrors HarmonyOS 6's
> understanding / planning / execution split; the 7 MB INT8 budget
> fits the Smart Hanhan companion form-factor; the 64-dim user
> embedding is exactly the size of payload HarmonyOS's distributed
> databus is designed to sync silently across a user's device
> constellation. The accessibility story is the most important of
> these: the system adapts without a settings menu, which is the
> product behaviour the accessibility statement in
> `docs/responsible_ai/` establishes as the bar — detection is
> one signal among many, it complements explicit settings rather
> than replacing them, and it fades with recovery.

---

## Honest caveats

A short list of things this summary has not said, which anyone
evaluating the work should hear.

- **It is a prototype.** The model writes competent short-turn
  replies. It does not write like a frontier model.
- **The user-model training data is synthetic.** It is derived
  from published keystroke-dynamics archetypes in the research
  literature, not from real users at scale. Results on real users
  will differ and will need calibration.
- **The "runs on a Kirin NPU" number is extrapolated, not
  measured.** The memory claim is measured. The per-turn latency
  band of 50–80 ms on a flagship NPU is a calculation from a
  laptop measurement, not a deployment proof. Running on silicon
  is next.
- **Accessibility detection from typing alone is incomplete.**
  It cannot observe screen-reader or voice-control users. The
  adaptation is therefore a *bias* — a nudge toward shorter,
  simpler responses — never a gate that removes a capability.
  User-visible opt-out is part of the design.

---

## What to ask next

Three questions this summary should make it possible to ask
usefully:

1. *"How does the system handle a user who is typing slowly
   because they are distracted, not because they are struggling?"*
   — A: the four-axis design lets cognitive-load rise without the
   accessibility-simplification axis engaging; the two are
   separately mapped from different signal conjunctions.
2. *"What stops this from becoming surveillance-by-typing?"* —
   A: raw text is never stored, the statistics that are stored
   are per-user and encrypted, the model never leaves the device
   when the topic is sensitive, and the 64-dim embedding that
   could move between devices is a lossy abstract summary, not a
   recording.
3. *"If the cross-attention tokens are only four, is that
   enough?"* — A: it is the minimum that encodes the four
   behavioural axes plus a summary of the long-term user state.
   The number is a deliberate knob, not a scientific constant.
   Larger token counts trade parameter overhead for finer-grained
   conditioning.

---

*— End of executive summary. See the research paper
(`docs/paper/I3_research_paper.md`) for technical depth and the
provisional patent disclosure (`docs/patent/provisional_disclosure.md`)
for the formal invention claim.*
