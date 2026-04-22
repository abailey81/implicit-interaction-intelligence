# Accessibility Statement

*A statement of what the I³ accessibility signal is, what it is not, and
the architectural guarantees that separate the two.*

This document exists because the accessibility beat of the I³ live demo
is the single most value-laden moment in the system. It is also the
moment where the system is most at risk of *overclaiming* —
misrepresenting a statistical correlation as a diagnostic, or a
behavioural suggestion as a label. Getting the framing right is the
difference between a system that helps and a system that patronises.

The position this document takes, stated plainly up front:
**accessibility is a property of the interaction, not a label on the
person.**

---

## 1. What the System Detects

The I³ system observes behavioural signals drawn from a user's typing
behaviour over time. The observable quantities are timing (inter-key
intervals, pause lengths, burst durations), editing behaviour
(backspace rate, correction patterns, composition time), and message
structure (message length, vocabulary diversity, word complexity).
These are fused into a 32-dimensional `InteractionFeatureVector` at
each message boundary, encoded into a 64-dim embedding by the TCN,
and translated into an 8-dim `AdaptationVector` by the adaptation
controller.

One component of that adaptation vector — the single scalar
`accessibility` ∈ [0, 1] — is what this document is principally about.
The scalar rises when the `AccessibilityAdapter` in
`i3/adaptation/dimensions.py` observes elevated correction rate,
slower inter-key intervals, shorter bursts, or a sustained drop in
message length. The exact rule is a weighted combination of those
features thresholded against the user's own long-term baseline, and
the combination is smoothed by an exponential moving average so that
the signal fades on recovery rather than pinning a static label.

That is everything the "accessibility detection" in this system is:
a behavioural scalar over behavioural signals, with EMA-based decay,
grounded in a small and specific body of keystroke-dynamics research
(Epp et al. 2011; Vizer 2009; Zimmermann 2014).

## 2. What the System Does Not Do

The following list is exhaustive for the current prototype and is
intended to be treated as a binding constraint on any downstream use:

- **It does not infer clinical diagnoses.** The system does not and
  will not attempt to identify dyslexia, dyspraxia, dysgraphia,
  tremor, Parkinson's disease, arthritis, ADHD, depression, anxiety,
  or any other clinical condition. There is no classifier in the
  pipeline whose output corresponds to any such label.
- **It does not infer protected characteristics.** The system does
  not infer age, gender, ethnicity, sexual orientation, religion,
  disability status as a legal category, or any other characteristic
  protected under anti-discrimination legislation (UK Equality Act
  2010, EU Charter of Fundamental Rights, US ADA).
- **It does not make decisions with legal effect.** No content is
  withheld, no account is flagged, no service is denied based on the
  accessibility scalar or any other signal from this system.
- **It does not replace explicit accessibility settings.** If the
  user has enabled a screen reader, voice control, a screen
  magnifier, or any operating-system-level accessibility feature,
  those features must continue to work unmodified. The I³
  adaptation is additive to explicit user control, never substitutive.
- **It does not persist raw text.** The diary stores embeddings,
  adaptation vectors, and TF-IDF topic keywords. No message text
  reaches disk, and no message text is sent to a cloud LLM without
  PII sanitisation (`i3/privacy/sanitizer.py`) running first.
- **It does not retain the accessibility scalar as an identity
  attribute.** The scalar is recomputed from recent behaviour at
  every message. It has no privileged "sticky" status in the user
  profile; the long-term EMA that holds it has a decay half-life
  measured in sessions, not days.

## 3. Opt-Out and User Control

The following opt-out guarantees are part of the architectural
commitment, not a nice-to-have:

1. **Detection is opt-out per user.** Any production derivative of
   this prototype must surface a clearly-labelled toggle that
   disables the accessibility adapter entirely. When disabled, the
   scalar is held at 0 and no adaptation occurs on that axis.
2. **Adaptation is visible to the user.** The user-facing dashboard
   exposes the current accessibility scalar in the live demo. A
   production system must do the same — users should be able to see
   that an adaptation is active, what it is doing, and why.
3. **The user can reset session state.** The demo dashboard exposes
   a "reset session" button. A production system must expose an
   equivalent affordance.
4. **Long-term profile is user-owned.** Profiles persist in a local
   SQLite database encrypted with a user-controlled Fernet key
   (`I3_ENCRYPTION_KEY`). Deletion is a file-level operation. On a
   HarmonyOS device this maps cleanly to the user's Distributed
   Data Management controls.
5. **No covert adaptation.** A future product decision not to show
   the adaptation to the user is not a change we endorse. The
   "it just adapts, no settings menu" narrative is about
   *explicit-configuration overhead*, not about concealment; the
   adaptation is always visible on request.

## 4. The Detection ≠ Diagnosis Boundary

The accessibility signal derives from behaviour that correlates
statistically with motor difficulty, cognitive load, and fatigue.
The overlap is real; it is what makes the signal useful at all. It is
also the overlap that makes framing mistakes dangerous.

Consider three users who all produce the same keystroke signature —
slow inter-key intervals, many backspaces, short fragmentary
messages:

- User A is an experienced typist fighting a stiff mechanical
  keyboard on an unfamiliar laptop.
- User B has a mild tremor related to a long-term condition and
  types this way every day.
- User C is composing a difficult emotional message and is pausing
  to think rather than to type.

The system sees the same 32-dim feature vector for all three users
in that window. The responsible behaviour is the same for all three:
reduce the verbosity of responses, simplify vocabulary, offer
shorter confirmations. That behaviour helps each user for different
reasons, and crucially it does **not** require the system to decide
which user it is talking to.

The architectural commitment is to make the helpful adaptation
without making the diagnostic inference. The accessibility scalar
is the amount of help; it is not a class label.

This is why the scalar is an EMA with decay rather than a sticky
flag: if User A switches keyboards and the signal fades, the
adaptation fades. If User C finishes the emotional message and
starts typing fluently again, the adaptation fades. If User B types
at that speed every day, the long-term EMA stabilises and the
adaptation is effectively always on for them — which is exactly
what a well-designed accessibility feature should do.

## 5. Relationship to WCAG 2.2, ARIA, and POUR

The I³ prototype does not replace the Web Content Accessibility
Guidelines (WCAG 2.2), the Accessible Rich Internet Applications
specification (ARIA), or the Perceivable / Operable / Understandable
/ Robust (POUR) framework. Those are the primary accessibility
commitments of any user-facing interface and must be satisfied by
the host application regardless of what I³ does.

The web demo in `web/` aims to be POUR-compliant in its own right:

- **Perceivable** — colour contrast on the dark theme is checked
  against WCAG 2.2 AA (contrast ratio ≥ 4.5:1 for normal text); the
  embedding visualisation provides text descriptions of state
  transitions; gauge values are readable as numbers, not only as
  bar widths.
- **Operable** — keyboard navigation works; the "reset session"
  button is reachable via Tab; WebSocket reconnection happens
  automatically without user intervention.
- **Understandable** — the adaptation dashboard labels what each
  gauge represents in plain English, not jargon.
- **Robust** — ARIA roles on the dashboard elements let assistive
  technology identify them correctly.

Where I³'s adaptation signal interacts with explicit accessibility
settings, the priority is always: **explicit setting wins**. If a
screen-reader-user has set a preference for short responses, I³'s
accessibility adapter does not add further simplification on top;
the explicit preference is the ceiling.

A screen-reader user is also a case where I³'s keystroke-based
signal is effectively blind: screen-reader-driven input does not
produce the kind of keystroke timing I³ is calibrated for. This is a
real limitation — the current prototype cannot detect accessibility
needs in users whose primary input modality is not a keyboard. The
correct response to this limitation is not to ship the prototype as
an accessibility product, but to combine it with explicit
accessibility settings and with signal modalities that do cover
those users.

## 6. Lived-Experience Accessibility and the Standard to Meet

I acknowledge that real accessibility expertise lives with people who
ship accessibility features to real users every day — and that
accessible text-based systems have had to get this right under live
conditions because their users' ability to participate depends on it.
The I³ prototype is nowhere near that standard. It is a research
prototype with a 17-day build window and no real accessibility user
study.

What the prototype does try to honour is a specific principle:
**the interaction should adapt to the person, but the person should
not have to become a label to get the adaptation**. That is what the
EMA-based decay is for; that is what "accessibility is a property of
the interaction, not a label on the person" means in practice. It is
also why this document exists as a statement rather than as a feature.

## 7. Known Limitations of the Current Prototype

For an honest account, the prototype has the following known
limitations in its accessibility behaviour:

- **Keystroke-only signal.** Users who do not type with a keyboard
  are invisible to the current pipeline. Voice-control, eye-tracking,
  switch-access, and other modalities are outside scope. A
  multi-modal extension is future work (`docs/ARCHITECTURE.md` §12).
- **Synthetic training data.** The TCN encoder is trained on
  synthetic data that approximates keystroke-dynamics archetypes
  from published studies. Those studies have small, demographically
  narrow participant pools. The encoder's behaviour on users outside
  those distributions is not well characterised. See `data_card.md`
  §1.
- **English-only.** The linguistic feature extraction (formality,
  valence lexicon, Flesch-Kincaid) is English-specific. A non-English
  user gets a degraded feature vector; the accessibility signal is
  still partially available from the modality-agnostic timing
  features, but the linguistic component is silent.
- **No formal accessibility user study.** The prototype has not been
  evaluated with users from any of the populations the accessibility
  adapter nominally helps. That evaluation is a necessary
  pre-condition for shipping any product derivative and is not a
  step that can be skipped on the basis of the prototype's
  synthetic-data numbers.
- **No clinical-partnership review.** Any claim that this system
  helps users with a named clinical condition would require review
  by professionals qualified to assess that claim. No such review
  has taken place, and no such claim is made.
- **The scalar is a suggestion, not a guarantee.** The
  `AccessibilityAdapter` is a smoothed weighted sum over noisy
  features. It will be wrong sometimes. The architectural mitigation
  is that being wrong in the direction of *more* simplification is
  almost always harmless (a fluent typist who gets a shorter,
  clearer reply is not worse off), while being wrong in the direction
  of *too little* simplification means the user got a default
  response — which is the fallback behaviour of any system that did
  not attempt adaptation at all.

## 8. Closing Statement

The accessibility adapter in I³ is a small, behaviourally-driven
signal with bounded claims and an explicit architectural commitment
to decay on recovery, to never replace explicit settings, to never
infer protected characteristics, to never persist raw text, and to
always be visible to the user.

It is offered in the spirit of making helpful adaptations available
without making the user identify themselves as needing help. The
helpful part is real; the label part is not attempted. That
distinction is what makes this signal worth shipping in the
prototype and what would make it worth shipping in a product.

**Accessibility is a property of the interaction, not a label on the
person.**
