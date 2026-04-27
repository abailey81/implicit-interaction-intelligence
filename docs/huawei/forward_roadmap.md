# Forward roadmap — what I³ becomes at Huawei

> **Iter 51 (2026-04-27).**  What I would build next at HMI Lab,
> tied to Huawei's public 2025–2026 product / research roadmap.
> Closes the JD's *"applied innovation environments"* desired bullet
> and demonstrates the forward-thinking the lab will look for.

---

## Why this section exists

A reviewer asks two questions about the project:

1. *"What's done?"* — the rest of the docs answer that.
2. *"What would you do next, given six months at the lab?"* — this
   document answers that.

The plan below is sequenced for a hypothetical 6-month internship,
phased so each item *unblocks* the next, and explicitly tied to
Huawei's announced product trajectory (HarmonyOS 6, Kirin NPU
roadmap, Smart Hanhan band, AI Glasses 12-MP camera, Pangu series).

---

## Month 1 — productionisation

Goal: take what's already in the repo and make it deployable on a
real Kirin device, not just demo-able in a browser.

### M1.1 — MindSpore Lite conversion

The custom transformer (`AdaptiveTransformerV2`) and TCN encoder
already export cleanly to ONNX.  Next step: convert ONNX → MindSpore
Lite (`.ms` format) for native Kirin NPU execution.

* Validate Pre-LN transformer + cross-attention conditioning is
  representable in MindSpore Lite's op set.
* Validate INT8 quantisation per-tensor preserves the
  cross-attention path.
* Benchmark Kirin 9000 P50 latency vs the published extrapolation
  (50–80 ms band).
* Files to add: `i3/edge/mindspore_export.py`, `tests/test_mindspore_export.py`.

### M1.2 — TrustZone-backed key storage

The Fernet encryption key currently lives in the
`I3_ENCRYPTION_KEY` env var.  Production needs hardware-backed key
storage — Kirin TrustZone, Samsung Knox, or equivalent.

* Plumb the existing `ModelEncryptor` to consume keys from a
  hardware key handle rather than an env var.
* `i3/privacy/encryption.py` already abstracts the key source.
* New file: `i3/privacy/trustzone_key_provider.py`.

### M1.3 — Smart Hanhan band split

The technical report ([`docs/paper/I3_research_paper.md`](../paper/I3_research_paper.md)
§6.4) proposes a TCN-on-band, SLM-on-paired-phone split for the
32 MB Smart Hanhan budget.  Build it.

* Wire the BLE sync layer in `i3/crossdevice/`.
* Compress the encoder embedding to 16 bytes for transmission.
* Stress-test with simulated band/phone latency in the 100–300 ms
  band.

---

## Month 2 — multi-device profile sync

Goal: the user's adaptation profile follows them across phone,
watch, glasses, laptop.

### M2.1 — HarmonyOS Distributed Data Management integration

The `i3/huawei/hmaf_adapter.py` and `i3/huawei/agentic_core_runtime.py`
modules are scaffolded.  Wire to real DDS:

* Per-user adaptation vector synced via DDS (DDS handles conflict
  resolution).
* Encrypted per-biometric LoRA adapter synced too — but *only*
  when the user's current device passes biometric authentication.

### M2.2 — Cross-device coreference

The on-device entity tracker is currently per-session.  With
DDS sync, the same active topic could persist across the user
moving from phone to watch to glasses.

* Promote `EntityTracker` state to the synced layer.
* Add cross-device topic-anchor TTL to avoid stale topics surviving
  context shifts (user obviously switched task between morning phone
  use and afternoon watch use).

---

## Month 3 — voice-first wearable surface

Goal: HMI Lab's actual product target is wearables.  Voice + intent
parsing is the dominant interaction.

### M3.1 — Streaming voice-prosody fusion

`i3/multimodal/prosody.py` (761 LOC) extracts 8 prosody scalars
from audio.  Currently consumed at end-of-turn.  Wearable use needs
streaming fusion:

* Process audio in 100 ms windows.
* Stream prosody updates into the TCN encoder mid-turn.
* Re-run the adaptation projection on each window — the model can
  start with a baseline and converge as the user speaks.

### M3.2 — On-device intent parser deployed on Kirin A2

The iter-51 Qwen3-1.7B + LoRA intent parser fits at INT8 quantised.
Validate it actually runs on Kirin A2-class hardware:

* Quantise the LoRA-merged Qwen3-1.7B to INT8.
* Convert to MindSpore Lite.
* Benchmark P50 / P95 against the 100 ms wearable-budget target.

### M3.3 — Voice-output adaptation conditioning

`i3/tts/engine.py` already exists; condition the TTS prosody on
the same 8-axis adaptation vector that conditions the SLM.  When
the model is generating short, direct text, the voice should also
be faster and lower-pitched.

---

## Month 4 — vision modality on AI Glasses

Goal: add the camera-side analogue of keystroke dynamics for the
12-MP AI Glasses surface.

### M4.1 — Real-time gaze-aware UI

`i3/multimodal/gaze_classifier.py` (1014 LOC) is the foundation.
Add:

* Gaze-zone tracking (where on the AR display is the user looking?).
* Adaptation-axis "gaze-load" — high gaze-load (rapid saccades, dwell
  times < 200 ms) → bias the SLM toward shorter sentences.
* Gaze-driven UI prompts: when the user looks at a notification,
  surface a 1-line response option.

### M4.2 — Facial-affect-conditioned TTS pacing

Wire `i3/multimodal/vision.py`'s 8 facial-affect features into the
TTS pacing.  When the user looks confused (high brow-furrow AU4),
the TTS slows down.  When the user looks relaxed, normal pace.

### M4.3 — Multimodal user-state learning study

Run a 50-participant 4-week study (synthetic personas already
scaffolded in `tests/test_simulation_personas.py`) measuring whether
the multimodal user state significantly improves adaptation quality
over keystroke-alone.

---

## Month 5 — agentic / tool-use layer

Goal: `i3/huawei/agentic_core_runtime.py` (657 LOC) is scaffolded
but not exercised end-to-end.  Make it the canonical action-routing
layer for the assistant.

### M5.1 — Wire the intent parser into the agentic runtime

The iter-51 intent parser produces `{action, params}` JSON.  The
agentic runtime should:

* Look up the action's handler in a tool registry.
* Validate params against the action's schema (already in
  `i3/intent/types.py:ACTION_SLOTS`).
* Execute the handler (often via HMAF) and return a confirmation.

### M5.2 — Multi-step plans

For compound utterances ("set a 10 min timer and text Sarah I'll be
late"), generate a plan of 2+ tool calls and execute sequentially
with user confirmation per step.

### M5.3 — Adapter for HarmonyOS Agentic Framework (HMAF)

The HMAF adapter exists at `i3/huawei/hmaf_adapter.py` (452 LOC).
Wire it as the production execution backend so I³ tools dispatch
through HMAF.

---

## Month 6 — research paper + open release

Goal: publish.

### M6.1 — Real user study

The synthetic-persona work scales to a real study.  Recruit 50–100
participants, 4-week diary study with weekly surveys + endline NPS.
Measure: task-completion delta vs ChatGPT baseline; perceived
trust delta; time-to-personalisation convergence.

### M6.2 — arXiv preprint

[`docs/paper/I3_research_paper.md`](../paper/I3_research_paper.md) is
already at draft quality (461 lines, full results section).  Final
pass to bring up to NeurIPS / CHI submission standard.

### M6.3 — Open-source release

Move the repo from private to public on GitHub under MIT.  Issue a
v1.0.0 tag.  HuggingFace model card for `AdaptiveTransformerV2`.
Reference implementation for the HMI Lab's research findings.

---

## What this roadmap signals to the recruiter

* I've thought about the project beyond "iteration 51" — there's
  6 months of clear next work, sequenced for impact.
* The roadmap is anchored in Huawei's actual public direction
  (Kirin, HarmonyOS, HMAF, AI Glasses, Smart Hanhan).
* It's calibrated to an internship's scope — six months, deliverable,
  measurable.
* It demonstrates I would *contribute*, not just *consume*, the lab's
  research output.
