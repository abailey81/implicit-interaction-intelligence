# Adaptive Text-to-Speech — The Output-Side Half of Mirroring

*An integration note on conditioning speech synthesis on the same
`AdaptationVector` that conditions response generation, so the output
modality mirrors the input modality.*

---

## 1. Motivation — symmetry, not novelty

Implicit Interaction Intelligence (I³) reads *how* a user types: the
inter-key intervals, the backspace cadence, the vocabulary drift, the
engagement arc. It converts that stream of behavioural signals into an
eight-dimensional `AdaptationVector` whose scalars determine what the
system says back — response length, vocabulary complexity, prosodic
warmth of the written reply, and so on.

Everything up to the point the reply lands on the screen is
well-trodden ground for I³. The gap this note fills is on the other
side of the loop: once the system has decided *what* to say, it can
also decide *how* to say it when the output modality is speech, not
text. A user who types slowly should not be met with a voice that
rattles through the answer at 220 words per minute. A user whose
behaviour indicates elevated cognitive load should hear longer
inter-sentence pauses. A user whose typing cadence signals emotional
warmth should hear a warmer prosody back.

This is a symmetry claim, not a novelty claim. Sagisaka and Campbell
argued for prosody as the primary carrier of speaker attitude in 2004;
Hochman et al. and, later, the Apple AVSpeechSynthesizer
documentation, made the same point for mobile assistants. What I³ adds
is the architectural hookup: a single conditioning vector drives both
the content of the reply *and* the voice of the reply, with the same
fade-on-recovery principle that `docs/responsible_ai/accessibility_
statement.md` documents for the content side.

The motivation also pulls the AI Glasses launch (20 Apr 2026) into
scope. Huawei's AI Glasses ship with real-time translation across
twenty languages; the output modality is speech, not text. Any system
I³ might plug into on that device is speech-first by construction, so
not having a conditioned output path would leave the output modality
speaking in a voice that had no relationship to the user's state. This
document describes the minimum-viable conditioning path that closes
that gap.

## 2. The conditioning map

`i3/tts/conditioning.py` holds the derivation as plain, tested linear
functions. The mapping has five rules, each of which corresponds to
one dimension of the `AdaptationVector`:

**Rule 1 — Cognitive load slows the voice.** A baseline rate of 180
words per minute is reduced linearly by up to 70 wpm across the
`cognitive_load` axis: at `cognitive_load = 0.0` the voice speaks at
180 wpm, at `cognitive_load = 1.0` it speaks at 110 wpm. The
inter-sentence pause is raised from 120 ms to 500 ms over the same
range. Research on cognitive-load-responsive speech interfaces
(Mayer et al., 2010; Vizer, 2009) supports this direction: heavier
cognitive demand tolerates less speech-rate bandwidth.

**Rule 2 — Accessibility dominates.** When `accessibility` exceeds
0.6, the rate is clamped to a hard ceiling of 120 wpm regardless of
what the other dimensions would have produced, the pause floor is
raised to 600 ms, and the enunciation mode is set to `"maximum"`. The
rationale matches `accessibility_statement.md`: the accessibility
signal is the one that most plausibly correlates with a user for whom
the cost of missing information is highest, so its effect overrides
warming or formalising nudges.

Crucially — and this is the subtle part — the system does **not**
flip on a "clear speech mode" flag that stays on and labels the user.
The accessibility scalar itself is an exponential moving average with
a decay half-life of a few sessions. If the user types slowly for
three exchanges and then recovers, the EMA fades and the voice speeds
back up. The same fade-with-recovery logic that prevents content-side
stigmatisation also prevents voice-side stigmatisation.

**Rule 3 — Emotional tone shapes pitch.** The `emotional_tone`
dimension runs from 0.0 (fully warm / supportive) to 1.0 (fully
neutral / objective). Pitch is modulated linearly from −40 cents at
the warmest end to +40 cents at the neutral end, with the
`style_mirror.emotionality` dimension amplifying the swing (a high-
emotionality user hears a larger pitch variance, a low-emotionality
user hears a flatter delivery). The warm end also carries a small
rate nudge (up to +5 % of the rate computed by rule 1) because
perceptual studies of warmth consistently associate it with slightly
higher speaking rate within a conversational register.

**Rule 4 — Style formality applies a ±5 % rate tilt.** A fully formal
user style (`formality = 1.0`) slows the delivery by 5 %; a fully
casual style (`formality = 0.0`) speeds it up by 5 %. This is a small
effect, and intentionally so — the dominant rate driver is cognitive
load, not formality. The purpose of the tilt is to match the
existing content-side style-mirror: if the written words are more
formal, the spoken delivery should feel slightly more measured.

**Rule 5 — Verbosity is NOT a TTS dimension.** The `style_mirror.
verbosity` axis is a content-length signal, and it is already
consumed by `i3/cloud/postprocess.py` to cap the number of sentences
in the reply. Passing it again into the TTS layer would double-count
it. The TTS layer therefore ignores it explicitly in its docstring.

`derive_tts_params(adaptation, base_rate=180)` returns a Pydantic
`TTSParams` instance with every field clipped into its valid range
(rate ∈ [80, 220], pitch ∈ [−100, +100] cents, pause ∈ [100, 1000]
ms). Every clip is tested in `tests/test_tts_conditioning.py`. A
small helper, `explain_params(params, adaptation)`, produces a
one-sentence natural-language explanation — e.g. *"Speech rate set
to 118 wpm with 530 ms inter-sentence pause and maximum enunciation
— accessibility is elevated (0.82), so enunciation is maximum and
the rate is capped at 118 wpm."* — which the web UI renders in an
`aria-live="polite"` region below the "Speak response" button so the
user can see why the voice is behaving the way it is.

## 3. Integration with AI Glasses, Celia, Smart Hanhan

The engine abstraction in `i3/tts/engine.py` holds four pluggable
backends, each of them soft-imported so the module loads on a stock
install that has none of them present.

`PyttsxBackend` uses `pyttsx3` and the operating system's native
speech engine (SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak
on Linux). The prosody control exposed by these engines is the lowest
common denominator — rate and volume universally, pitch on some
platforms — which is why the module intentionally keeps its
conditioning to rate, pitch, volume, and inter-sentence pause rather
than the full set of SSML features the research literature discusses.

`PiperBackend` wraps the Rhasspy Piper neural TTS. Piper is CPU-only,
runs at roughly real-time on a modern laptop without a GPU, and
produces raw int16 PCM that the module wraps in a WAV container.
This is the preferred backend for anything shipping inside a
HarmonyOS-class device: Piper's small voice models (tens of MB) fit
in the edge-device memory envelope and there is no network round
trip. The `PiperVoice` abstraction accepts a `length_scale` parameter
that the module maps directly from `rate_wpm`.

`KokoroBackend` wraps the Kokoro 82 M-param open-weight TTS released
by Hexgrad in 2025. Kokoro's quality at 82 M parameters is
comparable to commercial cloud TTS, which makes it the preferred
option when the target has the resources to run an 82 M-param model
but still wants on-device speech. Celia on a flagship HarmonyOS phone
is the obvious fit.

`WebSpeechApiBackend` is the server-side stub: rather than producing
audio, it returns a directive dictionary the browser uses to drive
`window.speechSynthesis.speak(new SpeechSynthesisUtterance(...))`.
This is the zero-server-cost path and the default on a stock install
where none of the heavy backends are present. The browser runtime
handles prosody directly; the server has only communicated the
AdaptationVector-derived rate / pitch / volume scalars.

The binding to the Huawei 2025-2026 roadmap is straightforward. On
**AI Glasses** the device itself is too small for on-glass synthesis,
so the architecture is encoder-only on-glass with the paired phone
running Piper or Kokoro and returning audio over the HarmonyOS DDM
bus. On **Smart Hanhan** the 64 MB RAM budget means Piper's smallest
voice (around 20 MB) is the only neural option that fits; pyttsx3 on
the paired phone is the fallback. On **Celia** the full Kokoro model
is an option because the target hardware is a flagship phone with
several GB of RAM and a dedicated NPU. The module does not ship a
MindSpore Lite conversion path for Kokoro — that is future work — but
the path is the same as for the I³ SLM: export to ONNX, convert via
`converter_lite --fmk=ONNX`.

## 4. Accessibility — extending the output-side principle

The accessibility statement's central commitment is that
*accessibility is a property of the interaction, not a label on the
person*. The TTS layer has to honour that commitment or the output
modality becomes the place where the principle is broken.

The commitment in practice is three things. First, there is no
"clear speech mode" that the user has to opt into — the adaptation
is continuous and EMA-smoothed, so a user who types more slowly for
three exchanges hears a slightly slower voice for those three
exchanges and a gradual return to normal thereafter. There is no
flag, no profile, no setting menu to label.

Second, the adaptation is visible. The same
`aria-live="polite"` caption that surfaces the content-side
explanation below the chat panel also surfaces the output-side one
("Speech rate set to 118 wpm because cognitive load is 0.82"). The
user can see that the voice has changed, what it has changed to, and
why. This is mandatory under the accessibility statement's visibility
principle and it is also the mechanism by which the user can notice
that something has drifted in a direction they do not want — at which
point the next commitment applies.

Third, the adaptation defers to explicit accessibility settings.
`docs/responsible_ai/accessibility_statement.md` §3.4 says: *"if a
user has enabled a screen reader or voice control, those features
must continue to work unmodified"*. On the output side, that means
the Web Speech API respects the browser-level voice rate preference
when the browser exposes one; the pyttsx3 path respects the OS-level
speech rate override; the Piper and Kokoro paths do the same by
letting the user pin a `voice_id` that they control. The rule is
identical to the content side: explicit setting wins.

## 5. Known limitations — threats to the prototype

Three limitations are worth calling out explicitly so the interviewer
does not have to surface them.

**Backend availability is not guaranteed.** On a bare install none of
pyttsx3, Piper, or Kokoro are present, and the server would otherwise
fail the TTS call. The Web Speech API fallback closes that gap for
any browser client, but a CLI consumer of the library would see a
clear `RuntimeError` with install hints. This is deliberate: the
library raises on `speak()`, not on import, so the rest of the
service is unaffected by a missing TTS backend.

**Prosody control fidelity is uneven.** Pyttsx3 exposes rate and
volume reliably but not pitch; SAPI5 on Windows has no pause-between-
sentences control at the API level. Piper and Kokoro expose a
`length_scale` / `speed` parameter that the module maps from
`rate_wpm`, but the rate-to-length-scale curve is non-linear and the
module's mapping is a first-order approximation. For the prototype
this is acceptable; for a production deployment on AI Glasses the
curve would need to be calibrated per voice.

**English-first.** The conditioning rules assume English
conversational rates (180 wpm baseline, 120 wpm lower end). For
Mandarin the baseline should be closer to 200 syllables per minute
and the rules need different clip bounds; the module has a hook for
this (`base_rate` is a parameter to `derive_tts_params`) but no
calibration data for non-English targets. The AI Glasses launch
supports twenty languages; the prototype does not.

## 6. Operational footprint

No new *required* Python dependency is added. The three heavy
backends live in an optional Poetry group:

```toml
[tool.poetry.group.tts.dependencies]
pyttsx3   = ">=2.90"
piper-tts = ">=1.2"
kokoro    = ">=0.1"
```

Installed via `poetry install --with tts` only when the operator
wants server-side audio; a stock `poetry install` still builds, still
passes every test, still serves the Web Speech API directive path,
and adds zero install latency for the common case. This matches the
supply-chain posture of the rest of the project's optional groups
(observability, mlops, analytics) and the explicit non-goal in
`the advancement plan` §4 to avoid adding more required dependencies.

The route surface is three endpoints — `POST /api/tts`,
`GET /api/tts/backends`, `GET /api/tts/preview?archetype=<N>` —
protected by the same middleware stack that already rate-limits and
size-bounds the other routes. The body cap is 8 KiB, the text cap is
2 000 characters, and PII sanitisation happens before any backend
touches the input. The CLI demo
`scripts/demos/tts.py` renders all eight archetype vectors into
`reports/tts_demo/`, producing one WAV per archetype plus a
parameter-comparison table, so a reviewer can hear the difference
between the neutral baseline and the accessibility-elevated case
without wiring the web UI.

## 7. Citations

- Sagisaka, Y. and Campbell, N. (2004). *Using prosody in speech
  technology*. Speech Communication, 42 (1), 1–4.
- Mayer, R.E., Dow, G.T., and Mayer, S. (2003). *Multimedia learning
  in an interactive self-explaining environment: what works in the
  design of agent-based microworlds?* Journal of Educational
  Psychology, 95 (4), 806–812.
- Vizer, L.M. (2009). *Detecting cognitive and physical stress
  through typing behavior*. CHI '09 Extended Abstracts.
- Epp, C., Lippold, M., and Mandryk, R.L. (2011). *Identifying
  emotional states using keystroke dynamics*. CHI '11.
- Kokoro-82M model card (Hexgrad, 2025). https://huggingface.co/
  hexgrad/Kokoro-82M.
- Piper TTS (Rhasspy project). https://github.com/rhasspy/piper.
- Rhasspy voice assistant toolkit. https://rhasspy.readthedocs.io.
- Apple, *AVSpeechSynthesizer — AVFoundation*, Apple Developer
  Documentation. https://developer.apple.com/documentation/
  avfoundation/avspeechsynthesizer.
- W3C, *Web Speech API specification*, Working Draft, 2020.
  https://wicg.github.io/speech-api/.
- Huawei, *AI Glasses product announcement*, 20 April 2026.
- `docs/responsible_ai/accessibility_statement.md` (this repository).
- `docs/huawei/harmonyos6_ai_glasses_alignment.md` (this repository).
