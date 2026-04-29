"""Individual adaptation dimension computers.

Each adapter class is responsible for a single axis of the
:class:`~src.adaptation.types.AdaptationVector`.  They consume user
interaction features and deviation metrics to produce a scalar or vector
that the :class:`~src.adaptation.controller.AdaptationController`
assembles into the full adaptation specification.

Design philosophy
~~~~~~~~~~~~~~~~~
- **Reactive, not prescriptive**: Adapters respond to observed behaviour
  rather than predicting future intent.
- **Smooth transitions**: All dimensions use clamping and rate-limited
  updates to prevent jarring switches.
- **Transparent mapping**: Each adapter's logic is self-contained and
  easily auditable -- critical for HMI systems where users must trust
  that the system's behaviour is appropriate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from i3.adaptation.types import StyleVector, _clamp

if TYPE_CHECKING:
    from i3.config import (
        AccessibilityConfig,
        CognitiveLoadConfig,
        EmotionalToneConfig,
        StyleMirrorConfig,
    )
    from i3.interaction.types import InteractionFeatureVector
    from i3.user_model.types import DeviationMetrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: object, default: float = 0.0) -> float:
    """Coerce *value* to ``float`` while neutralising ``NaN`` and ``None``.

    Adapter inputs come from upstream feature extractors that *should*
    produce finite floats, but defensive coding prevents a single bad
    feature from corrupting the entire :class:`AdaptationVector`.
    """
    # SEC: NaN/None guard at the boundary of every adapter -- without this
    # a NaN feature would silently propagate through ``np.mean`` and
    # ultimately end up in the SLM conditioning tensor.
    if value is None:
        return float(default)
    if not isinstance(value, (int, float, str)):
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(v):
        return float(default)
    if math.isinf(v):
        return float(1.0 if v > 0 else 0.0)
    return v


# ---------------------------------------------------------------------------
# Cognitive Load Adapter
# ---------------------------------------------------------------------------

class CognitiveLoadAdapter:
    """Adapt response complexity to match the user's current cognitive capacity.

    When the user's vocabulary, sentence length, and overall linguistic
    complexity are **declining** relative to their personal baseline, it is
    a strong implicit signal that they are experiencing cognitive overload,
    fatigue, or distraction.  The system should respond with simpler
    language to reduce friction.

    Conversely, when the user engages at or above their baseline complexity,
    the system can provide richer, more nuanced responses.

    The adapter combines four normalised complexity signals from the
    interaction feature vector and adjusts the output based on the
    magnitude and direction of the ``complexity_deviation`` z-score.

    Parameters:
        config: :class:`~src.config.CognitiveLoadConfig` with response
            length bounds and vocabulary level settings.
    """

    def __init__(self, config: CognitiveLoadConfig) -> None:
        self.config = config

    def compute(
        self,
        features: InteractionFeatureVector,
        deviation: DeviationMetrics,
    ) -> float:
        """Compute the target cognitive load level.

        Args:
            features: Current interaction feature vector (32-dim).
            deviation: Deviation metrics from the user's personal baseline.

        Returns:
            A float in [0, 1] indicating the user's measured cognitive
            load.  Iter 38 + iter 53 unified the semantic across the
            pipeline:

              * 0.0  - 0.4  → user has spare bandwidth; reply may be
                              detailed, multi-sentence.
              * 0.4  - 0.6  → moderate; standard reply length.
              * 0.6  - 1.0  → user is stressed / rushed / showing
                              motor-rhythm signs of overload.  Reply
                              should be tight (1–2 sentences).

            Both the cloud prompt-builder and the post-processor's
            length-tiering key off this scale; pre-iter-53 they had
            opposite semantics (the prompt-builder treated high cl
            as "give richer detail" — the now-removed convention).
        """
        # SEC: NaN-safe coercion at the input boundary; downstream arithmetic
        # then never sees ``NaN`` even if a feature extractor mis-fires.
        ttr = _safe_float(features.type_token_ratio)
        mean_word_length = _safe_float(features.mean_word_length)
        flesch_kincaid = _safe_float(features.flesch_kincaid)
        message_length = _safe_float(features.message_length)
        complexity_dev = _safe_float(deviation.complexity_deviation)

        # Iter 38 — typing-rhythm signals also feed cognitive_load.
        # Before: cognitive_load was driven *only* by message-content
        # complexity, so a stressed user typing a short message
        # ("ugh just tell me") got the same cognitive_load as a calm
        # user typing the same short message.  The post-processor's
        # length tiering keys off cognitive_load — so stressed users
        # never got the short replies they need.
        #
        # Now we also incorporate keystroke-rhythm stress signals:
        #   * editing_effort — high edit ratio suggests cognitive
        #     load (struggling to compose).
        #   * backspace_ratio — uncertainty / motor difficulty.
        #   * iki deviation (positive = typing slower than baseline) —
        #     hesitation / fatigue.
        #
        # We combine via ``max()`` rather than ``mean()``: any single
        # signal indicating stress is sufficient evidence — averaging
        # in zero-valued signals (especially iki_deviation, which
        # collapses to 0 with degenerate baselines) dilutes a real
        # stress reading from edits alone.
        editing_effort = _safe_float(features.editing_effort)
        backspace_ratio = _safe_float(features.backspace_ratio)
        iki_dev_pos = max(0.0, _safe_float(deviation.iki_deviation))
        rhythm_stress = max(
            _clamp(editing_effort),
            _clamp(backspace_ratio),
            _clamp(iki_dev_pos),
        )

        # Iter 34 — DYNAMIC RANGE FIX (preserved):
        # Removed the spurious /10 and /20 divisions on already-
        # normalised features so every signal contributes its full
        # [0, 1] range.
        complexity_signals = [
            _clamp(ttr),                          # Vocabulary richness
            _clamp(mean_word_length),             # Word sophistication
            _clamp(flesch_kincaid),               # Readability grade
            _clamp(message_length),               # Normalised length
        ]
        user_complexity = float(np.mean(complexity_signals))

        # If complexity is dropping substantially below baseline, simplify
        # our responses to be *simpler than* the user's current level.
        if complexity_dev < -0.5:
            return _clamp(max(0.2, user_complexity - 0.2))

        # The load is the GREATER of content complexity and typing-
        # rhythm stress — both are independent reasons to return a
        # tighter reply.  A stressed user typing a short message
        # gets the rhythm-driven cognitive_load (content is low but
        # rhythm signals load).  A calm user typing a complex
        # message gets the content-driven cognitive_load.
        #
        # The rhythm stream gets a +0.20 boost above its raw value
        # only once it crosses a meaningful threshold (>= 0.20) —
        # otherwise tiny incidental edits would inflate cl on
        # untroubled users.  This boost is what makes a 4-edit
        # message ("ugh just tell me") cross from the 4-sentence
        # tier into the 2-sentence tier.
        if rhythm_stress >= 0.20:
            rhythm_term = rhythm_stress + 0.20
        else:
            rhythm_term = 0.0
        combined = max(user_complexity, rhythm_term)
        return _clamp(min(0.95, combined + 0.10))


# ---------------------------------------------------------------------------
# Style Mirror Adapter
# ---------------------------------------------------------------------------

class StyleMirrorAdapter:
    """Mirror the user's communication style with a smoothing lag.

    Humans naturally converge communication styles during conversation
    (linguistic accommodation / Communication Accommodation Theory).  This
    adapter replicates that process by observing the user's current style
    along four dimensions and exponentially smoothing toward it.

    The ``adaptation_rate`` (default 0.2) controls how quickly the system
    converges.  A low rate prevents jarring style shifts when the user
    sends a single atypical message; a high rate makes the system feel
    more responsive to deliberate style changes.

    Parameters:
        config: :class:`~src.config.StyleMirrorConfig` with the number of
            style dimensions and the adaptation rate.
    """

    def __init__(self, config: StyleMirrorConfig) -> None:
        self.config = config

    def compute(
        self,
        features: InteractionFeatureVector,
        current_style: StyleVector,
    ) -> StyleVector:
        """Compute the updated style vector.

        Args:
            features: Current interaction feature vector.
            current_style: The style vector from the previous adaptation
                cycle (or :meth:`StyleVector.default` on cold-start).

        Returns:
            A new :class:`StyleVector` smoothed toward the user's observed
            style.  All dimensions are clamped to [0, 1].
        """
        # SEC: NaN-safe coercion of every input feature.
        formality = _safe_float(features.formality)
        message_length = _safe_float(features.message_length)
        emoji_density = _safe_float(features.emoji_density)
        sentiment_valence = _safe_float(features.sentiment_valence)
        question_ratio = _safe_float(features.question_ratio)

        # Derive the user's observed style from interaction features.
        #
        # Iter 35 — VERBOSITY CALIBRATION FIX:
        #
        # Before: ``verbosity = message_length / 0.7``.  Since
        # ``message_length`` is normalised against a 500-word ceiling
        # (FeatureExtractor's ``_MAX_MSG_LEN_WORDS``), real chat
        # messages of 5–50 words produce ``message_length`` of
        # 0.01–0.10, which mapped to verbosity 0.014–0.143.
        # Result: every chat-sized message read as "very low
        # verbosity" → the post-processor's hedge-stripping path
        # always fired regardless of how the user actually typed.
        # The adapter never adapted.
        #
        # After: ``verbosity = message_length / 0.10``.  Calibrated for
        # chat-sized messages — 5-word msgs land near 0.10 (terse
        # → strip hedges), 25-word msgs near 0.5 (default), 50-word
        # msgs near 1.0 (verbose → append follow-up).  The adapter
        # actually adapts.
        observed = StyleVector(
            formality=_clamp(formality),
            verbosity=_clamp(message_length / 0.10),
            emotionality=_clamp(
                max(emoji_density * 5.0, abs(sentiment_valence))
            ),
            # Iter 36 — DIRECTNESS RANGE FIX:
            #
            # Before: ``question_ratio * 0.3 + (1 - question_ratio) * 0.7``
            # capped directness at 0.7 (when question_ratio=0).  But the
            # cloud prompt-builder gates the "be more direct"
            # instruction on ``directness > 0.7`` — strict inequality —
            # so the path was unreachable: statements never produced an
            # adjustment.
            #
            # After: ``0.85 - 0.7 * question_ratio`` covers [0.15, 0.85].
            # Pure statements push directness above the 0.7 threshold
            # (instruction fires), pure questions drop below 0.3
            # (counter-instruction fires), mixed sits at the default
            # 0.5.
            directness=_clamp(0.85 - 0.7 * question_ratio),
        )

        # SEC: defensively clamp the adaptation rate to [0, 1] -- a
        # mis-configured ``rate < 0`` would push the style *away* from
        # observation, and ``rate > 1`` would overshoot and oscillate.
        rate = _clamp(_safe_float(self.config.adaptation_rate, default=0.2))

        # Exponential smoothing toward observed style.  We construct a NEW
        # ``StyleVector`` rather than mutating ``current_style`` so that the
        # caller can keep a reference to the previous style for diagnostics.
        return StyleVector(
            formality=_clamp(
                current_style.formality + rate * (observed.formality - current_style.formality)
            ),
            verbosity=_clamp(
                current_style.verbosity + rate * (observed.verbosity - current_style.verbosity)
            ),
            emotionality=_clamp(
                current_style.emotionality + rate * (observed.emotionality - current_style.emotionality)
            ),
            directness=_clamp(
                current_style.directness + rate * (observed.directness - current_style.directness)
            ),
        )


# ---------------------------------------------------------------------------
# Emotional Tone Adapter
# ---------------------------------------------------------------------------

class EmotionalToneAdapter:
    """Adjust the warmth and supportiveness of responses.

    The emotional tone axis ranges from 0.0 (most warm / supportive) to
    1.0 (most neutral / objective).  The adapter monitors four distress
    signals:

    1. **Engagement dropping** -- the user is sending fewer messages or
       shorter responses than their baseline.
    2. **Typing slower** -- elevated inter-key interval suggests
       hesitation, frustration, or fatigue.
    3. **Vocabulary simplifying** -- the user is using simpler words than
       usual, often a sign of cognitive or emotional overload.
    4. **Negative sentiment** -- explicit negative affect in the message
       content.

    When distress is detected, the tone shifts toward warmth and support.
    In neutral conditions, the tone defaults to 0.5 (balanced).

    This is critical for HMI because an AI companion that remains clinically
    neutral when a user is struggling feels cold and unhelpful, while one
    that is always warm feels patronising when the user is fine.

    Parameters:
        config: :class:`~src.config.EmotionalToneConfig` with warmth range
            bounds and default tone.
    """

    def __init__(self, config: EmotionalToneConfig) -> None:
        self.config = config

    def compute(
        self,
        features: InteractionFeatureVector,
        deviation: DeviationMetrics,
    ) -> float:
        """Compute the target emotional tone.

        Args:
            features: Current interaction feature vector.
            deviation: Deviation metrics from the user's personal baseline.

        Returns:
            A float in [0, 1] where 0 = most warm/supportive and
            1 = most neutral/objective.
        """
        # SEC: NaN-safe coercion of every distress input.  ``np.mean`` would
        # otherwise propagate ``NaN`` straight through to the SLM tensor.
        engagement_dev = _safe_float(deviation.engagement_deviation)
        iki_dev = _safe_float(deviation.iki_deviation)
        vocab_dev = _safe_float(deviation.vocab_deviation)
        sentiment_valence = _safe_float(features.sentiment_valence)

        distress_signals = [
            max(0.0, -engagement_dev),     # Engagement dropping
            max(0.0, iki_dev),             # Typing slower (positive = slower)
            max(0.0, -vocab_dev),          # Vocabulary simplifying
            max(0.0, -sentiment_valence),  # Negative sentiment
        ]
        distress_score = float(np.mean(distress_signals))

        # Iter 43 — neutrality drive.  Pre-iter-43 the formula was
        # ``tone = 0.5 - distress*0.5`` so ``emotional_tone`` could
        # never exceed 0.5 in practice — meaning the post-processor's
        # ``emotional_tone > 0.7`` warmth-stripping branch was dead
        # code.  Strong positive sentiment is now a "this user is fine,
        # be clinical" signal that can lift the tone above 0.5 toward
        # 1.0.
        #
        # Iter 44 — re-enabled the formality contribution now that
        # ``LinguisticAnalyzer.formality_score`` is properly calibrated
        # (was purely subtractive; every plain chat read 1.0).  Both
        # signals additively drive neutrality, mean-aggregated so a
        # single signal can't dominate.
        formality = _safe_float(features.formality)
        neutrality_signals = [
            max(0.0, sentiment_valence),         # positive sentiment
            max(0.0, (formality - 0.6) * 2.5),   # formality > 0.6 -> [0,1]
        ]
        neutrality_score = float(np.mean(neutrality_signals))

        # SEC: defensively unpack the configured warmth range so that an
        # inverted (lo > hi) tuple is normalised by ``_clamp`` rather than
        # silently producing an empty interval.
        try:
            lo, hi = self.config.warmth_range
        except (TypeError, ValueError):
            lo, hi = 0.0, 1.0
        # tone in [0, 1]:
        #   pure distress + no positive sentiment -> 0.0  (very warm)
        #   neither signal active                  -> 0.5  (balanced)
        #   strong positive sentiment, no distress -> 1.0  (clinical)
        return _clamp(
            0.5 + 0.5 * neutrality_score - 0.5 * distress_score,
            lo=lo,
            hi=hi,
        )


# ---------------------------------------------------------------------------
# Accessibility Adapter
# ---------------------------------------------------------------------------

class AccessibilityAdapter:
    """Detect motor or cognitive difficulty and trigger simplification.

    Some users may experience temporary or persistent motor difficulties
    (e.g., injury, fatigue, disability) or cognitive challenges that make
    standard interaction taxing.  This adapter monitors four difficulty
    signals:

    1. **IKI deviation** -- typing significantly slower than baseline.
    2. **Speed deviation** -- overall composition speed dropping.
    3. **Backspace ratio** -- frequent corrections suggest motor difficulty
       or uncertainty.
    4. **Editing effort** -- high edit-distance ratio indicates struggling
       to compose messages.

    When the aggregate difficulty score exceeds a configurable threshold,
    the adapter activates accessibility mode, which downstream modules use
    to:
    - Shorten and simplify responses.
    - Use larger logical chunks.
    - Reduce the need for follow-up questions.
    - Prefer yes/no or multiple-choice interactions.

    Below the threshold, accessibility remains at 0.0 (standard mode) to
    avoid unnecessary condescension.

    Parameters:
        config: :class:`~src.config.AccessibilityConfig` with the detection
            threshold and number of simplification levels.
    """

    def __init__(self, config: AccessibilityConfig) -> None:
        self.config = config

    def compute(
        self,
        features: InteractionFeatureVector,
        deviation: DeviationMetrics,
        threshold: float | None = None,
    ) -> float:
        """Compute the accessibility simplification level.

        Args:
            features: Current interaction feature vector.
            deviation: Deviation metrics from the user's personal baseline.
            threshold: Override for the detection threshold.  If ``None``,
                uses ``self.config.detection_threshold``.

        Returns:
            A float in [0, 1] where 0 = standard mode and
            1 = maximum simplification.  Returns 0.0 if the difficulty
            score is below the threshold.
        """
        if threshold is None:
            threshold = self.config.detection_threshold
        # SEC: guard against a NaN/None override or config.
        threshold = _safe_float(threshold, default=0.7)

        # SEC: NaN-safe coercion at the input boundary.
        iki_dev = _safe_float(deviation.iki_deviation)
        speed_dev = _safe_float(deviation.speed_deviation)
        backspace_ratio = _safe_float(features.backspace_ratio)
        editing_effort = _safe_float(features.editing_effort)

        # Iter 48 — switched from ``mean()`` to ``max()`` for the same
        # reason iter 38 made the cognitive_load rhythm-stress signal
        # use max(): any single strong difficulty signal is sufficient
        # evidence of motor trouble, and averaging dilutes the reading
        # whenever a baseline-derived signal (iki_deviation,
        # speed_deviation) collapses to 0 because the user's prior
        # turns were too uniform to define a meaningful std.  Pre-fix,
        # an editing_effort of 0.8 + backspace_ratio of 0.33 averaged
        # to 0.28 when the deviation channels were 0 — below the
        # threshold — and the path silently failed to fire on a user
        # who clearly needed simplification.
        difficulty_signals = [
            max(0.0, iki_dev),               # Typing slower than baseline
            max(0.0, -speed_dev),            # Composition speed dropping
            _clamp(backspace_ratio),         # Frequent corrections
            _clamp(editing_effort),          # High editing effort
        ]
        difficulty_score = max(difficulty_signals)

        if difficulty_score > threshold:
            return _clamp(difficulty_score)
        return 0.0
