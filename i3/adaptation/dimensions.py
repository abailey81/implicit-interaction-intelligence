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
            A float in [0, 1] where 0 = simplest possible response and
            1 = richest/most complex response.
        """
        # SEC: NaN-safe coercion at the input boundary; downstream arithmetic
        # then never sees ``NaN`` even if a feature extractor mis-fires.
        ttr = _safe_float(features.type_token_ratio)
        mean_word_length = _safe_float(features.mean_word_length)
        flesch_kincaid = _safe_float(features.flesch_kincaid)
        message_length = _safe_float(features.message_length)
        complexity_dev = _safe_float(deviation.complexity_deviation)

        # Combine four normalised complexity signals from the user's current
        # message.  Each is scaled to roughly [0, 1].  The constant
        # denominators (10, 20) are intentional and tolerate the rare case
        # where upstream forwards an unscaled raw value.
        complexity_signals = [
            _clamp(ttr),                          # Vocabulary richness
            _clamp(mean_word_length / 10.0),      # Word sophistication
            _clamp(flesch_kincaid / 20.0),        # Readability grade
            _clamp(message_length),               # Normalised length
        ]
        user_complexity = float(np.mean(complexity_signals))

        # If complexity is dropping substantially below baseline, simplify
        # our responses to be *simpler than* the user's current level.
        if complexity_dev < -0.5:
            return _clamp(max(0.2, user_complexity - 0.2))

        # Otherwise, respond at slightly higher complexity to keep the
        # conversation stimulating without overwhelming.
        return _clamp(min(0.9, user_complexity + 0.1))


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
        # ``message_length`` is already in [0, 1]; the /0.7 saturation
        # constant means verbosity hits 1.0 when the user fills 70% of the
        # max-length budget.  The constant is non-zero so division is safe.
        observed = StyleVector(
            formality=_clamp(formality),
            verbosity=_clamp(message_length / 0.7),
            emotionality=_clamp(
                max(emoji_density * 5.0, abs(sentiment_valence))
            ),
            directness=_clamp(
                question_ratio * 0.3 + (1.0 - question_ratio) * 0.7
            ),
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

        # More distress -> lower tone value -> warmer/more supportive.
        # The mapping is: tone = 0.5 - distress * 0.5, clamped to [0, 1].
        # SEC: defensively unpack the configured warmth range so that an
        # inverted (lo > hi) tuple is normalised by ``_clamp`` rather than
        # silently producing an empty interval.
        try:
            lo, hi = self.config.warmth_range
        except (TypeError, ValueError):
            lo, hi = 0.0, 1.0
        return _clamp(0.5 - distress_score * 0.5, lo=lo, hi=hi)


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

        difficulty_signals = [
            max(0.0, iki_dev),               # Typing slower than baseline
            max(0.0, -speed_dev),            # Composition speed dropping
            _clamp(backspace_ratio),         # Frequent corrections
            _clamp(editing_effort),          # High editing effort
        ]
        difficulty_score = float(np.mean(difficulty_signals))

        if difficulty_score > threshold:
            return _clamp(difficulty_score)
        return 0.0
