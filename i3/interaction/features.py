"""Feature extraction and baseline tracking for I3 interaction vectors.

This module converts raw keystroke metrics, message text, and session history
into the 32-dimensional :class:`InteractionFeatureVector` consumed by
downstream adaptation modules.

Key classes:

* :class:`BaselineTracker` -- accumulates running statistics and computes
  z-score deviations from a user's established baseline.
* :class:`FeatureExtractor` -- orchestrates the full extraction pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from i3.interaction.linguistic import LinguisticAnalyzer
from i3.interaction.types import InteractionFeatureVector

# ====================================================================
# BaselineTracker
# ====================================================================

@dataclass
class BaselineTracker:
    """Accumulates running mean/std for each feature and reports deviations.

    The baseline is considered *established* once at least ``warmup``
    feature vectors have been observed.  Before that point, all deviation
    queries return ``0.0``.

    Internally the tracker uses Welford's online algorithm for numerically
    stable running variance.

    Attributes:
        warmup: Number of observations required before the baseline is
            considered established.
        count: Number of observations seen so far.
    """

    warmup: int = 5
    count: int = field(default=0, init=False)
    _mean: dict[str, float] = field(default_factory=dict, init=False)
    _m2: dict[str, float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def is_established(self) -> bool:
        """``True`` once enough observations have been accumulated."""
        return self.count >= self.warmup

    def update(self, fv: InteractionFeatureVector) -> None:
        """Incorporate a new feature vector into the running statistics.

        Args:
            fv: The latest feature vector for this user.
        """
        self.count += 1
        from i3.interaction.types import FEATURE_NAMES  # avoid circular at module level

        for name in FEATURE_NAMES:
            x = getattr(fv, name)
            if name not in self._mean:
                self._mean[name] = 0.0
                self._m2[name] = 0.0
            old_mean = self._mean[name]
            self._mean[name] += (x - old_mean) / self.count
            self._m2[name] += (x - old_mean) * (x - self._mean[name])

    def deviation(self, feature_name: str, value: float) -> float:
        """Return the z-score deviation of *value* from the baseline.

        Uses **Bessel-corrected sample variance** (``m2 / (n - 1)``)
        rather than the population estimator (``m2 / n``), so the z-
        score is unbiased at small sample sizes.  At early-session
        counts (5–10 messages) the difference is meaningful: the
        population estimator under-estimates noise and inflates z-
        scores.

        Returns:
            0.0 when the baseline is not yet established, when the
            sample size is below 2 (no defined sample variance), or
            when the standard deviation is near zero (degenerate).
            Otherwise the z-score, clamped to ±3σ and normalised to
            ``[-1, 1]``.
        """
        if not self.is_established:
            return 0.0
        if self.count < 2:
            # No defined sample variance with a single observation.
            return 0.0
        mean = self._mean.get(feature_name, 0.0)
        # Bessel correction: divide by (n - 1) for the unbiased
        # sample variance estimator.
        variance = self._m2.get(feature_name, 0.0) / (self.count - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std < 1e-9:
            return 0.0
        z = (value - mean) / std
        # Clamp to [-3, 3] then normalise to [-1, 1].
        return max(-1.0, min(1.0, z / 3.0))

    def get_mean(self, feature_name: str) -> float:
        """Return the running mean for *feature_name* (0.0 if unseen)."""
        return self._mean.get(feature_name, 0.0)

    def get_std(self, feature_name: str) -> float:
        """Return the running sample (Bessel-corrected) std for *feature_name*.

        Returns ``0.0`` when fewer than 2 observations have been seen
        (sample variance is undefined for n=1).
        """
        if self.count < 2:
            return 0.0
        variance = self._m2.get(feature_name, 0.0) / (self.count - 1)
        return math.sqrt(variance) if variance > 0 else 0.0

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.count = 0
        self._mean.clear()
        self._m2.clear()


# ====================================================================
# FeatureExtractor
# ====================================================================

# Normalisation constants (used for min-max scaling)
_MAX_IKI_MS: float = 2000.0          # 2 seconds between keys
_MAX_BURST_LEN: float = 50.0         # 50 characters in one burst
_MAX_PAUSE_MS: float = 10_000.0      # 10 s pause between bursts
_MAX_COMP_SPEED: float = 15.0        # 15 chars/s fast typist
_MAX_PAUSE_SEND_MS: float = 30_000.0 # 30 s hesitation before send
_MAX_MSG_LEN_WORDS: float = 500.0    # 500 words very long message
_MAX_FK_GRADE: float = 20.0          # FK grade level cap
_MAX_WORD_LEN: float = 15.0          # average word length cap
_MAX_EMOJI_DENSITY: float = 1.0      # emoji per word cap
_MAX_ENG_VEL: float = 10.0           # messages per minute cap


class FeatureExtractor:
    """Compute the 32-dim :class:`InteractionFeatureVector`.

    This class is stateless except for its reference to a
    :class:`LinguisticAnalyzer` instance.  Session-level and deviation
    features require the caller to supply the history window and
    :class:`BaselineTracker`.

    Typical usage::

        extractor = FeatureExtractor()
        fv = extractor.extract(
            keystroke_metrics=ks_metrics,
            message_text="Hello there!",
            history=history_window,
            baseline=baseline_tracker,
            session_start_ts=t0,
            current_ts=now,
        )
    """

    def __init__(self) -> None:
        self._ling = LinguisticAnalyzer()

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        keystroke_metrics: dict[str, float],
        message_text: str,
        history: list[InteractionFeatureVector],
        baseline: BaselineTracker,
        session_start_ts: float = 0.0,
        current_ts: float = 0.0,
        expected_session_length: float = 600.0,
    ) -> InteractionFeatureVector:
        """Build a complete 32-dim feature vector.

        Args:
            keystroke_metrics: Raw keystroke statistics for the current
                message.  Expected keys (all optional, default 0.0):
                ``mean_iki_ms``, ``std_iki_ms``, ``mean_burst_length``,
                ``mean_pause_duration_ms``, ``backspace_ratio``,
                ``composition_speed_cps``, ``pause_before_send_ms``,
                ``editing_effort``.
            message_text: The submitted message text.
            history: Sliding window of the last N feature vectors for this
                user in the current session (may be empty).
            baseline: The user's :class:`BaselineTracker`.
            session_start_ts: Unix timestamp when the session began.
            current_ts: Unix timestamp of the current message.
            expected_session_length: Expected session length in seconds
                (used for ``session_progress``).

        Returns:
            A fully populated :class:`InteractionFeatureVector`.
        """
        # --- 1. Keystroke dynamics (8) -----------------------------------
        ks = self._extract_keystroke(keystroke_metrics)

        # --- 2. Message content (8) --------------------------------------
        mc = self._extract_message(message_text)

        # --- 3. Session dynamics (8) -------------------------------------
        sd = self._extract_session(
            mc, history, session_start_ts, current_ts, expected_session_length, message_text,
        )

        # Merge groups into a preliminary feature vector
        fv = InteractionFeatureVector(
            # keystroke
            mean_iki=ks["mean_iki"],
            std_iki=ks["std_iki"],
            mean_burst_length=ks["mean_burst_length"],
            mean_pause_duration=ks["mean_pause_duration"],
            backspace_ratio=ks["backspace_ratio"],
            composition_speed=ks["composition_speed"],
            pause_before_send=ks["pause_before_send"],
            editing_effort=ks["editing_effort"],
            # message
            message_length=mc["message_length"],
            type_token_ratio=mc["type_token_ratio"],
            mean_word_length=mc["mean_word_length"],
            flesch_kincaid=mc["flesch_kincaid"],
            question_ratio=mc["question_ratio"],
            formality=mc["formality"],
            emoji_density=mc["emoji_density"],
            sentiment_valence=mc["sentiment_valence"],
            # session
            length_trend=sd["length_trend"],
            latency_trend=sd["latency_trend"],
            vocab_trend=sd["vocab_trend"],
            engagement_velocity=sd["engagement_velocity"],
            topic_coherence=sd["topic_coherence"],
            session_progress=sd["session_progress"],
            time_deviation=sd["time_deviation"],
            response_depth=sd["response_depth"],
        )

        # --- 4. Deviation metrics (8) ------------------------------------
        devs = self._extract_deviations(fv, baseline)
        fv.iki_deviation = devs["iki_deviation"]
        fv.length_deviation = devs["length_deviation"]
        fv.vocab_deviation = devs["vocab_deviation"]
        fv.formality_deviation = devs["formality_deviation"]
        fv.speed_deviation = devs["speed_deviation"]
        fv.engagement_deviation = devs["engagement_deviation"]
        fv.complexity_deviation = devs["complexity_deviation"]
        fv.pattern_deviation = devs["pattern_deviation"]

        return fv

    # ------------------------------------------------------------------ #
    # Keystroke features                                                   #
    # ------------------------------------------------------------------ #

    def _extract_keystroke(self, km: dict[str, float]) -> dict[str, float]:
        """Normalise raw keystroke metrics to [0, 1]."""
        return {
            "mean_iki": _clamp01(km.get("mean_iki_ms", 0.0) / _MAX_IKI_MS),
            "std_iki": _clamp01(km.get("std_iki_ms", 0.0) / _MAX_IKI_MS),
            "mean_burst_length": _clamp01(
                km.get("mean_burst_length", 0.0) / _MAX_BURST_LEN
            ),
            "mean_pause_duration": _clamp01(
                km.get("mean_pause_duration_ms", 0.0) / _MAX_PAUSE_MS
            ),
            "backspace_ratio": _clamp01(km.get("backspace_ratio", 0.0)),
            "composition_speed": _clamp01(
                km.get("composition_speed_cps", 0.0) / _MAX_COMP_SPEED
            ),
            "pause_before_send": _clamp01(
                km.get("pause_before_send_ms", 0.0) / _MAX_PAUSE_SEND_MS
            ),
            "editing_effort": _clamp01(km.get("editing_effort", 0.0)),
        }

    # ------------------------------------------------------------------ #
    # Message content features                                             #
    # ------------------------------------------------------------------ #

    def _extract_message(self, text: str) -> dict[str, float]:
        """Compute linguistic features from message text."""
        ling = self._ling.compute_all(text)
        return {
            "message_length": _clamp01(ling["message_length"] / _MAX_MSG_LEN_WORDS),
            "type_token_ratio": _clamp01(ling["type_token_ratio"]),
            "mean_word_length": _clamp01(ling["mean_word_length"] / _MAX_WORD_LEN),
            "flesch_kincaid": _clamp01(ling["flesch_kincaid"] / _MAX_FK_GRADE),
            "question_ratio": _clamp01(ling["question_ratio"]),
            "formality": _clamp01(ling["formality"]),
            "emoji_density": _clamp01(ling["emoji_density"] / _MAX_EMOJI_DENSITY),
            "sentiment_valence": max(-1.0, min(1.0, ling["sentiment_valence"])),
        }

    # ------------------------------------------------------------------ #
    # Session dynamics features                                            #
    # ------------------------------------------------------------------ #

    def _extract_session(
        self,
        current_msg: dict[str, float],
        history: list[InteractionFeatureVector],
        session_start_ts: float,
        current_ts: float,
        expected_length: float,
        message_text: str,
    ) -> dict[str, float]:
        """Compute session-level trend and engagement features."""
        n = len(history)

        # -- Trends (linear regression slope over the window) -------------
        length_trend = 0.0
        latency_trend = 0.0
        vocab_trend = 0.0

        if n >= 2:
            lengths = [h.message_length for h in history] + [current_msg["message_length"]]
            length_trend = _normalised_slope(lengths)

            latencies = [h.composition_speed for h in history] + [current_msg.get("composition_speed", 0.0)]
            latency_trend = _normalised_slope(latencies)

            vocabs = [h.type_token_ratio for h in history] + [current_msg["type_token_ratio"]]
            vocab_trend = _normalised_slope(vocabs)

        # -- Engagement velocity (messages per minute) --------------------
        elapsed = max(1.0, current_ts - session_start_ts)
        msg_count = n + 1
        eng_vel = _clamp01((msg_count / (elapsed / 60.0)) / _MAX_ENG_VEL)

        # -- Topic coherence (cosine similarity with previous message) ----
        #
        # Iter 4 precision improvement:
        #
        # Before: rounding-Jaccard at 0.1 resolution — a 0.05 shift
        # across all three (type_token_ratio, formality, flesch_kincaid)
        # could cross every rounding boundary and collapse coherence
        # from 1.0 to 0.0.  Discontinuous, brittle, and visibly wrong
        # to a reader inspecting trajectories.
        #
        # After: cosine similarity over the same three-feature
        # signature, with each feature centred at 0.5 first so the
        # measure is *direction* of deviation rather than raw magnitude.
        # This produces a smooth [0, 1] coherence score that:
        #   * is 1.0 when the signature vectors point the same way,
        #   * decays gradually as the signatures diverge,
        #   * is 0.0 only when the signatures are orthogonal /
        #     anti-correlated.
        topic_coherence = 0.0
        if n >= 1:
            prev = history[-1]
            prev_sig = (
                prev.type_token_ratio - 0.5,
                prev.formality - 0.5,
                prev.flesch_kincaid - 0.5,
            )
            cur_sig = (
                current_msg["type_token_ratio"] - 0.5,
                current_msg["formality"] - 0.5,
                current_msg["flesch_kincaid"] - 0.5,
            )
            topic_coherence = _cosine_similarity_unit(prev_sig, cur_sig)

        # -- Session progress [0, 1] --------------------------------------
        session_progress = _clamp01(elapsed / max(1.0, expected_length))

        # -- Time deviation (how far current inter-message time deviates
        #    from the user's mean, normalised) ----------------------------
        time_deviation = 0.0
        if n >= 2:
            speeds = [h.composition_speed for h in history]
            mean_speed = sum(speeds) / len(speeds)
            std_speed = _std(speeds)
            if std_speed > 1e-9:
                current_speed = current_msg.get("composition_speed", mean_speed)
                time_deviation = _clamp_neg1_1((current_speed - mean_speed) / std_speed / 3.0)

        # -- Response depth (proxy: normalised message length relative to
        #    the median of the window) ------------------------------------
        response_depth = current_msg["message_length"]
        if n >= 1:
            median_len = sorted([h.message_length for h in history])[n // 2]
            if median_len > 1e-9:
                response_depth = _clamp01(current_msg["message_length"] / (2.0 * median_len))

        return {
            "length_trend": _clamp_neg1_1(length_trend),
            "latency_trend": _clamp_neg1_1(latency_trend),
            "vocab_trend": _clamp_neg1_1(vocab_trend),
            "engagement_velocity": eng_vel,
            "topic_coherence": _clamp01(topic_coherence),
            "session_progress": session_progress,
            "time_deviation": time_deviation,
            "response_depth": _clamp01(response_depth),
        }

    # ------------------------------------------------------------------ #
    # Deviation features                                                   #
    # ------------------------------------------------------------------ #

    def _extract_deviations(
        self,
        fv: InteractionFeatureVector,
        baseline: BaselineTracker,
    ) -> dict[str, float]:
        """Compute z-score deviations from the user baseline.

        All values are 0.0 if the baseline is not yet established.
        """
        return {
            "iki_deviation": baseline.deviation("mean_iki", fv.mean_iki),
            "length_deviation": baseline.deviation("message_length", fv.message_length),
            "vocab_deviation": baseline.deviation("type_token_ratio", fv.type_token_ratio),
            "formality_deviation": baseline.deviation("formality", fv.formality),
            "speed_deviation": baseline.deviation("composition_speed", fv.composition_speed),
            "engagement_deviation": baseline.deviation(
                "engagement_velocity", fv.engagement_velocity
            ),
            "complexity_deviation": baseline.deviation("flesch_kincaid", fv.flesch_kincaid),
            "pattern_deviation": baseline.deviation("mean_burst_length", fv.mean_burst_length),
        }


# ====================================================================
# Utility helpers
# ====================================================================

def _clamp01(v: float) -> float:
    """Clamp *v* to [0, 1]."""
    return max(0.0, min(1.0, v))


def _clamp_neg1_1(v: float) -> float:
    """Clamp *v* to [-1, 1]."""
    return max(-1.0, min(1.0, v))


def _normalised_slope(values: list[float]) -> float:
    """Simple linear regression slope, normalised by the mean.

    Returns a value roughly in [-1, 1] representing the direction and
    magnitude of the trend.  Returns 0.0 for fewer than 2 data points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(values):
        dx = i - x_mean
        numerator += dx * (y - y_mean)
        denominator += dx * dx
    if denominator < 1e-12:
        return 0.0
    slope = numerator / denominator
    # Normalise by mean to get relative slope
    if abs(y_mean) > 1e-9:
        return slope / abs(y_mean)
    return slope


def _std(values: list[float]) -> float:
    """Population standard deviation of *values*."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _cosine_similarity_unit(
    a: tuple[float, ...], b: tuple[float, ...]
) -> float:
    """Cosine similarity, mapped from [-1, 1] to [0, 1].

    Used by the topic-coherence feature so the resulting score is a
    valid similarity in ``[0, 1]`` (matching the 0.0=different / 1.0=
    identical convention of the rest of the feature vector).

    * Returns ``1.0`` when both vectors are zero (e.g. all features
      centred — interpreted as "no signal of difference between turns").
    * Returns ``0.5`` when one vector is zero and the other is not
      (orthogonal in the standard cosine sense).
    """
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = math.sqrt(sum(x * x for x in a[:n]))
    norm_b = math.sqrt(sum(x * x for x in b[:n]))
    # Both vectors zero ⇒ identical "no-signal" turns.
    if norm_a < 1e-9 and norm_b < 1e-9:
        return 1.0
    # One zero, one non-zero ⇒ no signal to compare; midpoint.
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.5
    cosine = dot / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    # Map [-1, 1] -> [0, 1].
    return 0.5 * (cosine + 1.0)
