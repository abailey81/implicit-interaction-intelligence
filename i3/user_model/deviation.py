"""Deviation computation for the persistent user model.

Computes how the current user state differs from established baselines at
both the embedding level (cosine distance) and the per-feature level
(z-score deviations).

This module operates exclusively on numerical tensors and scalar metrics --
no raw text is ever processed or stored.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from i3.user_model.types import DeviationMetrics

if TYPE_CHECKING:
    from i3.interaction.types import InteractionFeatureVector

logger = logging.getLogger(__name__)

# Feature names from InteractionFeatureVector used for engagement scoring.
_ENGAGEMENT_FEATURES: list[tuple[str, float]] = [
    # (feature_name, weight)
    ("composition_speed", 0.25),
    ("message_length", 0.25),
    ("type_token_ratio", 0.20),
    ("topic_coherence", 0.15),
    ("engagement_velocity", 0.15),
]

# SEC: Per-feature z-scores are clamped to this symmetric range to prevent
# outlier values from dominating downstream pattern-deviation magnitudes.
_Z_SCORE_CLAMP: float = 5.0


class DeviationComputer:
    """Computes deviation metrics between current state and baselines.

    All operations are pure functions on tensors and scalars.  The class
    is stateless and can be shared across user models.
    """

    # ------------------------------------------------------------------
    # Static geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine distance between two vectors.

        Returns ``1 - cosine_similarity(a, b)`` clamped to [0, 2].

        Args:
            a: First embedding vector (1-D).
            b: Second embedding vector (1-D).

        Returns:
            Cosine distance as a Python float. Returns 0.0 if either vector
            is all-zero (an undefined cosine), so the metric never leaks NaN.
        """
        a_flat = a.detach().float().flatten()
        b_flat = b.detach().float().flatten()
        # SEC: Guard against zero-norm vectors. F.cosine_similarity returns
        # NaN when either input has zero norm; that NaN propagates into the
        # downstream router and trips threshold gates. Treat the undefined
        # case as "no deviation".
        a_norm = torch.linalg.norm(a_flat).item()
        b_norm = torch.linalg.norm(b_flat).item()
        if a_norm < 1e-12 or b_norm < 1e-12:
            return 0.0
        similarity = F.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)
        ).item()
        if not math.isfinite(similarity):
            return 0.0
        return float(max(0.0, min(2.0, 1.0 - similarity)))

    # ------------------------------------------------------------------
    # Per-feature z-score deviations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_feature_deviations(
        current_features: InteractionFeatureVector,
        baseline_mean: dict[str, float],
        baseline_std: dict[str, float],
    ) -> dict[str, float]:
        """Compute z-score deviations of current features from the baseline.

        For each feature in *baseline_mean*, the z-score is computed as::

            z = (current - mean) / max(std, 1e-6)

        Features present in the baseline but missing from *current_features*
        are silently skipped.

        Args:
            current_features: The current interaction feature vector.
            baseline_mean: Per-feature running mean.
            baseline_std: Per-feature running standard deviation.

        Returns:
            Dict mapping feature name to its z-score.
        """
        deviations: dict[str, float] = {}
        for name, mean_val in baseline_mean.items():
            current_val = getattr(current_features, name, None)
            if current_val is None:
                continue
            current_f = float(current_val)
            # SEC: Skip non-finite current values; do not pollute z-scores
            # with NaN/Inf.
            if not math.isfinite(current_f) or not math.isfinite(mean_val):
                continue
            std_val = max(baseline_std.get(name, 1e-6), 1e-6)
            z = (current_f - mean_val) / std_val
            # SEC: Clamp z-scores to a sane range so outliers cannot blow up
            # downstream pattern_deviation or trip routing thresholds.
            if z > _Z_SCORE_CLAMP:
                z = _Z_SCORE_CLAMP
            elif z < -_Z_SCORE_CLAMP:
                z = -_Z_SCORE_CLAMP
            deviations[name] = z
        return deviations

    # ------------------------------------------------------------------
    # Engagement score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_engagement_score(features: InteractionFeatureVector) -> float:
        """Derive an engagement score in [0, 1] from interaction features.

        The score is a weighted average of normalised signals reflecting
        response speed, message length, vocabulary richness, and topic
        coherence.

        Args:
            features: Current interaction feature vector.

        Returns:
            Engagement score clamped to [0, 1].
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for feature_name, weight in _ENGAGEMENT_FEATURES:
            value = getattr(features, feature_name, 0.0)
            value_f = float(value)
            # SEC: Treat non-finite values as missing rather than letting
            # NaN/Inf poison the weighted average.
            if not math.isfinite(value_f):
                continue
            # All feature values should already be normalised to ~[0, 1]
            clamped = max(0.0, min(1.0, value_f))
            weighted_sum += weight * clamped
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return max(0.0, min(1.0, weighted_sum / total_weight))

    # ------------------------------------------------------------------
    # Aggregate pattern deviation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pattern_deviation(feature_deviations: dict[str, float]) -> float:
        """Compute overall pattern deviation as RMS of per-feature z-scores.

        Formula::

            pattern_deviation = sqrt( sum(z_i^2) / n )

        Args:
            feature_deviations: Dict of feature-name to z-score.

        Returns:
            RMS deviation magnitude (non-negative float).
        """
        if not feature_deviations:
            return 0.0
        squared = [z * z for z in feature_deviations.values()]
        return math.sqrt(sum(squared) / len(squared))

    # ------------------------------------------------------------------
    # Full deviation metrics
    # ------------------------------------------------------------------

    def compute(
        self,
        current_embedding: torch.Tensor,
        session_embedding: torch.Tensor | None,
        baseline_embedding: torch.Tensor | None,
        features: InteractionFeatureVector,
        baseline_mean: dict[str, float] | None,
        baseline_std: dict[str, float] | None,
    ) -> DeviationMetrics:
        """Compute complete deviation metrics for the current state.

        If the baseline has not yet been established (i.e. *baseline_embedding*
        or *baseline_mean* is ``None``), all baseline-relative metrics are
        returned as zero.

        Args:
            current_embedding: 64-dim embedding from the encoder.
            session_embedding: Current session EMA embedding (may be ``None``
                at session start).
            baseline_embedding: Long-term baseline embedding (``None`` before
                warm-up).
            features: Current interaction feature vector.
            baseline_mean: Per-feature running mean (``None`` before warm-up).
            baseline_std: Per-feature running std (``None`` before warm-up).

        Returns:
            Fully populated :class:`DeviationMetrics`.
        """
        # Cosine distances
        current_vs_baseline = 0.0
        if baseline_embedding is not None:
            current_vs_baseline = self.cosine_distance(
                current_embedding, baseline_embedding
            )

        current_vs_session = 0.0
        if session_embedding is not None:
            current_vs_session = self.cosine_distance(
                current_embedding, session_embedding
            )

        # Engagement
        engagement_score = self.compute_engagement_score(features)

        # Per-feature z-scores
        if baseline_mean is not None and baseline_std is not None:
            feature_devs = self.compute_feature_deviations(
                features, baseline_mean, baseline_std
            )
        else:
            feature_devs = {}

        pattern_dev = self.compute_pattern_deviation(feature_devs)

        return DeviationMetrics(
            current_vs_baseline=current_vs_baseline,
            current_vs_session=current_vs_session,
            engagement_score=engagement_score,
            magnitude=pattern_dev,
            iki_deviation=feature_devs.get("mean_iki", 0.0),
            length_deviation=feature_devs.get("message_length", 0.0),
            vocab_deviation=feature_devs.get("type_token_ratio", 0.0),
            formality_deviation=feature_devs.get("formality", 0.0),
            speed_deviation=feature_devs.get("composition_speed", 0.0),
            engagement_deviation=feature_devs.get("engagement_velocity", 0.0),
            complexity_deviation=feature_devs.get("flesch_kincaid", 0.0),
            pattern_deviation=pattern_dev,
        )
