"""Post-training feature analysis for cross-attention SAEs.

Given a trained :class:`~i3.interpretability.sparse_autoencoder.SparseAutoencoder`
and its activation cache, this module correlates every learned feature
against each :class:`~i3.adaptation.types.AdaptationVector` dimension to
expose which features encode which user-state axis. Features whose
activation correlates above a threshold with a single dimension are
labelled *monosemantic* in the Bricken 2023 sense; the remaining features
are left unlabelled (polysemantic) and reported with their top-three
correlations for follow-up.

The module also exposes :func:`feature_steering_vector` — a thin helper
that extracts a decoder column to produce a steering vector for
activation patching / ``ActivationAddition`` (Turner et al., 2023).

References
----------
- Bricken, T. et al. (2023). *Towards Monosemanticity.*
- Templeton, A. et al. (2024). *Scaling Monosemanticity.*
- Turner, A. et al. (2023). *Activation Addition: Steering Language
  Models Without Optimisation.* arXiv:2308.10248.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector
from i3.interpretability.feature_attribution import ADAPTATION_DIMS
from i3.interpretability.sparse_autoencoder import (
    FeatureDictionary,
    SparseAutoencoder,
)

# Soft import scipy for Spearman; fall back to the bundled torch impl.
try:  # pragma: no cover - exercised at import time
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]

    _SCIPY_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - optional dependency
    _scipy_stats = None
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# FeatureSemantics.
# ---------------------------------------------------------------------------


class FeatureSemantics(BaseModel):
    """Per-feature semantic summary produced by correlation analysis.

    Attributes:
        feature_idx: Zero-based feature index in ``[0, d_dict)``.
        top_dimension_correlations: List of
            ``(dimension_name, correlation_coefficient)`` pairs for the
            three highest-magnitude Pearson correlations.
        mean_activation: Mean activation across the cached samples.
        max_activation: Maximum activation across the cached samples.
        sparsity: Fraction of samples on which the feature is zero.
        dimension_label: Auto-assigned label when one dimension's
            ``|correlation|`` exceeds the monosemanticity threshold;
            ``None`` otherwise.
        spearman: Spearman correlations keyed by dimension name (all
            eight dimensions, for diagnostic plots).
    """

    model_config = ConfigDict(frozen=True)

    feature_idx: int = Field(ge=0)
    top_dimension_correlations: list[tuple[str, float]]
    mean_activation: float
    max_activation: float
    sparsity: float = Field(ge=0.0, le=1.0)
    dimension_label: Optional[str] = None
    spearman: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# compute_per_feature_semantics.
# ---------------------------------------------------------------------------


def _adaptation_matrix(
    adaptation_vectors: Sequence[AdaptationVector],
) -> torch.Tensor:
    """Stack a list of :class:`AdaptationVector` into a ``[N, 8]`` tensor."""
    if not adaptation_vectors:
        raise ValueError("adaptation_vectors must not be empty")
    rows = [v.to_tensor() for v in adaptation_vectors]
    return torch.stack(rows, dim=0)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the Pearson correlation coefficient between two 1-D tensors.

    Args:
        x: 1-D tensor of observations.
        y: 1-D tensor of observations (same length as ``x``).

    Returns:
        Pearson correlation in ``[-1, 1]``. Returns ``0.0`` if either
        input is constant (std = 0) to avoid division by zero.
    """
    x = x.flatten().float()
    y = y.flatten().float()
    if x.numel() != y.numel() or x.numel() < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    x_c = x - x_mean
    y_c = y - y_mean
    denom = x_c.norm() * y_c.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    r = float((x_c.dot(y_c) / denom).item())
    if math.isnan(r):
        return 0.0
    return max(-1.0, min(1.0, r))


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Rank-based Spearman correlation.

    Uses :mod:`scipy.stats` when available; otherwise falls back to
    computing Pearson on ranked data.

    Args:
        x: 1-D tensor of observations.
        y: 1-D tensor of observations.

    Returns:
        Spearman correlation in ``[-1, 1]``.
    """
    if x.numel() < 2:
        return 0.0
    if _SCIPY_AVAILABLE:
        rho, _ = _scipy_stats.spearmanr(
            x.detach().cpu().numpy(), y.detach().cpu().numpy()
        )
        if rho is None or (isinstance(rho, float) and math.isnan(rho)):
            return 0.0
        return float(rho)
    # Fallback: Pearson on ranks.
    x_ranked = torch.argsort(torch.argsort(x.flatten())).float()
    y_ranked = torch.argsort(torch.argsort(y.flatten())).float()
    return _pearson(x_ranked, y_ranked)


def compute_per_feature_semantics(
    sae: SparseAutoencoder,
    cache: torch.Tensor,
    adaptation_vectors: Sequence[AdaptationVector],
    monosemanticity_threshold: float = 0.7,
) -> list[FeatureSemantics]:
    """Correlate every SAE feature with each AdaptationVector dimension.

    The function is deliberately vectorised: one forward pass through the
    SAE to obtain the ``[N, d_dict]`` feature matrix, then an 8-column
    correlation sweep against each of the ``AdaptationVector`` dimensions.

    Args:
        sae: A trained :class:`SparseAutoencoder`.
        cache: Activation tensor of shape ``[N, d_model]`` used as
            training-time input. Each row must correspond to the
            identically-indexed :class:`AdaptationVector` in
            ``adaptation_vectors``.
        adaptation_vectors: List/tuple of :class:`AdaptationVector` with
            length ``N``.
        monosemanticity_threshold: ``|r|`` above which a feature is
            auto-labelled with the dominating dimension. Defaults to
            ``0.7`` (Bricken 2023 §4.2 empirical choice).

    Returns:
        List of :class:`FeatureSemantics`, one entry per feature, in
        feature-index order.

    Raises:
        ValueError: If the cache and adaptation list have mismatched
            length, or if either is empty.
    """
    if cache.dim() != 2:
        raise ValueError(
            f"cache must be 2-D, got shape {tuple(cache.shape)}"
        )
    if cache.size(0) != len(adaptation_vectors):
        raise ValueError(
            f"cache rows ({cache.size(0)}) must equal "
            f"len(adaptation_vectors) ({len(adaptation_vectors)})"
        )

    dim_matrix = _adaptation_matrix(adaptation_vectors)  # [N, 8]

    sae.eval()
    with torch.no_grad():
        features = sae.encode(cache).cpu()  # [N, d_dict]

    result: list[FeatureSemantics] = []
    for f in range(sae.d_dict):
        col = features[:, f]
        per_dim: dict[str, float] = {}
        per_dim_spearman: dict[str, float] = {}
        for d_idx, dim_name in enumerate(ADAPTATION_DIMS):
            per_dim[dim_name] = _pearson(col, dim_matrix[:, d_idx])
            per_dim_spearman[dim_name] = _spearman(col, dim_matrix[:, d_idx])

        ranked = sorted(
            per_dim.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        top3 = ranked[:3]

        label: Optional[str] = None
        if ranked and abs(ranked[0][1]) >= monosemanticity_threshold:
            label = ranked[0][0]

        result.append(
            FeatureSemantics(
                feature_idx=f,
                top_dimension_correlations=[
                    (name, float(val)) for name, val in top3
                ],
                mean_activation=float(col.mean().item()),
                max_activation=float(col.max().item()),
                sparsity=float((col == 0).float().mean().item()),
                dimension_label=label,
                spearman=per_dim_spearman,
            )
        )
    return result


# ---------------------------------------------------------------------------
# identify_monosemantic_features.
# ---------------------------------------------------------------------------


def identify_monosemantic_features(
    semantics: Sequence[FeatureSemantics],
    threshold: float = 0.7,
) -> list[FeatureSemantics]:
    """Filter a semantics list to features with a single dominant dimension.

    A feature is monosemantic when its top Pearson correlation exceeds
    ``threshold`` in absolute value. Note that :func:`compute_per_feature_semantics`
    already assigns ``dimension_label`` against a (possibly different)
    monosemanticity threshold; this function lets downstream analyses
    redo the filter at a stricter or looser boundary without recomputing
    correlations.

    Args:
        semantics: Sequence of :class:`FeatureSemantics`.
        threshold: ``|r|`` above which a feature qualifies.

    Returns:
        List of :class:`FeatureSemantics` whose best correlation has
        absolute value ``>= threshold``. Relative ordering is preserved.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1]``.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold}"
        )
    out: list[FeatureSemantics] = []
    for sem in semantics:
        if not sem.top_dimension_correlations:
            continue
        _, r = sem.top_dimension_correlations[0]
        if abs(r) >= threshold:
            out.append(sem)
    return out


# ---------------------------------------------------------------------------
# feature_steering_vector.
# ---------------------------------------------------------------------------


def feature_steering_vector(
    sae: SparseAutoencoder,
    feature_idx: int,
) -> torch.Tensor:
    """Extract the decoder column for use as a steering vector.

    The decoder column of a trained SAE is the direction in residual
    space that the feature "points at", and adding a scalar multiple of
    it to a mid-forward-pass hidden state is the core move of
    :class:`~i3.interpretability.activation_steering.ActivationSteerer`
    (Turner et al., 2023).

    Args:
        sae: Trained :class:`SparseAutoencoder`.
        feature_idx: Zero-based feature index in ``[0, d_dict)``.

    Returns:
        Detached 1-D tensor of length ``d_model``.

    Raises:
        ValueError: If ``feature_idx`` is out of range.
    """
    if not 0 <= feature_idx < sae.d_dict:
        raise ValueError(
            f"feature_idx {feature_idx} out of range [0, {sae.d_dict})"
        )
    return sae.decoder.weight.detach()[:, feature_idx].clone()


# ---------------------------------------------------------------------------
# Helper for reports.
# ---------------------------------------------------------------------------


def top_features_per_dimension(
    semantics: Sequence[FeatureSemantics],
    k: int = 5,
) -> dict[str, list[tuple[int, float]]]:
    """Return the top-``k`` features per AdaptationVector dimension.

    Args:
        semantics: Output of :func:`compute_per_feature_semantics`.
        k: Number of top features per dimension.

    Returns:
        Mapping ``dimension_name -> [(feature_idx, pearson_r), ...]``
        sorted in descending absolute-correlation order. Exactly eight
        keys, matching ``ADAPTATION_DIMS``.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    scores: dict[str, list[tuple[int, float]]] = {
        dim: [] for dim in ADAPTATION_DIMS
    }
    for sem in semantics:
        top_map = dict(sem.top_dimension_correlations)
        for dim in ADAPTATION_DIMS:
            r = top_map.get(dim, 0.0)
            scores[dim].append((sem.feature_idx, float(r)))

    truncated: dict[str, list[tuple[int, float]]] = {}
    for dim, pairs in scores.items():
        pairs.sort(key=lambda fr: abs(fr[1]), reverse=True)
        truncated[dim] = pairs[:k]
    return truncated


def decoder_cosine_similarity_matrix(
    dictionary: FeatureDictionary,
    max_features: Optional[int] = None,
) -> torch.Tensor:
    """Cosine-similarity matrix between all (or first N) decoder columns.

    Args:
        dictionary: A populated :class:`FeatureDictionary`.
        max_features: Optional upper bound on feature count — useful for
            plotting heatmaps on a large dictionary without exceeding
            memory / pixels.

    Returns:
        ``[M, M]`` tensor where ``M = min(max_features, d_dict)``.
    """
    w = dictionary.sae.decoder.weight.detach()  # [d_model, d_dict]
    m = w.size(1)
    if max_features is not None:
        m = min(m, int(max_features))
    cols = w[:, :m]
    norms = cols.norm(dim=0, keepdim=True).clamp(min=1e-12)
    normed = cols / norms
    return normed.t() @ normed


__all__ = [
    "FeatureSemantics",
    "compute_per_feature_semantics",
    "decoder_cosine_similarity_matrix",
    "feature_steering_vector",
    "identify_monosemantic_features",
    "top_features_per_dimension",
]
