"""Integrated-gradients feature attribution for the adaptation pathway.

Attributes each of the 32 :class:`~i3.interaction.types.InteractionFeatureVector`
dimensions to each of the 8
:class:`~i3.adaptation.types.AdaptationVector` dimensions using the
integrated-gradients method of Sundararajan, Taly & Yan (2017):

.. math::

    \\text{IG}_i(x) = (x_i - x'_i) \\cdot
    \\int_{\\alpha=0}^{1}
    \\frac{\\partial F(x' + \\alpha(x - x'))}{\\partial x_i} d\\alpha

where ``x`` is the observed feature vector, ``x'`` is a baseline (all
zeros by default) and ``F`` is any scalar-valued differentiable mapping
from ``x`` to the target AdaptationVector dimension.

The integral is approximated by a Riemann sum with 50 interpolation
steps, per the authors' recommendation for neural networks.

References
----------
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution
  for Deep Networks*.  ICML 2017.
  https://proceedings.mlr.press/v70/sundararajan17a.html
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

# Canonical feature names matching ``InteractionFeatureVector`` order.
# Duplicated here (rather than imported from ``i3.interaction.types``)
# so this module is importable in isolation in tests.
FEATURE_NAMES: list[str] = [
    # Keystroke dynamics (8)
    "mean_iki",
    "std_iki",
    "mean_burst_length",
    "mean_pause_duration",
    "backspace_ratio",
    "composition_speed",
    "pause_before_send",
    "editing_effort",
    # Message content (8)
    "message_length",
    "type_token_ratio",
    "mean_word_length",
    "flesch_kincaid",
    "question_ratio",
    "formality",
    "emoji_density",
    "sentiment_valence",
    # Session dynamics (8)
    "length_trend",
    "latency_trend",
    "vocab_trend",
    "engagement_velocity",
    "topic_coherence",
    "session_progress",
    "time_deviation",
    "response_depth",
    # Deviation metrics (8)
    "iki_deviation",
    "length_deviation",
    "vocab_deviation",
    "formality_deviation",
    "speed_deviation",
    "engagement_deviation",
    "complexity_deviation",
    "pattern_deviation",
]
"""Ordered list of the 32 feature names tracked by this module."""

ADAPTATION_DIMS: list[str] = [
    "cognitive_load",
    "formality",
    "verbosity",
    "emotionality",
    "directness",
    "emotional_tone",
    "accessibility",
    "reserved",
]
"""Ordered list of the 8 AdaptationVector output dimensions."""


class FeatureAttributor:
    """Integrated-gradients attributor for the feature->adaptation mapping.

    Given a scalar-to-vector map from a 32-dim feature vector to an 8-dim
    adaptation vector, this class returns a dictionary of per-feature
    per-dimension contributions using the integrated-gradients method
    (Sundararajan et al., 2017).

    Because the adaptation pipeline's :class:`AdaptationController` is
    *not* end-to-end differentiable (it contains thresholds and
    classifiers), this attributor accepts a ``mapping_fn`` callable.
    Typical use-cases instantiate the callable as either:

    - A surrogate :class:`nn.Module` trained to predict the
      ``AdaptationVector`` from feature vectors (for production
      attribution), or
    - An ``AdaptiveSLM.conditioning_projector`` composed with a small
      adapter (for probing learned conditioning), or
    - The identity (in tests: ``f(x) -> x[:8]``).

    Parameters
    ----------
    mapping_fn : Callable[[torch.Tensor], torch.Tensor]
        A *differentiable* function mapping a 1-D feature tensor of
        length ``feature_dim`` (default 32) to a 1-D adaptation tensor of
        length ``adaptation_dim`` (default 8).  Must accept a 2-D
        ``[batch, feature_dim]`` tensor too.
    feature_dim : int, default 32
        Dimensionality of the input feature vector.
    adaptation_dim : int, default 8
        Dimensionality of the output adaptation vector.
    n_steps : int, default 50
        Number of interpolation steps for the Riemann-sum approximation
        of the integral.  50 is the value recommended by
        Sundararajan et al. (2017) for deep networks.

    Attributes
    ----------
    feature_names : list[str]
        32 feature names keyed in the returned attribution dict.
    adaptation_dims : list[str]
        8 adaptation-dimension names keyed in the returned dict.
    n_steps : int
        Configured number of integration steps.
    """

    def __init__(
        self,
        mapping_fn: Callable[[torch.Tensor], torch.Tensor],
        feature_dim: int = 32,
        adaptation_dim: int = 8,
        n_steps: int = 50,
    ) -> None:
        if not callable(mapping_fn):
            raise TypeError(
                f"mapping_fn must be callable, got {type(mapping_fn).__name__}"
            )
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {feature_dim}")
        if adaptation_dim <= 0:
            raise ValueError(
                f"adaptation_dim must be > 0, got {adaptation_dim}"
            )
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")

        self._mapping_fn = mapping_fn
        self.feature_dim: int = int(feature_dim)
        self.adaptation_dim: int = int(adaptation_dim)
        self.n_steps: int = int(n_steps)
        # Use module-level names but truncate / pad if caller configured
        # different dims -- keeps this class usable with custom mappings.
        self.feature_names: list[str] = (
            FEATURE_NAMES[:feature_dim]
            if feature_dim <= len(FEATURE_NAMES)
            else FEATURE_NAMES + [f"feat_{i}" for i in range(len(FEATURE_NAMES), feature_dim)]
        )
        self.adaptation_dims: list[str] = (
            ADAPTATION_DIMS[:adaptation_dim]
            if adaptation_dim <= len(ADAPTATION_DIMS)
            else ADAPTATION_DIMS
            + [f"adapt_{i}" for i in range(len(ADAPTATION_DIMS), adaptation_dim)]
        )

    def attribute(
        self,
        feature_vector: torch.Tensor,
        adaptation_vector: Optional[torch.Tensor] = None,
    ) -> dict[str, dict[str, float]]:
        """Compute integrated-gradients attributions.

        Args:
            feature_vector: 1-D tensor of length ``feature_dim``.  Can be
                on any device; attribution is computed on the same
                device.  Is NOT modified.
            adaptation_vector: Optional 1-D tensor of length
                ``adaptation_dim``.  Ignored for the attribution
                computation itself (integrated gradients does not need
                a reference output), but accepted so the signature
                matches the interpretability-API contract described in
                ``docs/research/stretch_goals.md``.

        Returns:
            Nested dict keyed first by feature name then by adaptation
            dimension name.  Each innermost value is a Python ``float``
            representing the contribution of that feature to that
            adaptation dimension.  Positive values push the adaptation
            up; negative values push it down.

        Raises:
            ValueError: If ``feature_vector`` is not 1-D or has the wrong
                length.
        """
        if feature_vector.dim() != 1:
            raise ValueError(
                "feature_vector must be 1-D, got shape "
                f"{tuple(feature_vector.shape)}"
            )
        if feature_vector.numel() != self.feature_dim:
            raise ValueError(
                "feature_vector length mismatch: expected "
                f"{self.feature_dim}, got {feature_vector.numel()}"
            )
        # ``adaptation_vector`` is accepted for API symmetry but not used
        # by integrated gradients (IG needs only the input and a baseline).
        del adaptation_vector

        device = feature_vector.device
        dtype = (
            feature_vector.dtype
            if feature_vector.is_floating_point()
            else torch.float32
        )

        # Zero baseline per the brief.  Shape: [feature_dim].
        baseline = torch.zeros(self.feature_dim, device=device, dtype=dtype)
        target = feature_vector.to(dtype=dtype)

        # Interpolation coefficients alpha_0..alpha_{n-1} uniformly on [0, 1].
        alphas = torch.linspace(
            0.0, 1.0, steps=self.n_steps, device=device, dtype=dtype
        )  # [n_steps]

        # Build interpolated batch: [n_steps, feature_dim].
        # x_alpha = baseline + alpha * (target - baseline)
        diff = target - baseline
        interpolated = (
            baseline.unsqueeze(0)
            + alphas.unsqueeze(1) * diff.unsqueeze(0)
        )
        interpolated.requires_grad_(True)

        # Forward through the mapping.  Some callers may provide a
        # mapping_fn that only accepts 1-D input; in that case fall back
        # to a per-step loop.
        try:
            outputs = self._mapping_fn(interpolated)
        except Exception:
            outputs = torch.stack(
                [self._mapping_fn(x) for x in interpolated], dim=0
            )
        if outputs.dim() == 1:
            outputs = torch.stack(
                [self._mapping_fn(x) for x in interpolated], dim=0
            )
        if outputs.shape[0] != self.n_steps:
            raise ValueError(
                f"mapping_fn must return a leading batch dim of {self.n_steps}, "
                f"got shape {tuple(outputs.shape)}"
            )
        if outputs.shape[-1] != self.adaptation_dim:
            raise ValueError(
                f"mapping_fn output must have last dim {self.adaptation_dim}, "
                f"got shape {tuple(outputs.shape)}"
            )

        # Per-output-dim integrated gradients.
        result: dict[str, dict[str, float]] = {
            name: {dim: 0.0 for dim in self.adaptation_dims}
            for name in self.feature_names
        }

        for j, adapt_name in enumerate(self.adaptation_dims):
            # Sum outputs[:, j] over the batch -- summing gives a scalar
            # whose gradient w.r.t. each interpolated point is the
            # gradient at that point.
            scalar_j = outputs[:, j].sum()
            grads = torch.autograd.grad(
                scalar_j,
                interpolated,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grads is None:
                # mapping_fn did not depend on the input -- zero gradient.
                continue
            # Riemann-sum average over alphas, then multiply by (x - x').
            # Shape: [feature_dim].
            avg_grad = grads.mean(dim=0)
            attributions_j = (diff * avg_grad).detach().cpu().tolist()
            for i, feat_name in enumerate(self.feature_names):
                result[feat_name][adapt_name] = float(attributions_j[i])
        return result


# Convenience: a linear surrogate for unit tests / quick demos.
class LinearFeatureAdapter(nn.Module):
    """Identity-like linear surrogate used in tests and examples.

    This is the smallest possible differentiable mapping from a 32-dim
    feature vector to an 8-dim adaptation vector: a bias-free linear
    layer.  The completeness axiom of integrated gradients
    (Sundararajan et al., 2017, Prop 1) guarantees that for a linear
    model ``f(x) = W x``, ``IG_i = W_{:, i} x_i`` and therefore the sum
    of attributions along any output dimension exactly equals
    ``f(x) - f(0) = f(x)``.  Tests exploit this for exact-sum checks.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        adaptation_dim: int = 8,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, adaptation_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply the linear map."""
        return self.linear(x)


__all__ = [
    "ADAPTATION_DIMS",
    "FEATURE_NAMES",
    "FeatureAttributor",
    "LinearFeatureAdapter",
]
