"""Per-decision counterfactual explanations for the adaptation pathway.

Motivation
----------

After the adaptation controller has produced an
:class:`~i3.adaptation.types.AdaptationVector`, the user (and the
research panel) may reasonably ask: *"What feature of my behaviour
caused this particular adaptation? If that feature had been different,
what would the adaptation have been?"* That is the contract of a
counterfactual explanation in the sense of Wachter, Mittelstadt &
Russell (2018): a minimal perturbation of the input features that
would have produced a meaningfully different output.

Two implementation choices matter here:

1. **Locality.** The adaptation pipeline contains non-differentiable
   thresholds (the :class:`~i3.adaptation.dimensions.AccessibilityAdapter`
   in particular) so a global optimisation over the full input space is
   neither needed nor well-behaved. We instead use a *local gradient-
   based* method: we linearise the supplied ``mapping_fn`` at the
   observed feature vector, and for each (feature, dimension) pair
   compute the minimum perturbation along that feature that would
   change that dimension by a fixed target delta. The top ``k`` pairs
   by sensitivity are returned.

2. **Soft-fail on gradient gaps.** When the user supplies a surrogate
   that is only piecewise differentiable, ``torch.autograd`` may
   return ``None`` or a zero gradient at the observed point. In that
   case the counterfactual for that pair is silently dropped from the
   ranking (never exposed to the UI as a bogus explanation).

References
----------
- Wachter, S., Mittelstadt, B., & Russell, C. (2018). *Counterfactual
  Explanations Without Opening the Black Box: Automated Decisions and
  the GDPR.* Harvard Journal of Law & Technology 31 (2).
  https://arxiv.org/abs/1711.00399
- Verma, S., Dickerson, J., & Hines, K. (2022). *Counterfactual
  Explanations for Machine Learning: A Review.* ACM Computing
  Surveys. https://arxiv.org/abs/2010.10596
- Mothilal, R. K., Sharma, A., & Tan, C. (2020). *Explaining Machine
  Learning Classifiers through Diverse Counterfactual Explanations*
  (DiCE). FAccT 2020. https://arxiv.org/abs/1905.07697
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from i3.interpretability.feature_attribution import (
    ADAPTATION_DIMS,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counterfactual schema
# ---------------------------------------------------------------------------


class Counterfactual(BaseModel):
    """One counterfactual explanation row.

    Represents the statement: *"If ``feature_name`` had been
    ``counterfactual_value`` instead of ``current_value``, the
    adaptation dimension ``dimension_affected`` would have been
    ``counterfactual_dimension`` instead of ``current_dimension``."*

    Attributes:
        feature_name: Canonical feature name (matches
            :data:`FEATURE_NAMES`).
        current_value: The feature's observed value.
        counterfactual_value: The hypothetical feature value that
            produces the counterfactual adaptation dimension. Always
            different from ``current_value``.
        dimension_affected: Canonical adaptation-dimension name
            (matches :data:`ADAPTATION_DIMS`).
        current_dimension: The adaptation dimension's observed value.
        counterfactual_dimension: The dimension value the surrogate
            predicts under the counterfactual feature value.
        sensitivity: Magnitude of the local gradient,
            ``|∂dim/∂feature|``. Larger values mean the feature has a
            stronger *local* influence on the adaptation dimension —
            this is the ranking key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: str = Field(..., min_length=1, max_length=64)
    current_value: float
    counterfactual_value: float
    dimension_affected: str = Field(..., min_length=1, max_length=64)
    current_dimension: float
    counterfactual_dimension: float
    sensitivity: float = Field(..., ge=0.0)

    @field_validator("feature_name")
    @classmethod
    def _check_feature_name(cls, v: str) -> str:
        """Reject feature names not in :data:`FEATURE_NAMES`."""
        if v not in FEATURE_NAMES:
            raise ValueError(f"unknown feature_name {v!r}")
        return v

    @field_validator("dimension_affected")
    @classmethod
    def _check_dim_name(cls, v: str) -> str:
        """Reject dimension names not in :data:`ADAPTATION_DIMS`."""
        if v not in ADAPTATION_DIMS:
            raise ValueError(f"unknown dimension_affected {v!r}")
        return v

    @field_validator("counterfactual_value")
    @classmethod
    def _must_differ(cls, v: float, info) -> float:
        """Require the counterfactual feature value to differ from the observed one."""
        current = info.data.get("current_value")
        if current is not None and math.isclose(
            float(v), float(current), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "counterfactual_value must differ from current_value "
                "(minimum separation 1e-12)."
            )
        return float(v)


# ---------------------------------------------------------------------------
# CounterfactualExplainer
# ---------------------------------------------------------------------------


class CounterfactualExplainer:
    """Local-gradient counterfactual explainer.

    The explainer linearises a user-supplied ``mapping_fn`` — a
    differentiable surrogate of the feature-to-adaptation pipeline —
    at the observed feature point and returns the ``k`` feature /
    dimension pairs with the largest local sensitivity.

    Args:
        mapping_fn: Differentiable callable mapping a length-``32``
            tensor to a length-``8`` tensor. May be an
            :class:`~torch.nn.Module`, a lambda, or any other callable.
            The same convention is used in
            :class:`~i3.interpretability.feature_attribution.FeatureAttributor`.
        feature_names: Override for the canonical feature-name list.
            Must have length 32.
        dimension_names: Override for the canonical dimension-name
            list. Must have length 8.
        target_delta: Target change in the adaptation dimension used
            to size the counterfactual step
            (``Δfeature = target_delta / gradient``). Default 0.2.

    Raises:
        TypeError: If ``mapping_fn`` is not callable.
        ValueError: If ``target_delta`` is non-positive or the name
            lists have the wrong length.
    """

    def __init__(
        self,
        mapping_fn: Callable[[torch.Tensor], torch.Tensor],
        feature_names: Sequence[str] | None = None,
        dimension_names: Sequence[str] | None = None,
        target_delta: float = 0.2,
    ) -> None:
        if not callable(mapping_fn):
            raise TypeError(
                f"mapping_fn must be callable, got {type(mapping_fn).__name__}"
            )
        if target_delta <= 0.0 or not math.isfinite(target_delta):
            raise ValueError(
                f"target_delta must be a positive finite float, got {target_delta!r}"
            )

        fnames: Sequence[str] = (
            list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        )
        dnames: Sequence[str] = (
            list(dimension_names) if dimension_names is not None else list(ADAPTATION_DIMS)
        )
        if len(fnames) != 32:
            raise ValueError(
                f"feature_names must have length 32, got {len(fnames)}"
            )
        if len(dnames) != 8:
            raise ValueError(
                f"dimension_names must have length 8, got {len(dnames)}"
            )

        self._mapping_fn = mapping_fn
        self._feature_names: list[str] = list(fnames)
        self._dimension_names: list[str] = list(dnames)
        self._target_delta: float = float(target_delta)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @property
    def target_delta(self) -> float:
        """Configured target change in the adaptation dimension."""
        return self._target_delta

    def explain(
        self,
        feature_window: torch.Tensor,
        adaptation: torch.Tensor,
        k: int = 3,
    ) -> list[Counterfactual]:
        """Return the top ``k`` counterfactuals for ``feature_window``.

        Args:
            feature_window: A feature tensor. Either shape ``[32]``
                (a single feature vector) or ``[seq_len, 32]`` (a
                window whose *last row* is taken as the current
                feature vector, matching the convention of
                :mod:`i3.adaptation.uncertainty`).
            adaptation: The observed adaptation vector as a length-8
                tensor. Used to populate
                :attr:`Counterfactual.current_dimension` and to define
                the starting point of the counterfactual arrow.
            k: Number of counterfactuals to return. Must satisfy
                ``1 <= k <= 32 * 8``.

        Returns:
            List of :class:`Counterfactual` instances, ordered by
            decreasing sensitivity. The list length is at most ``k``
            — shorter if the surrogate returns a zero/None gradient
            for some pairs.

        Raises:
            ValueError: If the input shapes are wrong or ``k`` is out
                of range.
        """
        x = self._unwrap_feature_window(feature_window)
        adapt_vec = self._check_adaptation(adaptation)
        if k < 1 or k > len(self._feature_names) * len(self._dimension_names):
            raise ValueError(
                f"k must be between 1 and {len(self._feature_names) * len(self._dimension_names)}, got {k}"
            )

        sensitivities = self._compute_sensitivity(x)  # [8, 32]
        ranked = self._rank_pairs(sensitivities, k_cap=k)

        results: list[Counterfactual] = []
        for dim_idx, feat_idx, grad_val in ranked:
            # Minimum feature perturbation that moves the dimension by
            # ``target_delta``. Guard against microscopic gradients.
            if abs(grad_val) < 1e-8:
                continue
            delta_feature = self._target_delta / grad_val
            new_feature_value = float(x[feat_idx].item()) + float(delta_feature)
            new_dim_value = float(adapt_vec[dim_idx].item()) + self._target_delta

            # Iter 41: clamp the linearly-extrapolated counterfactual values
            # so the natural-language sentence never reports "formality
            # would have been 1.089" — every feature *and* every adaptation
            # dimension lives in [0, 1] downstream, so promising a value
            # above 1.0 is misleading.  We clamp tightly here while the
            # underlying linear sensitivity stays unaffected.
            new_feature_value = min(1.0, max(0.0, new_feature_value))
            new_dim_value = min(1.0, max(0.0, new_dim_value))

            current_value = float(x[feat_idx].item())
            if math.isclose(new_feature_value, current_value, rel_tol=0.0, abs_tol=1e-12):
                continue

            try:
                cf = Counterfactual(
                    feature_name=self._feature_names[feat_idx],
                    current_value=current_value,
                    counterfactual_value=new_feature_value,
                    dimension_affected=self._dimension_names[dim_idx],
                    current_dimension=float(adapt_vec[dim_idx].item()),
                    counterfactual_dimension=new_dim_value,
                    sensitivity=abs(float(grad_val)),
                )
            except ValueError:
                # Pydantic rejected the pair (e.g. duplicate values after
                # rounding). Skip and move on.
                continue
            results.append(cf)
            if len(results) >= k:
                break
        return results

    @staticmethod
    def to_natural_language(cf: Counterfactual) -> str:
        """Render a single counterfactual as a natural-language sentence.

        Example output (exact phrasing depends on the feature):

            *"If your pause-ratio had been 0.24 instead of 0.54, the
            cognitive_load dimension would have been 0.42 instead of
            0.68."*

        Args:
            cf: A :class:`Counterfactual` instance.

        Returns:
            A plain-English sentence ending in a period. Never empty.

        Raises:
            TypeError: If ``cf`` is not a :class:`Counterfactual`.
        """
        if not isinstance(cf, Counterfactual):
            raise TypeError(
                f"cf must be a Counterfactual, got {type(cf).__name__}"
            )
        return (
            f"If your {cf.feature_name.replace('_', '-')} had been "
            f"{cf.counterfactual_value:.3f} instead of {cf.current_value:.3f}, "
            f"the {cf.dimension_affected} adaptation would have been "
            f"{cf.counterfactual_dimension:.3f} instead of "
            f"{cf.current_dimension:.3f}."
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _unwrap_feature_window(
        self, feature_window: torch.Tensor
    ) -> torch.Tensor:
        """Validate and unwrap ``feature_window`` into a length-32 vector.

        Args:
            feature_window: Input tensor of rank 1 (``[32]``) or
                rank 2 (``[seq_len, 32]``).

        Returns:
            A length-32 tensor with ``requires_grad=True`` flipped on
            *after* detachment so the gradient graph is local to this
            explanation call and does not pollute any caller state.

        Raises:
            ValueError: If the shape is not one of the accepted forms.
            TypeError: If the argument is not a tensor.
        """
        if not isinstance(feature_window, torch.Tensor):
            raise TypeError(
                "feature_window must be a torch.Tensor, got "
                f"{type(feature_window).__name__}"
            )
        if feature_window.dim() == 1:
            if feature_window.numel() != len(self._feature_names):
                raise ValueError(
                    "1-D feature_window must have length "
                    f"{len(self._feature_names)}, got {feature_window.numel()}"
                )
            vec = feature_window
        elif feature_window.dim() == 2:
            if feature_window.shape[1] != len(self._feature_names):
                raise ValueError(
                    "2-D feature_window must have shape "
                    f"[seq_len, {len(self._feature_names)}], got "
                    f"{tuple(feature_window.shape)}"
                )
            vec = feature_window[-1]
        else:
            raise ValueError(
                "feature_window must be 1-D or 2-D, got rank "
                f"{feature_window.dim()}"
            )
        return vec.detach().clone().to(dtype=torch.float32)

    def _check_adaptation(self, adaptation: torch.Tensor) -> torch.Tensor:
        """Validate ``adaptation`` as an 8-element tensor."""
        if not isinstance(adaptation, torch.Tensor):
            raise TypeError(
                "adaptation must be a torch.Tensor, got "
                f"{type(adaptation).__name__}"
            )
        if adaptation.dim() != 1 or adaptation.numel() != len(self._dimension_names):
            raise ValueError(
                "adaptation must be a 1-D tensor of length "
                f"{len(self._dimension_names)}, got shape {tuple(adaptation.shape)}"
            )
        return adaptation.detach().clone().to(dtype=torch.float32)

    def _compute_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the local Jacobian ``∂adaptation/∂feature``.

        Returns:
            A tensor of shape ``[n_dims, n_features]`` (``[8, 32]``)
            where entry ``[j, i]`` is ``∂dim_j / ∂feature_i``.
        """
        x_var = x.detach().clone().requires_grad_(True)
        try:
            out = self._mapping_fn(x_var)
        except TypeError:
            # Some callables only accept 2-D batched input.
            out = self._mapping_fn(x_var.unsqueeze(0)).squeeze(0)
        if out.dim() != 1:
            out = out.reshape(-1)
        if out.numel() != len(self._dimension_names):
            raise ValueError(
                "mapping_fn must return a length-"
                f"{len(self._dimension_names)} tensor, got {out.numel()}"
            )

        n_dims = len(self._dimension_names)
        n_feats = len(self._feature_names)
        jac = torch.zeros(n_dims, n_feats, dtype=torch.float32)
        for j in range(n_dims):
            grad = torch.autograd.grad(
                out[j],
                x_var,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                continue
            jac[j] = grad.detach()
        return jac

    @staticmethod
    def _rank_pairs(
        sensitivities: torch.Tensor, k_cap: int
    ) -> list[tuple[int, int, float]]:
        """Return (dim, feat, grad) triples sorted by ``|grad|`` desc.

        The slice is over-sampled (``2 * k_cap`` up to the full grid)
        so that callers can skip microscopic-gradient pairs while still
        returning a list of length ``k_cap`` in the common case.
        """
        abs_sens = sensitivities.abs()
        flat = abs_sens.flatten()
        n = int(flat.numel())
        # Over-sample so the caller can skip degenerate entries.
        k_search = min(n, max(k_cap * 2, k_cap + 4))
        topk = torch.topk(flat, k=k_search, largest=True, sorted=True)
        n_feats = sensitivities.shape[1]

        triples: list[tuple[int, int, float]] = []
        for flat_idx in topk.indices.tolist():
            dim_idx = flat_idx // n_feats
            feat_idx = flat_idx % n_feats
            triples.append(
                (int(dim_idx), int(feat_idx), float(sensitivities[dim_idx, feat_idx].item()))
            )
        return triples


__all__ = [
    "Counterfactual",
    "CounterfactualExplainer",
]
