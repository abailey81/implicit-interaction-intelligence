"""Optional SHAP adapter with integrated-gradients fallback.

This module wraps the optional :mod:`shap` third-party library
(Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model
Predictions") so that, if installed, :class:`SHAPAdapter` transparently
delegates to SHAP's Kernel / DeepExplainer.  When ``shap`` is NOT
installed (the default for the I^3 demo to keep the dependency
footprint small), :class:`SHAPAdapter` falls back to the in-house
integrated-gradients attributor :class:`~i3.interpretability.feature_attribution.FeatureAttributor`.

Both methods produce attributions with the *completeness* property -- the
sum of attributions equals ``f(x) - f(0)`` -- so swapping between the
two does not change the qualitative interpretation of the result, only
the noise characteristics.

References
----------
- Lundberg, S. M., & Lee, S.-I. (2017).  *A Unified Approach to
  Interpreting Model Predictions*.  NeurIPS 2017.
- Sundararajan, M., Taly, A., & Yan, Q. (2017).  *Axiomatic Attribution
  for Deep Networks*.  ICML 2017.  (Fallback method.)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from i3.interpretability.feature_attribution import (
    ADAPTATION_DIMS,
    FEATURE_NAMES,
    FeatureAttributor,
)

logger = logging.getLogger(__name__)


def shap_available() -> bool:
    """Return True iff the third-party :mod:`shap` package is importable.

    Returns:
        ``True`` if ``import shap`` succeeds, ``False`` otherwise.
        The result is *not* cached so callers can perform a fresh check
        after installing the package mid-session.
    """
    try:
        import importlib

        importlib.import_module("shap")
    except ImportError:
        return False
    except Exception:  # pragma: no cover - defensive
        return False
    return True


class SHAPAdapter:
    """Adapter that uses SHAP when available and integrated gradients otherwise.

    Parameters
    ----------
    mapping_fn : Callable[[torch.Tensor], torch.Tensor]
        Differentiable mapping from a 1-D (or 2-D batched) feature tensor
        to an adaptation-vector tensor.  Forwarded to the fallback
        :class:`FeatureAttributor`; also used as SHAP's prediction
        function when SHAP is available.
    feature_dim : int, default 32
        Dimensionality of the input feature space.
    adaptation_dim : int, default 8
        Dimensionality of the output adaptation space.
    n_background_samples : int, default 16
        Number of background samples used to construct the SHAP
        Kernel / Deep explainer's reference distribution.  Ignored when
        SHAP is unavailable.

    Notes
    -----
    SHAP (Shapley values) is a coalitional-game-theoretic attribution
    method with the unique fairness axioms proved by Shapley (1953).
    Integrated gradients gives the same completeness guarantee but
    relies on path integrals rather than feature permutations.  For
    high-dimensional feature vectors both methods have been shown to
    agree closely in practice (Lundberg & Lee, 2017, Fig. 2).
    """

    def __init__(
        self,
        mapping_fn: Callable[[torch.Tensor], torch.Tensor],
        feature_dim: int = 32,
        adaptation_dim: int = 8,
        n_background_samples: int = 16,
    ) -> None:
        self._mapping_fn = mapping_fn
        self.feature_dim: int = int(feature_dim)
        self.adaptation_dim: int = int(adaptation_dim)
        self.n_background_samples: int = int(n_background_samples)
        self._fallback = FeatureAttributor(
            mapping_fn,
            feature_dim=feature_dim,
            adaptation_dim=adaptation_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute SHAP-or-IG attributions for the given feature vector.

        Args:
            feature_vector: 1-D tensor of length :attr:`feature_dim`.
            adaptation_vector: Optional reference adaptation vector
                (unused by SHAP and by integrated gradients; accepted
                for API symmetry with
                :class:`FeatureAttributor.attribute`).

        Returns:
            Nested dict of per-feature per-dimension contributions,
            identical in structure to the fallback's output.
        """
        if shap_available():
            try:
                return self._shap_attribute(feature_vector)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "SHAP attribution failed; falling back to integrated gradients"
                )
        return self._fallback.attribute(feature_vector, adaptation_vector)

    # ------------------------------------------------------------------
    # SHAP implementation (only called when shap is importable)
    # ------------------------------------------------------------------

    def _shap_attribute(
        self, feature_vector: torch.Tensor
    ) -> dict[str, dict[str, float]]:
        """Compute SHAP values via :class:`shap.KernelExplainer`.

        Uses the model-agnostic KernelExplainer so this works even when
        ``mapping_fn`` is a non-differentiable wrapper.  The background
        distribution is a batch of zeros (matching the IG baseline).

        Args:
            feature_vector: 1-D tensor of length :attr:`feature_dim`.

        Returns:
            Nested attributions dict.
        """
        import numpy as np
        import shap  # type: ignore[import-untyped]

        if feature_vector.dim() != 1:
            raise ValueError(
                "feature_vector must be 1-D, got shape "
                f"{tuple(feature_vector.shape)}"
            )

        # Convert mapping_fn to a numpy-in/numpy-out callable as required
        # by shap.KernelExplainer.
        def _predict(x_np: np.ndarray) -> np.ndarray:
            t = torch.as_tensor(x_np, dtype=torch.float32)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            with torch.no_grad():
                y = self._mapping_fn(t)
            return y.detach().cpu().numpy()

        background = np.zeros(
            (self.n_background_samples, self.feature_dim), dtype="float32"
        )
        explainer = shap.KernelExplainer(_predict, background)
        values: Any = explainer.shap_values(
            feature_vector.detach().cpu().numpy().reshape(1, -1),
            silent=True,
        )

        # KernelExplainer returns a list[array] of length ``adaptation_dim``
        # for multi-output models; shape of each array is (1, feature_dim).
        if isinstance(values, list):
            arrays = values
        else:
            arrays = [values]
        # Pad to adaptation_dim if SHAP returned fewer outputs.
        while len(arrays) < self.adaptation_dim:
            arrays.append(np.zeros((1, self.feature_dim), dtype="float32"))

        result: dict[str, dict[str, float]] = {
            feat: dict.fromkeys(ADAPTATION_DIMS[:self.adaptation_dim], 0.0)
            for feat in FEATURE_NAMES[: self.feature_dim]
        }
        for j, adapt_name in enumerate(ADAPTATION_DIMS[: self.adaptation_dim]):
            arr = arrays[j].reshape(-1)
            for i in range(self.feature_dim):
                result[FEATURE_NAMES[i]][adapt_name] = float(arr[i])
        return result


__all__ = ["SHAPAdapter", "shap_available"]
