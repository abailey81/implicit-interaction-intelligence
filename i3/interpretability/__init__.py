"""Interpretability utilities for the Implicit Interaction Intelligence SLM.

This package provides tooling for explaining *why* the adaptation
controller produced a given :class:`AdaptationVector` and *how* the
cross-attention conditioning pathway influenced generation.

Submodules
----------

- :mod:`.feature_attribution` -- integrated gradients
  (Sundararajan, Taly & Yan, 2017) for attributing each of the 32
  :class:`InteractionFeatureVector` dimensions to each of the 8
  :class:`AdaptationVector` dimensions.
- :mod:`.attention_extractor` -- context-manager hook-based extractor of
  cross-attention weights from every
  :class:`~i3.slm.cross_attention.MultiHeadCrossAttention` module in a
  loaded :class:`~i3.slm.model.AdaptiveSLM`.
- :mod:`.token_heatmap` -- JSON-serialisable representation of per-token
  attention weights suitable for a front-end heatmap.
- :mod:`.shap_adapter` -- optional soft-import of the SHAP library
  (Lundberg & Lee, 2017); falls back to integrated gradients when
  unavailable.

References
----------
- Sundararajan, M., Taly, A., & Yan, Q. (2017).  *Axiomatic Attribution
  for Deep Networks*.  ICML 2017.
- Lundberg, S. M., & Lee, S.-I. (2017).  *A Unified Approach to
  Interpreting Model Predictions*.  NeurIPS 2017.
"""

from i3.interpretability.attention_extractor import (
    CrossAttentionExtractor,
    ExtractedAttention,
)
from i3.interpretability.feature_attribution import (
    FeatureAttributor,
    FEATURE_NAMES,
    ADAPTATION_DIMS,
)
from i3.interpretability.token_heatmap import TokenHeatmap
from i3.interpretability.shap_adapter import SHAPAdapter, shap_available

__all__ = [
    "ADAPTATION_DIMS",
    "CrossAttentionExtractor",
    "ExtractedAttention",
    "FEATURE_NAMES",
    "FeatureAttributor",
    "SHAPAdapter",
    "TokenHeatmap",
    "shap_available",
]
