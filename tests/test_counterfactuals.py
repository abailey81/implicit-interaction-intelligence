"""Unit tests for :mod:`i3.interpretability.counterfactuals`.

Covers the counterfactual explainer:
    * Exactly ``k`` rows returned on a non-degenerate surrogate.
    * Counterfactual values differ from the observed ones.
    * Sensitivity ordering (descending).
    * Natural-language rendering non-empty and contains the feature
      name.
    * Invalid input-shape handling.
    * Canonical name validation on the Pydantic schema.
"""

from __future__ import annotations

import pytest
import torch

from i3.interpretability.counterfactuals import (
    Counterfactual,
    CounterfactualExplainer,
)
from i3.interpretability.feature_attribution import (
    ADAPTATION_DIMS,
    FEATURE_NAMES,
    LinearFeatureAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def explainer() -> CounterfactualExplainer:
    """A :class:`CounterfactualExplainer` over a deterministic linear surrogate."""
    torch.manual_seed(0)
    adapter = LinearFeatureAdapter(32, 8)
    return CounterfactualExplainer(
        mapping_fn=adapter,
        target_delta=0.2,
    )


@pytest.fixture
def feature_vector() -> torch.Tensor:
    """A deterministic length-32 feature vector."""
    gen = torch.Generator().manual_seed(7)
    return torch.rand(32, generator=gen, dtype=torch.float32)


@pytest.fixture
def adaptation_vector() -> torch.Tensor:
    """A deterministic length-8 adaptation vector."""
    gen = torch.Generator().manual_seed(11)
    return torch.rand(8, generator=gen, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CounterfactualExplainer
# ---------------------------------------------------------------------------


class TestCounterfactualExplainer:
    """Unit tests for :class:`CounterfactualExplainer`."""

    def test_returns_exactly_k(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        cfs = explainer.explain(feature_vector, adaptation_vector, k=3)
        assert len(cfs) == 3

    def test_returns_at_most_k_with_large_k(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        """With a generous k, the result list length is bounded by the grid."""
        cfs = explainer.explain(feature_vector, adaptation_vector, k=16)
        assert len(cfs) <= 16

    def test_counterfactual_differs_from_current(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        cfs = explainer.explain(feature_vector, adaptation_vector, k=3)
        for cf in cfs:
            assert cf.counterfactual_value != cf.current_value

    def test_sensitivity_monotone(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        """Rows must be ordered by decreasing sensitivity."""
        cfs = explainer.explain(feature_vector, adaptation_vector, k=5)
        sensitivities = [cf.sensitivity for cf in cfs]
        assert sensitivities == sorted(sensitivities, reverse=True)

    def test_names_are_canonical(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        """Every row must reference a canonical feature and dimension name."""
        cfs = explainer.explain(feature_vector, adaptation_vector, k=5)
        for cf in cfs:
            assert cf.feature_name in FEATURE_NAMES
            assert cf.dimension_affected in ADAPTATION_DIMS

    def test_natural_language_non_empty(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        cfs = explainer.explain(feature_vector, adaptation_vector, k=1)
        assert cfs, "expected at least one counterfactual"
        sentence = CounterfactualExplainer.to_natural_language(cfs[0])
        assert isinstance(sentence, str)
        assert sentence.strip() != ""
        # The feature name (with underscore replaced) must appear.
        feature_fragment = cfs[0].feature_name.replace("_", "-")
        assert feature_fragment in sentence
        assert sentence.endswith(".")

    def test_natural_language_type_guard(self) -> None:
        """Passing a non-Counterfactual must raise :class:`TypeError`."""
        with pytest.raises(TypeError):
            CounterfactualExplainer.to_natural_language("not a cf")  # type: ignore[arg-type]

    def test_invalid_feature_shape_raises(
        self,
        explainer: CounterfactualExplainer,
        adaptation_vector: torch.Tensor,
    ) -> None:
        with pytest.raises(ValueError):
            explainer.explain(torch.randn(31), adaptation_vector, k=1)

    def test_invalid_feature_rank_raises(
        self,
        explainer: CounterfactualExplainer,
        adaptation_vector: torch.Tensor,
    ) -> None:
        with pytest.raises(ValueError):
            explainer.explain(torch.randn(2, 3, 32), adaptation_vector, k=1)

    def test_invalid_adaptation_shape_raises(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
    ) -> None:
        with pytest.raises(ValueError):
            explainer.explain(feature_vector, torch.randn(7), k=1)

    def test_invalid_k_raises(
        self,
        explainer: CounterfactualExplainer,
        feature_vector: torch.Tensor,
        adaptation_vector: torch.Tensor,
    ) -> None:
        with pytest.raises(ValueError):
            explainer.explain(feature_vector, adaptation_vector, k=0)
        with pytest.raises(ValueError):
            explainer.explain(feature_vector, adaptation_vector, k=9999)

    def test_invalid_target_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            CounterfactualExplainer(
                mapping_fn=LinearFeatureAdapter(32, 8),
                target_delta=0.0,
            )
        with pytest.raises(ValueError):
            CounterfactualExplainer(
                mapping_fn=LinearFeatureAdapter(32, 8),
                target_delta=-1.0,
            )

    def test_non_callable_mapping_raises(self) -> None:
        with pytest.raises(TypeError):
            CounterfactualExplainer(mapping_fn=42)  # type: ignore[arg-type]

    def test_feature_window_2d_accepted(
        self,
        explainer: CounterfactualExplainer,
        adaptation_vector: torch.Tensor,
    ) -> None:
        """A 2-D feature window is accepted (last row is used)."""
        w = torch.rand(5, 32)
        cfs = explainer.explain(w, adaptation_vector, k=2)
        assert len(cfs) == 2


# ---------------------------------------------------------------------------
# Counterfactual Pydantic schema
# ---------------------------------------------------------------------------


class TestCounterfactualSchema:
    """Schema-level tests for :class:`Counterfactual`."""

    def test_rejects_unknown_feature_name(self) -> None:
        with pytest.raises(ValueError):
            Counterfactual(
                feature_name="not_a_feature",
                current_value=0.1,
                counterfactual_value=0.2,
                dimension_affected="cognitive_load",
                current_dimension=0.5,
                counterfactual_dimension=0.7,
                sensitivity=0.4,
            )

    def test_rejects_unknown_dimension(self) -> None:
        with pytest.raises(ValueError):
            Counterfactual(
                feature_name="mean_iki",
                current_value=0.1,
                counterfactual_value=0.2,
                dimension_affected="not_a_dim",
                current_dimension=0.5,
                counterfactual_dimension=0.7,
                sensitivity=0.4,
            )

    def test_rejects_identical_values(self) -> None:
        """The counterfactual value must differ from the observed one."""
        with pytest.raises(ValueError):
            Counterfactual(
                feature_name="mean_iki",
                current_value=0.3,
                counterfactual_value=0.3,
                dimension_affected="cognitive_load",
                current_dimension=0.5,
                counterfactual_dimension=0.7,
                sensitivity=0.4,
            )

    def test_rejects_negative_sensitivity(self) -> None:
        with pytest.raises(ValueError):
            Counterfactual(
                feature_name="mean_iki",
                current_value=0.1,
                counterfactual_value=0.2,
                dimension_affected="cognitive_load",
                current_dimension=0.5,
                counterfactual_dimension=0.7,
                sensitivity=-0.1,
            )

    def test_is_frozen(self) -> None:
        cf = Counterfactual(
            feature_name="mean_iki",
            current_value=0.1,
            counterfactual_value=0.2,
            dimension_affected="cognitive_load",
            current_dimension=0.5,
            counterfactual_dimension=0.7,
            sensitivity=0.4,
        )
        with pytest.raises(Exception):
            cf.sensitivity = 0.99  # type: ignore[misc]
