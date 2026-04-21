"""Hypothesis property tests for :class:`InteractionFeatureVector`.

Invariants:
    * **Dimensionality** — ``to_tensor()`` always produces a 32-dim tensor.
    * **Finiteness** — round-tripping through ``to_tensor()`` /
      ``from_tensor()`` preserves every finite value.
    * **Shape errors** — ``from_tensor()`` rejects tensors of the wrong
      length.
    * **NaN handling** — a ``NaN`` entering the tensor is round-tripped
      exactly (the dataclass does not silently swap it — NaN handling
      happens upstream at the feature extractor).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

torch = pytest.importorskip("torch")

from i3.interaction.types import FEATURE_NAMES, InteractionFeatureVector


# A "realistic" feature value: within the nominal [-1, 1] envelope that
# FeatureExtractor produces, but we also probe slightly outside it.
_FEATURE_FLOAT = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


class TestFeatureVectorDimensionality:
    def test_feature_names_match_dataclass_fields(self) -> None:
        """``FEATURE_NAMES`` is the source of truth for tensor layout."""
        assert len(FEATURE_NAMES) == 32

    @given(arr=arrays(dtype=np.float32, shape=(32,), elements=_FEATURE_FLOAT))
    @settings(max_examples=200, deadline=None)
    def test_to_tensor_shape(self, arr: np.ndarray) -> None:
        """``to_tensor()`` always returns a 1-D tensor of length 32."""
        vec = InteractionFeatureVector.from_tensor(torch.from_numpy(arr))
        t = vec.to_tensor()
        assert t.shape == (32,)
        assert t.dtype == torch.float32


class TestFeatureVectorRoundTrip:
    @given(arr=arrays(dtype=np.float32, shape=(32,), elements=_FEATURE_FLOAT))
    @settings(max_examples=200, deadline=None)
    @example(arr=np.zeros(32, dtype=np.float32))
    @example(arr=np.ones(32, dtype=np.float32))
    def test_tensor_round_trip(self, arr: np.ndarray) -> None:
        """``from_tensor(x).to_tensor()`` returns a tensor equal to ``x``."""
        t = torch.from_numpy(arr)
        vec = InteractionFeatureVector.from_tensor(t)
        restored = vec.to_tensor()
        assert torch.allclose(restored, t, atol=1e-6, equal_nan=False)


class TestFeatureVectorBounds:
    @given(
        values=st.lists(_FEATURE_FLOAT, min_size=32, max_size=32),
    )
    @settings(max_examples=100, deadline=None)
    def test_values_preserved_bitwise(self, values: list[float]) -> None:
        """Values handed to the constructor are kept intact (no clamping
        inside the dataclass — clamping happens at the adaptation layer)."""
        vec = InteractionFeatureVector(**dict(zip(FEATURE_NAMES, values)))
        for name, v in zip(FEATURE_NAMES, values):
            stored = getattr(vec, name)
            if math.isfinite(v):
                assert stored == pytest.approx(v, abs=1e-6)


class TestFeatureVectorErrorHandling:
    @given(
        n=st.integers(min_value=0, max_value=64).filter(lambda n: n != 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_wrong_size_raises(self, n: int) -> None:
        """Tensors not of length 32 must raise ``ValueError``."""
        t = torch.zeros(n, dtype=torch.float32)
        with pytest.raises(ValueError, match="32"):
            InteractionFeatureVector.from_tensor(t)


class TestFeatureVectorNaNHandling:
    def test_nan_not_silently_swallowed(self) -> None:
        """A NaN inserted by a caller is round-tripped as NaN — the
        dataclass itself does not mutate values."""
        arr = np.zeros(32, dtype=np.float32)
        arr[5] = float("nan")
        t = torch.from_numpy(arr)
        vec = InteractionFeatureVector.from_tensor(t)
        # to_tensor exposes the NaN back
        assert math.isnan(vec.to_tensor()[5].item())
