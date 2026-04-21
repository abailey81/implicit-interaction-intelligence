"""Hypothesis property tests for ``i3.adaptation.types``.

Invariants verified:
    * Every dimension of :class:`AdaptationVector` and :class:`StyleVector`
      is clamped to ``[0, 1]`` regardless of the input value (including
      NaN, ±∞, negative, huge).
    * ``to_dict()`` / ``from_dict()`` and ``to_tensor()`` / ``from_tensor()``
      round-trips are value-preserving up to float tolerance.
    * The 8-dim tensor layout is stable: the final dimension is always 0.0.
"""

from __future__ import annotations

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import example, given, settings, strategies as st

torch = pytest.importorskip("torch")

from i3.adaptation.types import AdaptationVector, StyleVector


# Strategy covering ordinary, boundary, and pathological inputs.  We allow
# NaN / Inf explicitly so the clamping logic is exercised.
_ANY_FLOAT = st.floats(
    allow_nan=True,
    allow_infinity=True,
    min_value=-1e6,
    max_value=1e6,
    width=32,
)

_UNIT_FLOAT = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


# ─────────────────────────────────────────────────────────────────────────
#  StyleVector
# ─────────────────────────────────────────────────────────────────────────


class TestStyleVectorClamping:
    @given(
        f=_ANY_FLOAT, v=_ANY_FLOAT, e=_ANY_FLOAT, d=_ANY_FLOAT,
    )
    @settings(max_examples=200, deadline=None)
    @example(f=float("nan"), v=0.5, e=0.5, d=0.5)
    @example(f=float("inf"), v=-float("inf"), e=2.0, d=-3.0)
    def test_dimensions_in_unit_interval(
        self, f: float, v: float, e: float, d: float
    ) -> None:
        """All four style dimensions lie in ``[0, 1]`` after construction."""
        s = StyleVector(formality=f, verbosity=v, emotionality=e, directness=d)
        for val in (s.formality, s.verbosity, s.emotionality, s.directness):
            assert 0.0 <= val <= 1.0
            assert math.isfinite(val)


class TestStyleVectorRoundTrip:
    @given(
        f=_UNIT_FLOAT, v=_UNIT_FLOAT, e=_UNIT_FLOAT, d=_UNIT_FLOAT,
    )
    @settings(max_examples=200, deadline=None)
    def test_dict_round_trip(
        self, f: float, v: float, e: float, d: float
    ) -> None:
        """``from_dict(to_dict(s))`` recovers the original style vector."""
        s = StyleVector(formality=f, verbosity=v, emotionality=e, directness=d)
        s2 = StyleVector.from_dict(s.to_dict())
        assert s2.formality == pytest.approx(s.formality, abs=1e-6)
        assert s2.verbosity == pytest.approx(s.verbosity, abs=1e-6)
        assert s2.emotionality == pytest.approx(s.emotionality, abs=1e-6)
        assert s2.directness == pytest.approx(s.directness, abs=1e-6)

    @given(
        f=_UNIT_FLOAT, v=_UNIT_FLOAT, e=_UNIT_FLOAT, d=_UNIT_FLOAT,
    )
    @settings(max_examples=100, deadline=None)
    def test_tensor_round_trip(
        self, f: float, v: float, e: float, d: float
    ) -> None:
        """``from_tensor(to_tensor(s))`` recovers the original style vector."""
        s = StyleVector(formality=f, verbosity=v, emotionality=e, directness=d)
        t = s.to_tensor()
        assert t.shape == (4,)
        s2 = StyleVector.from_tensor(t)
        for a, b in zip(
            (s.formality, s.verbosity, s.emotionality, s.directness),
            (s2.formality, s2.verbosity, s2.emotionality, s2.directness),
        ):
            assert a == pytest.approx(b, abs=1e-6)

@pytest.mark.parametrize("bad_size", [0, 1, 2, 3, 5, 6, 8, 16])
def test_style_vector_from_tensor_rejects_bad_size(bad_size: int) -> None:
    """Explicit non-Hypothesis check for wrong-length tensors."""
    t = torch.zeros(bad_size, dtype=torch.float32)
    with pytest.raises(ValueError, match="4"):
        StyleVector.from_tensor(t)


# ─────────────────────────────────────────────────────────────────────────
#  AdaptationVector
# ─────────────────────────────────────────────────────────────────────────


class TestAdaptationVectorClamping:
    @given(
        cog=_ANY_FLOAT, tone=_ANY_FLOAT, acc=_ANY_FLOAT,
        sf=_ANY_FLOAT, sv=_ANY_FLOAT, se=_ANY_FLOAT, sd=_ANY_FLOAT,
    )
    @settings(max_examples=200, deadline=None)
    @example(
        cog=float("nan"), tone=float("nan"), acc=float("nan"),
        sf=float("nan"), sv=float("nan"), se=float("nan"), sd=float("nan"),
    )
    @example(
        cog=-1e9, tone=1e9, acc=1e9,
        sf=-1e9, sv=1e9, se=0.0, sd=1.0,
    )
    def test_scalar_bounds(
        self,
        cog: float, tone: float, acc: float,
        sf: float, sv: float, se: float, sd: float,
    ) -> None:
        """Every scalar field of :class:`AdaptationVector` is in ``[0, 1]``."""
        vec = AdaptationVector(
            cognitive_load=cog,
            style_mirror=StyleVector(
                formality=sf, verbosity=sv, emotionality=se, directness=sd,
            ),
            emotional_tone=tone,
            accessibility=acc,
        )
        for v in (
            vec.cognitive_load,
            vec.emotional_tone,
            vec.accessibility,
            vec.style_mirror.formality,
            vec.style_mirror.verbosity,
            vec.style_mirror.emotionality,
            vec.style_mirror.directness,
        ):
            assert 0.0 <= v <= 1.0
            assert math.isfinite(v)


class TestAdaptationVectorTensor:
    @given(
        cog=_UNIT_FLOAT, tone=_UNIT_FLOAT, acc=_UNIT_FLOAT,
        sf=_UNIT_FLOAT, sv=_UNIT_FLOAT, se=_UNIT_FLOAT, sd=_UNIT_FLOAT,
    )
    @settings(max_examples=200, deadline=None)
    def test_tensor_layout(
        self,
        cog: float, tone: float, acc: float,
        sf: float, sv: float, se: float, sd: float,
    ) -> None:
        """8-dim tensor layout, with reserved slot 7 == 0.0."""
        vec = AdaptationVector(
            cognitive_load=cog,
            style_mirror=StyleVector(
                formality=sf, verbosity=sv, emotionality=se, directness=sd,
            ),
            emotional_tone=tone,
            accessibility=acc,
        )
        t = vec.to_tensor()
        assert t.shape == (8,)
        assert t[7].item() == 0.0, "reserved slot must be 0.0"
        # Spot-check layout
        assert t[0].item() == pytest.approx(vec.cognitive_load, abs=1e-6)
        assert t[1].item() == pytest.approx(vec.style_mirror.formality, abs=1e-6)
        assert t[5].item() == pytest.approx(vec.emotional_tone, abs=1e-6)
        assert t[6].item() == pytest.approx(vec.accessibility, abs=1e-6)


class TestAdaptationVectorRoundTrip:
    @given(
        cog=_UNIT_FLOAT, tone=_UNIT_FLOAT, acc=_UNIT_FLOAT,
        sf=_UNIT_FLOAT, sv=_UNIT_FLOAT, se=_UNIT_FLOAT, sd=_UNIT_FLOAT,
    )
    @settings(max_examples=200, deadline=None)
    def test_dict_round_trip(
        self,
        cog: float, tone: float, acc: float,
        sf: float, sv: float, se: float, sd: float,
    ) -> None:
        """``AdaptationVector.from_dict(v.to_dict())`` is value-preserving."""
        vec = AdaptationVector(
            cognitive_load=cog,
            style_mirror=StyleVector(
                formality=sf, verbosity=sv, emotionality=se, directness=sd,
            ),
            emotional_tone=tone,
            accessibility=acc,
        )
        restored = AdaptationVector.from_dict(vec.to_dict())
        assert restored.cognitive_load == pytest.approx(vec.cognitive_load, abs=1e-6)
        assert restored.emotional_tone == pytest.approx(vec.emotional_tone, abs=1e-6)
        assert restored.accessibility == pytest.approx(vec.accessibility, abs=1e-6)
        for a, b in zip(
            (
                vec.style_mirror.formality,
                vec.style_mirror.verbosity,
                vec.style_mirror.emotionality,
                vec.style_mirror.directness,
            ),
            (
                restored.style_mirror.formality,
                restored.style_mirror.verbosity,
                restored.style_mirror.emotionality,
                restored.style_mirror.directness,
            ),
        ):
            assert a == pytest.approx(b, abs=1e-6)

    @given(
        cog=_UNIT_FLOAT, tone=_UNIT_FLOAT, acc=_UNIT_FLOAT,
        sf=_UNIT_FLOAT, sv=_UNIT_FLOAT, se=_UNIT_FLOAT, sd=_UNIT_FLOAT,
    )
    @settings(max_examples=100, deadline=None)
    def test_tensor_round_trip(
        self,
        cog: float, tone: float, acc: float,
        sf: float, sv: float, se: float, sd: float,
    ) -> None:
        """``AdaptationVector.from_tensor(v.to_tensor())`` is value-preserving."""
        vec = AdaptationVector(
            cognitive_load=cog,
            style_mirror=StyleVector(
                formality=sf, verbosity=sv, emotionality=se, directness=sd,
            ),
            emotional_tone=tone,
            accessibility=acc,
        )
        restored = AdaptationVector.from_tensor(vec.to_tensor())
        for a, b in zip(vec.to_tensor()[:7], restored.to_tensor()[:7]):
            assert a.item() == pytest.approx(b.item(), abs=1e-6)


@pytest.mark.parametrize("bad_size", [0, 1, 4, 7, 9, 16])
def test_adaptation_vector_from_tensor_rejects_bad_size(bad_size: int) -> None:
    """Non-8-element tensors must raise ``ValueError``."""
    t = torch.zeros(bad_size, dtype=torch.float32)
    with pytest.raises(ValueError, match="8"):
        AdaptationVector.from_tensor(t)
