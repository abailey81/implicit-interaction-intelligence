"""Iter 73 — AdaptationVector / StyleVector contract + clamping tests."""
from __future__ import annotations

import pytest
import torch

from i3.adaptation.types import AdaptationVector, StyleVector


# ---------------------------------------------------------------------------
# StyleVector
# ---------------------------------------------------------------------------

def test_style_vector_default_in_unit_interval():
    sv = StyleVector.default()
    for k in ("formality", "verbosity", "emotionality", "directness"):
        v = getattr(sv, k)
        assert 0.0 <= v <= 1.0


def test_style_vector_to_dict_shape():
    sv = StyleVector.default()
    d = sv.to_dict()
    assert set(d.keys()) == {"formality", "verbosity",
                             "emotionality", "directness"}


def test_style_vector_from_tensor_clamps():
    t = torch.tensor([5.0, -2.0, 0.5, 0.5])
    sv = StyleVector.from_tensor(t)
    assert sv.formality == 1.0
    assert sv.verbosity == 0.0
    assert 0.0 <= sv.emotionality <= 1.0


def test_style_vector_from_tensor_rejects_wrong_shape():
    with pytest.raises(ValueError):
        StyleVector.from_tensor(torch.tensor([0.5, 0.5]))


# ---------------------------------------------------------------------------
# AdaptationVector — clamping
# ---------------------------------------------------------------------------

def test_adaptation_vector_clamps_cognitive_load():
    a = AdaptationVector(
        cognitive_load=5.0,
        style_mirror=StyleVector.default(),
        emotional_tone=0.5,
        accessibility=0.0,
    )
    assert a.cognitive_load == 1.0


def test_adaptation_vector_clamps_negative():
    a = AdaptationVector(
        cognitive_load=-2.0,
        style_mirror=StyleVector.default(),
        emotional_tone=-1.0,
        accessibility=-99.0,
    )
    assert a.cognitive_load == 0.0
    assert a.emotional_tone == 0.0
    assert a.accessibility == 0.0


def test_adaptation_vector_clamps_huge_value():
    a = AdaptationVector(
        cognitive_load=1e9,
        style_mirror=StyleVector.default(),
        emotional_tone=1e9,
        accessibility=1e9,
    )
    assert a.cognitive_load == 1.0
    assert a.emotional_tone == 1.0
    assert a.accessibility == 1.0


# ---------------------------------------------------------------------------
# Tensor serialisation
# ---------------------------------------------------------------------------

def test_to_tensor_shape_is_8():
    a = AdaptationVector(
        cognitive_load=0.5,
        style_mirror=StyleVector.default(),
        emotional_tone=0.5,
        accessibility=0.0,
    )
    t = a.to_tensor()
    assert t.shape == (8,)
    assert t.dtype == torch.float32


def test_to_tensor_layout_matches_doc():
    sv = StyleVector(formality=0.1, verbosity=0.2,
                     emotionality=0.3, directness=0.4)
    a = AdaptationVector(
        cognitive_load=0.7,
        style_mirror=sv,
        emotional_tone=0.8,
        accessibility=0.9,
    )
    t = a.to_tensor()
    # Layout from the docstring: [load, F, V, E, D, tone, access, 0]
    assert t[0].item() == pytest.approx(0.7)
    assert t[1].item() == pytest.approx(0.1)
    assert t[2].item() == pytest.approx(0.2)
    assert t[3].item() == pytest.approx(0.3)
    assert t[4].item() == pytest.approx(0.4)
    assert t[5].item() == pytest.approx(0.8)
    assert t[6].item() == pytest.approx(0.9)
    assert t[7].item() == pytest.approx(0.0)


def test_to_tensor_no_nan_under_hostile_input():
    a = AdaptationVector(
        cognitive_load=float("nan"),
        style_mirror=StyleVector.default(),
        emotional_tone=float("nan"),
        accessibility=float("nan"),
    )
    t = a.to_tensor()
    # Clamp must reject NaN — every cell finite
    assert torch.isfinite(t).all()
