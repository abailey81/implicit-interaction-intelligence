"""Iter 68 — multimodal payload validator tests.

Pins the contracts of ``validate_prosody_payload`` and
``validate_gaze_payload``: both accept the documented schema, both
return ``None`` (degrade-to-keystroke) on malformed input, and both
clamp out-of-range / hostile values without crashing the server.
"""
from __future__ import annotations

import math

import pytest

from i3.multimodal import (
    GAZE_FEATURE_KEYS,
    PROSODY_FEATURE_KEYS,
    validate_gaze_payload,
    validate_prosody_payload,
)


# ---------------------------------------------------------------------------
# Prosody
# ---------------------------------------------------------------------------

def _good_prosody(value=0.5):
    base = {k: float(value) for k in PROSODY_FEATURE_KEYS}
    base["samples_count"] = 100
    base["captured_seconds"] = 2.5
    return base


def test_prosody_accepts_well_formed_payload():
    out = validate_prosody_payload(_good_prosody())
    assert out is not None
    # Round-trips into a dataclass with PROSODY_FEATURE_KEYS attrs
    for k in PROSODY_FEATURE_KEYS:
        assert hasattr(out, k)


def test_prosody_rejects_non_dict():
    for bad in (None, [], "hello", 42, 3.14, object()):
        assert validate_prosody_payload(bad) is None


def test_prosody_rejects_missing_key():
    p = _good_prosody()
    del p[PROSODY_FEATURE_KEYS[0]]
    assert validate_prosody_payload(p) is None


def test_prosody_clamps_out_of_range():
    p = _good_prosody()
    p[PROSODY_FEATURE_KEYS[0]] = 1e30
    out = validate_prosody_payload(p)
    assert out is not None
    val = getattr(out, PROSODY_FEATURE_KEYS[0])
    assert 0.0 <= val <= 1.0, f"expected clamp, got {val}"


def test_prosody_clamps_negative():
    p = _good_prosody()
    p[PROSODY_FEATURE_KEYS[0]] = -5.0
    out = validate_prosody_payload(p)
    assert out is not None
    assert getattr(out, PROSODY_FEATURE_KEYS[0]) >= 0.0


def test_prosody_rejects_nan():
    p = _good_prosody()
    p[PROSODY_FEATURE_KEYS[0]] = float("nan")
    out = validate_prosody_payload(p)
    # Either rejects entirely OR clamps NaN to a sane default.
    if out is not None:
        v = getattr(out, PROSODY_FEATURE_KEYS[0])
        assert not math.isnan(v)


def test_prosody_rejects_string_value():
    p = _good_prosody()
    p[PROSODY_FEATURE_KEYS[0]] = "not a number"
    out = validate_prosody_payload(p)
    # Either rejects or coerces to a default; never raises.
    if out is not None:
        v = getattr(out, PROSODY_FEATURE_KEYS[0])
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Gaze
# ---------------------------------------------------------------------------

def _good_gaze():
    out = {k: 0.5 for k in GAZE_FEATURE_KEYS}
    out["label"] = "centre"
    out["confidence"] = 0.8
    out["label_probs"] = {"centre": 0.8, "left": 0.1, "right": 0.1}
    out["presence"] = True
    out["samples_count"] = 30
    out["captured_seconds"] = 5.0
    return out


def test_gaze_accepts_well_formed_payload():
    out = validate_gaze_payload(_good_gaze())
    assert out is not None
    assert isinstance(out, dict)


def test_gaze_rejects_non_dict():
    for bad in (None, [], "hi", 42):
        assert validate_gaze_payload(bad) is None


def test_gaze_handles_missing_label_probs():
    p = _good_gaze()
    del p["label_probs"]
    # Should not crash; either returns None or a dict with sensible
    # default for label_probs.
    out = validate_gaze_payload(p)
    assert out is None or isinstance(out, dict)


def test_gaze_clamps_out_of_range_confidence():
    p = _good_gaze()
    p["confidence"] = 5.0
    out = validate_gaze_payload(p)
    if out is not None:
        c = out.get("confidence")
        if c is not None:
            assert 0.0 <= c <= 1.0
