"""Iter 135 — BiometricMatch dataclass + WS-shape tests."""
from __future__ import annotations

import json

import pytest

from i3.biometric.keystroke_auth import BiometricMatch


def _ok():
    return BiometricMatch(
        state="registered",
        similarity=0.9,
        confidence=0.85,
        threshold=0.65,
        enrolment_progress=10,
        enrolment_target=10,
        is_owner=True,
        drift_alert=False,
    )


def test_minimal_construction():
    m = _ok()
    assert m.state == "registered"
    assert m.is_owner is True
    assert m.diverged_signals == []
    assert m.notes == ""


def test_to_dict_keys():
    m = _ok()
    d = m.to_dict()
    expected = {"state", "similarity", "confidence", "threshold",
                "enrolment_progress", "enrolment_target",
                "is_owner", "drift_alert", "diverged_signals",
                "ewma_iki_mean", "ewma_iki_std",
                "ewma_composition_ms", "ewma_edit_rate", "notes"}
    assert set(d.keys()) == expected


def test_to_dict_types_json_safe():
    m = _ok()
    d = m.to_dict()
    parsed = json.loads(json.dumps(d))
    assert parsed["state"] == "registered"
    assert parsed["is_owner"] is True


def test_diverged_signals_factory_isolation():
    a = _ok()
    b = _ok()
    a.diverged_signals.append("iki_mean_drift")
    assert b.diverged_signals == []


@pytest.mark.parametrize("state", [
    "unregistered", "registering", "registered",
    "verifying", "mismatch",
])
def test_documented_states_accepted(state):
    m = BiometricMatch(
        state=state, similarity=0.0, confidence=0.0, threshold=0.65,
        enrolment_progress=0, enrolment_target=10,
        is_owner=False, drift_alert=False,
    )
    assert m.state == state
