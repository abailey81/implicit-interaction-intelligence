"""Iter 134 — AccessibilityModeState dataclass + WS contract."""
from __future__ import annotations

import json

import pytest

from i3.affect.accessibility_mode import AccessibilityModeState


def test_default_inactive_state():
    s = AccessibilityModeState(active=False)
    assert s.active is False
    assert s.activated_this_turn is False
    assert s.deactivated_this_turn is False
    assert s.confidence == 0.0
    assert s.sentence_cap == 3
    assert s.simplify_vocab is False
    assert s.tts_rate_multiplier == 1.0
    assert s.font_scale == 1.0


def test_active_state_with_overrides():
    s = AccessibilityModeState(
        active=True,
        activated_this_turn=True,
        confidence=0.9,
        reason="elevated edit-rate over 3 turns",
        sentence_cap=1,
        simplify_vocab=True,
        tts_rate_multiplier=0.6,
        font_scale=1.25,
    )
    assert s.active is True
    assert s.sentence_cap == 1


def test_to_dict_keys():
    s = AccessibilityModeState(active=False)
    d = s.to_dict()
    expected = {"active", "activated_this_turn", "deactivated_this_turn",
                "confidence", "reason", "sentence_cap",
                "simplify_vocab", "tts_rate_multiplier", "font_scale"}
    assert set(d.keys()) == expected


def test_to_dict_types_json_safe():
    s = AccessibilityModeState(
        active=True, confidence=0.7, reason="test",
        sentence_cap=2, simplify_vocab=True,
        tts_rate_multiplier=0.6, font_scale=1.25,
    )
    d = s.to_dict()
    s_json = json.dumps(d)
    parsed = json.loads(s_json)
    assert parsed["active"] is True
    assert parsed["sentence_cap"] == 2


def test_inactive_defaults_match_normal_ui():
    """When inactive, TTS / font / sentence cap should be the
    "normal mode" defaults (no UI shift)."""
    s = AccessibilityModeState(active=False)
    assert s.tts_rate_multiplier == 1.0
    assert s.font_scale == 1.0
    assert s.sentence_cap == 3
