"""Iter 95 — PipelineInput dataclass contract tests."""
from __future__ import annotations

import pytest

from i3.pipeline.types import PipelineInput


def test_minimal_construction():
    inp = PipelineInput(
        user_id="alice",
        session_id="s1",
        message_text="hello",
        timestamp=1000.0,
        composition_time_ms=1500.0,
        edit_count=0,
        pause_before_send_ms=200.0,
    )
    assert inp.user_id == "alice"
    assert inp.message_text == "hello"


def test_keystroke_timings_default_empty():
    inp = PipelineInput(
        user_id="u", session_id="s", message_text="hi",
        timestamp=0.0, composition_time_ms=0.0,
        edit_count=0, pause_before_send_ms=0.0,
    )
    assert inp.keystroke_timings == []


def test_optional_multimodal_defaults_none():
    inp = PipelineInput(
        user_id="u", session_id="s", message_text="hi",
        timestamp=0.0, composition_time_ms=0.0,
        edit_count=0, pause_before_send_ms=0.0,
    )
    assert inp.prosody_features is None
    assert inp.gaze_features is None
    assert inp.playground_overrides is None


def test_keystroke_timings_passed_through():
    inp = PipelineInput(
        user_id="u", session_id="s", message_text="hi",
        timestamp=0.0, composition_time_ms=0.0,
        edit_count=0, pause_before_send_ms=0.0,
        keystroke_timings=[120.0, 80.0, 95.0],
    )
    assert inp.keystroke_timings == [120.0, 80.0, 95.0]


def test_prosody_features_passed_through():
    inp = PipelineInput(
        user_id="u", session_id="s", message_text="hi",
        timestamp=0.0, composition_time_ms=0.0,
        edit_count=0, pause_before_send_ms=0.0,
        prosody_features={"speech_rate_wpm_norm": 0.5, "samples_count": 100},
    )
    assert inp.prosody_features["speech_rate_wpm_norm"] == 0.5


def test_each_instance_has_its_own_keystroke_list():
    """Default factory must not share state across instances."""
    a = PipelineInput(
        user_id="u", session_id="s", message_text="hi",
        timestamp=0.0, composition_time_ms=0.0,
        edit_count=0, pause_before_send_ms=0.0,
    )
    b = PipelineInput(
        user_id="u", session_id="s", message_text="hi",
        timestamp=0.0, composition_time_ms=0.0,
        edit_count=0, pause_before_send_ms=0.0,
    )
    a.keystroke_timings.append(100.0)
    assert b.keystroke_timings == []
