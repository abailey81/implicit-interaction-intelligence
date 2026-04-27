"""Unit tests for the iter-51 Qwen-intent cascade arm.

These tests exercise the *cheap* parts of the intent cascade — the
``Pipeline._looks_like_command`` regex gate and the
``_maybe_handle_intent_command`` dispatch path with a stubbed parser
— so they run in <0.1 s on CI without loading Qwen3-1.7B.

The full Qwen LoRA inference is exercised separately by
``training/eval_intent.py``.
"""
from __future__ import annotations

import pytest

from i3.intent.gemini_inference import (
    _coerce_duration_seconds,
    _normalize_slots,
)
from i3.pipeline.engine import Pipeline


def _bare_pipeline() -> Pipeline:
    """Construct a Pipeline shell suitable for testing helper methods.

    We bypass ``__init__`` because the full constructor loads the SLM,
    encoder, retrieval index, etc. and would dominate test runtime.
    The helpers we test only need a few attributes set.
    """
    p = Pipeline.__new__(Pipeline)
    p._stated_facts = {}
    p._intent_parser_qwen = None
    return p


# ---------------------------------------------------------------------------
# 1. Cheap regex gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "set timer for 5 minutes",
    "set a 30 second timer",
    # Phase 5: polite / OOD phrasings the gate should now accept.
    "start a timer",
    "start a five minute timer",
    "could you start a five minute timer please",
    "start an alarm for 7am",
    "play jazz",
    "play taylor swift",
    "skip this song",
    "next track",
    "previous track",
    "pause",
    "stop the music",
    "turn the volume up",
    "volume down",
    "set volume to 50 percent",
    "what's the weather like",
    "set an alarm for 7am",
    "remind me to call mum at 4pm",
    "send a message to alex",
    "text bob hello",
    "call dad",
    "video call sam",
    "open spotify",
    "open netflix",
    "navigate home",
    "directions to the airport",
    "turn on the lights",
    "turn off the bedroom lamp",
    "set the thermostat to 21",
    "lock the doors",
    "unlock the front door",
])
def test_command_detected(msg):
    p = _bare_pipeline()
    assert p._looks_like_command(msg), f"should detect command in {msg!r}"


@pytest.mark.parametrize("msg", [
    "what is photosynthesis",
    "hello there",
    "how are you doing today",
    "tell me about transformers",
    "thanks",
    "I'm feeling tired today",
    "explain entropy",
    "who was alan turing",
    "summarise our conversation",
    "what's my name",
])
def test_chat_not_detected_as_command(msg):
    p = _bare_pipeline()
    assert not p._looks_like_command(msg), f"chat misread as command: {msg!r}"


def test_empty_and_overlong_rejected():
    p = _bare_pipeline()
    assert not p._looks_like_command("")
    assert not p._looks_like_command("   ")
    assert not p._looks_like_command("x" * 250)


# ---------------------------------------------------------------------------
# 2. Cascade dispatch with a stubbed parser
# ---------------------------------------------------------------------------

class _StubIntentResult:
    """Minimal IntentResult stand-in matching the contract the helper expects."""

    def __init__(self, *, action="set_timer", params=None,
                 valid_action=True, valid_slots=True, confidence=0.95):
        self.action = action
        self.params = params or {}
        self.valid_action = valid_action
        self.valid_slots = valid_slots
        self.confidence = confidence

    def to_dict(self):
        return {
            "action": self.action,
            "params": self.params,
            "valid_action": self.valid_action,
            "valid_slots": self.valid_slots,
            "confidence": self.confidence,
        }


class _StubParser:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def parse(self, text):
        self.calls += 1
        return self._result


def test_cascade_short_circuits_on_valid_action():
    p = _bare_pipeline()
    p._intent_parser_qwen = _StubParser(
        _StubIntentResult(action="set_timer",
                          params={"duration_seconds": 300}),
    )
    out = p._maybe_handle_intent_command(
        message="set timer for 5 minutes",
        user_id="u", session_id="s",
    )
    assert out is not None
    assert "5 min" in out.lower()
    assert p._last_intent_result is not None
    assert p._last_intent_result["action"] == "set_timer"


def test_cascade_falls_through_on_invalid_action(monkeypatch):
    # Phase 4 added a Gemini backup that fires when GEMINI_API_KEY is
    # set; this test pins the *no-backup* path, so unset the key.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = _bare_pipeline()
    p._intent_parser_qwen = _StubParser(
        _StubIntentResult(valid_action=False, valid_slots=False),
    )
    out = p._maybe_handle_intent_command(
        message="set timer for 5 minutes",
        user_id="u", session_id="s",
    )
    assert out is None
    # The IntentResult is still stashed (so the dashboard can show
    # "we tried, here's what came back").
    assert p._last_intent_result is not None


def test_cascade_skipped_on_chat_text():
    p = _bare_pipeline()
    parser = _StubParser(_StubIntentResult())
    p._intent_parser_qwen = parser
    out = p._maybe_handle_intent_command(
        message="what is photosynthesis",
        user_id="u", session_id="s",
    )
    assert out is None
    # The cheap regex should have prevented even calling parse().
    assert parser.calls == 0


@pytest.mark.parametrize("action,params,must_contain", [
    ("set_timer", {"duration_seconds": 300}, "5 min"),
    ("play_music", {"genre": "jazz"}, "jazz"),
    ("send_message", {"recipient": "Alex"}, "alex"),
    ("navigate", {"destination": "home"}, "home"),
    ("set_alarm", {"time": "07:00"}, "07:00"),
    ("set_volume", {"level": 50}, "50"),
    ("control_device", {"device": "lights", "verb": "on"}, "on"),
    ("weather", {"location": "London"}, "london"),
    ("remind", {"task": "call mum"}, "call mum"),
    ("cancel", {}, "cancel"),
    ("unsupported", {}, "can't"),
])
def test_cascade_acks(action, params, must_contain, monkeypatch):
    # The 'unsupported' case relies on Qwen's stub ack short-circuiting
    # to "I can't action that one." — but Phase 4 now consults Gemini
    # as a backup whenever the primary parse is unsupported.  Unset
    # GEMINI_API_KEY so we exercise the no-backup ack path.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = _bare_pipeline()
    p._intent_parser_qwen = _StubParser(
        _StubIntentResult(action=action, params=params),
    )
    # Pick a message that triggers the gate so the dispatch runs.
    msg = {
        "set_timer": "set timer for 5 minutes",
        "play_music": "play jazz",
        "send_message": "send a message to Alex",
        "navigate": "navigate home",
        "set_alarm": "set an alarm for 7am",
        "set_volume": "set volume to 50",
        "control_device": "turn on the lights",
        "weather": "what's the weather in London",
        "remind": "remind me to call mum",
        "cancel": "cancel the timer",
        "unsupported": "set timer for 5 minutes",
    }[action]
    out = p._maybe_handle_intent_command(
        message=msg, user_id="u", session_id="s",
    )
    assert out is not None
    assert must_contain.lower() in out.lower()


# ---------------------------------------------------------------------------
# 3. Gemini slot normalisation (Phase 5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("5 minutes", 300),
    ("30 seconds", 30),
    ("1 hour", 3600),
    ("1 hour 30 minutes", 5400),
    ("2 hrs", 7200),
    (300, 300),
    ("300", 300),
    ("0", None),
    ("", None),
    (None, None),
    (True, None),
])
def test_coerce_duration_seconds(raw, expected):
    assert _coerce_duration_seconds(raw) == expected


@pytest.mark.parametrize("action,raw,expected", [
    ("set_timer", {"duration": "5 minutes"}, {"duration_seconds": 300}),
    ("set_timer", {"duration_seconds": 300}, {"duration_seconds": 300}),
    ("set_timer", {"duration": "garbage"}, {}),
    ("navigate", {"destination": "Trafalgar Square"},
        {"location": "Trafalgar Square"}),
    ("weather_query", {"city": "London"}, {"location": "London"}),
    ("send_message", {"to": "Mum", "text": "on my way"},
        {"recipient": "Mum", "message": "on my way"}),
    ("call", {"name": "Dad", "video_call": "yes"},
        {"recipient": "Dad", "video": True}),
    ("call", {"recipient": "Dad", "video": False},
        {"recipient": "Dad", "video": False}),
    ("control_device", {"device_name": "lights", "on_off": "on"},
        {"device": "lights", "state": "on"}),
    ("set_reminder", {"what": "call mum", "at": "6pm"},
        {"task": "call mum", "time": "6pm"}),
    ("play_music", {"singer": "Adele"}, {"artist": "Adele"}),
    ("cancel", {}, {}),
])
def test_normalize_slots(action, raw, expected):
    assert _normalize_slots(action, raw) == expected
