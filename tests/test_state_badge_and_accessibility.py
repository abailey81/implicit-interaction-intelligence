"""Tests for the Live State Badge + Accessibility Mode showpiece.

Two coupled features that share per-session sliding-window
infrastructure (see ``i3/affect/state_classifier.py`` and
``i3/affect/accessibility_mode.py``).  These tests cover:

* The 6-state classifier at its characteristic operating points.
* The accessibility-mode auto-activation cycle (calm → elevated
  for 3 turns → activated → calm for 4 turns → deactivated).
* The manual-toggle path used by the REST endpoint.
* Determinism — same inputs always produce the same outputs.
* Reasoning-trace surfacing of both signals.
"""

from __future__ import annotations

from i3.affect.accessibility_mode import (
    AccessibilityController,
    AccessibilityModeState,
)
from i3.affect.state_classifier import (
    UserStateLabel,
    classify_user_state,
)
from i3.explain.reasoning_trace import build_reasoning_trace


# ---------------------------------------------------------------------------
# State classifier
# ---------------------------------------------------------------------------


def _classify(**overrides) -> UserStateLabel:
    """Helper — start from a calm baseline and override individual signals."""
    base = dict(
        adaptation={
            "cognitive_load": 0.3,
            "formality": 0.5,
            "accessibility": 0.0,
            "verbosity": 0.5,
        },
        composition_time_ms=1200.0,
        edit_count=0,
        iki_mean=100.0,
        iki_std=10.0,
        engagement_score=0.6,
        deviation_from_baseline=0.0,
        baseline_established=True,
        messages_in_session=5,
    )
    base.update(overrides)
    return classify_user_state(**base)


def test_state_classifier_warming_up_first_message() -> None:
    label = _classify(baseline_established=False, messages_in_session=1)
    assert label.state == "warming up"
    assert label.confidence > 0.5


def test_state_classifier_calm_baseline() -> None:
    label = _classify()
    assert label.state == "calm"
    assert label.confidence > 0.6


def test_state_classifier_stressed() -> None:
    label = _classify(
        adaptation={
            "cognitive_load": 0.85,
            "formality": 0.5,
            "accessibility": 0.5,
            "verbosity": 0.5,
        },
        composition_time_ms=4500.0,
        edit_count=4,
        iki_mean=180.0,
        iki_std=50.0,
        engagement_score=0.5,
        deviation_from_baseline=0.3,
    )
    assert label.state == "stressed"
    assert "elevated edit count" in label.contributing_signals
    assert "high cognitive load" in label.contributing_signals


def test_state_classifier_tired() -> None:
    label = _classify(
        adaptation={
            "cognitive_load": 0.55,
            "formality": 0.5,
            "accessibility": 0.0,
            "verbosity": 0.5,
        },
        composition_time_ms=6000.0,
        edit_count=1,
        iki_mean=220.0,
        iki_std=30.0,
        engagement_score=0.25,
    )
    assert label.state == "tired"


def test_state_classifier_distracted() -> None:
    label = _classify(
        adaptation={
            "cognitive_load": 0.5,
            "formality": 0.5,
            "accessibility": 0.0,
            "verbosity": 0.5,
        },
        composition_time_ms=3000.0,
        edit_count=2,
        iki_mean=130.0,
        iki_std=70.0,
        engagement_score=0.5,
    )
    assert label.state == "distracted"


def test_state_classifier_focused() -> None:
    label = _classify(
        adaptation={
            "cognitive_load": 0.5,
            "formality": 0.7,
            "accessibility": 0.0,
            "verbosity": 0.5,
        },
        composition_time_ms=2500.0,
        edit_count=0,
        iki_mean=110.0,
        iki_std=15.0,
        engagement_score=0.7,
    )
    assert label.state == "focused"


def test_state_classifier_deterministic() -> None:
    """Same inputs produce the same label, confidence, signals."""
    a = _classify()
    b = _classify()
    assert a.state == b.state
    assert a.confidence == b.confidence
    assert a.contributing_signals == b.contributing_signals


def test_state_classifier_to_dict_shape() -> None:
    label = _classify()
    d = label.to_dict()
    assert set(d.keys()) == {
        "state",
        "confidence",
        "secondary_state",
        "contributing_signals",
    }
    assert isinstance(d["confidence"], float)
    assert isinstance(d["contributing_signals"], list)


def test_state_classifier_handles_missing_inputs() -> None:
    """Defensive coercion: NaN / non-finite values fall through gracefully."""
    label = classify_user_state(
        adaptation={},
        composition_time_ms=float("nan"),
        edit_count=-1,
        iki_mean=float("inf"),
        iki_std=0.0,
        engagement_score=0.0,
        deviation_from_baseline=0.0,
        baseline_established=False,
        messages_in_session=0,
    )
    assert label.state in {
        "calm", "focused", "stressed", "tired", "distracted", "warming up"
    }


# ---------------------------------------------------------------------------
# Accessibility controller
# ---------------------------------------------------------------------------


def _calm_obs(c: AccessibilityController, sid: str = "s1") -> AccessibilityModeState:
    return c.observe(
        "u1",
        sid,
        edit_count=0,
        iki_mean=100.0,
        iki_std=10.0,
        cognitive_load=0.3,
        accessibility_axis=0.0,
    )


def _elevated_obs(c: AccessibilityController, sid: str = "s1") -> AccessibilityModeState:
    return c.observe(
        "u1",
        sid,
        edit_count=4,
        iki_mean=180.0,
        iki_std=50.0,
        cognitive_load=0.7,
        accessibility_axis=0.5,
    )


def test_accessibility_starts_inactive() -> None:
    c = AccessibilityController()
    state = _calm_obs(c)
    assert state.active is False
    assert state.font_scale == 1.0
    assert state.tts_rate_multiplier == 1.0


def test_accessibility_activates_on_sustained_elevation() -> None:
    c = AccessibilityController()
    # Establish baseline
    for _ in range(5):
        _calm_obs(c)
    # 3 elevated turns → trigger
    final = None
    for _ in range(3):
        final = _elevated_obs(c)
    assert final is not None
    assert final.active is True
    assert "sustained" in final.reason
    assert final.font_scale == 1.25
    assert final.tts_rate_multiplier == 0.6


def test_accessibility_deactivates_on_recovery_window() -> None:
    c = AccessibilityController()
    for _ in range(5):
        _calm_obs(c)
    for _ in range(3):
        _elevated_obs(c)
    # 4 calm turns → deactivate
    final = None
    for _ in range(4):
        final = _calm_obs(c)
    assert final is not None
    assert final.active is False
    assert final.deactivated_this_turn is True


def test_accessibility_sticky_does_not_flap_on_one_calm_turn() -> None:
    c = AccessibilityController()
    for _ in range(5):
        _calm_obs(c)
    for _ in range(3):
        _elevated_obs(c)
    # One calm turn — must remain active.
    state = _calm_obs(c)
    assert state.active is True


def test_accessibility_force_activates_immediately() -> None:
    c = AccessibilityController()
    state = c.force("u1", "s1", force=True)
    assert state.active is True
    assert state.activated_this_turn is True
    assert state.confidence == 1.0


def test_accessibility_force_clears_override() -> None:
    c = AccessibilityController()
    c.force("u1", "s1", force=True)
    state = c.force("u1", "s1", force=None)
    # No rolling window evidence yet → falls back to inactive.
    assert state.active is False


def test_accessibility_lru_eviction() -> None:
    """LRU cap prevents unbounded growth."""
    c = AccessibilityController(max_sessions=3)
    for i in range(5):
        c.observe(
            f"u{i}", f"s{i}",
            edit_count=0, iki_mean=100, iki_std=10,
        )
    # Only the 3 most recent sessions are retained.
    assert len(c._sessions) == 3


def test_accessibility_to_dict_shape() -> None:
    c = AccessibilityController()
    state = c.force("u1", "s1", force=True)
    d = state.to_dict()
    assert d["active"] is True
    assert d["sentence_cap"] in {1, 2, 3}
    assert d["font_scale"] == 1.25
    assert d["tts_rate_multiplier"] == 0.6


# ---------------------------------------------------------------------------
# Reasoning-trace integration
# ---------------------------------------------------------------------------


def test_reasoning_trace_surfaces_state_label() -> None:
    trace = build_reasoning_trace(
        keystroke_metrics={
            "composition_time_ms": 4500,
            "edit_count": 4,
            "pause_before_send_ms": 200,
            "keystroke_timings": [180.0] * 20,
        },
        adaptation={"cognitive_load": 0.85, "verbosity": 0.3, "formality": 0.5},
        adaptation_changes=[],
        engagement_score=0.5,
        deviation_from_baseline=0.3,
        messages_in_session=6,
        baseline_established=True,
        routing_confidence={"local_slm": 1.0, "cloud_llm": 0.0},
        response_path="retrieval",
        retrieval_score=1.0,
        user_message_preview="hi",
        response_preview="ok",
        user_state_label={
            "state": "stressed",
            "confidence": 0.83,
            "secondary_state": None,
            "contributing_signals": [
                "high cognitive load",
                "elevated edit count",
                "IKI std +35% vs baseline",
            ],
        },
    )
    paragraphs = trace["narrative_paragraphs"]
    para2 = paragraphs[1]
    assert "stressed" in para2
    assert "0.83" in para2
    assert "high cognitive load" in para2


def test_reasoning_trace_surfaces_accessibility_when_active() -> None:
    trace = build_reasoning_trace(
        keystroke_metrics={
            "composition_time_ms": 4500,
            "edit_count": 4,
            "pause_before_send_ms": 200,
            "keystroke_timings": [180.0] * 20,
        },
        adaptation={"cognitive_load": 0.85, "verbosity": 0.2, "accessibility": 0.95},
        adaptation_changes=[],
        engagement_score=0.5,
        deviation_from_baseline=0.3,
        messages_in_session=6,
        baseline_established=True,
        routing_confidence={"local_slm": 1.0, "cloud_llm": 0.0},
        response_path="retrieval",
        retrieval_score=1.0,
        user_message_preview="hi",
        response_preview="ok",
        accessibility={
            "active": True,
            "activated_this_turn": True,
            "deactivated_this_turn": False,
            "confidence": 0.85,
            "reason": "elevated edit-rate sustained over 3 turns",
            "sentence_cap": 1,
            "simplify_vocab": True,
            "tts_rate_multiplier": 0.6,
            "font_scale": 1.25,
        },
    )
    para3 = trace["narrative_paragraphs"][2]
    assert "Accessibility mode" in para3
    assert "force-overridden" in para3
    assert "sustained over 3 turns" in para3


def test_toggle_endpoint_round_trip() -> None:
    """The REST toggle endpoint accepts force=true|false|null and returns
    the post-toggle state in the same dict shape the WS layer uses."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.routes_accessibility import include_accessibility_routes

    class _Pipe:
        def force_accessibility_mode(self, *, user_id, session_id, force):
            return {
                "active": bool(force) if force is not None else False,
                "activated_this_turn": bool(force) if force else False,
                "deactivated_this_turn": (
                    not bool(force) if force is not None else False
                ),
                "confidence": 1.0,
                "reason": f"manual override (force={force})",
                "sentence_cap": 1 if force else 3,
                "simplify_vocab": bool(force),
                "tts_rate_multiplier": 0.6 if force else 1.0,
                "font_scale": 1.25 if force else 1.0,
            }

    app = FastAPI()
    app.state.pipeline = _Pipe()
    include_accessibility_routes(app)
    client = TestClient(app)

    r = client.post(
        "/api/accessibility/u1/toggle",
        json={"session_id": "s1", "force": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is True
    assert body["font_scale"] == 1.25

    r = client.post(
        "/api/accessibility/u1/toggle",
        json={"session_id": "s1", "force": False},
    )
    assert r.status_code == 200
    assert r.json()["active"] is False

    # Invalid user_id (path regex on the route — FastAPI returns 404).
    r = client.post(
        "/api/accessibility/!@#bad/toggle",
        json={"session_id": "s1", "force": True},
    )
    assert r.status_code in (404, 422)


def test_toggle_endpoint_503_when_pipeline_missing() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server.routes_accessibility import include_accessibility_routes

    app = FastAPI()
    include_accessibility_routes(app)
    client = TestClient(app)
    r = client.post(
        "/api/accessibility/u1/toggle",
        json={"session_id": "s1", "force": True},
    )
    assert r.status_code == 503


def test_reasoning_trace_omits_accessibility_when_inactive() -> None:
    trace = build_reasoning_trace(
        keystroke_metrics={"composition_time_ms": 1000, "edit_count": 0,
                          "pause_before_send_ms": 100, "keystroke_timings": []},
        adaptation={"cognitive_load": 0.3, "verbosity": 0.5, "formality": 0.5},
        adaptation_changes=[],
        engagement_score=0.5,
        deviation_from_baseline=0.0,
        messages_in_session=5,
        baseline_established=True,
        routing_confidence={"local_slm": 1.0, "cloud_llm": 0.0},
        response_path="retrieval",
        retrieval_score=1.0,
        user_message_preview="hi",
        response_preview="ok",
        accessibility={
            "active": False,
            "activated_this_turn": False,
            "deactivated_this_turn": False,
            "confidence": 0.0,
            "reason": "",
            "sentence_cap": 3,
            "simplify_vocab": False,
            "tts_rate_multiplier": 1.0,
            "font_scale": 1.0,
        },
    )
    para3 = trace["narrative_paragraphs"][2]
    assert "Accessibility mode" not in para3
