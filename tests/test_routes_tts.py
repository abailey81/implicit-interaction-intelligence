"""Tests for :mod:`server.routes_tts`.

Covers the happy path, body-size + text-length caps, the backend
enumeration endpoint, the archetype preview endpoint, and the PII
redaction guarantee (verified by intercepting the engine call).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from i3.tts import TTSOutput, TTSParams
from server import routes_tts
from server.routes_tts import (
    MAX_BODY_BYTES,
    MAX_TEXT_CHARS,
    include_tts_routes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Stand-in engine that records every ``speak`` call for assertions.

    Always returns a Web-Speech-style directive output (no WAV) so the
    test suite does not depend on any heavy backend being installed.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def available_backends(self) -> list[Any]:  # pragma: no cover - not used
        return []

    def speak(
        self,
        text: str,
        params: TTSParams,
        backend_hint: str | None = None,
    ) -> TTSOutput:
        self.calls.append(
            {"text": text, "params": params, "backend_hint": backend_hint}
        )
        return TTSOutput(
            audio_wav_base64=None,
            directive={
                "kind": "speech_synthesis_utterance",
                "text": text,
                "rate": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
            },
            sample_rate_hz=0,
            duration_ms=1000,
            backend_name="test_engine",
            params_used=params,
        )


@pytest.fixture()
def recording_engine(monkeypatch: pytest.MonkeyPatch) -> _RecordingEngine:
    """Swap the module-level engine for a recording stub."""
    engine = _RecordingEngine()
    monkeypatch.setattr(routes_tts, "_ENGINE", engine)
    return engine


@pytest.fixture()
def tts_app(recording_engine: _RecordingEngine) -> TestClient:
    """Minimal FastAPI app with only the TTS router + stub engine."""
    app = FastAPI()
    include_tts_routes(app)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_directive(
    tts_app: TestClient, recording_engine: _RecordingEngine
) -> None:
    """A valid request yields a 200 with the directive payload."""
    resp = tts_app.post(
        "/api/tts",
        json={
            "user_id": "demo_user",
            "text": "Hello there, this is a test.",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["backend_name"] == "test_engine"
    assert data["directive"]["text"].endswith("test.")
    assert data["audio_wav_base64"] is None
    assert data["sample_rate_hz"] == 0
    assert "adaptation_applied" in data
    assert "cognitive_load" in data["adaptation_applied"]
    assert data["explanation"]
    # Engine was called exactly once with the sanitised text.
    assert len(recording_engine.calls) == 1


def test_override_adaptation_changes_rate(
    tts_app: TestClient, recording_engine: _RecordingEngine
) -> None:
    """Raising cognitive_load via override slows the rate."""
    resp = tts_app.post(
        "/api/tts",
        json={
            "user_id": "demo_user",
            "text": "This should be slow.",
            "override_adaptation": {"cognitive_load": 1.0},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["adaptation_applied"]["cognitive_load"] == 1.0
    assert data["params_used"]["rate_wpm"] <= 120


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_oversized_body_rejected_with_413(tts_app: TestClient) -> None:
    """Bodies above MAX_BODY_BYTES return 413."""
    big_text = "a" * (MAX_BODY_BYTES + 100)
    payload = json.dumps({"user_id": "demo_user", "text": big_text})
    resp = tts_app.post(
        "/api/tts",
        content=payload.encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413


def test_text_over_2000_chars_rejected_with_422(tts_app: TestClient) -> None:
    """Text fields longer than MAX_TEXT_CHARS yield 422 (even if body fits)."""
    # Stay under the body cap but over the text cap.
    text = "b" * (MAX_TEXT_CHARS + 1)
    resp = tts_app.post(
        "/api/tts",
        json={"user_id": "demo_user", "text": text},
    )
    assert resp.status_code == 422


def test_empty_text_rejected(tts_app: TestClient) -> None:
    """All-whitespace text is refused at validation."""
    resp = tts_app.post(
        "/api/tts", json={"user_id": "demo_user", "text": "   "}
    )
    assert resp.status_code == 422


def test_invalid_user_id_rejected(tts_app: TestClient) -> None:
    """Path-traversal-style user ids fail the regex."""
    resp = tts_app.post(
        "/api/tts",
        json={"user_id": "../../etc/passwd", "text": "Hello"},
    )
    assert resp.status_code == 422


def test_unknown_backend_hint_rejected(tts_app: TestClient) -> None:
    """A non-whitelisted backend_hint yields 422."""
    resp = tts_app.post(
        "/api/tts",
        json={
            "user_id": "demo_user",
            "text": "Hello",
            "backend_hint": "festival",
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_pii_is_scrubbed_before_engine(
    tts_app: TestClient, recording_engine: _RecordingEngine
) -> None:
    """Email / phone in the input must never reach the engine raw."""
    resp = tts_app.post(
        "/api/tts",
        json={
            "user_id": "demo_user",
            "text": "Please email me at secret@example.com or call +1 555 123 4567.",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["pii_redactions"] >= 2

    # The engine received the sanitised text only.
    assert len(recording_engine.calls) == 1
    seen = recording_engine.calls[0]["text"]
    assert "secret@example.com" not in seen
    assert "[EMAIL]" in seen


# ---------------------------------------------------------------------------
# Backend enumeration
# ---------------------------------------------------------------------------


def test_backends_endpoint_shape(tts_app: TestClient) -> None:
    """GET /api/tts/backends returns a list of backends with the right keys."""
    resp = tts_app.get("/api/tts/backends")
    assert resp.status_code == 200
    data = resp.json()
    assert "backends" in data
    names = {b["name"] for b in data["backends"]}
    assert {"pyttsx3", "piper", "kokoro", "web_speech_api"} <= names
    for b in data["backends"]:
        assert set(b.keys()) == {
            "name",
            "display_name",
            "available",
            "install_hint",
        }
        assert isinstance(b["available"], bool)
    # Web Speech API is always available by construction.
    web = next(b for b in data["backends"] if b["name"] == "web_speech_api")
    assert web["available"] is True


# ---------------------------------------------------------------------------
# Archetype preview
# ---------------------------------------------------------------------------


def test_preview_accessibility_high_caps_rate(tts_app: TestClient) -> None:
    """preview?archetype=accessibility_high yields rate_wpm <= 120."""
    resp = tts_app.get("/api/tts/preview", params={"archetype": "accessibility_high"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["params_used"]["rate_wpm"] <= 120
    assert data["params_used"]["enunciation"] == "maximum"


def test_preview_unknown_archetype_404(tts_app: TestClient) -> None:
    """An unknown archetype name returns 404."""
    resp = tts_app.get("/api/tts/preview", params={"archetype": "nonesuch"})
    assert resp.status_code == 404


def test_preview_malformed_archetype_422(tts_app: TestClient) -> None:
    """An archetype name with invalid characters yields 422."""
    resp = tts_app.get("/api/tts/preview", params={"archetype": "Bad-Name"})
    assert resp.status_code == 422
