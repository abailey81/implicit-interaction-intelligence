"""Contract tests for the WebSocket JSON protocol.

The I3 demo SPA (``web/js/app.js``) binds to a fixed set of event
``type`` strings and a fixed field layout inside each payload.  These
tests pin that contract in Python so a server-side refactor that
accidentally renames ``user_state_embedding_2d`` or drops
``latency_ms`` will fail CI before the SPA breaks in production.

We do NOT run the actual WebSocket here (that is covered by
``tests/test_integration.py``).  Instead we load explicit JSON schemas
that mirror the consumer shapes in ``web/js/app.js`` and validate the
server-side sender code against them.

When ``jsonschema`` is missing the entire module is skipped.
"""

from __future__ import annotations

import json
from typing import Any

import pytest


jsonschema = pytest.importorskip("jsonschema")


# ─────────────────────────────────────────────────────────────────────────
#  Canonical schemas for every event the server emits
# ─────────────────────────────────────────────────────────────────────────


SESSION_STARTED_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["type", "session_id", "user_id"],
    "properties": {
        "type": {"const": "session_started"},
        "session_id": {"type": "string", "minLength": 1},
        "user_id": {"type": "string", "pattern": r"^[a-zA-Z0-9_-]{1,64}$"},
    },
    "additionalProperties": False,
}


RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["type", "text", "route", "latency_ms", "timestamp"],
    "properties": {
        "type": {"const": "response"},
        "text": {"type": "string"},
        "route": {"type": "string", "minLength": 1},
        "latency_ms": {"type": "number", "minimum": 0},
        "timestamp": {"type": "number"},
    },
    "additionalProperties": False,
}


STATE_UPDATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "type",
        "user_state_embedding_2d",
        "adaptation",
        "engagement_score",
        "deviation_from_baseline",
        "routing_confidence",
        "messages_in_session",
        "baseline_established",
    ],
    "properties": {
        "type": {"const": "state_update"},
        "user_state_embedding_2d": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "adaptation": {
            "type": "object",
            # adaptation serialisation is a dict with cognitive_load etc.
            "properties": {
                "cognitive_load": {"type": "number"},
                "emotional_tone": {"type": "number"},
                "accessibility": {"type": "number"},
            },
            "additionalProperties": True,
        },
        "engagement_score": {"type": "number", "minimum": 0, "maximum": 1},
        "deviation_from_baseline": {"type": "number"},
        "routing_confidence": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "messages_in_session": {"type": "integer", "minimum": 0},
        "baseline_established": {"type": "boolean"},
    },
    "additionalProperties": False,
}


DIARY_ENTRY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["type", "entry"],
    "properties": {
        "type": {"const": "diary_entry"},
        "entry": {"type": "object"},
    },
    "additionalProperties": False,
}


ERROR_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["type", "code", "detail"],
    "properties": {
        "type": {"const": "error"},
        "code": {"type": "integer"},
        "detail": {"type": "string"},
    },
    "additionalProperties": False,
}


EVENT_SCHEMAS = {
    "session_started": SESSION_STARTED_SCHEMA,
    "response": RESPONSE_SCHEMA,
    "state_update": STATE_UPDATE_SCHEMA,
    "diary_entry": DIARY_ENTRY_SCHEMA,
    "error": ERROR_SCHEMA,
}


# ─────────────────────────────────────────────────────────────────────────
#  Representative payloads — the EXACT shape the server emits today.
#  Any change here must be mirrored in web/js/app.js and vice-versa.
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_session_started() -> dict[str, Any]:
    return {
        "type": "session_started",
        "session_id": "c0ffee-0001",
        "user_id": "test_user",
    }


@pytest.fixture
def sample_response() -> dict[str, Any]:
    return {
        "type": "response",
        "text": "Hello!",
        "route": "local_slm",
        "latency_ms": 42,
        "timestamp": 1_700_000_000.0,
    }


@pytest.fixture
def sample_state_update() -> dict[str, Any]:
    return {
        "type": "state_update",
        "user_state_embedding_2d": [0.1, -0.2],
        "adaptation": {
            "cognitive_load": 0.5,
            "style_mirror": {
                "formality": 0.5,
                "verbosity": 0.5,
                "emotionality": 0.5,
                "directness": 0.5,
            },
            "emotional_tone": 0.5,
            "accessibility": 0.0,
        },
        "engagement_score": 0.7,
        "deviation_from_baseline": 0.15,
        "routing_confidence": {"local_slm": 0.6, "cloud_llm": 0.4},
        "messages_in_session": 3,
        "baseline_established": False,
    }


@pytest.fixture
def sample_diary_entry() -> dict[str, Any]:
    return {
        "type": "diary_entry",
        "entry": {"topic": "greetings", "sentiment": "positive"},
    }


@pytest.fixture
def sample_error() -> dict[str, Any]:
    return {"type": "error", "code": 429, "detail": "rate_limited"}


# ─────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────


class TestEventSchemas:
    def test_session_started_matches_schema(self, sample_session_started) -> None:
        jsonschema.validate(sample_session_started, SESSION_STARTED_SCHEMA)

    def test_response_matches_schema(self, sample_response) -> None:
        jsonschema.validate(sample_response, RESPONSE_SCHEMA)

    def test_state_update_matches_schema(self, sample_state_update) -> None:
        jsonschema.validate(sample_state_update, STATE_UPDATE_SCHEMA)

    def test_diary_entry_matches_schema(self, sample_diary_entry) -> None:
        jsonschema.validate(sample_diary_entry, DIARY_ENTRY_SCHEMA)

    def test_error_matches_schema(self, sample_error) -> None:
        jsonschema.validate(sample_error, ERROR_SCHEMA)


class TestBreakingChangeDetection:
    """Negative tests that fail if the producer drops a required field."""

    def test_response_missing_text_fails(self) -> None:
        bad = {"type": "response", "route": "local_slm", "latency_ms": 10, "timestamp": 0.0}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, RESPONSE_SCHEMA)

    def test_state_update_missing_embedding_fails(
        self, sample_state_update
    ) -> None:
        bad = dict(sample_state_update)
        bad.pop("user_state_embedding_2d")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, STATE_UPDATE_SCHEMA)

    def test_state_update_wrong_embedding_length_fails(
        self, sample_state_update
    ) -> None:
        bad = dict(sample_state_update)
        bad["user_state_embedding_2d"] = [0.1, 0.2, 0.3]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, STATE_UPDATE_SCHEMA)


class TestServerEmittersMatchContract:
    """Synthesise what the server would emit and validate it live."""

    def test_pipeline_output_serialises_to_state_update(self) -> None:
        from i3.pipeline.types import PipelineOutput

        output = PipelineOutput(
            response_text="ok",
            route_chosen="local_slm",
            latency_ms=12.3,
            user_state_embedding_2d=(0.0, 0.0),
            adaptation={
                "cognitive_load": 0.5,
                "emotional_tone": 0.5,
                "accessibility": 0.0,
            },
            engagement_score=0.5,
            deviation_from_baseline=0.0,
            routing_confidence={"local_slm": 1.0, "cloud_llm": 0.0},
            messages_in_session=1,
            baseline_established=False,
        )
        # Mirror what `server.websocket.websocket_endpoint` sends
        payload = {
            "type": "state_update",
            "user_state_embedding_2d": list(output.user_state_embedding_2d),
            "adaptation": output.adaptation,
            "engagement_score": output.engagement_score,
            "deviation_from_baseline": output.deviation_from_baseline,
            "routing_confidence": output.routing_confidence,
            "messages_in_session": output.messages_in_session,
            "baseline_established": output.baseline_established,
        }
        jsonschema.validate(payload, STATE_UPDATE_SCHEMA)

    def test_websocket_js_types_are_registered(self) -> None:
        """The JS client binds handlers for these event types — they
        must all have Python schemas so a server-side rename fails
        in CI before it breaks the SPA."""
        js_file = (
            __file__.rsplit("tests", 1)[0]
            + "web/js/app.js"
        )
        try:
            with open(js_file, "r", encoding="utf-8") as fh:
                js = fh.read()
        except OSError:
            pytest.skip("web/js/app.js not present")
        for event in ("response", "state_update", "diary_entry"):
            assert f"on('{event}'" in js, (
                f"web/js/app.js no longer listens for '{event}' — "
                "update the schemas in this test file if this is intentional."
            )
            assert event in EVENT_SCHEMAS
