"""Tests for :mod:`server.routes_translate`.

Covers happy path, body-size cap (413), unsupported language (422),
validation edge cases, fallback-mode output when the cloud client is
absent, and the AdaptationVector echoed back in the response.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes_translate import (
    MAX_BODY_BYTES,
    include_translate_routes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def translate_app() -> TestClient:
    """Return a minimal FastAPI TestClient with only the translate router."""
    app = FastAPI()
    include_translate_routes(app)
    client = TestClient(app)
    return client


@pytest.fixture()
def translate_app_with_pipeline() -> TestClient:
    """TestClient with a mocked pipeline exposing no cloud client.

    The pipeline is installed on ``app.state`` so the route's internal
    helper can look up a (fake) user model without finding one -- this
    exercises the graceful-default branch.
    """
    app = FastAPI()
    include_translate_routes(app)

    class _FakePipeline:
        """Minimal pipeline shape -- no cloud client, empty user models."""

        cloud_client = None
        user_models: dict[str, object] = {}

    app.state.pipeline = _FakePipeline()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_returns_fallback_when_cloud_absent(
    translate_app: TestClient,
) -> None:
    """With no pipeline/cloud, the endpoint returns a fallback translation."""
    resp = translate_app.post(
        "/api/translate",
        json={
            "user_id": "demo_user",
            "text": "Hello world",
            "target_language": "fr",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_mode"] is True
    assert data["translated_text"].startswith("[fr] ")
    assert data["target_language"] == "fr"
    assert data["source_language"] is None
    assert "adaptation_applied" in data
    assert "cognitive_load" in data["adaptation_applied"]
    assert data["latency_ms"] >= 0.0


def test_happy_path_with_source_language(
    translate_app: TestClient,
) -> None:
    """Supplying an explicit source language is accepted and echoed."""
    resp = translate_app.post(
        "/api/translate",
        json={
            "user_id": "alice",
            "text": "The quick brown fox",
            "target_language": "de",
            "source_language": "en",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_language"] == "en"
    assert data["target_language"] == "de"


def test_happy_path_with_installed_pipeline(
    translate_app_with_pipeline: TestClient,
) -> None:
    """An installed pipeline without cloud still yields fallback mode."""
    resp = translate_app_with_pipeline.post(
        "/api/translate",
        json={
            "user_id": "bob",
            "text": "Good morning",
            "target_language": "ja",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_mode"] is True


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_unsupported_language_rejected(
    translate_app: TestClient,
) -> None:
    """A language not in the enum yields HTTP 422."""
    resp = translate_app.post(
        "/api/translate",
        json={
            "user_id": "demo_user",
            "text": "hello",
            "target_language": "xx",
        },
    )
    assert resp.status_code == 422


def test_oversized_body_rejected_with_413(
    translate_app: TestClient,
) -> None:
    """Request bodies above MAX_BODY_BYTES return 413."""
    # Build a body whose serialised form exceeds MAX_BODY_BYTES.
    big_text = "a" * (MAX_BODY_BYTES + 100)
    payload = json.dumps(
        {
            "user_id": "demo_user",
            "text": big_text,
            "target_language": "en",
        }
    )
    resp = translate_app.post(
        "/api/translate",
        content=payload.encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413


def test_empty_text_rejected(translate_app: TestClient) -> None:
    """All-whitespace text is rejected at validation time."""
    resp = translate_app.post(
        "/api/translate",
        json={
            "user_id": "demo_user",
            "text": "   ",
            "target_language": "en",
        },
    )
    assert resp.status_code == 422


def test_invalid_user_id_rejected(translate_app: TestClient) -> None:
    """User ids containing slashes / spaces fail the pattern check."""
    resp = translate_app.post(
        "/api/translate",
        json={
            "user_id": "../../etc/passwd",
            "text": "hello",
            "target_language": "en",
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_pii_is_redacted_before_translation(
    translate_app: TestClient,
) -> None:
    """Embedded email addresses are stripped and counted as redactions."""
    resp = translate_app.post(
        "/api/translate",
        json={
            "user_id": "demo_user",
            "text": "Contact me at test@example.com please",
            "target_language": "fr",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pii_redactions"] >= 1
    assert "test@example.com" not in data["translated_text"]
    assert "[EMAIL]" in data["translated_text"]


# ---------------------------------------------------------------------------
# Languages endpoint
# ---------------------------------------------------------------------------


def test_languages_endpoint_lists_supported_set(
    translate_app: TestClient,
) -> None:
    """``/languages`` returns every :class:`LanguageCode` value."""
    resp = translate_app.get("/api/translate/languages")
    assert resp.status_code == 200
    codes = {item["code"] for item in resp.json()["languages"]}
    assert {"zh", "en", "fr", "de", "es", "it", "pt", "ja", "ko"} <= codes
