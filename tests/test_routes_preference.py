"""Tests for :mod:`server.routes_preference`.

Covers the happy path, invalid winner / oversized body, the query and
stats endpoints, and the 404 path for an unknown user.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes_preference import (
    MAX_BODY_BYTES,
    _CACHE,
    include_preference_routes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pref_app() -> TestClient:
    """Minimal FastAPI app wired with only the preference router."""
    # Clear the module-level cache so tests don't leak state.
    _CACHE._store.clear()
    app = FastAPI()
    include_preference_routes(app)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_record_happy_path(pref_app: TestClient) -> None:
    """POST /record accepts a valid payload and reports one pair stored."""
    resp = pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "demo",
            "prompt": "How are you?",
            "response_a": "I'm fine",
            "response_b": "Fine thanks",
            "winner": "a",
            "context": [0.1] * 12,
            "response_a_features": [0.5] * 12,
            "response_b_features": [0.4] * 12,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "demo"
    assert body["accepted"] is True
    assert body["pairs_collected"] == 1


def test_record_without_vectors_zero_pads(pref_app: TestClient) -> None:
    """Missing context / features default to zero vectors server-side."""
    resp = pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "demo",
            "prompt": "Q",
            "response_a": "A",
            "response_b": "B",
            "winner": "b",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["pairs_collected"] == 1


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_invalid_winner_returns_422(pref_app: TestClient) -> None:
    """A winner outside {a, b, tie} is rejected."""
    resp = pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "demo",
            "prompt": "p",
            "response_a": "a",
            "response_b": "b",
            "winner": "probably_a",
        },
    )
    assert resp.status_code == 422


def test_invalid_user_id_returns_422(pref_app: TestClient) -> None:
    """Non-regex user IDs are rejected by the pydantic pattern."""
    resp = pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "bad user!",  # whitespace + '!'
            "prompt": "p",
            "response_a": "a",
            "response_b": "b",
            "winner": "a",
        },
    )
    assert resp.status_code == 422


def test_oversized_body_returns_413(pref_app: TestClient) -> None:
    """A body past MAX_BODY_BYTES is rejected with 413."""
    fat_prompt = "x" * (MAX_BODY_BYTES + 1024)
    resp = pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "demo",
            "prompt": fat_prompt,
            "response_a": "a",
            "response_b": "b",
            "winner": "a",
        },
    )
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


def test_query_returns_candidate_pair(pref_app: TestClient) -> None:
    """GET /query always returns a candidate (possibly fabricated)."""
    resp = pref_app.get("/api/preference/query/demo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "demo"
    assert "information_gain" in body
    assert isinstance(body["response_a_features"], list)
    assert isinstance(body["response_b_features"], list)
    assert body["prompt"] != ""


def test_query_after_record_uses_last_candidate(pref_app: TestClient) -> None:
    """After a POST the GET reuses the most recent labelled pair."""
    pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "demo",
            "prompt": "unique prompt",
            "response_a": "A",
            "response_b": "B",
            "winner": "tie",
            "context": [0.7] * 12,
            "response_a_features": [0.2] * 12,
            "response_b_features": [0.8] * 12,
        },
    )
    resp = pref_app.get("/api/preference/query/demo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompt"] == "unique prompt"


def test_query_invalid_user_id_returns_422(pref_app: TestClient) -> None:
    """Path parameter regex rejects malformed user IDs."""
    resp = pref_app.get("/api/preference/query/bad user!")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------


def test_stats_unknown_user_returns_404(pref_app: TestClient) -> None:
    """Stats for a user with no state returns 404."""
    resp = pref_app.get("/api/preference/stats/never_seen")
    assert resp.status_code == 404


def test_stats_shape_after_record(pref_app: TestClient) -> None:
    """After recording a pair, stats returns the expected shape."""
    pref_app.post(
        "/api/preference/record",
        json={
            "user_id": "demo",
            "prompt": "p",
            "response_a": "a",
            "response_b": "b",
            "winner": "a",
        },
    )
    resp = pref_app.get("/api/preference/stats/demo")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "user_id",
        "pairs_collected",
        "reward_model_ready",
        "reward_model_accuracy",
        "estimated_active_budget_remaining",
        "learned_reward_uses",
        "fallback_reward_uses",
    ):
        assert key in body, f"missing key {key!r}"
    assert body["pairs_collected"] == 1
    assert body["reward_model_ready"] is False  # <8 pairs
    assert body["estimated_active_budget_remaining"] >= 0
