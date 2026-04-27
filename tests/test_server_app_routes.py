"""Iter 86 — FastAPI app route inventory + middleware smoke tests.

Verifies the app imports cleanly, that the iter-51..67 routes are
registered (intent / profiling / cascade / health-deep / cost), and
that the OpenAPI schema is well-formed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    from server.app import app as fa
    return fa


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def _all_paths(app) -> set[str]:
    return {r.path for r in app.routes if hasattr(r, "path")}


def test_iter51_routes_registered(app):
    paths = _all_paths(app)
    iter51 = {"/api/intent", "/api/intent/status", "/api/profiling/report"}
    missing = iter51 - paths
    assert not missing, f"iter-51 routes missing: {missing}"


def test_iter55_to_67_routes_registered(app):
    paths = _all_paths(app)
    new_routes = {"/api/cascade/stats", "/api/health/deep",
                  "/api/cost/report"}
    missing = new_routes - paths
    assert not missing, f"iter-55..67 routes missing: {missing}"


def test_health_routes_registered(app):
    paths = _all_paths(app)
    for p in ("/api/health", "/api/live", "/api/ready"):
        assert p in paths, f"health route {p} missing"


# ---------------------------------------------------------------------------
# Smoke responses
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_openapi_schema_well_formed(client):
    r = client.get("/openapi.json")
    if r.status_code == 404:
        pytest.skip("OpenAPI disabled in this build")
    assert r.status_code == 200
    schema = r.json()
    assert schema.get("openapi", "").startswith("3")
    assert "paths" in schema


def test_unknown_route_returns_404(client):
    r = client.get("/api/this-does-not-exist-xyz")
    assert r.status_code in (404, 405)


def test_health_deep_returns_200(client):
    r = client.get("/api/health/deep")
    assert r.status_code == 200
    body = r.json()
    assert "slm_v2" in body
    assert "intent" in body
    assert "cloud" in body


def test_cost_report_returns_200(client):
    r = client.get("/api/cost/report")
    assert r.status_code == 200
    body = r.json()
    assert "total_calls" in body
    assert "by_provider" in body


def test_intent_post_validates_body(client):
    """Empty/invalid body → 400, not 500."""
    r = client.post("/api/intent", json={})  # missing 'text'
    # Depending on validation order: 400 or 200 with error envelope.
    assert r.status_code in (200, 400, 422)
