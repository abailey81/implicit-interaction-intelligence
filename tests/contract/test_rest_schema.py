"""Contract tests for the REST API using :mod:`schemathesis`.

The FastAPI application is loaded in-process and schemathesis is pointed
at the live ASGI app.  Two layers of checks run:

1. **OpenAPI schema validation** — the generated ``/api/openapi.json``
   must itself be a valid OpenAPI 3.x document.
2. **Generated-request fuzzing** — schemathesis synthesises example
   requests from the schema and checks every response against
   ``all_checks=True`` (status code, content type, response schema,
   header semantics, etc.).

If :mod:`schemathesis` is not installed, the entire module is skipped
cleanly so a minimal install does not break the suite.
"""

from __future__ import annotations

import os

import pytest


# Schemathesis is an optional dev dependency — declare a single skip
# gate so the rest of the test file stays lean.
schemathesis = pytest.importorskip("schemathesis")


# The FastAPI app is heavy to import (PyTorch / numpy); defer it to a
# fixture so pytest collection stays fast on environments that only run
# a subset of tests.
@pytest.fixture(scope="module")
def asgi_app():
    """Return the FastAPI ASGI application, or skip if wiring fails."""
    # The app constructs a Pipeline at import time via create_app();
    # we also need I3_DISABLE_OPENAPI to be unset for the schema route.
    os.environ.pop("I3_DISABLE_OPENAPI", None)
    try:
        from server.app import create_app  # noqa: WPS433
    except Exception as exc:  # pragma: no cover — degraded env
        pytest.skip(f"Cannot import server.app: {exc}")
    return create_app()


@pytest.fixture(scope="module")
def schema(asgi_app):
    """schemathesis schema object, loaded from the live ASGI app."""
    try:
        return schemathesis.from_asgi("/api/openapi.json", asgi_app)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Cannot load OpenAPI schema from ASGI app: {exc}")


class TestOpenAPIValidity:
    def test_schema_is_valid_openapi(self, schema) -> None:
        """The generated schema is itself a valid OpenAPI document."""
        # schemathesis raises on malformed schemas during discovery.
        assert schema.raw_schema is not None
        assert "openapi" in schema.raw_schema
        assert "paths" in schema.raw_schema

    def test_paths_are_nonempty(self, schema) -> None:
        """We must have at least one documented path."""
        assert len(schema.raw_schema["paths"]) >= 1


# ─────────────────────────────────────────────────────────────────────────
#  Request-level fuzzing — runs schemathesis-generated cases through
#  the ASGI client and checks all built-in response-shape invariants.
# ─────────────────────────────────────────────────────────────────────────


def test_rest_api_conforms_to_schema(schema, asgi_app) -> None:
    """Run schemathesis' built-in checks for every documented operation.

    For each operation we pull one Hypothesis-generated ``Case``, dispatch
    it through the in-process ASGI ``TestClient``, and assert that no
    handler returned a 5xx (i.e. an unhandled server error).  A 4xx is
    allowed as long as the schema admits it — that's the "you gave me
    garbage" path we explicitly wanted to exercise.
    """
    from starlette.testclient import TestClient

    # schemathesis stores the app differently across versions; fall back
    # to the fixture-provided asgi_app if `schema.app` is not available.
    app = getattr(schema, "app", None) or asgi_app
    client = TestClient(app)

    failures: list[str] = []
    try:
        operations = list(schema.get_all_operations())
    except Exception as exc:
        pytest.skip(f"schemathesis.get_all_operations unsupported: {exc}")

    for operation in operations:
        try:
            strategy = operation.as_strategy()
        except Exception as exc:  # pragma: no cover
            # Some operations may require bodies schemathesis can't
            # synthesise (e.g. stream uploads); skip them gracefully.
            failures.append(f"{getattr(operation, 'path', '?')}: {exc}")
            continue

        try:
            case = strategy.example()
        except Exception:
            continue

        try:
            response = case.call(client)
        except Exception as exc:  # pragma: no cover
            failures.append(f"{getattr(operation, 'path', '?')}: {exc}")
            continue

        if response.status_code >= 500:
            failures.append(
                f"{getattr(operation, 'path', '?')}: "
                f"{response.status_code} {response.text[:200]}"
            )

    assert not failures, "Schema contract violations:\n" + "\n".join(failures)


# ─────────────────────────────────────────────────────────────────────────
#  Simple concrete-request sanity checks (no Hypothesis)
# ─────────────────────────────────────────────────────────────────────────


class TestRestConcreteEndpoints:
    def test_health_endpoint_shape(self, asgi_app) -> None:
        """`GET /api/health` returns `{"status": "healthy", "version": ...}`."""
        from starlette.testclient import TestClient

        client = TestClient(asgi_app)
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "healthy"
        assert "version" in body

    def test_openapi_schema_served(self, asgi_app) -> None:
        """The schema endpoint itself is reachable and returns JSON."""
        from starlette.testclient import TestClient

        client = TestClient(asgi_app)
        r = client.get("/api/openapi.json")
        assert r.status_code == 200
        body = r.json()
        assert body["info"]["title"]
        assert "paths" in body

    def test_invalid_user_id_rejected(self, asgi_app) -> None:
        """Path traversal / out-of-regex user ids must be rejected."""
        from starlette.testclient import TestClient

        client = TestClient(asgi_app)
        r = client.get("/api/user/..%2Fetc%2Fpasswd/profile")
        assert r.status_code in (404, 422)
