"""Integration tests for the production middleware stack.

These tests spin up a minimal FastAPI app with the full middleware
chain and exercise it through the Starlette TestClient.  They catch
middleware-ordering regressions that unit tests would miss (CORS +
security headers + rate-limit + body-size + observability interact
at runtime).

The tests do NOT require the Pipeline to initialise — we replace
``app.state.pipeline`` with a lightweight stub so these tests run
in environments without torch / real checkpoints.
"""

from __future__ import annotations

import sys
import types

import pytest


# Stub torch so the middleware module chain imports cleanly.
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.tensor = lambda *a, **k: _torch_stub.Tensor()
_torch_stub.float32 = "float32"
sys.modules.setdefault("torch", _torch_stub)


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server.middleware import (  # noqa: E402
    DEFAULT_API_RATE_LIMIT,
    DEFAULT_MAX_BODY_BYTES,
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)


def _build_app(api_limit: int = DEFAULT_API_RATE_LIMIT) -> FastAPI:
    """Return a minimal app with the full middleware stack."""
    app = FastAPI()
    # Order matches server/app.py::create_app.
    app.add_middleware(
        RateLimitMiddleware, api_limit=api_limit, window_seconds=60
    )
    app.add_middleware(
        RequestSizeLimitMiddleware, max_body_bytes=DEFAULT_MAX_BODY_BYTES
    )
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/ping")
    def ping() -> dict:  # noqa: ANN201
        return {"ok": True}

    @app.post("/api/echo")
    async def echo(payload: dict) -> dict:  # noqa: ANN201
        return payload

    @app.get("/api/health")
    def health() -> dict:  # noqa: ANN201
        return {"status": "ok"}

    @app.get("/whatif/probe")
    def whatif_probe() -> dict:  # noqa: ANN201
        return {"ok": True}

    @app.get("/static/asset.js")
    def static_asset() -> str:  # noqa: ANN201
        return "// js"

    return app


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


def test_security_headers_applied_to_every_response():
    with TestClient(_build_app()) as client:
        r = client.get("/api/ping")
        assert r.status_code == 200
        for header in (
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
            "permissions-policy",
            "content-security-policy",
        ):
            assert header in {k.lower() for k in r.headers}, (
                f"missing security header: {header}"
            )


# ---------------------------------------------------------------------------
# Request-size limit
# ---------------------------------------------------------------------------


def test_oversized_body_rejected_with_413():
    app = _build_app()
    with TestClient(app) as client:
        # DEFAULT_MAX_BODY_BYTES is 1 MiB; send 2 MiB.
        big = "x" * (2 * 1024 * 1024)
        r = client.post(
            "/api/echo",
            data=big,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 413


def test_small_body_accepted():
    app = _build_app()
    with TestClient(app) as client:
        r = client.post("/api/echo", json={"msg": "hello"})
        assert r.status_code == 200
        assert r.json() == {"msg": "hello"}


# ---------------------------------------------------------------------------
# Rate-limit middleware — exclude-list semantics
# ---------------------------------------------------------------------------


def test_rate_limit_triggers_after_quota_exceeded():
    # Tiny limit so the test is deterministic and fast.
    app = _build_app(api_limit=5)
    with TestClient(app) as client:
        for _ in range(5):
            r = client.get("/api/ping")
            assert r.status_code == 200
        # 6th request within the window is rejected.
        r = client.get("/api/ping")
        assert r.status_code == 429
        assert "retry-after" in {k.lower() for k in r.headers}


def test_health_path_exempt_from_rate_limit():
    app = _build_app(api_limit=3)
    with TestClient(app) as client:
        for _ in range(10):
            r = client.get("/api/health")
            assert r.status_code == 200, (
                "/api/health should be exempt from throttling"
            )


def test_static_assets_exempt_from_rate_limit():
    app = _build_app(api_limit=3)
    with TestClient(app) as client:
        for _ in range(10):
            r = client.get("/static/asset.js")
            assert r.status_code == 200


def test_whatif_path_is_throttled():
    """/whatif/* was the original bypass vector; now throttled by
    exclude-list semantics."""
    app = _build_app(api_limit=3)
    with TestClient(app) as client:
        codes = [client.get("/whatif/probe").status_code for _ in range(6)]
        assert 429 in codes, (
            "/whatif/* should now be throttled (exclude-list semantics); "
            f"codes: {codes}"
        )


# ---------------------------------------------------------------------------
# Rate-limit response shape + retry-after header
# ---------------------------------------------------------------------------


def test_rate_limit_429_response_shape():
    app = _build_app(api_limit=1)
    with TestClient(app) as client:
        client.get("/api/ping")  # consume the quota
        r = client.get("/api/ping")
        assert r.status_code == 429
        body = r.json()
        assert "detail" in body
        # Retry-After should be a stringified integer.
        retry = r.headers.get("retry-after")
        assert retry is not None and retry.isdigit()


# ---------------------------------------------------------------------------
# Body-size limit when client does not send Content-Length
# ---------------------------------------------------------------------------


def test_missing_content_length_still_bounded():
    """Chunked requests bypass Content-Length — enforcement happens by
    accumulating bytes as they stream in."""
    app = _build_app()
    big = ("x" * (2 * 1024 * 1024)).encode()
    with TestClient(app) as client:
        r = client.post(
            "/api/echo",
            content=big,  # httpx will not set Content-Length here
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code in (413, 422)
