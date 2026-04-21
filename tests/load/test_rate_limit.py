"""Verify the REST rate limiter returns 429 once the per-IP quota is exhausted.

The middleware default is ``DEFAULT_API_RATE_LIMIT=60`` requests / min /
IP.  We send 120 requests in a tight loop through ``TestClient`` (which
presents a constant client IP) and expect at least some to come back as
429.  We intentionally do **not** assert the exact threshold because the
sliding window can drift by a few requests across runs.
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.load, pytest.mark.slow]


@pytest.fixture(scope="module")
def client():
    """A TestClient wrapping the real FastAPI application."""
    try:
        from starlette.testclient import TestClient

        from server.app import create_app
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Cannot build FastAPI app: {exc}")
    return TestClient(create_app())


def test_rest_rate_limit_returns_429(client) -> None:
    """Sending > DEFAULT_API_RATE_LIMIT requests/min yields at least one 429."""
    from server.middleware import DEFAULT_API_RATE_LIMIT

    status_codes: list[int] = []
    # Send twice the configured budget in one tight loop.
    for _ in range(DEFAULT_API_RATE_LIMIT * 2 + 5):
        r = client.get("/api/health")
        status_codes.append(r.status_code)

    assert 429 in status_codes, (
        f"No 429 returned after {len(status_codes)} requests "
        f"(limit={DEFAULT_API_RATE_LIMIT}/min). "
        f"Status codes: {set(status_codes)}"
    )

    # At least *some* 200s must precede the first 429 (otherwise the
    # limiter is broken in the other direction).
    assert 200 in status_codes


def test_429_response_has_headers(client) -> None:
    """A 429 response must still carry the security headers."""
    from server.middleware import DEFAULT_API_RATE_LIMIT

    # Drain the budget first
    for _ in range(DEFAULT_API_RATE_LIMIT + 5):
        client.get("/api/health")
    r = client.get("/api/health")
    if r.status_code != 429:
        # Sliding window slack — try a few more
        for _ in range(20):
            r = client.get("/api/health")
            if r.status_code == 429:
                break
    assert r.status_code == 429
    # Security headers must not be skipped on error paths
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
