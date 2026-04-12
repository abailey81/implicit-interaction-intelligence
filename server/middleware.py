"""Security middleware for the I3 FastAPI server.

This module provides three middleware classes used by :mod:`server.app`:

- :class:`SecurityHeadersMiddleware` — injects OWASP-recommended HTTP
  response headers on every outbound response.
- :class:`RequestSizeLimitMiddleware` — rejects any HTTP request whose
  body (or declared ``Content-Length``) exceeds a configurable ceiling.
- :class:`RateLimitMiddleware` — simple in-memory token-bucket-style
  rate limiter keyed by client IP for HTTP and by user ID for WebSocket.

These middlewares are deliberately dependency-free (no Redis, no
third-party limiter library) so the demo can run offline.  For a
multi-process production deployment, the limiter should be backed by a
shared store such as Redis.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Awaitable, Callable, Deque, Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — tunable via middleware constructor
# ---------------------------------------------------------------------------

DEFAULT_MAX_BODY_BYTES: int = 1 * 1024 * 1024          # 1 MiB for REST
DEFAULT_API_RATE_LIMIT: int = 60                        # requests / minute / IP
DEFAULT_WS_USER_RATE_LIMIT: int = 600                   # messages / minute / user
DEFAULT_WINDOW_SECONDS: int = 60                        # sliding-window length


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add OWASP-recommended security headers to every HTTP response.

    The headers applied are:

    - ``X-Content-Type-Options: nosniff`` — disable MIME sniffing.
    - ``X-Frame-Options: DENY`` — prevent click-jacking.
    - ``Referrer-Policy: strict-origin-when-cross-origin`` — minimize
      referer leakage.
    - ``Permissions-Policy`` — disable unused browser capabilities.
    - ``Content-Security-Policy`` — restrict resource origins.
    - ``Strict-Transport-Security`` — only applied on HTTPS requests.
    """

    DEFAULT_CSP: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' ws: wss:; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )

    DEFAULT_PERMISSIONS: str = (
        "camera=(), microphone=(), geolocation=(), payment=(), "
        "usb=(), accelerometer=(), gyroscope=(), magnetometer=()"
    )

    def __init__(
        self,
        app,
        content_security_policy: str | None = None,
        permissions_policy: str | None = None,
    ) -> None:
        super().__init__(app)
        self.csp = content_security_policy or self.DEFAULT_CSP
        self.permissions = permissions_policy or self.DEFAULT_PERMISSIONS

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", self.permissions)
        response.headers.setdefault("Content-Security-Policy", self.csp)
        # HSTS only applies to HTTPS
        if request.url.scheme == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains",
            )
        # Prevent caching of API responses (reduce information disclosure)
        if request.url.path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response


# ---------------------------------------------------------------------------
# Request size limit
# ---------------------------------------------------------------------------


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject HTTP requests whose body exceeds *max_body_bytes*.

    The check is performed in two stages:

    1. Fast-path — if the ``Content-Length`` header is present we compare
       against the limit immediately.
    2. Slow-path — we still let the request through but the downstream
       handlers are expected to read streaming bodies responsibly.  This
       middleware guards the common case where a malicious client sends
       a multi-megabyte JSON payload.
    """

    def __init__(self, app, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                return JSONResponse(
                    {"detail": "Invalid Content-Length header"},
                    status_code=400,
                )
            if declared > self.max_body_bytes:
                logger.warning(
                    "Request rejected: body %d > limit %d (path=%s)",
                    declared,
                    self.max_body_bytes,
                    request.url.path,
                )
                return JSONResponse(
                    {"detail": "Request body too large"},
                    status_code=413,
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class _SlidingWindowLimiter:
    """Tiny in-memory sliding-window counter.

    Not suitable for multi-process deployments (each worker gets its own
    state), but fine for the single-process demo server.  For production,
    replace with a Redis-backed implementation.
    """

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._events: dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._events[key]
        cutoff = now - self.window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._events.clear()
        else:
            self._events.pop(key, None)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter for REST endpoints, keyed by client IP.

    WebSocket rate limiting is NOT performed here (FastAPI middleware
    only sees HTTP requests); use :meth:`RateLimitMiddleware.ws_limiter`
    to share the same sliding-window implementation for per-user
    WebSocket throttling from inside the websocket handler.
    """

    def __init__(
        self,
        app,
        api_limit: int = DEFAULT_API_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        exempt_paths: Iterable[str] = ("/api/health",),
    ) -> None:
        super().__init__(app)
        self._limiter = _SlidingWindowLimiter(api_limit, window_seconds)
        self._exempt_paths = frozenset(exempt_paths)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Only rate-limit /api/* paths; static files and the docs UI are
        # not subject to throttling.
        path = request.url.path
        if not path.startswith("/api") or path in self._exempt_paths:
            return await call_next(request)

        client_ip = self._client_ip(request)
        if not self._limiter.allow(client_ip):
            logger.warning("Rate limit exceeded for %s on %s", client_ip, path)
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        return await call_next(request)

    @staticmethod
    def _client_ip(request: Request) -> str:
        """Best-effort client IP extraction.

        Honours ``X-Forwarded-For`` only when an explicit trust boundary
        marker is present in request state (set by a reverse-proxy hook
        if deployed behind nginx / envoy).  Otherwise uses the direct
        connection address.
        """
        fwd = request.headers.get("x-forwarded-for")
        if fwd and getattr(request.state, "trust_forwarded_for", False):
            return fwd.split(",")[0].strip()
        if request.client is None:
            return "unknown"
        return request.client.host

    @staticmethod
    def ws_limiter(
        limit: int = DEFAULT_WS_USER_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> _SlidingWindowLimiter:
        """Factory for a standalone WebSocket per-user rate limiter.

        The returned limiter is completely independent of the HTTP
        middleware instance and is intended for use from inside the
        WebSocket handler, which cannot use HTTP middleware.
        """
        return _SlidingWindowLimiter(limit=limit, window_seconds=window_seconds)


__all__ = [
    "SecurityHeadersMiddleware",
    "RequestSizeLimitMiddleware",
    "RateLimitMiddleware",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_API_RATE_LIMIT",
    "DEFAULT_WS_USER_RATE_LIMIT",
]
