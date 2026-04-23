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
import threading
import time
from collections import OrderedDict, deque
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

# SEC: Cap on the number of distinct rate-limit keys we track in-memory.
# Without this bound the limiter dict grows unbounded for every distinct
# client IP that has ever hit the API, which is a DoS vector (memory
# exhaustion).  When the cap is hit we evict the oldest idle key.
DEFAULT_MAX_TRACKED_KEYS: int = 10_000


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

    # SEC: 'unsafe-inline' is required for both script-src and style-src
    # because web/index.html embeds a critical style block (lines 8-13) and
    # a DOMContentLoaded inline script (lines 146-151).  Removing
    # 'unsafe-inline' would break the demo UI.  Trade-off: this weakens
    # XSS defence-in-depth.  To tighten in production:
    #   1. Move both inline blocks into /static/css/critical.css and
    #      /static/js/bootstrap.js, OR
    #   2. Generate a per-request nonce and rewrite the HTML on the fly,
    #      then emit `script-src 'self' 'nonce-<n>'`.
    # Option (1) is the recommended hardening path.
    DEFAULT_CSP: str = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self' ws: wss:; "
        "font-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )

    # SEC: fullscreen=(self) added per OWASP guidance — the demo dashboard
    # may use fullscreen on the embedding canvas; geolocation/camera/etc.
    # are denied entirely.
    DEFAULT_PERMISSIONS: str = (
        "camera=(), microphone=(), geolocation=(), payment=(), "
        "usb=(), accelerometer=(), gyroscope=(), magnetometer=(), "
        "fullscreen=(self)"
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
        # SEC: X-Permitted-Cross-Domain-Policies blocks Adobe Flash / PDF
        # cross-domain policy files, which is the safe default for an API.
        response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
        # SEC: Cross-Origin-Opener-Policy isolates the browsing context so
        # that a malicious cross-origin window cannot use window.opener
        # tricks against the demo UI.
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        # SEC: HSTS only applies to HTTPS.  When deployed behind a TLS
        # terminating reverse proxy (nginx/envoy) the wire-level scheme is
        # "http" and we must consult X-Forwarded-Proto.  We only honour
        # the forwarded header when an explicit trust marker has been set
        # in request.state by the proxy bootstrap (mirrors the rate-limit
        # _client_ip handling).
        if self._is_https(request):
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains",
            )
        # SEC: API responses must not be cached by intermediaries — they
        # may contain user-specific profile data (PII).  Static assets are
        # served from a different mount and are unaffected.
        if request.url.path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-store")
            response.headers.setdefault("Pragma", "no-cache")
        return response

    @staticmethod
    def _is_https(request: Request) -> bool:
        """Detect HTTPS, honouring trusted X-Forwarded-Proto."""
        if request.url.scheme == "https":
            return True
        # SEC: only trust the forwarded header when the proxy bootstrap
        # has explicitly opted-in by setting request.state.trust_proxy.
        if getattr(request.state, "trust_proxy", False):
            xfp = request.headers.get("x-forwarded-proto", "")
            if xfp.split(",")[0].strip().lower() == "https":
                return True
        return False


# ---------------------------------------------------------------------------
# Request size limit
# ---------------------------------------------------------------------------


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject HTTP requests whose body exceeds *max_body_bytes*.

    The check is performed in two stages:

    1. Fast-path — if the ``Content-Length`` header is present we compare
       against the limit immediately and reject before any body is read.
    2. Slow-path — for chunked / streaming requests with no declared
       length we reject up-front because the demo API has no legitimate
       streaming endpoints.  Switching to a buffered enforcement (read
       the body and bail mid-stream once the threshold is crossed) is a
       documented future enhancement.
    """

    # SEC: methods that may legitimately carry a body — GET/HEAD/DELETE
    # are allowed through without inspection because they should not have
    # one (and Starlette will surface any oddity to the route).
    _BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

    def __init__(self, app, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        super().__init__(app)
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.max_body_bytes = max_body_bytes

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # SEC: only inspect methods that can carry a body.  GET/HEAD with
        # a Content-Length is unusual but not inherently dangerous, so we
        # let it through unchecked rather than risk false positives.
        if request.method.upper() not in self._BODY_METHODS:
            return await call_next(request)

        content_length = request.headers.get("content-length")
        transfer_encoding = request.headers.get("transfer-encoding", "").lower()

        # SEC: chunked encoding does not declare a length up-front.  We
        # reject because none of the demo API endpoints accept streaming
        # uploads, so any chunked POST is either misconfigured or hostile.
        if "chunked" in transfer_encoding and content_length is None:
            logger.warning(
                "Request rejected: chunked encoding with no Content-Length (path=%s)",
                request.url.path,
            )
            return JSONResponse(
                {"detail": "Chunked transfer encoding is not supported"},
                status_code=411,  # Length Required
            )

        if content_length is None:
            # SEC: no Content-Length and no chunked encoding for a body
            # method is itself an HTTP protocol violation.  Reject.
            return JSONResponse(
                {"detail": "Length Required"},
                status_code=411,
            )

        try:
            declared = int(content_length)
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid Content-Length header"},
                status_code=400,
            )

        # SEC: a negative Content-Length is a smuggling indicator.
        if declared < 0:
            logger.warning(
                "Request rejected: negative Content-Length %d (path=%s)",
                declared,
                request.url.path,
            )
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

    SEC: this implementation guarantees three security properties:

    1. **Concurrency safety** — all reads and writes happen under a
       :class:`threading.Lock`, which protects against both asyncio
       coroutine interleaving (defence-in-depth — there is no ``await``
       inside the critical section) and the rare case of multiple
       uvicorn worker threads sharing process state.
    2. **Bounded memory** — the dict size is capped at *max_keys*; when
       full, the oldest *idle* keys are evicted.  Without this an
       attacker could enumerate IPs to exhaust server memory.
    3. **Idle key pruning** — keys whose buckets become empty after
       window expiry are removed lazily on each ``allow()`` call when
       the cap is hit (amortised O(1)).

    ``allow()`` is intentionally synchronous so it can be invoked from
    the WebSocket handler (which holds the event loop and cannot ``await``
    a separate limiter without restructuring its read loop).
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        max_keys: int = DEFAULT_MAX_TRACKED_KEYS,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_keys <= 0:
            raise ValueError("max_keys must be positive")
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        # SEC/PERF: insertion-ordered dict gives amortised O(1) oldest-key
        # eviction (``popitem(last=False)``) instead of the O(n) scan that a
        # plain dict + ``min(..., key=...)`` required.  Under hostile key
        # churn this drops the per-accept cost from O(n) to O(1).
        self._events: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window

            bucket = self._events.get(key)
            if bucket is None:
                # SEC: enforce the global key cap *before* we insert.
                if len(self._events) >= self.max_keys:
                    self._evict_idle(cutoff)
                    if len(self._events) >= self.max_keys:
                        # All keys are still active — fall back to
                        # evicting the single oldest entry by head time.
                        self._evict_oldest()
                bucket = deque()
                self._events[key] = bucket
            else:
                while bucket and bucket[0] < cutoff:
                    bucket.popleft()
                # SEC: an idle key whose bucket emptied out can be
                # dropped immediately to keep the dict from accumulating
                # one entry per ever-seen IP.
                if not bucket:
                    self._events.pop(key, None)
                    bucket = deque()
                    self._events[key] = bucket
                else:
                    # PERF: refresh LRU ordering so active keys migrate to
                    # the most-recently-used end, keeping eviction fair.
                    self._events.move_to_end(key)

            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True

    def _evict_idle(self, cutoff: float) -> None:
        """Drop any key whose entire bucket is older than *cutoff*."""
        stale = [k for k, b in self._events.items() if not b or b[-1] < cutoff]
        for k in stale:
            self._events.pop(k, None)

    def _evict_oldest(self) -> None:
        """Evict the least-recently-used key in O(1).

        Uses ``OrderedDict.popitem(last=False)`` instead of a linear
        ``min()`` scan — amortised O(1) per accept call.
        """
        if not self._events:
            return
        self._events.popitem(last=False)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._events.clear()
        else:
            self._events.pop(key, None)

    def __len__(self) -> int:  # diagnostics / tests
        return len(self._events)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter for REST endpoints, keyed by client IP.

    WebSocket rate limiting is NOT performed here (Starlette HTTP
    middleware never sees the ``websocket`` ASGI scope); use
    :meth:`RateLimitMiddleware.ws_limiter` to share the same
    sliding-window implementation for per-user WebSocket throttling from
    inside the websocket handler.

    Trust model
    -----------
    By default the limiter keys on the immediate TCP peer
    (``request.client.host``) and **does not** trust ``X-Forwarded-For``.
    To deploy behind nginx / envoy, set
    ``request.state.trust_forwarded_for = True`` from a proxy bootstrap
    middleware that has already validated the upstream IP.
    """

    # SEC: prefixes that serve static / read-only assets and should NOT
    # be throttled.  This is an **exclude-list** — every request is
    # throttled by default, which means any newly-added route family
    # inherits the limiter without having to remember to prefix itself
    # under ``/api``.  An include-list ("only throttle /api/*") was the
    # prior design; it silently missed ``/whatif/*`` (see H-1 of the
    # 2026-04-23 security audit).
    DEFAULT_EXEMPT_PREFIXES: tuple[str, ...] = (
        "/ws/",              # WebSocket upgrades bypass BaseHTTPMiddleware
        "/static/",          # Bundled CSS / JS / fonts
        "/_static/",         # Starlette-mounted assets
        "/assets/",          # Front-end build output
        # SEC (M-12, 2026-04-23 audit): docs are mounted under /api/ in
        # ``server/app.py::create_app`` (``docs_url="/api/docs"``), so
        # the exempt prefix must match the real mount point.  Bare
        # ``/docs`` / ``/redoc`` previously never fired.
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/favicon",          # Favicons
    )

    def __init__(
        self,
        app,
        api_limit: int = DEFAULT_API_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        exempt_paths: Iterable[str] = ("/api/health", "/api/live", "/api/ready"),
        max_tracked_keys: int = DEFAULT_MAX_TRACKED_KEYS,
        exempt_prefixes: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._limiter = _SlidingWindowLimiter(
            api_limit, window_seconds, max_keys=max_tracked_keys
        )
        self._exempt_paths = frozenset(exempt_paths)
        self._exempt_prefixes: tuple[str, ...] = tuple(
            exempt_prefixes
            if exempt_prefixes is not None
            else self.DEFAULT_EXEMPT_PREFIXES
        )
        # SEC: cache the Retry-After value so we don't string-format
        # on every rejected request.
        self._retry_after = str(window_seconds)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # SEC: every request is throttled except static assets, health
        # probes, WebSocket upgrades, and the OpenAPI/docs surfaces.  The
        # exclude-list is intentional so that any newly-added route
        # family is throttled by default (see H-1 / M-5 in the
        # 2026-04-23 security audit).
        path = request.url.path
        if (
            path in self._exempt_paths
            or any(path.startswith(p) for p in self._exempt_prefixes)
        ):
            return await call_next(request)

        client_ip = self._client_ip(request)
        if not self._limiter.allow(client_ip):
            logger.warning("Rate limit exceeded for %s on %s", client_ip, path)
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": self._retry_after},
            )
        return await call_next(request)

    @staticmethod
    def _client_ip(request: Request) -> str:
        """Best-effort client IP extraction.

        SEC: ``X-Forwarded-For`` is **only** honoured when an explicit
        trust boundary marker (``request.state.trust_forwarded_for``) is
        set by a reverse-proxy bootstrap.  Otherwise the header is
        ignored entirely — accepting it unconditionally would let any
        client trivially bypass the limiter by spoofing the header.
        """
        if getattr(request.state, "trust_forwarded_for", False):
            fwd = request.headers.get("x-forwarded-for")
            if fwd:
                # The first entry is the original client per RFC 7239.
                return fwd.split(",")[0].strip() or "unknown"
        if request.client is None:
            return "unknown"
        return request.client.host

    @staticmethod
    def ws_limiter(
        limit: int = DEFAULT_WS_USER_RATE_LIMIT,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        max_keys: int = DEFAULT_MAX_TRACKED_KEYS,
    ) -> _SlidingWindowLimiter:
        """Factory for a standalone WebSocket per-user rate limiter.

        The returned limiter is completely independent of the HTTP
        middleware instance and is intended for use from inside the
        WebSocket handler, which cannot use HTTP middleware.
        """
        return _SlidingWindowLimiter(
            limit=limit, window_seconds=window_seconds, max_keys=max_keys
        )


__all__ = [
    "SecurityHeadersMiddleware",
    "RequestSizeLimitMiddleware",
    "RateLimitMiddleware",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_API_RATE_LIMIT",
    "DEFAULT_WS_USER_RATE_LIMIT",
]
