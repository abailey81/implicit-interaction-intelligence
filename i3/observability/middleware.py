"""Request-correlation + metrics middleware for FastAPI.

Responsibilities
----------------
1. Generate (or propagate) an ``X-Request-ID`` header using UUID4.
2. Bind ``request_id``, ``trace_id`` (if a span is active), and
   ``client_ip`` to the structlog context for the lifetime of the request.
3. Emit a ``request.start`` / ``request.end`` pair of structured log
   records with the wall-clock duration.
4. Record HTTP metrics (count + latency histogram) labelled by method,
   route template, and status code.
5. Always clean up the context vars on response — even if the handler
   raises.

This middleware is deliberately implemented with the raw ASGI interface
to avoid the overhead of ``BaseHTTPMiddleware`` and to preserve
streaming responses.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from i3.observability.context import bind_context, reset_context
from i3.observability.metrics import (
    HTTP_REQUESTS_IN_PROGRESS,
    record_http,
)
from i3.observability.tracing import current_trace_id

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = b"x-request-id"
_REQUEST_ID_HEADER_OUT = "x-request-id"


def _extract_client_ip(scope: Scope) -> str:
    """Prefer ``X-Forwarded-For`` first IP, else the direct peer."""
    for name, value in scope.get("headers", []) or []:
        if name == b"x-forwarded-for":
            raw = value.decode("latin-1", errors="replace")
            return raw.split(",")[0].strip()
    client = scope.get("client")
    if client and isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return ""


def _extract_request_id(scope: Scope) -> str:
    for name, value in scope.get("headers", []) or []:
        if name == _REQUEST_ID_HEADER:
            raw = value.decode("latin-1", errors="replace").strip()
            # SEC: bound the header length so a malicious client cannot
            # blow up our log payloads.
            if 0 < len(raw) <= 128:
                return raw
    return uuid.uuid4().hex


def _route_template(scope: Scope) -> str:
    """Return the matched route path template, else the raw path.

    Using the template (e.g. ``/api/users/{user_id}``) instead of the raw
    path keeps Prometheus cardinality bounded.
    """
    route = scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if path:
            return str(path)
    return str(scope.get("path", ""))


class ObservabilityMiddleware:
    """ASGI middleware that emits per-request telemetry."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            # WebSocket / lifespan events bypass this middleware; WS
            # connection counting happens in :func:`track_websocket`.
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        client_ip = _extract_client_ip(scope)
        request_id = _extract_request_id(scope)

        tokens = bind_context(
            request_id=request_id,
            client_ip=client_ip,
        )

        # The OTel trace id is only available once the server-span has been
        # opened by the FastAPI instrumentor; read it as late as possible
        # but still inside the request lifetime.
        trace_id = current_trace_id()
        if trace_id:
            tokens.extend(bind_context(trace_id=trace_id))

        start = time.perf_counter()
        status_holder: dict[str, int] = {"status": 500}
        route_holder: dict[str, str] = {"route": str(scope.get("path", ""))}

        try:
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method).inc()
        except Exception:
            pass

        logger.info(
            "request.start",
            extra={
                "method": method,
                "path": scope.get("path", ""),
                "request_id": request_id,
                "client_ip": client_ip,
            },
        )

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message.get("status", 0))
                # Refresh route template now that the router has matched.
                route_holder["route"] = _route_template(scope)
                headers = list(message.get("headers") or [])
                # Inject X-Request-ID on the response for client correlation.
                headers.append(
                    (
                        _REQUEST_ID_HEADER_OUT.encode("latin-1"),
                        request_id.encode("latin-1"),
                    )
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Record 5xx and re-raise; FastAPI's exception handlers will
            # still produce a proper response.
            status_holder["status"] = 500
            raise
        finally:
            duration = time.perf_counter() - start
            try:
                HTTP_REQUESTS_IN_PROGRESS.labels(method=method).dec()
            except Exception:
                pass
            try:
                record_http(
                    method=method,
                    route=route_holder["route"] or scope.get("path", ""),
                    status=status_holder["status"],
                    duration_seconds=duration,
                )
            except Exception:
                pass
            logger.info(
                "request.end",
                extra={
                    "method": method,
                    "path": scope.get("path", ""),
                    "route": route_holder["route"],
                    "status": status_holder["status"],
                    "duration_ms": round(duration * 1000.0, 3),
                    "request_id": request_id,
                },
            )
            reset_context(tokens)


def install(app: Any) -> None:
    """Attach :class:`ObservabilityMiddleware` to a FastAPI app.

    Kept separate from the class so ``instrumentation.setup_observability``
    can call a single function regardless of how the middleware evolves.
    """
    app.add_middleware(ObservabilityMiddleware)
