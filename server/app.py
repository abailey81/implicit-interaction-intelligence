"""FastAPI application for Implicit Interaction Intelligence (I3).

Serves the web interface, REST API, and real-time WebSocket endpoint
for the live demo.

Security hardening (see SECURITY.md):
    * CORS origins come from ``config.server.cors_origins`` (not ``*``).
    * OWASP security headers are injected by :class:`SecurityHeadersMiddleware`.
    * Request bodies are capped by :class:`RequestSizeLimitMiddleware`.
    * REST API traffic is rate-limited by :class:`RateLimitMiddleware`.
    * Exception handlers return sanitised JSON error bodies — no stack
      traces or internal paths are ever exposed to the client.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from i3.config import load_config
from i3.pipeline.engine import Pipeline
from server.middleware import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize pipeline on startup, cleanup on shutdown.

    SEC: pipeline initialisation failure must propagate so the worker
    refuses to come online with a half-built state.  Shutdown is wrapped
    in best-effort cleanup so that a misbehaving cloud client cannot
    block process exit (and is idempotent — calling cleanup twice is
    harmless).
    """
    pipeline: Pipeline | None = None
    try:
        config = load_config("configs/default.yaml")
        pipeline = Pipeline(config)
        await pipeline.initialize()
        app.state.pipeline = pipeline
        app.state.config = config
        logger.info("I3 Pipeline initialized")
    except Exception:
        # SEC: initialisation failure must abort startup.  We tear down
        # any partially-built pipeline before re-raising so file handles
        # / sockets are not leaked.
        logger.exception("I3 Pipeline initialisation failed; aborting startup")
        if pipeline is not None:
            await _safe_pipeline_close(pipeline)
        raise

    try:
        yield
    finally:
        await _safe_pipeline_close(pipeline)
        # SEC: drop the reference so any post-shutdown handler that
        # accidentally touches app.state.pipeline gets a clean error.
        app.state.pipeline = None
        logger.info("I3 Pipeline shut down")


async def _safe_pipeline_close(pipeline: Pipeline | None) -> None:
    """Idempotent best-effort pipeline teardown.

    Wraps the cloud client close so a misbehaving SDK cannot raise out
    of the lifespan context — that would leave the worker stuck in the
    "shutting down" state and prevent uvicorn from exiting.
    """
    if pipeline is None:
        return
    cloud_client = getattr(pipeline, "cloud_client", None)
    if cloud_client is None:
        return
    close = getattr(cloud_client, "close", None)
    if close is None:
        return
    try:
        result = close()
        # ``close`` may be sync or async depending on the cloud client.
        if hasattr(result, "__await__"):
            await result
    except Exception:  # pragma: no cover - defensive
        logger.exception("cloud_client.close() raised during shutdown")


def _resolve_cors_origins(config) -> list[str]:
    """Determine the final list of allowed CORS origins.

    Precedence (highest first):
        1. ``I3_CORS_ORIGINS`` environment variable (comma-separated).
        2. ``config.server.cors_origins`` from the YAML configuration.
        3. A conservative localhost-only default.

    A wildcard (``*``) is permitted only when ``I3_ALLOW_CORS_WILDCARD``
    is explicitly set to ``"1"``.  Otherwise the wildcard is silently
    replaced with the localhost default and a warning is logged.
    """
    env_origins = os.environ.get("I3_CORS_ORIGINS")
    if env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    else:
        origins = list(getattr(config.server, "cors_origins", []) or [])

    if not origins:
        origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

    if "*" in origins and os.environ.get("I3_ALLOW_CORS_WILDCARD") != "1":
        logger.warning(
            "CORS wildcard '*' configured but I3_ALLOW_CORS_WILDCARD!=1; "
            "falling back to localhost-only origins."
        )
        origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
    return origins


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    # SEC: ``I3_DISABLE_OPENAPI=1`` removes the schema endpoint and the
    # interactive docs UIs.  Production deployments should set this — the
    # OpenAPI schema is an information-disclosure vector that catalogues
    # every internal route, parameter shape, and error model.
    if os.environ.get("I3_DISABLE_OPENAPI") == "1":
        openapi_url = None
        docs_url = None
        redoc_url = None
    else:
        # Namespaced under /api so a wildcard reverse-proxy block on
        # /openapi.json (a common scanning target) does not accidentally
        # break the dev UX.
        openapi_url = "/api/openapi.json"
        docs_url = "/api/docs"
        redoc_url = "/api/redoc"

    app = FastAPI(
        title="Implicit Interaction Intelligence (I3)",
        description="AI companion that adapts to how you interact",
        version="1.0.0",
        lifespan=lifespan,
        openapi_url=openapi_url,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )

    # ------------------------------------------------------------------
    # Load config *once* for the middleware stack (the lifespan reloads
    # it again for the pipeline — cheap and keeps layers decoupled).
    # ------------------------------------------------------------------
    config = load_config("configs/default.yaml")
    cors_origins = _resolve_cors_origins(config)
    logger.info("CORS origins: %s", cors_origins)

    # ------------------------------------------------------------------
    # Middleware stack — order matters.
    # ------------------------------------------------------------------
    # In Starlette, ``add_middleware`` *wraps* the existing app, so the
    # LAST middleware added is the OUTERMOST layer (it sees the request
    # first and the response last).  The desired request flow is:
    #
    #     client -> CORS -> SecurityHeaders -> RequestSizeLimit
    #            -> RateLimit -> route handler
    #
    # so we add them in REVERSE order below.
    #
    # SEC: rate-limit FIRST (innermost) so an over-quota client still
    # receives the security headers + CORS wrapping on the 429 response.
    # SEC: CORS LAST (outermost) so its preflight short-circuit returns
    # a CORS-conformant response without needing the inner stack.
    # SEC: SecurityHeaders sits INSIDE CORS so headers are applied to
    # OPTIONS responses generated by the inner stack (note: CORS
    # preflights short-circuit *before* this layer — see SECURITY.md).

    # 1. Rate limiter (innermost — runs just before the route handler).
    app.add_middleware(RateLimitMiddleware)

    # 2. Request size limit (rejects oversized bodies before parsing).
    app.add_middleware(RequestSizeLimitMiddleware)

    # 3. Security headers on every response.
    app.add_middleware(SecurityHeadersMiddleware)

    # 4. CORS (outermost so preflight responses also get headers above).
    # SEC: allow_credentials=True is REQUIRED to be False whenever the
    # origin list contains a wildcard, per the CORS spec.  We belt-and-
    # braces enforce this here even though _resolve_cors_origins already
    # rejects wildcards unless I3_ALLOW_CORS_WILDCARD=1.
    allow_credentials = "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        # SEC: explicit method allow-list — never wildcard.
        allow_methods=["GET", "POST", "OPTIONS"],
        # SEC: explicit header allow-list — never wildcard.  Only
        # Content-Type and Authorization are required by the demo SPA.
        allow_headers=["Content-Type", "Authorization"],
        # SEC: minimal expose_headers — the SPA does not need to read
        # any custom server response headers.
        expose_headers=[],
        max_age=600,
    )

    # Observability (structlog + OTel + Prometheus + optional Sentry).
    from i3.observability.instrumentation import setup_observability
    setup_observability(config, app)

    # ------------------------------------------------------------------
    # Exception handlers — avoid leaking stack traces or internal paths.
    # ------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(_request: Request, exc: StarletteHTTPException):
        # SEC: only the curated ``exc.detail`` is exposed.  Routes are
        # responsible for ensuring detail strings never contain user
        # input or internal paths (see server/routes.py for examples).
        return JSONResponse(
            {"detail": exc.detail}, status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, _exc: RequestValidationError):
        # SEC: do not echo the raw payload or pydantic error trace.
        # Pydantic's default error format includes the offending field
        # values, which may contain PII or auth tokens.
        return JSONResponse(
            {"detail": "Invalid request payload"}, status_code=422
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        # SEC: full traceback goes to the structured log only.  The
        # client receives a constant string with no exception type, no
        # exc.args, and no filename — preventing stack-trace and
        # internal-path disclosure.
        logger.exception("Unhandled exception: %s", type(exc).__name__)
        return JSONResponse(
            {"detail": "Internal server error"}, status_code=500
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    from server.routes import router as api_router
    from server.websocket import router as ws_router

    app.include_router(api_router, prefix="/api")
    app.include_router(ws_router)

    # What-if / adaptation-override endpoints for the interpretability panel.
    from server.routes_whatif import include_whatif_routes
    include_whatif_routes(app)

    # Admin routes — gated by I3_DISABLE_ADMIN env var (see server/routes_admin.py).
    from server.routes_admin import include_admin_routes
    include_admin_routes(app)

    # Serve ONNX model blobs + cross-origin isolation headers for WebGPU/threaded WASM.
    from server.routes_inference import include_inference_routes
    include_inference_routes(app)

    # Real-time translation endpoint (AI Glasses parallel — see docs/huawei/
    # harmonyos6_ai_glasses_alignment.md §2).
    from server.routes_translate import include_translate_routes
    include_translate_routes(app)

    # ------------------------------------------------------------------
    # Static files -- serve the demo UI (must be mounted *last* so API
    # and WS routes take precedence)
    # ------------------------------------------------------------------
    app.mount("/", StaticFiles(directory="web", html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # SEC: bind to loopback by default; operators must explicitly set
    # I3_HOST=0.0.0.0 (or a specific NIC address) to expose the demo on
    # the network.  This avoids accidental public exposure when the
    # binary is launched on a multi-homed dev machine.
    host = os.environ.get("I3_HOST", "127.0.0.1")
    port = int(os.environ.get("I3_PORT", "8000"))
    if host == "0.0.0.0":
        logger.warning(
            "I3_HOST=0.0.0.0 — server is reachable on all interfaces. "
            "Ensure I3_CORS_ORIGINS is locked down and a reverse proxy "
            "with TLS is in front of this process."
        )
    # SEC: reload=False is mandatory in any non-dev launch to prevent
    # the auto-reloader from watching the working directory (an info
    # disclosure vector if the working directory contains secrets).
    uvicorn.run("server.app:app", host=host, port=port, reload=False)
