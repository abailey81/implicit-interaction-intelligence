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
    """Application lifespan: initialize pipeline on startup, cleanup on shutdown."""
    config = load_config("configs/default.yaml")
    pipeline = Pipeline(config)
    await pipeline.initialize()
    app.state.pipeline = pipeline
    app.state.config = config
    logger.info("I3 Pipeline initialized")
    yield
    await pipeline.cloud_client.close()
    logger.info("I3 Pipeline shut down")


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
    app = FastAPI(
        title="Implicit Interaction Intelligence (I3)",
        description="AI companion that adapts to how you interact",
        version="1.0.0",
        lifespan=lifespan,
        # Hide detailed validation errors in production by default.
        openapi_url="/api/openapi.json",
    )

    # ------------------------------------------------------------------
    # Load config *once* for the middleware stack (the lifespan reloads
    # it again for the pipeline — cheap and keeps layers decoupled).
    # ------------------------------------------------------------------
    config = load_config("configs/default.yaml")
    cors_origins = _resolve_cors_origins(config)
    logger.info("CORS origins: %s", cors_origins)

    # ------------------------------------------------------------------
    # Middleware stack — order matters: outermost added last.
    # ------------------------------------------------------------------
    # 1. Rate limiter (innermost relative to CORS — rejects before we do
    #    expensive work).
    app.add_middleware(RateLimitMiddleware)

    # 2. Request size limit.
    app.add_middleware(RequestSizeLimitMiddleware)

    # 3. Security headers on every response.
    app.add_middleware(SecurityHeadersMiddleware)

    # 4. CORS (outermost so preflight responses also get headers above).
    allow_credentials = "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=600,
    )

    # ------------------------------------------------------------------
    # Exception handlers — avoid leaking stack traces or internal paths.
    # ------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(_request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            {"detail": exc.detail}, status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, _exc: RequestValidationError):
        # Do not echo raw payloads back — they may contain PII.
        return JSONResponse(
            {"detail": "Invalid request payload"}, status_code=422
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
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

    # ------------------------------------------------------------------
    # Static files -- serve the demo UI (must be mounted *last* so API
    # and WS routes take precedence)
    # ------------------------------------------------------------------
    app.mount("/", StaticFiles(directory="web", html=True), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Bind to loopback by default; operators opt in to 0.0.0.0 via
    # I3_HOST to avoid accidental public exposure in demo mode.
    host = os.environ.get("I3_HOST", "127.0.0.1")
    port = int(os.environ.get("I3_PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)
