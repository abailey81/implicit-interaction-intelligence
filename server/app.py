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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path


def _load_dotenv_into_os_environ() -> None:
    """Populate ``os.environ`` from a ``.env`` file in the repo root.

    SEC: ``pydantic-settings`` supports ``env_file=`` but our
    :mod:`i3.config` reads several variables — most notably
    ``I3_ENCRYPTION_KEY`` — directly via ``os.environ.get`` long before
    the Pydantic model is built.  If the process was launched by
    ``uvicorn`` (which does not auto-load ``.env``), those reads return
    ``None`` and the server warns about a missing Fernet key.  This
    loader closes that gap with a minimal stdlib parser so the project
    works without ``python-dotenv``.

    Rules:
        * Only lines of the form ``KEY=VALUE`` are parsed.
        * Existing ``os.environ`` entries take precedence (the process
          environment always wins over the file).
        * Lines starting with ``#`` and blank lines are skipped.
        * Trailing comments after an unquoted value are preserved as
          part of the value; users who need a ``#`` in a value should
          quote it (common ``.env`` convention).
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matching quote wrappers so VALUE="foo" → foo.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv_into_os_environ()


from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Receive, Scope, Send


class SafeStaticFiles(StaticFiles):
    """StaticFiles that ignores non-http scopes.

    The upstream ``StaticFiles.__call__`` asserts ``scope["type"] == "http"``.
    Because the root demo-UI mount is last in the router, any WebSocket or
    lifespan request that doesn't match an earlier route falls through to
    StaticFiles and raises ``AssertionError``, flooding the server log.
    We intercept non-http scopes and close them cleanly instead of asserting.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return
        await super().__call__(scope, receive, send)

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
        # PERF (M-1): reuse the config already loaded in ``create_app``
        # instead of re-parsing the YAML (which would also re-seed the
        # global RNGs).  Fall back to loading fresh if no config is
        # attached — mostly for tests that instantiate ``lifespan``
        # directly.
        config = getattr(app.state, "config", None)
        if config is None:
            _cfg_path = os.environ.get("I3_CONFIG_PATH", "configs/default.yaml")
            config = load_config(_cfg_path, set_seeds=False)
        # SEC (M-3, 2026-04-23 audit): the in-memory sliding-window rate
        # limiter keeps per-process state.  If operators scale to
        # ``I3_WORKERS > 1`` without a shared store, each worker
        # independently enforces the per-IP limit, so the effective rate
        # silently multiplies by worker count.  Require an explicit
        # opt-in override so the failure mode is *loud* rather than
        # silent.
        workers_env = os.environ.get("I3_WORKERS", "1").strip()
        try:
            workers_n = int(workers_env) if workers_env else 1
        except ValueError:
            workers_n = 1
        if workers_n > 1:
            allow_local = (
                os.environ.get("I3_ALLOW_LOCAL_LIMITER", "").strip() == "1"
            )
            if not allow_local:
                raise RuntimeError(
                    f"I3_WORKERS={workers_n} but no shared rate-limit store is "
                    "configured.  The in-memory limiter is per-process, so "
                    "per-IP limits multiply by worker count under this config. "
                    "Set I3_ALLOW_LOCAL_LIMITER=1 to override (accepting the "
                    "risk), or migrate to a Redis-backed limiter for "
                    "production multi-worker deployments."
                )
            logger.warning(
                "I3_WORKERS=%d with per-process limiter; effective per-IP "
                "rate multiplies by worker count. I3_ALLOW_LOCAL_LIMITER=1 "
                "is the explicit operator override.",
                workers_n,
            )
        pipeline = Pipeline(config)
        await pipeline.initialize()
        app.state.pipeline = pipeline
        app.state.config = config
        logger.info("I3 Pipeline initialized")

        # Optional Qwen LoRA warmup — eliminates the ~30 s cold-start
        # delay on the *first* HMI command turn.  Off by default so
        # `pytest` and CI don't pay the load cost.  Operators turn it
        # on in production with ``I3_PRELOAD_QWEN=1``.  Wrapped so a
        # missing weights file or a transformers-version mismatch
        # never blocks server startup — the chat path still works
        # without it; only the first command will be slow.
        if os.environ.get("I3_PRELOAD_QWEN", "").lower() in ("1", "true", "yes"):
            try:
                logger.info("Pre-loading Qwen LoRA intent parser…")
                import time as _t
                _t0 = _t.time()
                from i3.intent.qwen_inference import QwenIntentParser
                # Iter 51 phase 7: ``I3_QWEN_DEVICE`` overrides device
                # autodetection (default: CUDA if available, else CPU).
                # Set to "cpu" on a tight 6 GB laptop GPU where the
                # SLM v2 + Qwen 1.7 B can't both fit.  CPU adds ~6 s
                # per intent call but the cascade still works and the
                # cloud Gemini-backup arm catches anything Qwen flunks.
                qwen_device = os.environ.get("I3_QWEN_DEVICE") or None
                parser = QwenIntentParser(device=qwen_device)
                parser._ensure_loaded()
                pipeline._intent_parser_qwen = parser
                logger.info(
                    "Qwen LoRA pre-loaded in %.1fs (device=%s)",
                    _t.time() - _t0, qwen_device or "auto",
                )
            except Exception as exc:  # pragma: no cover - opt-in path
                logger.warning(
                    "Qwen LoRA pre-load failed (%s); first command "
                    "will pay the cold-start cost.", exc,
                )
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
    # PERF: flip on cuDNN benchmark + TF32 fast matmul once at import/
    # startup so every downstream forward pass (encoder, SLM, explain
    # endpoints, WebSocket inference) gets the GPU fast path for free.
    # Safe no-op when CUDA is not visible.
    from i3.runtime.device import enable_cuda_optimizations
    enable_cuda_optimizations()

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
    # Load config *once* for the whole process.  The lifespan previously
    # called ``load_config`` a second time (M-1, 2026-04-23 audit) which
    # re-seeded Python / NumPy / torch RNGs mid-startup and doubled the
    # config-parsing cost.  We load it here, stash it on ``app.state``,
    # and disable seed-setting for the second (no-op) reload.
    # ------------------------------------------------------------------
    # SEC: Allow operators to override the config path via env var
    # (defaults to the canonical configs/default.yaml). Useful when
    # standing up a parallel verification server alongside the
    # production one — they need different DB paths to avoid SQLite
    # write-lock contention on the shared diary database.
    _cfg_path = os.environ.get("I3_CONFIG_PATH", "configs/default.yaml")
    config = load_config(_cfg_path)
    app.state.config = config
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
    from server.routes_health import router as health_router
    from server.websocket import router as ws_router

    app.include_router(api_router, prefix="/api")
    # SEC: health, live, ready, metrics probes.  routes_health.py owns
    # the full /live and /ready logic (disk + encryption key + pipeline
    # readiness); server/routes.py only has a minimal /health alias.
    # Mount health_router *after* api_router so /api/health from
    # routes_health takes precedence for the richer payload.
    app.include_router(health_router, prefix="/api")
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

    # Adaptation-conditioned TTS — AI Glasses / Celia / Smart Hanhan output modality.
    from server.routes_tts import include_tts_routes
    include_tts_routes(app)

    # Adaptation uncertainty + counterfactual explanations.
    from server.routes_explain import include_explain_routes
    include_explain_routes(app)

    # Preference learning / active DPO feedback loop.
    from server.routes_preference import include_preference_routes
    include_preference_routes(app)

    # Edge-deployment proof dashboard — answers Huawei HMI Lab filter
    # question "ever deployed ML models to low-compute devices?". Reads
    # the cached report at ``reports/edge_profile.json``; the measurement
    # itself is run out-of-band via ``scripts/measure_edge.py``.
    from server.routes_edge import include_edge_routes
    include_edge_routes(app)

    # Accessibility-mode manual toggle — companion endpoint to the
    # auto-activation logic in i3.affect.accessibility_mode.  The UI's
    # nav-side [A] button hits this when the user wants to force the
    # mode on (or clear a prior override).
    from server.routes_accessibility import include_accessibility_routes
    include_accessibility_routes(app)

    # Identity Lock -- typing-biometric continuous authentication.
    # The HEADLINE I3 feature (Monrose-Rubin / Killourhy-Maxion) --
    # see i3.biometric.keystroke_auth for the full design rationale.
    from server.routes_biometric import include_biometric_routes
    include_biometric_routes(app)

    # Cognitive Profile snapshot -- per-(user, session) running stats
    # for the second flagship surface (the Profile tab).
    from server.routes_profile import include_profile_routes
    include_profile_routes(app)

    # Per-biometric LoRA personalisation -- the FLAGSHIP novelty
    # feature.  Each registered biometric template gets its own tiny
    # on-device LoRA adapter (~544 params) layered onto the base
    # AdaptationVector.  Trained online from the A/B preference picker;
    # never federated, never leaves the device.  See
    # i3.personalisation.lora_adapter for the design + Hu et al. 2021,
    # Houlsby et al. 2019 citations.
    from server.routes_personalisation import include_personalisation_routes
    include_personalisation_routes(app)

    # Live system-architecture Flow dashboard -- the third flagship
    # surface.  Per-turn pipeline trace with stage timings + arrow
    # flows ships on every WS response/response_done frame; these REST
    # routes back the "Recent turns" replay table.  See
    # i3.observability.pipeline_trace for the collector contract.
    from server.routes_flow import include_flow_routes
    include_flow_routes(app)

    # Vision fine-tuning showcase -- gaze classifier with frozen
    # MobileNetV3-small backbone + per-user fine-tuned head.
    # Closes the JD-gap "adapt or fine-tune pre-trained models" bullet.
    # See i3/multimodal/gaze_classifier.py for the full design.
    from server.routes_gaze import include_gaze_routes
    include_gaze_routes(app)

    # Cloud-vs-edge routing surface — per-session privacy budget,
    # cloud-route consent toggle, recent routing decisions for the
    # scatter plot.  Closes the JD-line "Build and fine-tune SLMs,
    # traditional ML models, or applications leveraging foundational
    # LLMs, depending on the use case" — the cloud LLM is the *opt-in*
    # fallback, the edge SLM is the default.
    from server.routes_routing import include_routing_routes
    include_routing_routes(app)

    # Quantitative benchmarks tab — latency, perplexity, coherence,
    # adaptation faithfulness, memory + size.  See i3/benchmarks/runner.py
    # and scripts/run_benchmarks.py for the offline CLI.
    from server.routes_benchmarks import include_benchmark_routes
    include_benchmark_routes(app)

    # Playground tab — manual override of every pipeline stage for
    # what-if exploration.  Capped at 100 calls per session.
    from server.routes_playground import include_playground_routes
    include_playground_routes(app)

    # ------------------------------------------------------------------
    # Static files -- serve the demo UI (must be mounted *last* so API
    # and WS routes take precedence).
    #
    # Order matters: the root ("/") mount is a catch-all, so every more
    # specific mount must precede it or Starlette routes everything to
    # the root mount.  The demo HTML also uses two URL-path conventions
    # for assets -- ``/static/css/*`` (core chat bundle) and ``/css/*``
    # (advanced-UI modules) -- so we expose the same ``web/`` directory
    # under both prefixes.
    # ------------------------------------------------------------------
    app.mount("/advanced", SafeStaticFiles(directory="web/advanced", html=True), name="advanced_ui")
    app.mount("/static", SafeStaticFiles(directory="web"), name="static_alias")
    app.mount("/", SafeStaticFiles(directory="web", html=True), name="static")

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
