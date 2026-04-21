"""Health / readiness / metrics endpoints.

Routes exposed under the ``/api`` prefix by :mod:`server.app`:

* ``GET /api/health`` — liveness probe.  Always 200 while the process
  answers.  No authentication is required — the response body contains
  only non-sensitive metadata (service version + uptime).
* ``GET /api/live`` — alias for ``/api/health``.
* ``GET /api/ready`` — readiness probe.  200 when the pipeline is
  initialised and the Fernet encryption key is configured; 503 with
  per-check detail otherwise.
* ``GET /api/metrics`` — Prometheus exposition format.  Gated by the
  ``I3_METRICS_ENABLED`` environment variable (default: enabled); when
  disabled this endpoint returns 404 so Prometheus skips the target
  cleanly.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from i3.observability.metrics import (
    metrics_enabled,
    render_prometheus,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_PROCESS_START_MONOTONIC = time.monotonic()


def _service_version(request: Request) -> str:
    """Best-effort lookup of the service version from app.state."""
    try:
        config = getattr(request.app.state, "config", None)
        if config is not None:
            project = getattr(config, "project", None)
            if project is not None:
                return str(getattr(project, "version", "0.0.0"))
    except Exception:  # pragma: no cover - defensive
        pass
    return os.environ.get("I3_VERSION", "0.0.0")


def _uptime_seconds() -> float:
    return round(time.monotonic() - _PROCESS_START_MONOTONIC, 3)


def _check_pipeline(request: Request) -> tuple[str, str | None]:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return "not_initialized", "pipeline missing from app.state"
    # Prefer an explicit readiness flag if the Pipeline exposes one.
    ready_flag = getattr(pipeline, "is_ready", None)
    if ready_flag is False:
        return "initializing", "pipeline.is_ready == False"
    initialized = getattr(pipeline, "initialized", None)
    if initialized is False:
        return "initializing", "pipeline.initialized == False"
    return "ok", None


def _check_encryption_key() -> tuple[str, str | None]:
    """Confirm a Fernet key is available.

    The Pipeline will refuse to decrypt user profiles without one, so
    the readiness probe should fail fast when it is missing.
    """
    for env_name in ("I3_ENCRYPTION_KEY", "I3_FERNET_KEY", "FERNET_KEY"):
        if os.environ.get(env_name):
            return "ok", None
    # Allow an opt-out for environments that intentionally disable encryption
    # during local development.
    if os.environ.get("I3_DISABLE_ENCRYPTION") == "1":
        return "disabled", "I3_DISABLE_ENCRYPTION=1"
    return "missing", "no encryption key in environment"


def _check_disk() -> tuple[str, str | None]:
    """Verify the working directory has a non-trivial amount of free space."""
    try:
        total, _used, free = shutil.disk_usage(os.getcwd())
    except Exception as exc:  # pragma: no cover - platform dependent
        return "unknown", f"disk_usage failed: {exc.__class__.__name__}"
    # 64 MiB minimum — small enough to succeed on CI runners, large enough
    # that persistent DB writes will not immediately fail.
    threshold = 64 * 1024 * 1024
    if free < threshold:
        return "low", f"free={free} bytes"
    return "ok", None


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe — returns 200 whenever the process answers."""
    return {
        "status": "ok",
        "version": _service_version(request),
        "uptime_s": _uptime_seconds(),
    }


@router.get("/live")
async def live(request: Request) -> dict[str, Any]:
    """Alias for /api/health."""
    return await health(request)


@router.get("/ready")
async def ready(request: Request) -> Response:
    """Readiness probe — 200 only when all critical checks pass."""
    pipeline_status, pipeline_detail = _check_pipeline(request)
    key_status, key_detail = _check_encryption_key()
    disk_status, disk_detail = _check_disk()

    checks: dict[str, str] = {
        "pipeline": pipeline_status,
        "encryption_key": key_status,
        "disk": disk_status,
    }
    details: dict[str, str] = {}
    for name, detail in (
        ("pipeline", pipeline_detail),
        ("encryption_key", key_detail),
        ("disk", disk_detail),
    ):
        if detail:
            details[name] = detail

    critical_ok = pipeline_status == "ok" and key_status in {"ok", "disabled"}
    status_code = 200 if critical_ok else 503

    body: dict[str, Any] = {
        "status": "ready" if critical_ok else "not_ready",
        "checks": checks,
        "version": _service_version(request),
        "uptime_s": _uptime_seconds(),
    }
    if details:
        body["details"] = details

    return JSONResponse(body, status_code=status_code)


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus text exposition.  Returns 404 when metrics are disabled."""
    if not metrics_enabled():
        return JSONResponse({"detail": "metrics disabled"}, status_code=404)
    payload, content_type = render_prometheus()
    return Response(content=payload, media_type=content_type)
