"""REST API endpoints for the I3 server.

These complement the WebSocket stream with request/response queries
for user profiles, diary entries, statistics, and demo utilities.

Security controls:
    * ``user_id`` is constrained by a strict, anchored regex via a FastAPI
      ``Path`` parameter (alphanumeric, underscore, dash; 1-64 chars).
    * Pagination parameters (``limit``, ``offset``) are bounded by ``Query``
      constraints with explicit integer types.
    * Error messages never echo user input, internal paths, or stack traces.
    * Pipeline access is mediated through ``_get_pipeline`` so a missing
      pipeline yields 503 (service unavailable) rather than 500.
    * Demo endpoints (``/demo/reset`` / ``/demo/seed``) are gated behind the
      ``I3_DEMO_MODE`` environment flag to prevent accidental data loss in
      production deployments.

Known limitation (demo build):
    There is no caller authentication, so any client can read any user_id's
    data. This is acceptable for the on-device demo but MUST be revisited
    before any multi-tenant deployment (e.g. require an authenticated subject
    claim that matches the path ``user_id``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from server.auth import require_user_identity

router = APIRouter()
logger = logging.getLogger(__name__)


# SEC: gate destructive demo endpoints behind an explicit env flag so a
# misconfigured production deployment cannot wipe state by accident.
def _demo_mode_enabled() -> bool:
    return os.environ.get("I3_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


# SEC: typed pipeline accessor. Distinguishes "service not initialised" (503)
# from "logic error" (500) and avoids raw AttributeError leaking through.
def _get_pipeline(request: Request) -> Any:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return pipeline


# ---------------------------------------------------------------------------
# Shared parameter declarations
# ---------------------------------------------------------------------------

# SEC: anchored at start (^) AND end ($) so embedded newlines or trailing
# bytes (e.g. "alice\n../etc/passwd") cannot satisfy the pattern. The class
# also forbids '/', '.', and whitespace, foreclosing path-traversal payloads.
USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

UserIdParam = Path(
    ...,
    pattern=USER_ID_REGEX,
    min_length=1,
    max_length=64,
    description="Alphanumeric user identifier (1-64 chars).",
)

# SEC: explicit ``int`` type + bounded range. FastAPI/Pydantic raises a
# 422 via the global validation handler in app.py if a caller passes a
# non-integer or out-of-range value, and that handler returns a generic
# message that does NOT echo the offending value back.
LimitParam = Query(
    10,
    ge=1,
    le=100,
    description="Maximum number of records to return (1-100).",
)

OffsetParam = Query(
    0,
    ge=0,
    le=10_000,
    description="Number of records to skip (0-10000).",
)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get("/health")
async def health_check() -> JSONResponse:
    """Liveness probe -- always returns 200 if the server is up.

    SEC: deliberately minimal payload. Does NOT expose pipeline internals,
    connection counts, memory usage, hostname, or config. A separate
    ``/health/detailed`` endpoint behind an internal-network ACL is the
    correct place for ops-grade diagnostics.
    """
    return JSONResponse({"status": "healthy", "version": "1.0.0"})


# ------------------------------------------------------------------
# User endpoints
# ------------------------------------------------------------------

@router.get(
    "/user/{user_id}/profile",
    dependencies=[Depends(require_user_identity)],
)
async def get_user_profile(
    request: Request,
    user_id: str = UserIdParam,
) -> JSONResponse:
    """Return the user profile (embeddings only, no raw text).

    The profile includes the behavioural baseline, long-term style
    vector, and relationship-strength score. Raw user text is never
    included in the response body.
    """
    pipeline = _get_pipeline(request)
    try:
        profile = await pipeline.get_user_profile(user_id)
    except HTTPException:
        raise
    except Exception:
        # SEC: never leak Python class names, args, or stack traces
        # through the wire. Log full detail server-side only.
        logger.exception("get_user_profile failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if profile is None:
        # SEC: generic message -- does NOT echo user_id, preventing user
        # enumeration via 404-message comparison.
        raise HTTPException(status_code=404, detail="Profile not found")
    return JSONResponse(profile)


@router.get(
    "/user/{user_id}/diary",
    dependencies=[Depends(require_user_identity)],
)
async def get_user_diary(
    request: Request,
    user_id: str = UserIdParam,
    limit: int = LimitParam,
    offset: int = OffsetParam,
) -> JSONResponse:
    """Return recent interaction diary entries for a user.

    Parameters
    ----------
    limit : int
        Maximum number of diary entries to return (newest first, 1-100).
    offset : int
        Number of entries to skip (0-10000).
    """
    pipeline = _get_pipeline(request)
    try:
        # SEC: request ``limit + offset`` from the pipeline so the local
        # slice below produces a window of the requested size. The previous
        # implementation requested only ``limit`` and then sliced ``[offset:]``,
        # which silently truncated results whenever offset >= limit.
        fetch_count = limit + offset
        entries = await pipeline.get_diary_entries(user_id, limit=fetch_count)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_user_diary failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if entries is None:
        raise HTTPException(status_code=404, detail="Diary not found")
    if offset:
        entries = entries[offset:]
    entries = entries[:limit]
    # SEC: do not echo the validated user_id in the body either, to keep
    # response shape uniform with the 404 path. Caller already knows it.
    return JSONResponse({"entries": entries, "count": len(entries)})


@router.get(
    "/user/{user_id}/stats",
    dependencies=[Depends(require_user_identity)],
)
async def get_user_stats(
    request: Request,
    user_id: str = UserIdParam,
) -> JSONResponse:
    """Return aggregate statistics for a user.

    Includes total sessions, message count, average engagement,
    baseline-deviation history, and top routing categories.
    """
    pipeline = _get_pipeline(request)
    try:
        stats = await pipeline.get_user_stats(user_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_user_stats failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if stats is None:
        raise HTTPException(status_code=404, detail="Stats not found")
    return JSONResponse(stats)


# ------------------------------------------------------------------
# Profiling / diagnostics
# ------------------------------------------------------------------

# SEC: explicit allow-list of fields the profiling endpoint may return.
# Anything outside this list (hostname, OS info, Python version, file paths,
# environment variables, model artefact paths, etc.) is dropped before
# serialisation, even if a future pipeline implementation accidentally
# includes it in the report.
_PROFILING_ALLOWED_FIELDS = frozenset(
    {
        "components",
        "total_latency_ms",
        "memory_mb",
        "fits_budget",
        "budget_ms",
        "device_class",
    }
)


@router.get("/profiling/report")
async def get_profiling_report(request: Request) -> JSONResponse:
    """Return the edge-feasibility profiling report.

    Shows per-component latency, memory footprint, and whether the
    full pipeline fits within the 200 ms budget on-device.

    SEC: filtered through ``_PROFILING_ALLOWED_FIELDS`` so it cannot
    leak hostname, paths, Python version, or other system identifiers
    even if the underlying pipeline returns them. Rate-limiting is
    enforced upstream by the middleware layer.
    """
    pipeline = _get_pipeline(request)
    try:
        report = await pipeline.get_profiling_report()
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_profiling_report failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if not isinstance(report, dict):
        # SEC: type guard -- never serialise non-dict objects whose repr
        # might leak class names or memory addresses.
        raise HTTPException(status_code=500, detail="Internal error")
    safe_report = {k: v for k, v in report.items() if k in _PROFILING_ALLOWED_FIELDS}
    return JSONResponse(safe_report)


# ------------------------------------------------------------------
# Demo utilities
# ------------------------------------------------------------------

@router.post("/demo/reset")
async def demo_reset(request: Request) -> JSONResponse:
    """Reset all demo state (user profiles, sessions, diary).

    SEC: gated behind the ``I3_DEMO_MODE`` env flag. Without it, this
    endpoint returns 403 so an accidentally-deployed production instance
    cannot be wiped by an unauthenticated POST. ``reset_all`` is a
    destructive, irreversible operation, so the gate is mandatory.
    """
    if not _demo_mode_enabled():
        raise HTTPException(status_code=403, detail="Demo mode disabled")
    pipeline = _get_pipeline(request)
    try:
        await pipeline.reset_all()
    except HTTPException:
        raise
    except Exception:
        logger.exception("demo_reset failed")
        raise HTTPException(status_code=500, detail="Internal error")
    logger.info("Demo state reset")
    return JSONResponse({"status": "reset"})


@router.post("/demo/seed")
async def demo_seed(request: Request) -> JSONResponse:
    """Seed the demo with pre-built user profiles and diary entries.

    Creates a *demo_user* with an established baseline and a few
    historical diary entries so the live demo shows adaptation from
    the very first message.

    SEC: gated behind ``I3_DEMO_MODE``. The ``seed_demo_data`` helper
    is expected to be idempotent (using upserts keyed on demo_user)
    so repeated calls do not duplicate data. ImportError on the demo
    module is caught and surfaced as a generic 503 rather than a 500
    leaking the missing-module name.
    """
    if not _demo_mode_enabled():
        raise HTTPException(status_code=403, detail="Demo mode disabled")
    try:
        from demo.seed_data import seed_demo_data
    except ImportError:
        logger.exception("demo seed module missing")
        raise HTTPException(status_code=503, detail="Demo data unavailable")

    pipeline = _get_pipeline(request)
    try:
        await seed_demo_data(pipeline)
    except HTTPException:
        raise
    except Exception:
        logger.exception("demo_seed failed")
        raise HTTPException(status_code=500, detail="Internal error")
    logger.info("Demo data seeded")
    # SEC: do NOT echo ``result`` into the response body -- it may contain
    # internal IDs, paths, or counts useful for fingerprinting. A bare
    # status keeps the response uniform with /demo/reset.
    return JSONResponse({"status": "seeded"})
