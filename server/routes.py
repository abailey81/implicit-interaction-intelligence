"""REST API endpoints for the I3 server.

These complement the WebSocket stream with request/response queries
for user profiles, diary entries, statistics, and demo utilities.

Security controls:
    * ``user_id`` is constrained by a strict regex via a FastAPI ``Path``
      parameter (alphanumeric, underscore, dash; 1-64 chars).
    * Pagination parameters (``limit``) are bounded by ``Query`` constraints.
    * Error messages never echo user input or internal paths.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Request

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared parameter declarations
# ---------------------------------------------------------------------------

USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

UserIdParam = Path(
    ...,
    pattern=USER_ID_REGEX,
    min_length=1,
    max_length=64,
    description="Alphanumeric user identifier (1-64 chars).",
)

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
async def health_check() -> dict[str, str]:
    """Liveness probe -- always returns 200 if the server is up."""
    return {"status": "healthy", "version": "1.0.0"}


# ------------------------------------------------------------------
# User endpoints
# ------------------------------------------------------------------

@router.get("/user/{user_id}/profile")
async def get_user_profile(
    request: Request,
    user_id: str = UserIdParam,
) -> dict[str, Any]:
    """Return the user profile (embeddings only, no raw text).

    The profile includes the behavioural baseline, long-term style
    vector, and relationship-strength score.
    """
    pipeline = request.app.state.pipeline
    profile = await pipeline.get_user_profile(user_id)
    if profile is None:
        # Intentionally do not echo the user_id back to the caller.
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.get("/user/{user_id}/diary")
async def get_user_diary(
    request: Request,
    user_id: str = UserIdParam,
    limit: int = LimitParam,
    offset: int = OffsetParam,
) -> dict[str, Any]:
    """Return recent interaction diary entries for a user.

    Parameters
    ----------
    limit : int
        Maximum number of diary entries to return (newest first, 1-100).
    offset : int
        Number of entries to skip.
    """
    pipeline = request.app.state.pipeline
    entries = await pipeline.get_diary_entries(user_id, limit=limit)
    # Apply offset locally in case the pipeline does not implement it.
    if offset:
        entries = entries[offset:]
    return {"user_id": user_id, "entries": entries}


@router.get("/user/{user_id}/stats")
async def get_user_stats(
    request: Request,
    user_id: str = UserIdParam,
) -> dict[str, Any]:
    """Return aggregate statistics for a user.

    Includes total sessions, message count, average engagement,
    baseline-deviation history, and top routing categories.
    """
    pipeline = request.app.state.pipeline
    stats = await pipeline.get_user_stats(user_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Stats not found")
    return stats


# ------------------------------------------------------------------
# Profiling / diagnostics
# ------------------------------------------------------------------

@router.get("/profiling/report")
async def get_profiling_report(request: Request) -> dict[str, Any]:
    """Return the edge-feasibility profiling report.

    Shows per-component latency, memory footprint, and whether the
    full pipeline fits within the 200 ms budget on-device.
    """
    pipeline = request.app.state.pipeline
    report = await pipeline.get_profiling_report()
    return report


# ------------------------------------------------------------------
# Demo utilities
# ------------------------------------------------------------------

@router.post("/demo/reset")
async def demo_reset(request: Request) -> dict[str, str]:
    """Reset all demo state (user profiles, sessions, diary).

    Useful between presentation runs so each demo starts clean.
    """
    pipeline = request.app.state.pipeline
    await pipeline.reset_all()
    logger.info("Demo state reset")
    return {"status": "reset"}


@router.post("/demo/seed")
async def demo_seed(request: Request) -> dict[str, Any]:
    """Seed the demo with pre-built user profiles and diary entries.

    Creates a *demo_user* with an established baseline and a few
    historical diary entries so the live demo shows adaptation from
    the very first message.
    """
    from demo.seed_data import seed_demo_data

    pipeline = request.app.state.pipeline
    result = await seed_demo_data(pipeline)
    logger.info("Demo data seeded: %s", result)
    return {"status": "seeded", "detail": result}
