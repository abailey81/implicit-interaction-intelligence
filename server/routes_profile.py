"""REST endpoint for the per-(user, session) Cognitive Profile snapshot.

Powers the front-end Profile tab.  Returns a JSON dict of running
aggregates accumulated over the live session by the pipeline's
``_profile_aggregator`` map (see :class:`i3.pipeline.Pipeline`):

* ``biometric``           — latest BiometricMatch dict
* ``iki``                 — mean / std / vs-baseline-pct / history
* ``composition``         — mean / fast-turn % / history
* ``edits``               — mean / peak / history
* ``cognitive_load``      — histogram + mean
* ``style_preferences``   — averaged formality / verbosity /
  accessibility
* ``state_distribution``  — fraction of turns in each discrete state
* ``affect_shifts``       — total + last-event timestamp
* ``biometric_drifts``    — running count of drift_alert turns
* ``session_messages``    — message count
* ``session_duration_seconds``

Mounted at ``GET /api/profile/{user_id}/{session_id}``.

Validation matches the rest of the server: alphanumeric + underscore
+ dash for user_id (1-64 chars), and a 1-128 char regex for
session_id.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


router = APIRouter()


_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


@router.get("/api/profile/{user_id}/{session_id}")
async def get_profile(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
    session_id: str = Path(..., min_length=1, max_length=128),
) -> JSONResponse:
    """Return the aggregated cognitive-profile snapshot for the session."""
    if not _USER_ID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="invalid user_id")
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="invalid session_id")

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")

    try:
        snapshot = pipeline.get_profile_snapshot(user_id, session_id)
    except Exception:
        logger.exception(
            "get_profile_snapshot failed for user_id=%s session_id=%s",
            user_id,
            session_id,
        )
        raise HTTPException(status_code=500, detail="snapshot_failed")

    return JSONResponse(snapshot)


def include_profile_routes(app: FastAPI) -> None:
    """Mount the profile router on *app*."""
    app.include_router(router)


__all__ = ["include_profile_routes"]
