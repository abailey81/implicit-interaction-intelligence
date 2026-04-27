"""REST endpoints for the typing-biometric Identity Lock.

Companion API to :class:`i3.biometric.KeystrokeAuthenticator` --
exposes a status read-out plus two demo helpers (reset, force-register)
so the front-end Identity Lock badge popover can clear / stamp a
template without typing five enrolment messages first.

Mounted on:

* ``GET  /api/biometric/{user_id}/status``        — current state
* ``POST /api/biometric/{user_id}/reset``         — clear template
* ``POST /api/biometric/{user_id}/force-register`` — stamp template

Response payloads mirror :class:`i3.biometric.BiometricMatch.to_dict`
so the caller can update its UI state without waiting for the next
WebSocket frame.

Cites: Monrose & Rubin (1997) and Killourhy & Maxion (2009) -- see the
:mod:`i3.biometric.keystroke_auth` module docstring for the full
references.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


router = APIRouter()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# Matches server.websocket._USER_ID_RE — alphanumeric + underscore +
# dash, 1-64 chars.  Keeps the API surface symmetric with the WS layer
# and the existing accessibility / preference endpoints.
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_user_id(user_id: str) -> None:
    """Raise HTTP 400 if *user_id* fails the standard regex."""
    if not _USER_ID_RE.match(user_id or ""):
        raise HTTPException(status_code=400, detail="invalid user_id")


def _get_pipeline(request: Request):
    """Return the live Pipeline or raise 503 if not yet ready."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")
    return pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/biometric/{user_id}/status")
async def get_biometric_status(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Read-only view of the typing-biometric state for *user_id*."""
    _validate_user_id(user_id)
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.get_biometric_status(user_id)
    except Exception:
        logger.exception("get_biometric_status failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="status_failed")
    return JSONResponse(payload)


@router.post("/api/biometric/{user_id}/reset")
async def reset_biometric(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Forget the registered template for *user_id*.

    Idempotent.  Returns the post-reset state (always
    ``unregistered``) so the caller can update its UI without a
    second round-trip.
    """
    _validate_user_id(user_id)
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.reset_biometric_for_user(user_id)
    except Exception:
        logger.exception("reset_biometric failed for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="reset_failed")
    return JSONResponse(payload)


@router.post("/api/biometric/{user_id}/force-register")
async def force_register_biometric(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Stamp the template from the most recent observations.

    Demo helper -- skips the 5-turn enrolment so the lock badge can
    transition out of the ``unregistered`` / ``registering`` state on
    a button click for the demo.  Returns the post-register state.
    """
    _validate_user_id(user_id)
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.force_register_biometric(user_id)
    except Exception:
        logger.exception(
            "force_register_biometric failed for user_id=%s", user_id
        )
        raise HTTPException(status_code=500, detail="force_register_failed")
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def include_biometric_routes(app: FastAPI) -> None:
    """Mount the biometric router on *app*."""
    app.include_router(router)


__all__ = ["include_biometric_routes"]
