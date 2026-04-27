"""REST endpoint for the manual accessibility-mode toggle.

Mounts a single endpoint:

* ``POST /api/accessibility/{user_id}/toggle`` — body
  ``{"session_id": str, "force": bool | null}``.

Used by the front-end's ``[A]`` toggle button in the nav and the
``[exit]`` link inside the accessibility-mode strip.  The ``force``
field follows the controller's manual-override contract:

* ``true`` — force-activate immediately (sticky until cleared).
* ``false`` — force-deactivate immediately (sticky until cleared).
* ``null`` — clear the override and resume auto-evaluation.

Response payload mirrors the WebSocket ``accessibility`` frame so the
caller can update its local UI state without waiting for the next
chat turn.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


# Mirror server.websocket._USER_ID_RE — alphanumeric + underscore +
# dash, 1–64 chars.  Keeps the API surface symmetric with the WS
# layer.
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class AccessibilityToggleRequest(BaseModel):
    """Body model for ``POST /api/accessibility/{user_id}/toggle``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1, max_length=128)
    force: bool | None = Field(default=None)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/api/accessibility/{user_id}/toggle")
async def toggle_accessibility(
    request: Request,
    body: AccessibilityToggleRequest,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Apply a manual accessibility-mode override.

    SEC: ``user_id`` and ``session_id`` are regex-validated so a
    crafted path or body value cannot punch through into the
    controller's session map.  Body size is implicitly capped by the
    Pydantic model + FastAPI's default 1 MB request budget.
    """
    if not _USER_ID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="invalid user_id")
    if not _SESSION_ID_RE.match(body.session_id):
        raise HTTPException(status_code=400, detail="invalid session_id")

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")

    try:
        state_dict = pipeline.force_accessibility_mode(
            user_id=user_id,
            session_id=body.session_id,
            force=body.force,
        )
    except Exception:
        logger.exception(
            "force_accessibility_mode failed for user_id=%s session_id=%s",
            user_id,
            body.session_id,
        )
        raise HTTPException(status_code=500, detail="toggle_failed")

    return JSONResponse(state_dict)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def include_accessibility_routes(app: FastAPI) -> None:
    """Mount the accessibility-toggle router on ``app``."""
    app.include_router(router)


__all__ = [
    "AccessibilityToggleRequest",
    "include_accessibility_routes",
]
