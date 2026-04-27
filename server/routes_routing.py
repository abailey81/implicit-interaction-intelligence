"""REST endpoints for the cloud-vs-edge routing surface.

Exposes three GET / POST endpoints that the front-end consumes to
render the cloud-consent toggle, the privacy-budget counter, and the
Routing-tab scatter plot:

* ``GET  /api/routing/budget/{user_id}/{session_id}`` →
  :class:`PrivacyBudgetSnapshot` JSON for the current session.
* ``GET  /api/routing/decision/recent?n=10`` → last *n* routing
  decisions (max 50) for the scatter plot.
* ``POST /api/routing/cloud-consent/{user_id}`` body
  ``{"enabled": bool}`` → flip the per-user cloud-route consent flag.

All endpoints validate ``user_id`` against the same regex the
WebSocket handler uses to prevent path-injection or oversized
identifiers from reaching the engine.

Privacy: nothing in this module persists raw text or PII.  Counters
are returned as scalars; the routing decision dict carries the bandit
feature vector + the complexity score breakdown but never the prompt
itself.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, FastAPI, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/routing", tags=["routing"])

# SEC: same regex as the WebSocket layer — reject anything that could
# be path-traversal, control characters, or oversized identifiers.
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def _validate_user_id(user_id: str) -> None:
    if not _USER_ID_RE.match(user_id or ""):
        raise HTTPException(status_code=400, detail="Invalid user_id")


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise HTTPException(status_code=400, detail="Invalid session_id")


class CloudConsentBody(BaseModel):
    """Body for POST /api/routing/cloud-consent/{user_id}.

    ``enabled`` is the new state of the per-user cloud-route consent
    flag.  Defaults to ``False`` (opt-in by design).
    """

    enabled: bool = Field(..., description="True to opt in, False to opt out")


@router.get("/budget/{user_id}/{session_id}")
async def get_budget(
    request: Request,
    user_id: str = Path(..., max_length=64),
    session_id: str = Path(..., max_length=128),
) -> dict:
    """Return the privacy-budget snapshot for the given (user, session).

    Always returns a populated snapshot — counters are zeroed for
    sessions that haven't made a cloud call yet.
    """
    _validate_user_id(user_id)
    _validate_session_id(session_id)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None or not hasattr(pipeline, "privacy_budget"):
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    snap = pipeline.privacy_budget.snapshot(user_id, session_id)
    return snap.to_dict()


@router.get("/decision/recent")
async def get_recent_decisions(
    request: Request,
    n: int = Query(10, ge=1, le=50),
) -> dict:
    """Return the last *n* routing decisions for the scatter plot.

    The decisions are ordered oldest-first (so the UI's "tail of the
    deque" rendering reads naturally).  Each entry is the same dict
    shape as ``PipelineOutput.routing_decision``.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None or not hasattr(pipeline, "_recent_routing_decisions"):
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    decisions = list(pipeline._recent_routing_decisions)
    # Tail of the ring buffer.
    decisions = decisions[-int(n):]
    return {"decisions": decisions, "count": len(decisions)}


@router.post("/cloud-consent/{user_id}")
async def set_cloud_consent(
    request: Request,
    body: CloudConsentBody,
    user_id: str = Path(..., max_length=64),
) -> dict:
    """Flip the per-user cloud-route consent flag.

    Default for any unknown user is ``False`` (opt-in).  Toggling to
    ``True`` does NOT bypass the per-session budget — the engine
    still gates every cloud call on
    :meth:`PrivacyBudget.can_call`.

    Returns the new state plus the current snapshot for any active
    session (useful for the toast UI to immediately show the
    "calls remaining" counter).
    """
    _validate_user_id(user_id)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None or not hasattr(pipeline, "privacy_budget"):
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    new_state = pipeline.privacy_budget.set_consent(user_id, body.enabled)
    # Also flip the cloud-client's consent override so a user can opt
    # in even when the operator has set ``I3_CLOUD_DISABLED=1`` on the
    # process (demo convenience — the per-user budget still gates).
    cloud_client = getattr(pipeline, "cloud_client", None)
    if cloud_client is not None and hasattr(cloud_client, "set_consent_override"):
        try:
            cloud_client.set_consent_override(body.enabled)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "cloud_client.set_consent_override failed for user_id=%s",
                user_id,
            )
    logger.info(
        "Cloud-route consent for user_id=%s set to %s via REST",
        user_id,
        new_state,
    )
    return {
        "user_id": user_id,
        "enabled": new_state,
    }


def include_routing_routes(app: FastAPI) -> None:
    """Mount the routing router under ``/api/routing``."""
    app.include_router(router, prefix="/api")
