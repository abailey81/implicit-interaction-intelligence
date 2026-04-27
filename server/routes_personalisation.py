"""REST endpoints for the per-biometric LoRA personalisation feature.

The flagship novelty surface: each registered biometric template gets
a tiny on-device LoRA adapter (~544 trainable parameters) layered on
top of the base :class:`~i3.adaptation.types.AdaptationVector`.  This
module exposes three endpoints so the front-end Profile tab can read
the live state, reset the adapter, and surface system-wide stats:

* ``GET  /api/personalisation/{user_id}/status``
* ``POST /api/personalisation/{user_id}/reset``
* ``GET  /api/personalisation/global/stats``

Cites Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large Language
Models" (arXiv:2106.09685) and Houlsby et al. 2019 "Parameter-
Efficient Transfer Learning for NLP" (ICML 2019).  See
:mod:`i3.personalisation.lora_adapter` for the full design rationale.

Design notes
------------

* All endpoints degrade gracefully when the user's biometric template
  is not yet registered: the status endpoint returns a zero-filled
  placeholder rather than 404 so the Profile tile can still render.
* No raw template embeddings are ever returned -- only the SHA-256
  hash that keys the per-user adapter on disk.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# Matches server.websocket._USER_ID_RE — alphanumeric + underscore +
# dash, 1-64 chars.  Symmetric with the biometric / preference / WS
# layers.
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


router = APIRouter()


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


@router.get("/api/personalisation/{user_id}/status")
async def get_personalisation_status(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Return the per-user LoRA adapter status.

    Always returns a valid status dict; the ``biometric_registered``
    field tells the caller whether the adapter is actually active.
    """
    _validate_user_id(user_id)
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.get_personalisation_status(user_id)
    except Exception:
        logger.exception(
            "get_personalisation_status failed for user_id=%s", user_id
        )
        raise HTTPException(status_code=500, detail="status_failed")
    return JSONResponse(payload)


@router.post("/api/personalisation/{user_id}/reset")
async def reset_personalisation(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Clear the per-user LoRA adapter.

    Idempotent.  Returns the post-reset status so the caller can
    update its UI without a second round-trip.  When the user has no
    biometric template registered the call still succeeds with
    ``reset=False`` and a ``reason`` explaining why.
    """
    _validate_user_id(user_id)
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.reset_personalisation_for_user(user_id)
    except Exception:
        logger.exception(
            "reset_personalisation failed for user_id=%s", user_id
        )
        raise HTTPException(status_code=500, detail="reset_failed")
    return JSONResponse(payload)


@router.get("/api/personalisation/global/stats")
async def global_personalisation_stats(request: Request) -> JSONResponse:
    """System-wide per-biometric adapter statistics.

    Useful as a "look how many users have personal weights" demo
    surface for the Huawei pitch.  Returns ``active_users`` (number
    of in-memory adapters), ``total_updates`` (cumulative SGD steps
    since process start), and the configured rank / lr / alpha.
    """
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.get_personalisation_global_stats()
    except Exception:
        logger.exception("get_personalisation_global_stats failed")
        raise HTTPException(status_code=500, detail="stats_failed")
    return JSONResponse(payload)


def include_personalisation_routes(app: FastAPI) -> None:
    """Mount the personalisation router on *app*."""
    app.include_router(router)


__all__ = [
    "include_personalisation_routes",
    "router",
]
