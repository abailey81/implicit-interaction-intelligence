"""REST endpoints for the live Flow dashboard.

Companion to the WebSocket-driven ``pipeline_trace`` frame: when the
user wants to retrospectively replay any past turn (or the first page
load needs an initial set of traces before any chat happens), these
routes serve the bounded in-memory deque maintained by the pipeline's
:class:`i3.observability.pipeline_trace.PipelineTraceCollector`.

Mounted at:

* ``GET /api/flow/recent?n=10&user_id=<id>`` — most-recent N traces
* ``GET /api/flow/turn/{turn_id}`` — one specific trace by id

The user_id query param is OPTIONAL: when present we filter the deque
to that user; when absent we return everything in the buffer (the demo
runs single-tenant so this keeps the URL convenient for the front
end's first-load fetch).  ``n`` is clamped to 1..200.

Validation matches the rest of the server: alphanumeric + underscore +
dash for user_id (1-64 chars), and a UUID-ish 1-128 char regex for
turn_id (the collector returns ``str(uuid.uuid4())`` so 36 chars is the
canonical case).
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_TURN_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


@router.get("/api/flow/recent")
async def get_recent_traces(
    request: Request,
    n: int = Query(10, ge=1, le=200, description="Number of traces"),
    user_id: str | None = Query(None, max_length=64),
) -> JSONResponse:
    """Return the most-recent ``n`` traces, newest first.

    ``user_id`` is optional; when omitted the buffer contents are
    returned unfiltered (handy for the dashboard's first paint).
    """
    if user_id is not None and not _USER_ID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="invalid user_id")

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")

    collector = getattr(pipeline, "_trace_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="trace_collector_not_ready")

    try:
        traces = collector.recent(user_id=user_id, n=int(n))
    except Exception:
        logger.exception(
            "flow.recent failed (user_id=%s n=%d)", user_id, n,
        )
        raise HTTPException(status_code=500, detail="recent_failed")

    return JSONResponse({"count": len(traces), "traces": traces})


@router.get("/api/flow/turn/{turn_id}")
async def get_turn_trace(
    request: Request,
    turn_id: str = Path(..., min_length=1, max_length=128),
) -> JSONResponse:
    """Return one specific trace by ``turn_id``.

    Returns 404 when the trace has aged out of the in-memory deque
    (default capacity 200 turns).
    """
    if not _TURN_ID_RE.match(turn_id):
        raise HTTPException(status_code=400, detail="invalid turn_id")

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")

    collector = getattr(pipeline, "_trace_collector", None)
    if collector is None:
        raise HTTPException(status_code=503, detail="trace_collector_not_ready")

    try:
        trace = collector.get_turn(turn_id)
    except Exception:
        logger.exception("flow.get_turn failed (turn_id=%s)", turn_id)
        raise HTTPException(status_code=500, detail="get_turn_failed")

    if trace is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return JSONResponse(trace)


def include_flow_routes(app: FastAPI) -> None:
    """Mount the flow router on *app*."""
    app.include_router(router)


__all__ = ["include_flow_routes"]
