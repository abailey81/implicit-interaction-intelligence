"""REST endpoint for the Playground UI tab.

``POST /api/playground/whatif`` lets the operator manually override
every pipeline stage for a single forward pass — adaptation, biometric
state, accessibility, route, critique, coref, safety.

Distinct from ``/api/whatif/respond`` which only handles AdaptationVector
overrides; the Playground endpoint handles the full ``playground_overrides``
dict carried on :class:`i3.pipeline.types.PipelineInput`.

Hard-capped at 100 calls per session by the pipeline.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playground", tags=["playground"])

USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"
MAX_MESSAGE_CHARS = 4000
MAX_CALLS_PER_SESSION = 100


class PlaygroundOverrides(BaseModel):
    """Per-turn overrides; every field optional."""

    adaptation: dict | None = None
    biometric_state: str | None = Field(default=None, pattern=r"^(registered|mismatch|unregistered)$")
    accessibility: bool | None = None
    route: str | None = Field(default=None, pattern=r"^(edge|cloud)$")
    critique: bool | None = None
    coref: bool | None = None
    safety: bool | None = None


class PlaygroundRequest(BaseModel):
    user_id: str = Field(..., pattern=USER_ID_REGEX, min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)
    overrides: PlaygroundOverrides = Field(default_factory=PlaygroundOverrides)
    compare_baseline: bool = Field(default=False)

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("message must contain non-whitespace text")
        return s


@router.post("/whatif")
async def playground_whatif(request: Request, body: PlaygroundRequest) -> JSONResponse:
    """Run a single message through the pipeline with overrides applied.

    When ``compare_baseline=True`` the same prompt is run twice — once
    with overrides, once without — and the response includes both.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service unavailable")

    # Per-session call cap (separate from pipeline-internal counter for
    # robustness if the pipeline is recreated mid-session).
    counts = getattr(pipeline, "_playground_call_counts", None)
    if counts is None:
        counts = {}
        try:
            pipeline._playground_call_counts = counts
        except Exception:
            pass
    key = body.user_id
    counts[key] = counts.get(key, 0) + 1
    if counts[key] > MAX_CALLS_PER_SESSION:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Playground call cap reached "
                f"({MAX_CALLS_PER_SESSION} per session)."
            ),
        )

    overrides_dict: dict[str, Any] = {}
    o = body.overrides
    if o.adaptation is not None:
        overrides_dict["adaptation"] = o.adaptation
    if o.biometric_state is not None:
        overrides_dict["biometric_state"] = o.biometric_state
    if o.accessibility is not None:
        overrides_dict["accessibility"] = bool(o.accessibility)
    if o.route is not None:
        overrides_dict["route"] = o.route
    if o.critique is not None:
        overrides_dict["critique"] = bool(o.critique)
    if o.coref is not None:
        overrides_dict["coref"] = bool(o.coref)
    if o.safety is not None:
        overrides_dict["safety"] = bool(o.safety)

    from i3.pipeline.types import PipelineInput

    session_id = await pipeline.start_session(body.user_id)

    async def _drive(overrides: dict | None) -> dict[str, Any]:
        inp = PipelineInput(
            user_id=body.user_id,
            session_id=session_id,
            message_text=body.message,
            timestamp=time.time(),
            composition_time_ms=1500.0,
            edit_count=0,
            pause_before_send_ms=200.0,
            keystroke_timings=[80.0] * 10,
            playground_overrides=overrides,
        )
        started = time.perf_counter()
        out = await pipeline.process_message(inp)
        latency = (time.perf_counter() - started) * 1000.0
        return {
            "text": out.response_text,
            "route": out.route_chosen,
            "response_path": getattr(out, "response_path", "unknown"),
            "latency_ms": round(latency, 2),
            "adaptation": out.adaptation,
            "adaptation_changes": list(getattr(out, "adaptation_changes", []) or []),
            "safety": getattr(out, "safety", None),
            "biometric": getattr(out, "biometric", None),
            "accessibility": getattr(out, "accessibility", None),
            "critique": getattr(out, "critique", None),
        }

    try:
        primary = await _drive(overrides_dict if overrides_dict else None)
    except HTTPException:
        raise
    except Exception:
        logger.exception("playground.whatif failed")
        raise HTTPException(status_code=500, detail="Internal error")

    response: dict[str, Any] = {
        "request_id": uuid.uuid4().hex[:12],
        "overrides_applied": overrides_dict,
        "result": primary,
        "calls_used": counts[key],
        "calls_remaining": max(0, MAX_CALLS_PER_SESSION - counts[key]),
    }
    if body.compare_baseline:
        try:
            baseline = await _drive(None)
            response["baseline"] = baseline
        except Exception:
            logger.exception("playground.whatif baseline failed")
            response["baseline"] = None

    return JSONResponse(response)


def include_playground_routes(app: FastAPI) -> None:
    app.include_router(router, prefix="/api")


__all__ = ["include_playground_routes", "router"]
