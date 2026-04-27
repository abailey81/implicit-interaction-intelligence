"""Edge-profile endpoint — serves ``reports/edge_profile.json`` to the UI.

Exposes ``GET /api/edge/profile`` so the ``Edge`` tab in the SPA can
render the concrete on-device measurements (param counts, int8 size,
latency, peak RSS) produced by :class:`i3.edge.profiler.EdgeProfiler`.

The endpoint reads the cached JSON from disk; it *does not*
re-measure on every request because a full profile takes ~10 s even
on CPU and would block a request worker.  A caller can force a fresh
measurement by passing ``?run_now=true`` — intended for operator use,
not for the public web UI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# SEC: the router uses an explicit prefix so the endpoint is
# auto-namespaced under /api/edge/* and cannot collide with the
# static-files catch-all mount at ``/``.
router = APIRouter(prefix="/api/edge", tags=["edge"])


_REPORT_PATH = Path("reports/edge_profile.json")


def _load_cached_report() -> dict | None:
    """Return the cached edge profile, or ``None`` if missing / corrupt."""
    if not _REPORT_PATH.is_file():
        return None
    try:
        return json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("edge_profile.json unreadable: %s", exc)
        return None


@router.get("/profile")
async def get_edge_profile(
    run_now: bool = Query(
        False,
        description=(
            "Run the profiler synchronously instead of reading the cached "
            "report. Takes ~10 s on CPU — do NOT enable from the public UI."
        ),
    ),
) -> JSONResponse:
    """Return the latest edge-deployment profile.

    Response shape matches :class:`i3.edge.profiler.EdgeReport` (see
    ``reports/edge_profile.json``).  404 is returned when no profile
    has been measured yet, with a hint to run the measurement script.
    """
    if run_now:
        # Run the profiler synchronously.  Operator-only: the
        # measurement loads both models into memory and decodes 100
        # prompts, so it is not something we want to trigger from the
        # public web UI.  The route is still guarded by the global
        # rate limiter.
        from i3.edge.profiler import EdgeProfiler, _default_paths

        slm, tcn, tok = _default_paths()
        try:
            profiler = EdgeProfiler(slm, tcn, tok, device="cpu")
            data = profiler.measure()
            return JSONResponse(data)
        except FileNotFoundError as exc:
            logger.exception("Edge profile run_now failed — missing checkpoint")
            raise HTTPException(
                status_code=503,
                detail=f"Checkpoint missing: {exc!s}",
            ) from exc
        except Exception:  # pragma: no cover - defensive
            logger.exception("Edge profile run_now failed")
            raise HTTPException(
                status_code=500,
                detail="Edge profile measurement failed",
            )

    data = _load_cached_report()
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No edge profile yet - run scripts/measure_edge.py",
        )
    return JSONResponse(data)


def include_edge_routes(app: FastAPI) -> None:
    """Mount the edge profile router onto ``app``.

    Mirrors the include-pattern used by ``routes_tts.include_tts_routes``
    and ``routes_explain.include_explain_routes``.
    """
    app.include_router(router)


__all__ = ["include_edge_routes", "router"]
