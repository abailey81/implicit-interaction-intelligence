"""REST endpoints for the Benchmarks UI tab.

* ``GET  /api/benchmarks/latest``         → latest report dict
* ``GET  /api/benchmarks/svg/{plot_name}`` → latest SVG plot
* ``POST /api/benchmarks/run``             → kicks off a fresh run in
                                            the background (202 Accepted)

The tab loads ``latest`` on entry and renders the four embedded SVGs.
The Run button spawns a background task; the UI polls ``latest`` until
the timestamp changes.

Security
--------
These endpoints are read-only (except for ``run`` which only writes to
``reports/benchmarks/``).  ``run`` is guarded with a 60-second cooldown
so a malicious caller cannot force back-to-back full benchmark passes
in a tight loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPORT_DIR = _REPO_ROOT / "reports" / "benchmarks"
_ALLOWED_PLOTS = {
    "latency_breakdown.svg",
    "perplexity_curve.svg",
    "coherence_categories.svg",
    "adaptation_faithfulness.svg",
}

# Simple in-process cooldown — last run timestamp.
_last_run_started_at: float = 0.0
_RUN_COOLDOWN_SECONDS: float = 60.0
_run_in_progress: bool = False


@router.get("/latest")
async def latest() -> JSONResponse:
    """Return the most recent benchmark report dict (or 404 marker)."""
    path = _REPORT_DIR / "latest.json"
    if not path.exists():
        return JSONResponse(
            {
                "status": "not_run",
                "detail": (
                    "No benchmarks have been run on this server. "
                    "POST /api/benchmarks/run to populate."
                ),
            },
            status_code=200,
        )
    try:
        import json
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.exception("Failed to read benchmark report")
        raise HTTPException(status_code=500, detail="Report unreadable")


@router.get("/svg/{plot_name}")
async def svg(plot_name: str) -> Response:
    """Serve the latest SVG plot.

    SEC: ``plot_name`` is matched against an explicit allow-list so a
    malicious caller cannot path-traverse out of ``reports/benchmarks/``.
    """
    if plot_name not in _ALLOWED_PLOTS:
        raise HTTPException(status_code=404, detail="Unknown plot")
    path = _REPORT_DIR / plot_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not yet generated")
    try:
        body = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read SVG plot")
        raise HTTPException(status_code=500, detail="Plot unreadable")
    return Response(content=body, media_type="image/svg+xml")


@router.post("/run")
async def run() -> JSONResponse:
    """Kick off a fresh benchmark run in the background.

    Returns 202 with an estimated runtime; the UI polls ``/latest``
    until the timestamp changes.  Cooldown-throttled to one run per
    60 seconds.
    """
    global _last_run_started_at, _run_in_progress
    now = time.time()
    if _run_in_progress:
        return JSONResponse(
            {"status": "already_running"},
            status_code=409,
        )
    if now - _last_run_started_at < _RUN_COOLDOWN_SECONDS:
        retry = int(_RUN_COOLDOWN_SECONDS - (now - _last_run_started_at))
        return JSONResponse(
            {"status": "cooldown", "retry_after": retry},
            status_code=429,
        )
    _last_run_started_at = now
    _run_in_progress = True

    async def _run_in_thread() -> None:
        global _run_in_progress
        try:
            loop = asyncio.get_event_loop()
            from i3.benchmarks.runner import BenchmarkRunner
            runner = BenchmarkRunner(
                server_url=None,  # in-process to avoid recursive WS calls
                n_latency_prompts=20,
            )
            await loop.run_in_executor(None, runner.run_all)
        except Exception:
            logger.exception("Background benchmark run failed")
        finally:
            _run_in_progress = False

    asyncio.create_task(_run_in_thread())
    return JSONResponse(
        {
            "status": "started",
            "estimated_seconds": 60,
            "message": "Poll /api/benchmarks/latest until timestamp changes.",
        },
        status_code=202,
    )


def include_benchmark_routes(app: FastAPI) -> None:
    app.include_router(router, prefix="/api")


__all__ = ["include_benchmark_routes", "router"]
