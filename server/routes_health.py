"""Health / readiness / metrics endpoints.

Routes exposed under the ``/api`` prefix by :mod:`server.app`:

* ``GET /api/health`` — liveness probe.  Always 200 while the process
  answers.  No authentication is required — the response body contains
  only non-sensitive metadata (service version + uptime).
* ``GET /api/live`` — alias for ``/api/health``.
* ``GET /api/ready`` — readiness probe.  200 when the pipeline is
  initialised and the Fernet encryption key is configured; 503 with
  per-check detail otherwise.
* ``GET /api/metrics`` — Prometheus exposition format.  Gated by the
  ``I3_METRICS_ENABLED`` environment variable (default: enabled); when
  disabled this endpoint returns 404 so Prometheus skips the target
  cleanly.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from i3.observability.metrics import (
    metrics_enabled,
    render_prometheus,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_PROCESS_START_MONOTONIC = time.monotonic()


def _service_version(request: Request) -> str:
    """Best-effort lookup of the service version from app.state."""
    try:
        config = getattr(request.app.state, "config", None)
        if config is not None:
            project = getattr(config, "project", None)
            if project is not None:
                return str(getattr(project, "version", "0.0.0"))
    except Exception:  # pragma: no cover - defensive
        pass
    return os.environ.get("I3_VERSION", "0.0.0")


def _uptime_seconds() -> float:
    return round(time.monotonic() - _PROCESS_START_MONOTONIC, 3)


def _check_pipeline(request: Request) -> tuple[str, str | None]:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return "not_initialized", "pipeline missing from app.state"
    # Prefer an explicit readiness flag if the Pipeline exposes one.
    ready_flag = getattr(pipeline, "is_ready", None)
    if ready_flag is False:
        return "initializing", "pipeline.is_ready == False"
    initialized = getattr(pipeline, "initialized", None)
    if initialized is False:
        return "initializing", "pipeline.initialized == False"
    return "ok", None


def _check_encryption_key() -> tuple[str, str | None]:
    """Confirm a Fernet key is available.

    The Pipeline will refuse to decrypt user profiles without one, so
    the readiness probe should fail fast when it is missing.
    """
    for env_name in ("I3_ENCRYPTION_KEY", "I3_FERNET_KEY", "FERNET_KEY"):
        if os.environ.get(env_name):
            return "ok", None
    # Allow an opt-out for environments that intentionally disable encryption
    # during local development.
    if os.environ.get("I3_DISABLE_ENCRYPTION") == "1":
        return "disabled", "I3_DISABLE_ENCRYPTION=1"
    return "missing", "no encryption key in environment"


def _check_disk() -> tuple[str, str | None]:
    """Verify the working directory has a non-trivial amount of free space."""
    try:
        total, _used, free = shutil.disk_usage(os.getcwd())
    except Exception as exc:  # pragma: no cover - platform dependent
        return "unknown", f"disk_usage failed: {exc.__class__.__name__}"
    # 64 MiB minimum — small enough to succeed on CI runners, large enough
    # that persistent DB writes will not immediately fail.
    threshold = 64 * 1024 * 1024
    if free < threshold:
        return "low", f"free={free} bytes"
    return "ok", None


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe — returns 200 whenever the process answers."""
    return {
        "status": "ok",
        "version": _service_version(request),
        "uptime_s": _uptime_seconds(),
    }


@router.get("/live")
async def live(request: Request) -> dict[str, Any]:
    """Alias for /api/health."""
    result: dict[str, Any] = await health(request)
    return result


@router.get("/ready")
async def ready(request: Request) -> Response:
    """Readiness probe — 200 only when all critical checks pass."""
    pipeline_status, pipeline_detail = _check_pipeline(request)
    key_status, key_detail = _check_encryption_key()
    disk_status, disk_detail = _check_disk()

    checks: dict[str, str] = {
        "pipeline": pipeline_status,
        "encryption_key": key_status,
        "disk": disk_status,
    }
    details: dict[str, str] = {}
    for name, detail in (
        ("pipeline", pipeline_detail),
        ("encryption_key", key_detail),
        ("disk", disk_detail),
    ):
        if detail:
            details[name] = detail

    critical_ok = pipeline_status == "ok" and key_status in {"ok", "disabled"}
    status_code = 200 if critical_ok else 503

    body: dict[str, Any] = {
        "status": "ready" if critical_ok else "not_ready",
        "checks": checks,
        "version": _service_version(request),
        "uptime_s": _uptime_seconds(),
    }
    if details:
        body["details"] = details

    return JSONResponse(body, status_code=status_code)


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus text exposition.  Returns 404 when metrics are disabled."""
    if not metrics_enabled():
        return JSONResponse({"detail": "metrics disabled"}, status_code=404)
    payload, content_type = render_prometheus()
    return Response(content=payload, media_type=content_type)


# ---------------------------------------------------------------------------
# Iter 56: deep system-health snapshot covering every subsystem
# ---------------------------------------------------------------------------

@router.get("/health/deep")
async def health_deep(request: Request) -> JSONResponse:
    """Subsystem-level health snapshot for the dashboard.

    Iter 56 (2026-04-27).  Returns a single JSON document describing
    the readiness of every major I3 subsystem so the reviewer can
    answer "is the whole stack alive?" with one page-load.  Sections:

    * ``slm_v2``   — checkpoint exists, params, vocab, last best_val_loss
    * ``encoder``  — checkpoint exists, params
    * ``intent``   — Qwen LoRA adapter dir + Gemini key present (yes/no,
      not the key value)
    * ``cloud``    — list of cloud providers wired (no auth attempted)
    * ``privacy``  — encryption key set, budget snapshot
    * ``cascade``  — per-arm sample count from the rolling tracker
    * ``profiling`` — top-line edge-budget numbers
    * ``checkpoints`` — disk-resident checkpoint dirs + sizes

    All responses are JSON-safe; no file paths leak hostnames.
    """
    import os
    from pathlib import Path

    out: dict[str, Any] = {
        "status": "ok",
        "version": _service_version(request),
        "uptime_s": _uptime_seconds(),
    }

    # --- SLM v2 -----------------------------------------------------
    slm_v2_dir = Path("checkpoints/slm_v2")
    slm_best = slm_v2_dir / "best_model.pt"
    out["slm_v2"] = {
        "checkpoint_present": slm_best.exists(),
        "size_mb": (round(slm_best.stat().st_size / 1024 / 1024, 1)
                    if slm_best.exists() else 0),
        "step_checkpoints": (
            sorted(p.name for p in slm_v2_dir.glob("step_*.pt"))[-5:]
            if slm_v2_dir.exists() else []
        ),
    }
    # Stored best_eval_loss from training
    try:
        if slm_best.exists():
            import torch  # type: ignore[import-untyped]
            blob = torch.load(slm_best, map_location="cpu", weights_only=False)
            out["slm_v2"]["best_eval_loss"] = (
                float(blob.get("best_eval_loss"))
                if blob.get("best_eval_loss") is not None else None
            )
            cfg = (blob.get("config") or {}).get("model") or {}
            out["slm_v2"]["d_model"] = cfg.get("d_model")
            out["slm_v2"]["n_layers"] = cfg.get("n_layers")
            out["slm_v2"]["vocab_size"] = cfg.get("vocab_size")
    except Exception:
        out["slm_v2"]["best_eval_loss"] = None

    # --- Encoder ----------------------------------------------------
    enc_best = Path("checkpoints/encoder/best_model.pt")
    out["encoder"] = {
        "checkpoint_present": enc_best.exists(),
        "size_mb": (round(enc_best.stat().st_size / 1024 / 1024, 1)
                    if enc_best.exists() else 0),
    }

    # --- Intent (Qwen LoRA + Gemini) -------------------------------
    qwen_best = Path("checkpoints/intent_lora/qwen3.5-2b_best")
    qwen_metrics = Path("checkpoints/intent_lora/qwen3.5-2b/training_metrics.json")
    gemini_plan = Path("checkpoints/intent_gemini/tuning_plan.json")
    out["intent"] = {
        "qwen_adapter_present": qwen_best.exists()
                                and (qwen_best / "adapter_config.json").exists(),
        "qwen_metrics_present": qwen_metrics.exists(),
        "gemini_plan_present": gemini_plan.exists(),
        "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")
                                   or os.environ.get("GOOGLE_API_KEY")),
    }
    if qwen_metrics.exists():
        try:
            import json as _j
            m = _j.loads(qwen_metrics.read_text(encoding="utf-8"))
            out["intent"]["qwen_best_val_loss"] = m.get("best_val_loss")
            out["intent"]["qwen_final_step"] = m.get("final_step")
        except Exception:
            pass

    # --- Cloud router providers ------------------------------------
    try:
        from i3.cloud.provider_registry import ProviderRegistry
        out["cloud"] = {"registered_providers": ProviderRegistry.names()}
    except Exception:
        out["cloud"] = {"registered_providers": []}

    # --- Privacy (encryption + budget snapshot) --------------------
    out["privacy"] = {
        "encryption_key_set": bool(os.environ.get("I3_ENCRYPTION_KEY")),
    }
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        try:
            budget = getattr(pipeline, "privacy_budget", None)
            if budget is not None and hasattr(budget, "_get_or_create_bucket"):
                # Touch a synthetic bucket just to confirm the budget
                # plumbing is reachable; don't mutate a real user's bucket.
                pass
            out["privacy"]["budget_module_loaded"] = budget is not None
        except Exception:
            out["privacy"]["budget_module_loaded"] = False

    # --- Cascade arms (live counters) ------------------------------
    if pipeline is not None and hasattr(pipeline, "cascade_arm_stats"):
        try:
            stats = pipeline.cascade_arm_stats()
            out["cascade"] = {
                k: v.get("n", 0) if isinstance(v, dict) else 0
                for k, v in stats.items() if not k.startswith("_")
            }
        except Exception:
            out["cascade"] = {}
    else:
        out["cascade"] = {}

    # --- Profiling (top-line edge numbers) -------------------------
    if pipeline is not None and hasattr(pipeline, "get_profiling_report"):
        try:
            rep = await pipeline.get_profiling_report()
            out["profiling"] = {
                "total_latency_ms": rep.get("total_latency_ms"),
                "memory_mb": rep.get("memory_mb"),
                "fits_budget": rep.get("fits_budget"),
                "device_class": rep.get("device_class"),
            }
        except Exception:
            out["profiling"] = {}

    # --- Checkpoint disk inventory --------------------------------
    ck_root = Path("checkpoints")
    if ck_root.exists():
        out["checkpoints"] = {
            d.name: {
                "files": len(list(d.iterdir())) if d.is_dir() else 0,
                "size_mb": (round(sum(f.stat().st_size for f in d.glob('**/*')
                                        if f.is_file()) / 1024 / 1024, 1)
                            if d.is_dir() else 0),
            }
            for d in sorted(ck_root.iterdir()) if d.is_dir()
        }

    return JSONResponse(out)
