"""Admin endpoints. Disable in production via `I3_DISABLE_ADMIN=1`.

These endpoints exist only to support live-demo operations (reset, seed,
one-shot edge profiling runs, GDPR-style export / erase). They are gated
on a bearer token supplied via the ``I3_ADMIN_TOKEN`` environment variable
so a casually-exposed port cannot wipe user state.

Security model
~~~~~~~~~~~~~~
* The entire router is only mounted when ``I3_DISABLE_ADMIN`` is NOT
  ``"1"`` — operators can therefore hard-disable admin traffic in
  production without touching code.
* Every route depends on :func:`require_admin_token`, which compares
  the client-supplied ``Authorization: Bearer <token>`` header against
  ``os.environ["I3_ADMIN_TOKEN"]`` using :func:`secrets.compare_digest`
  (constant-time). The token is never echoed back.
* Responses carry only aggregate counts, never raw user text.
* Embeddings exported via :func:`admin_export` are base64-encoded so a
  human operator can ship the file as a single JSON document.

See ``docs/operations/`` for the runbook. The rationale for the export /
erase parity is GDPR right-to-export / right-to-erase.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Path as FPath, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# SEC: identical to the REST layer's user-id regex so admin traffic cannot
# smuggle path-traversal payloads or newline-bearing ids into downstream
# pipeline calls. Anchored start-to-end; forbids ``/``, ``.``, whitespace.
_USER_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"

_ADMIN_TOKEN_ENV = "I3_ADMIN_TOKEN"
_DISABLE_ENV = "I3_DISABLE_ADMIN"


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------


async def require_admin_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Validate the ``Authorization: Bearer <token>`` header.

    The admin token must be configured via the ``I3_ADMIN_TOKEN`` env var.
    Missing or mismatched credentials return ``401 Unauthorized`` with a
    constant-string detail; a missing server-side token is treated as
    ``missing_admin_token`` so misconfiguration does not silently expose
    the endpoints.

    Args:
        authorization: Raw ``Authorization`` header value, injected by
            FastAPI's :class:`Header` dependency.

    Raises:
        HTTPException: With status 401 when authentication fails.
    """
    expected = os.environ.get(_ADMIN_TOKEN_ENV, "")
    if not expected:
        logger.warning(
            "admin_auth.missing_server_token", extra={"event": "admin_auth"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # SEC: constant-time comparison to avoid timing oracles on the token.
    presented = authorization.split(" ", 1)[1].strip()
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_pipeline(request: Request) -> Any:
    """Return the initialised Pipeline or raise 503.

    Raises:
        HTTPException: With status 503 when the pipeline has not been
            attached to ``app.state``.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        )
    return pipeline


def _admin_disabled() -> bool:
    """Return ``True`` when ``I3_DISABLE_ADMIN=1`` is set in the environment."""
    return os.environ.get(_DISABLE_ENV, "").strip() == "1"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/reset", dependencies=[Depends(require_admin_token)])
async def admin_reset(request: Request) -> JSONResponse:
    """Reset the ``demo_user`` state.

    Wipes the persisted diary and user profile, resets the bandit
    posterior to its prior, and flushes in-memory engagement tracking.
    Idempotent — safe to call many times.

    Returns:
        JSON body with a ``status`` field and counts of the artefacts
        removed. No raw user data is ever echoed back.
    """
    pipeline = _get_pipeline(request)
    user_id = "demo_user"
    summary: dict[str, Any] = {"user_id": user_id, "status": "reset"}

    # 1. Clear in-memory per-user state --------------------------------
    try:
        pipeline.user_models.pop(user_id, None)
        pipeline._last_response_time.pop(user_id, None)
        pipeline._last_response_length.pop(user_id, None)
        pipeline._previous_engagement.pop(user_id, None)
        pipeline._previous_route.pop(user_id, None)
    except AttributeError:
        # Pipeline shape is stable; fall through without blocking the reset.
        logger.debug("admin_reset: pipeline missing in-memory maps")

    # 2. Reset the bandit posterior -----------------------------------
    try:
        reset_fn = getattr(pipeline.router, "reset", None)
        if callable(reset_fn):
            reset_fn()
            summary["bandit"] = "reset"
    except (AttributeError, RuntimeError) as exc:
        logger.warning(
            "admin_reset.bandit_reset_failed",
            extra={"event": "admin_reset", "err": type(exc).__name__},
        )

    # 3. Wipe persisted diary + user profile --------------------------
    diary_deleted = 0
    try:
        sessions = await pipeline.diary_store.get_user_sessions(user_id, limit=10_000)
        diary_deleted = len(sessions)
        if diary_deleted > 0:
            await pipeline.diary_store.prune_old_entries(user_id, max_entries=0)
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "admin_reset.diary_prune_failed",
            extra={"event": "admin_reset", "err": type(exc).__name__},
        )
    summary["diary_sessions_removed"] = diary_deleted

    profile_removed = False
    try:
        from i3.user_model.store import UserModelStore  # noqa: PLC0415

        db_path = getattr(pipeline.config.user_model, "db_path", "data/user_model.db")
        async with UserModelStore(db_path) as store:
            existing = await store.load_profile(user_id)
            if existing is not None:
                await store.delete_profile(user_id)
                profile_removed = True
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning(
            "admin_reset.profile_delete_failed",
            extra={"event": "admin_reset", "err": type(exc).__name__},
        )
    summary["profile_removed"] = profile_removed

    logger.info(
        "admin.reset",
        extra={
            "event": "admin_reset",
            "user_id": user_id,
            "diary_sessions_removed": diary_deleted,
            "profile_removed": profile_removed,
        },
    )
    return JSONResponse(summary)


@router.post("/profiling", dependencies=[Depends(require_admin_token)])
async def admin_profiling(request: Request) -> JSONResponse:
    """Trigger a one-shot edge profile run and return the JSON summary.

    Prefers the pipeline's own ``get_profiling_report`` coroutine when
    available; falls back to running :class:`EdgeProfiler` against any
    loaded encoder so the endpoint is useful even without a baked-in
    report.

    Returns:
        The profiling report as JSON. Never includes host paths or
        environment snapshots — only model-level metrics.
    """
    pipeline = _get_pipeline(request)

    # Fast path: pipeline exposes a native report coroutine.
    get_report = getattr(pipeline, "get_profiling_report", None)
    if callable(get_report):
        try:
            report = await get_report()
            if isinstance(report, dict):
                logger.info(
                    "admin.profiling.pipeline_report",
                    extra={"event": "admin_profiling", "mode": "pipeline"},
                )
                return JSONResponse(report)
        except (RuntimeError, OSError) as exc:
            logger.warning(
                "admin_profiling.pipeline_report_failed",
                extra={"event": "admin_profiling", "err": type(exc).__name__},
            )

    # Slow path: run EdgeProfiler against the loaded encoder, if any.
    encoder = getattr(pipeline, "_encoder", None)
    if encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No profiling target available",
        )

    try:
        import torch  # noqa: PLC0415

        from i3.profiling.report import EdgeProfiler  # noqa: PLC0415
    except ImportError as exc:
        logger.exception("admin_profiling.import_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profiling module unavailable",
        ) from exc

    try:
        profiler = EdgeProfiler()
        module = getattr(encoder, "model", encoder)
        input_dim = getattr(pipeline.config.encoder, "input_dim", 32)
        window = getattr(pipeline.config.interaction, "feature_window", 10)
        sample = torch.zeros(1, window, input_dim)
        report = profiler.profile_model(
            model=module, model_name="TCN Encoder", input_sample=sample
        )
        payload = report.to_dict()
    except (RuntimeError, ValueError) as exc:
        logger.exception("admin_profiling.run_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profiling failed",
        ) from exc

    logger.info(
        "admin.profiling.oneshot",
        extra={"event": "admin_profiling", "mode": "oneshot"},
    )
    return JSONResponse(payload)


@router.post("/seed", dependencies=[Depends(require_admin_token)])
async def admin_seed(request: Request) -> JSONResponse:
    """Re-run the demo pre-seed from :mod:`demo.pre_seed`.

    Imports lazily so the admin router does not drag the pre-seed helpers
    into every process that mounts the server.

    Returns:
        The dict returned by :func:`demo.pre_seed.seed_pipeline`.
    """
    pipeline = _get_pipeline(request)

    try:
        from demo.pre_seed import seed_pipeline  # noqa: PLC0415
    except ImportError as exc:
        logger.exception("admin_seed.import_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Seed module unavailable",
        ) from exc

    try:
        result = await seed_pipeline(pipeline, user_id="demo_user")
    except (RuntimeError, OSError, ValueError) as exc:
        logger.exception("admin_seed.run_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Seed failed",
        ) from exc

    logger.info(
        "admin.seed",
        extra={"event": "admin_seed", "user_id": "demo_user"},
    )
    return JSONResponse(
        {
            "status": "seeded",
            "user_id": "demo_user",
            "diary_sessions": int(result.get("diary_sessions", 0)),
            "diary_entries": int(result.get("diary_entries", 0)),
            "bandit_updates": int(result.get("bandit_updates", 0)),
        }
    )


@router.get(
    "/export/{user_id}",
    dependencies=[Depends(require_admin_token)],
)
async def admin_export(
    request: Request,
    user_id: str = FPath(
        ...,
        pattern=_USER_ID_PATTERN,
        min_length=1,
        max_length=64,
        description="Alphanumeric user identifier (1-64 chars).",
    ),
) -> JSONResponse:
    """GDPR-style export of a user's persisted data.

    Dumps the user's profile (embedding base64-encoded), diary metadata,
    and bandit-arm aggregate stats. **No raw text is included** because
    none is ever persisted — this is a structural guarantee of the
    pipeline, not a filtering choice.

    Args:
        user_id: Validated user identifier (alphanumeric, dash,
            underscore; 1-64 chars).

    Returns:
        JSON body with ``user_id``, ``profile``, ``diary``, and
        ``bandit_stats``.
    """
    pipeline = _get_pipeline(request)

    profile_payload: dict[str, Any] | None = None
    try:
        from i3.user_model.store import UserModelStore  # noqa: PLC0415

        db_path = getattr(pipeline.config.user_model, "db_path", "data/user_model.db")
        async with UserModelStore(db_path) as store:
            profile = await store.load_profile(user_id)
            if profile is not None:
                embedding_b64: str | None = None
                if profile.baseline_embedding is not None:
                    raw = profile.baseline_embedding.detach().cpu().numpy().tobytes()
                    embedding_b64 = base64.b64encode(raw).decode("ascii")
                profile_payload = {
                    "user_id": profile.user_id,
                    "baseline_embedding_b64": embedding_b64,
                    "baseline_embedding_dim": (
                        int(profile.baseline_embedding.numel())
                        if profile.baseline_embedding is not None
                        else None
                    ),
                    "baseline_features_mean": profile.baseline_features_mean,
                    "baseline_features_std": profile.baseline_features_std,
                    "total_sessions": profile.total_sessions,
                    "total_messages": profile.total_messages,
                    "relationship_strength": profile.relationship_strength,
                    "long_term_style": profile.long_term_style,
                    "baseline_established": profile.baseline_established,
                    "created_at": profile.created_at.isoformat(),
                    "updated_at": profile.updated_at.isoformat(),
                }
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning(
            "admin_export.profile_failed",
            extra={"event": "admin_export", "err": type(exc).__name__},
        )

    diary_payload: list[dict[str, Any]] = []
    try:
        sessions = await pipeline.diary_store.get_user_sessions(user_id, limit=1_000)
        for session in sessions:
            exchanges = await pipeline.diary_store.get_session_exchanges(
                session["session_id"]
            )
            safe_exchanges: list[dict[str, Any]] = []
            for ex in exchanges:
                embedding_b64 = None
                blob = ex.get("user_state_embedding")
                if isinstance(blob, (bytes, bytearray)):
                    embedding_b64 = base64.b64encode(bytes(blob)).decode("ascii")
                safe_exchanges.append(
                    {
                        "exchange_id": ex.get("exchange_id"),
                        "timestamp": ex.get("timestamp"),
                        "route_chosen": ex.get("route_chosen"),
                        "response_latency_ms": ex.get("response_latency_ms"),
                        "engagement_signal": ex.get("engagement_signal"),
                        "topics": ex.get("topics"),
                        "adaptation_vector": ex.get("adaptation_vector"),
                        "user_state_embedding_b64": embedding_b64,
                    }
                )
            diary_payload.append(
                {
                    "session_id": session.get("session_id"),
                    "start_time": session.get("start_time"),
                    "end_time": session.get("end_time"),
                    "message_count": session.get("message_count"),
                    "summary": session.get("summary"),
                    "dominant_emotion": session.get("dominant_emotion"),
                    "topics": session.get("topics"),
                    "mean_engagement": session.get("mean_engagement"),
                    "mean_cognitive_load": session.get("mean_cognitive_load"),
                    "mean_accessibility": session.get("mean_accessibility"),
                    "relationship_strength": session.get("relationship_strength"),
                    "exchanges": safe_exchanges,
                }
            )
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "admin_export.diary_failed",
            extra={"event": "admin_export", "err": type(exc).__name__},
        )

    bandit_stats: dict[str, Any] = {}
    try:
        get_stats = getattr(pipeline.router, "get_arm_stats", None)
        if callable(get_stats):
            bandit_stats = get_stats()
    except (AttributeError, RuntimeError):
        logger.debug("admin_export.bandit_stats_unavailable")

    # SEC (L-4, 2026-04-23 audit): when every signal is empty the user
    # genuinely does not exist in any backing store — return 404 rather
    # than an empty 200 shape so admin-token enumeration is not easier
    # than it needs to be.  The log still records ``profile_present``
    # for the present case; for the absent case we emit a dedicated
    # ``admin.export.not_found`` event.
    if profile_payload is None and not diary_payload and not bandit_stats:
        logger.info(
            "admin.export.not_found",
            extra={"event": "admin_export_nf", "user_id": user_id},
        )
        raise HTTPException(status_code=404, detail="Not found")

    logger.info(
        "admin.export",
        extra={
            "event": "admin_export",
            "user_id": user_id,
            "diary_sessions": len(diary_payload),
            "profile_present": profile_payload is not None,
        },
    )
    return JSONResponse(
        {
            "user_id": user_id,
            "profile": profile_payload,
            "diary": diary_payload,
            "bandit_stats": bandit_stats,
            "privacy_note": "Raw user text is never persisted; not in this export.",
        }
    )


@router.delete(
    "/user/{user_id}",
    dependencies=[Depends(require_admin_token)],
)
async def admin_delete_user(
    request: Request,
    user_id: str = FPath(
        ...,
        pattern=_USER_ID_PATTERN,
        min_length=1,
        max_length=64,
        description="Alphanumeric user identifier (1-64 chars).",
    ),
) -> JSONResponse:
    """GDPR-style deletion of a user's persisted data.

    Removes the profile row and all diary sessions + exchanges. Also
    purges in-memory engagement tracking so a running server does not
    resurrect the identifier on the next message.

    Args:
        user_id: Validated user identifier.

    Returns:
        JSON body summarising how many artefacts were removed.
    """
    pipeline = _get_pipeline(request)

    # In-memory purge -------------------------------------------------
    try:
        pipeline.user_models.pop(user_id, None)
        pipeline._last_response_time.pop(user_id, None)
        pipeline._last_response_length.pop(user_id, None)
        pipeline._previous_engagement.pop(user_id, None)
        pipeline._previous_route.pop(user_id, None)
    except AttributeError:
        logger.debug("admin_delete_user: pipeline missing in-memory maps")

    # Diary purge (sessions + exchanges cascade) ----------------------
    diary_removed = 0
    try:
        existing = await pipeline.diary_store.get_user_sessions(user_id, limit=10_000)
        diary_removed = len(existing)
        if diary_removed > 0:
            await pipeline.diary_store.prune_old_entries(user_id, max_entries=0)
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "admin_delete_user.diary_failed",
            extra={"event": "admin_delete_user", "err": type(exc).__name__},
        )

    # Profile purge ---------------------------------------------------
    profile_removed = False
    try:
        from i3.user_model.store import UserModelStore  # noqa: PLC0415

        db_path = getattr(pipeline.config.user_model, "db_path", "data/user_model.db")
        async with UserModelStore(db_path) as store:
            existing_profile = await store.load_profile(user_id)
            if existing_profile is not None:
                await store.delete_profile(user_id)
                profile_removed = True
    except (ImportError, RuntimeError, OSError) as exc:
        logger.warning(
            "admin_delete_user.profile_failed",
            extra={"event": "admin_delete_user", "err": type(exc).__name__},
        )

    logger.info(
        "admin.delete_user",
        extra={
            "event": "admin_delete_user",
            "user_id": user_id,
            "diary_sessions_removed": diary_removed,
            "profile_removed": profile_removed,
        },
    )
    return JSONResponse(
        {
            "user_id": user_id,
            "status": "deleted",
            "diary_sessions_removed": diary_removed,
            "profile_removed": profile_removed,
        }
    )


# ---------------------------------------------------------------------------
# Mount helper
# ---------------------------------------------------------------------------


def include_admin_routes(app: FastAPI) -> None:
    """Conditionally mount the admin router on *app*.

    The router is attached only when ``I3_DISABLE_ADMIN`` is NOT set to
    ``"1"``. This is a belt-and-braces guard: even if the container ships
    with the module imported, an operator can disable the whole surface
    area with a single environment flag.

    Args:
        app: The FastAPI application produced by :func:`server.app.create_app`.
    """
    # SEC: honour the env flag first so a disabled server never exposes /admin
    # paths under any circumstance, including behind a trusted reverse proxy.
    if _admin_disabled():
        logger.info(
            "admin.router.disabled",
            extra={"event": "admin_mount", "reason": "I3_DISABLE_ADMIN=1"},
        )
        return

    # Warn loudly when the token is missing — the dependency still rejects
    # every request, but the operator should see this at startup time.
    if not os.environ.get(_ADMIN_TOKEN_ENV):
        logger.warning(
            "admin.router.no_token",
            extra={
                "event": "admin_mount",
                "note": "I3_ADMIN_TOKEN unset; endpoints will always return 401.",
            },
        )

    app.include_router(router)
    logger.info(
        "admin.router.mounted",
        extra={"event": "admin_mount", "prefix": router.prefix},
    )


# ``reports`` directory is used by the offline GDPR export CLI. Create it
# lazily here so tools that import this module have a stable target path.
_REPORTS_DIR = Path("reports")
