"""One-shot bootstrap for the I3 observability stack.

Designed to be called *once* from ``server.app.create_app`` after the
middleware stack has been installed::

    from i3.observability.instrumentation import setup_observability
    setup_observability(config, app)

The function is idempotent and **never** raises on missing optional
dependencies — every subsystem (logging, tracing, metrics, Sentry) has
its own soft-fail path, and a single warning is logged summarising
which subsystems are active.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BOOTSTRAPPED: bool = False


def _service_version(config: Any) -> str:
    """Extract service version from the I3 config, with a sane default."""
    try:
        project = getattr(config, "project", None)
        if project is not None:
            version = getattr(project, "version", None)
            if version:
                return str(version)
    except Exception:
        pass
    return os.environ.get("I3_VERSION", "0.0.0")


def _load_observability_config() -> dict[str, Any]:
    """Load ``configs/observability.yaml`` if present.

    The file is optional.  Environment variables always win over file
    values — the file is a convenience for operators who want to commit
    a default profile.
    """
    path_env = os.environ.get("I3_OBSERVABILITY_CONFIG")
    candidates = []
    if path_env:
        candidates.append(Path(path_env))
    candidates.append(Path("configs/observability.yaml"))

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            import yaml

            with candidate.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                _apply_env_defaults(data)
                return data
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to load %s: %s", candidate, exc)
    return {}


def _apply_env_defaults(cfg: dict[str, Any]) -> None:
    """Promote file values to env vars *only if* env var is unset."""
    mapping = {
        ("logging", "level"): "I3_LOG_LEVEL",
        ("logging", "format"): "I3_LOG_FORMAT",
        ("tracing", "endpoint"): "OTEL_EXPORTER_OTLP_ENDPOINT",
        ("tracing", "sample_ratio"): "OTEL_TRACES_SAMPLER_ARG",
        ("tracing", "service_name"): "OTEL_SERVICE_NAME",
        ("tracing", "environment"): "OTEL_DEPLOYMENT_ENVIRONMENT",
        ("metrics", "enabled"): "I3_METRICS_ENABLED",
        ("sentry", "dsn"): "SENTRY_DSN",
        ("sentry", "traces_sample_rate"): "SENTRY_TRACES_SAMPLE_RATE",
        ("sentry", "environment"): "SENTRY_ENVIRONMENT",
    }
    for (section, key), env_name in mapping.items():
        if os.environ.get(env_name):
            continue
        section_obj = cfg.get(section)
        if not isinstance(section_obj, dict):
            continue
        value = section_obj.get(key)
        if value is None:
            continue
        os.environ[env_name] = str(value)


def setup_observability(config: Any, app: Any | None = None) -> None:
    """Configure logging, tracing, metrics, and Sentry.  Idempotent.

    Parameters
    ----------
    config:
        The loaded ``i3.config.Config`` (used only for ``project.version``
        at the moment; kept as a parameter so future knobs can be wired
        in without changing callers).
    app:
        Optional FastAPI application.  When supplied, the request-
        correlation middleware and auto-instrumentation are installed.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        if app is not None:
            # If a fresh app is registered after the first bootstrap
            # (common in tests), still attach the middleware.
            _install_middleware(app)
            _instrument_app(app)
        return

    # 1. File-based observability config -> env vars (non-destructive).
    _load_observability_config()

    version = _service_version(config)

    # 2. Structured logging first so subsequent subsystems log as JSON.
    try:
        from i3.observability.logging import configure_logging

        configure_logging()
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger().warning("structured logging init failed: %s", exc)

    # 3. Tracing.
    try:
        from i3.observability.tracing import configure_tracing, instrument_libraries

        configure_tracing(service_version=version)
        instrument_libraries(app=app)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("tracing init failed: %s", exc)

    # 4. Metrics (OTel meter provider + ensure Prometheus registry alive).
    try:
        from i3.observability.metrics import configure_otel_metrics

        configure_otel_metrics(service_version=version)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("metrics init failed: %s", exc)

    # 5. Sentry (optional, gated on SENTRY_DSN).
    try:
        from i3.observability.sentry import configure_sentry

        configure_sentry(service_version=version)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("sentry init failed: %s", exc)

    # 6. Middleware + health/metrics routes (only when an app is supplied).
    if app is not None:
        _install_middleware(app)
        _instrument_app(app)
        _install_health_router(app)

    _BOOTSTRAPPED = True
    logger.info(
        "observability bootstrap complete",
        extra={"service_version": version},
    )


def _install_middleware(app: Any) -> None:
    try:
        from i3.observability.middleware import install

        install(app)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("observability middleware install failed: %s", exc)


def _instrument_app(app: Any) -> None:
    try:
        from i3.observability.tracing import instrument_libraries

        instrument_libraries(app=app)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("re-instrument on app attach failed: %s", exc)


def _install_health_router(app: Any) -> None:
    """Mount /api/health, /api/ready, /api/live, /api/metrics if not already."""
    try:
        from server.routes_health import router as health_router
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("health router unavailable: %s", exc)
        return

    # Avoid double-registering when create_app has already included it.
    existing = {getattr(r, "path", None) for r in getattr(app, "routes", [])}
    health_paths = {"/api/health", "/api/live", "/api/ready", "/api/metrics"}
    if health_paths & existing:
        return
    try:
        app.include_router(health_router, prefix="/api")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("health router include failed: %s", exc)
