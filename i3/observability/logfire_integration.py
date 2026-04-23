"""Pydantic Logfire bootstrap for I3.

`Pydantic Logfire <https://logfire.pydantic.dev>`_ is an OpenTelemetry-
native observability platform from the Pydantic team.  Because it
speaks OTLP natively, traces emitted here are *also* consumed by any
OTel-compatible backend (Grafana Tempo, Honeycomb, Jaeger, SigNoz…),
so adopting Logfire does not create lock-in.

This module is deliberately thin: it exposes a single
:func:`configure_logfire` that is a **no-op when ``LOGFIRE_TOKEN`` is
absent**, instruments FastAPI / HTTPX / Pydantic only if Logfire is
actually active, and never raises on optional features.

References:
    * Pydantic Logfire docs (2025-01):
      https://logfire.pydantic.dev/docs/
    * OTel-native auto-instrumentation discussion:
      https://logfire.pydantic.dev/docs/integrations/

Install hint::

    pip install "logfire[fastapi,httpx,pydantic]>=0.55"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    import logfire  # type: ignore[import-not-found]

    _LOGFIRE_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    logfire = None  # type: ignore[assignment]
    _LOGFIRE_AVAILABLE = False


def is_available() -> bool:
    """Return ``True`` iff the ``logfire`` package is importable."""
    return _LOGFIRE_AVAILABLE


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass
class LogfireStatus:
    """Outcome of a :func:`configure_logfire` call.

    Attributes:
        enabled: ``True`` iff Logfire was actually initialised.
        reason: Human-readable explanation when *enabled* is ``False``.
        instrumented: Names of integrations successfully instrumented.
    """

    enabled: bool
    reason: str
    instrumented: list[str]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def configure_logfire(
    *,
    service_name: str = "i3",
    environment: str = "dev",
    fastapi_app: Any | None = None,
    httpx_client: Any | None = None,
    instrument_pydantic: bool = True,
) -> LogfireStatus:
    """Initialise Pydantic Logfire.

    When ``LOGFIRE_TOKEN`` is not set this function returns silently
    with ``enabled=False`` — the rest of the pipeline therefore remains
    functional on developer laptops or CI runners that do not have the
    Logfire credential.

    Args:
        service_name: ``service.name`` resource attribute.
        environment: ``deployment.environment`` resource attribute.
        fastapi_app: Optional FastAPI instance to instrument.
        httpx_client: Optional ``httpx.Client`` or ``httpx.AsyncClient``
            instance (or class) to instrument.
        instrument_pydantic: When ``True``, auto-instrument every
            :class:`pydantic.BaseModel` validation call.

    Returns:
        A :class:`LogfireStatus` describing what was activated.
    """
    if not _LOGFIRE_AVAILABLE:
        return LogfireStatus(
            enabled=False,
            reason="logfire package not installed",
            instrumented=[],
        )

    token = os.environ.get("LOGFIRE_TOKEN", "").strip()
    if not token:
        return LogfireStatus(
            enabled=False,
            reason="LOGFIRE_TOKEN not set",
            instrumented=[],
        )

    assert logfire is not None
    try:
        logfire.configure(
            service_name=service_name,
            environment=environment,
            token=token,
            send_to_logfire=True,
        )
    except Exception as exc:
        logger.warning("Logfire configure() failed: %s", exc)
        return LogfireStatus(
            enabled=False,
            reason=f"configure_failed:{type(exc).__name__}",
            instrumented=[],
        )

    instrumented: list[str] = []

    if fastapi_app is not None:
        try:
            logfire.instrument_fastapi(fastapi_app)
            instrumented.append("fastapi")
        except Exception as exc:
            logger.debug("Logfire FastAPI instrumentation skipped: %s", exc)

    if httpx_client is not None:
        try:
            logfire.instrument_httpx(httpx_client)
            instrumented.append("httpx")
        except Exception as exc:
            logger.debug("Logfire httpx instrumentation skipped: %s", exc)

    if instrument_pydantic:
        try:
            logfire.instrument_pydantic()
            instrumented.append("pydantic")
        except Exception as exc:
            logger.debug("Logfire Pydantic instrumentation skipped: %s", exc)

    logger.info(
        "Logfire enabled (service=%s, env=%s, instrumented=%s)",
        service_name,
        environment,
        instrumented,
    )
    return LogfireStatus(
        enabled=True,
        reason="ok",
        instrumented=instrumented,
    )


__all__ = [
    "LogfireStatus",
    "configure_logfire",
    "is_available",
]
