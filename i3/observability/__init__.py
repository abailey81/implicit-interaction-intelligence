"""Observability stack for Implicit Interaction Intelligence (I3).

Public surface:

* :func:`setup_observability` — one-shot bootstrap (call from
  ``server.app.create_app``).  Safe to call even if none of the optional
  dependencies are installed — each subsystem degrades gracefully.
* :func:`get_logger` — structlog-aware logger factory.
* Metrics singletons exposed in :mod:`i3.observability.metrics`.
* Context helpers in :mod:`i3.observability.context`.

All optional dependencies (``structlog``, ``opentelemetry-*``,
``prometheus_client``, ``sentry_sdk``) are **soft-imported**.  If a package
is missing, the corresponding feature becomes a no-op and a single warning
is emitted via the stdlib logger.  The main application must never crash
because of missing observability deps.
"""

from __future__ import annotations

from i3.observability.instrumentation import setup_observability
from i3.observability.logging import get_logger

__all__ = ["get_logger", "setup_observability"]
