"""Pyroscope continuous-profiling integration for I^3.

This module wires the Grafana Pyroscope Python SDK into the application so
that CPU and memory-allocation profiles are streamed to a Pyroscope server
on a rolling basis. Flame graphs are then queryable at
``https://<pyroscope>/``.

The integration is intentionally **soft**: if the ``pyroscope`` package is
not installed (for instance in unit tests or in the edge deployment), the
module degrades to a no-op so importing it is always safe.

References
----------
- Pyroscope docs (post-Grafana acquisition, 2023): https://grafana.com/docs/pyroscope/latest/
- Python SDK                                     : https://grafana.com/docs/pyroscope/latest/configure-client/language-sdks/python/
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

log = logging.getLogger(__name__)

try:
    import pyroscope  # type: ignore[import-untyped]

    _PYROSCOPE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    pyroscope = None  # type: ignore[assignment]
    _PYROSCOPE_AVAILABLE = False


__all__ = ["configure_pyroscope", "is_available"]


def is_available() -> bool:
    """Return True iff the ``pyroscope`` SDK is importable."""
    return _PYROSCOPE_AVAILABLE


def configure_pyroscope(
    service_name: str = "i3-server",
    server_url: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> bool:
    """Start the Pyroscope profiler.

    Parameters
    ----------
    service_name:
        Logical service identifier shown in the Pyroscope UI.
    server_url:
        Base URL of the Pyroscope server. Falls back to the
        ``PYROSCOPE_SERVER_URL`` environment variable, then to
        ``http://pyroscope:4040``.
    tags:
        Extra key/value labels attached to every profile sample (e.g.
        ``{"env": "production", "region": "eu-west-1"}``).

    Returns
    -------
    bool
        True if profiling started, False if the SDK was absent or the
        configuration call raised (no-op in that case).
    """
    if not _PYROSCOPE_AVAILABLE:
        log.debug(
            "pyroscope package not installed -- continuous profiling disabled. "
            "Install with: pip install pyroscope-io"
        )
        return False

    url = (
        server_url
        or os.environ.get("PYROSCOPE_SERVER_URL")
        or "http://pyroscope:4040"
    )
    merged_tags: dict[str, str] = {
        "env": os.environ.get("I3_ENV", "dev"),
        "service": service_name,
    }
    if tags:
        merged_tags.update(tags)

    try:
        # The SDK exposes memory/allocation profiling via detect_subprocesses
        # and sample_rate. We choose conservative defaults that add ~1% CPU.
        init_kwargs: dict[str, Any] = {
            "application_name": service_name,
            "server_address": url,
            "tags": merged_tags,
            "sample_rate": 100,  # 100 Hz
            "detect_subprocesses": True,
            "oncpu": True,  # CPU profile
            "gil_only": False,
            "enable_logging": False,
        }
        # Memory profiling is opt-in via an env flag -- it is heavier.
        if os.environ.get("PYROSCOPE_ENABLE_ALLOC", "true").lower() == "true":
            init_kwargs["profile_memory"] = True

        pyroscope.configure(**init_kwargs)  # type: ignore[union-attr]
        log.info(
            "pyroscope configured service=%s url=%s tags=%s",
            service_name,
            url,
            merged_tags,
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("failed to configure pyroscope: %s", exc)
        return False
