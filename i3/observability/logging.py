"""Structured logging for I3 (structlog + stdlib bridge).

Features
--------
* JSON output by default; ``I3_LOG_FORMAT=console`` switches to a
  developer-friendly colourised renderer.
* ``I3_LOG_LEVEL`` controls the root log level (default ``INFO``).
* A processor strips known-sensitive keys from any log record before it
  is serialised.  The list is conservative and can be extended via the
  ``I3_LOG_EXTRA_REDACTIONS`` environment variable (comma-separated).
* Every log record includes ISO8601 timestamp, level, logger name,
  module / function / line, the event string, and any structured kwargs.
* Stdlib ``logging`` is routed through structlog so existing
  ``logger.info(...)`` calls across the codebase emit JSON.
* ``contextvars`` (see :mod:`i3.observability.context`) are merged into
  every record, providing request_id / trace_id correlation for free.
* If ``structlog`` is not installed, :func:`configure_logging` falls
  back to a plain JSON formatter built from the stdlib so the server
  still starts.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "set-cookie",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "fernet",
        "fernet_key",
        "encryption",
        "encryption_key",
        "i3_encryption_key",
        "x-api-key",
        "anthropic_api_key",
    }
)

_REDACTED = "***REDACTED***"

_CONFIGURED: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extra_redactions() -> frozenset[str]:
    raw = os.environ.get("I3_LOG_EXTRA_REDACTIONS", "")
    if not raw:
        return frozenset()
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def _should_redact(key: str, extras: Iterable[str]) -> bool:
    lower = key.lower()
    if lower in _SENSITIVE_KEYS:
        return True
    if lower in extras:
        return True
    # Substring match for obvious markers (e.g. "session_token", "db_password").
    for needle in ("password", "secret", "token", "api_key", "authorization"):
        if needle in lower:
            return True
    return False


def _redact_mapping(obj: Any, extras: Iterable[str]) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if _should_redact(key, extras):
                out[key] = _REDACTED
            else:
                out[key] = _redact_mapping(v, extras)
        return out
    if isinstance(obj, list):
        return [_redact_mapping(item, extras) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_redact_mapping(item, extras) for item in obj)
    return obj


# ---------------------------------------------------------------------------
# Structlog processor
# ---------------------------------------------------------------------------


def _redact_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor that strips sensitive keys."""
    extras = _extra_redactions()
    return _redact_mapping(event_dict, extras)


def _add_source_location(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure module / function / line are present (structlog CallsiteParameterAdder
    is used when available; this is a belt-and-braces fallback)."""
    event_dict.setdefault("logger", event_dict.get("logger_name", ""))
    return event_dict


# ---------------------------------------------------------------------------
# Stdlib fallback JSON formatter (used when structlog is not installed)
# ---------------------------------------------------------------------------


class _FallbackJsonFormatter(logging.Formatter):
    """Minimal JSON formatter used when structlog cannot be imported."""

    def format(self, record: logging.LogRecord) -> str:
        extras = _extra_redactions()
        base: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "event": record.getMessage(),
        }
        # Merge context fields if present (best-effort, no import cycle).
        try:
            from i3.observability.context import snapshot

            base.update(snapshot())
        except Exception:
            pass
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in base or key.startswith("_"):
                continue
            if key in {
                "args",
                "msg",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
                "asctime",
                "message",
                "levelname",
            }:
                continue
            base[key] = value
        return json.dumps(_redact_mapping(base, extras), default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure structlog (+ stdlib bridge).  Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("I3_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.environ.get("I3_LOG_FORMAT", "json").lower()

    try:
        import structlog
    except Exception:
        # Fallback: configure stdlib logging with a JSON formatter.
        root = logging.getLogger()
        root.setLevel(level)
        # Remove existing handlers (idempotent across reloads).
        for h in list(root.handlers):
            root.removeHandler(h)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FallbackJsonFormatter())
        root.addHandler(handler)
        _CONFIGURED = True
        root.warning(
            "structlog not installed; falling back to stdlib JSON logging"
        )
        return

    # ---- structlog present --------------------------------------------------
    from structlog.contextvars import merge_contextvars

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    callsite = structlog.processors.CallsiteParameterAdder(
        parameters=[
            structlog.processors.CallsiteParameter.MODULE,
            structlog.processors.CallsiteParameter.FUNC_NAME,
            structlog.processors.CallsiteParameter.LINENO,
        ]
    )

    shared_processors: list[Any] = [
        merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        callsite,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_source_location,
        _redact_processor,
    ]

    if fmt == "console":
        renderer: Any = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    else:
        renderer = structlog.processors.JSONRenderer(sort_keys=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog's formatter so foreign loggers
    # (uvicorn, httpx, torch, etc.) also emit structured JSON.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    # Quiet down noisy third-party loggers that would otherwise drown out the
    # service's own signal at INFO level.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    """Return a structured logger.  Falls back to stdlib if structlog missing."""
    if not _CONFIGURED:
        configure_logging()
    try:
        import structlog

        return structlog.get_logger(name) if name else structlog.get_logger()
    except Exception:
        return logging.getLogger(name if name else "i3")
