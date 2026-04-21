"""Optional Sentry integration.

Activation criteria (all required):

1. ``sentry_sdk`` importable.
2. ``SENTRY_DSN`` environment variable is set and non-empty.

If either is missing, :func:`configure_sentry` returns silently (no
warning — Sentry is explicitly optional).  The ``before_send`` hook
strips PII using :class:`i3.privacy.sanitizer.PrivacySanitizer`
(imported lazily to avoid a circular import at module load).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_CONFIGURED: bool = False


def _lazy_sanitizer() -> Any:
    """Import :class:`PrivacySanitizer` on first use."""
    try:
        from i3.privacy.sanitizer import PrivacySanitizer

        return PrivacySanitizer()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("PrivacySanitizer unavailable (%s); Sentry PII strip skipped", exc)
        return None


def _scrub(value: Any, sanitizer: Any) -> Any:
    """Recursively sanitise strings inside an event / breadcrumb payload."""
    if sanitizer is None:
        return value
    if isinstance(value, str):
        try:
            result = sanitizer.sanitize(value)
            # SanitizationResult dataclass exposes ``sanitized_text``.
            return getattr(result, "sanitized_text", value)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _scrub(v, sanitizer) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(item, sanitizer) for item in value]
    return value


def _make_before_send() -> Any:
    """Return a ``before_send`` callback that strips PII from events."""
    sanitizer = _lazy_sanitizer()

    def before_send(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
        try:
            # Most commonly PII leaks via ``message``, breadcrumb text, and
            # ``extra``.  Scrub each conservatively.
            for key in ("message", "logentry", "extra", "tags", "breadcrumbs"):
                if key in event:
                    event[key] = _scrub(event[key], sanitizer)
            # Stack traces often contain local variables — drop them outright.
            for ex in (event.get("exception", {}) or {}).get("values", []) or []:
                stacktrace = ex.get("stacktrace") or {}
                for frame in stacktrace.get("frames", []) or []:
                    frame.pop("vars", None)
        except Exception:  # pragma: no cover - defensive
            # Never let the scrubber crash an outgoing event; it would be
            # dropped and cost us observability during an incident.
            logger.debug("Sentry before_send scrubber raised", exc_info=True)
        return event

    return before_send


def configure_sentry(service_version: str = "0.0.0") -> None:
    """Initialise Sentry if conditions are met.  Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        _CONFIGURED = True
        return

    try:
        import sentry_sdk
    except Exception:
        logger.info("sentry-sdk not installed; skipping Sentry init")
        _CONFIGURED = True
        return

    try:
        traces_rate = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    except ValueError:
        traces_rate = 0.1

    try:
        profiles_rate = float(os.environ.get("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))
    except ValueError:
        profiles_rate = 0.0

    environment = os.environ.get(
        "SENTRY_ENVIRONMENT",
        os.environ.get("I3_ENV", "development"),
    )

    # Integrations — soft-imported.  FastAPI integration is registered
    # automatically by sentry-sdk >= 1.30; asyncio integration must be
    # opted in.
    integrations: list[Any] = []
    try:
        from sentry_sdk.integrations.asyncio import AsyncioIntegration

        integrations.append(AsyncioIntegration())
    except Exception:  # pragma: no cover - import guarded
        pass
    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        integrations.append(FastApiIntegration())
    except Exception:  # pragma: no cover - import guarded
        pass
    try:
        from sentry_sdk.integrations.starlette import StarletteIntegration

        integrations.append(StarletteIntegration())
    except Exception:  # pragma: no cover - import guarded
        pass

    try:
        sentry_sdk.init(
            dsn=dsn,
            release=service_version,
            environment=environment,
            traces_sample_rate=max(0.0, min(1.0, traces_rate)),
            profiles_sample_rate=max(0.0, min(1.0, profiles_rate)),
            send_default_pii=False,
            attach_stacktrace=True,
            max_breadcrumbs=50,
            integrations=integrations,
            before_send=_make_before_send(),
        )
        logger.info(
            "Sentry initialised",
            extra={
                "environment": environment,
                "traces_sample_rate": traces_rate,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Sentry init failed: %s", exc)
    finally:
        _CONFIGURED = True


def capture_message(message: str, level: str = "info", **extra: Any) -> None:
    """Best-effort message capture; no-op if Sentry not configured."""
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for k, v in extra.items():
                try:
                    scope.set_extra(k, json.loads(json.dumps(v, default=str)))
                except Exception:
                    scope.set_extra(k, str(v))
            sentry_sdk.capture_message(message, level=level)
    except Exception:
        return
