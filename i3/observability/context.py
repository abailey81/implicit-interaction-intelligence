"""Per-request observability context (request_id, user_id, trace_id).

The context is stored in :mod:`contextvars` so it is automatically
propagated across ``await`` boundaries (including tasks spawned via
``asyncio.create_task`` from within the same Task).

Public API
----------
* :data:`request_id_ctx`, :data:`user_id_ctx`, :data:`trace_id_ctx`,
  :data:`client_ip_ctx` — the underlying :class:`contextvars.ContextVar`
  instances (exposed so tests can introspect them).
* :func:`bind_context` — bind one or more fields.  Returns a list of
  reset tokens that the caller must pass to :func:`reset_context`.
* :func:`reset_context` — reset previously-bound fields.
* :func:`snapshot` — return a dict of currently-bound, non-empty fields.

Binding these contextvars also binds them to :mod:`structlog` so every
log line emitted during the request lifetime carries the correlation
identifiers.  The :mod:`structlog` wiring is done lazily inside
``bind_context`` so that importing this module never forces a hard
dependency on structlog.
"""

from __future__ import annotations

import contextvars
from collections.abc import Mapping, MutableMapping
from typing import Any

# ---------------------------------------------------------------------------
# ContextVars
# ---------------------------------------------------------------------------

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "i3_request_id", default=""
)
user_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "i3_user_id", default=""
)
trace_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "i3_trace_id", default=""
)
client_ip_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "i3_client_ip", default=""
)

_CTX_MAP: Mapping[str, contextvars.ContextVar[str]] = {
    "request_id": request_id_ctx,
    "user_id": user_id_ctx,
    "trace_id": trace_id_ctx,
    "client_ip": client_ip_ctx,
}


# ---------------------------------------------------------------------------
# Soft structlog integration
# ---------------------------------------------------------------------------


def _structlog_bind(values: MutableMapping[str, Any]) -> None:
    """Bind *values* into structlog's contextvar store if available."""
    try:  # pragma: no cover - import guarded
        import structlog

        structlog.contextvars.bind_contextvars(**values)
    except Exception:
        # Structlog not installed or not initialised — silently ignore.
        # The stdlib contextvars still carry the data for any consumer
        # that wants it.
        pass


def _structlog_unbind(keys: list[str]) -> None:
    try:  # pragma: no cover - import guarded
        import structlog

        structlog.contextvars.unbind_contextvars(*keys)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def bind_context(**fields: str) -> list[tuple[contextvars.ContextVar[str], contextvars.Token]]:
    """Bind fields to the request context.

    Unknown fields are ignored (logged is not possible here without creating
    an import cycle, so callers should use the canonical field names).
    Returns a list of ``(var, token)`` pairs suitable for
    :func:`reset_context`.
    """
    tokens: list[tuple[contextvars.ContextVar[str], contextvars.Token]] = []
    slog_kwargs: dict[str, str] = {}
    for key, value in fields.items():
        var = _CTX_MAP.get(key)
        if var is None or value is None:
            continue
        tokens.append((var, var.set(str(value))))
        slog_kwargs[key] = str(value)
    if slog_kwargs:
        _structlog_bind(slog_kwargs)
    return tokens


def reset_context(
    tokens: list[tuple[contextvars.ContextVar[str], contextvars.Token]],
) -> None:
    """Reverse a previous :func:`bind_context` call."""
    # Reset in LIFO order for correctness when the same var is bound twice.
    keys: list[str] = []
    for var, token in reversed(tokens):
        try:
            var.reset(token)
        except ValueError:  # pragma: no cover - defensive
            # Token was created in a different Context; best-effort reset.
            var.set("")
        for key, candidate in _CTX_MAP.items():
            if candidate is var:
                keys.append(key)
                break
    if keys:
        _structlog_unbind(keys)


def snapshot() -> dict[str, str]:
    """Return currently-bound non-empty context fields."""
    return {k: v.get() for k, v in _CTX_MAP.items() if v.get()}
