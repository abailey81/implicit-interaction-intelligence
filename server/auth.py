"""Shared authentication dependencies for per-user REST routes.

The server historically shipped with *no* per-user authentication — any
client that knew a ``user_id`` could read that user's profile, diary,
preference history, or TTS adaptation. That posture is documented as a
"Known limitation" in :mod:`server.routes` and was flagged as M-2 / H-2
in the 2026-04-23 security audit.

This module provides an opt-in caller-identity gate that can be placed
on any route whose path contains ``{user_id}``. It is:

* **Off by default** — set ``I3_REQUIRE_USER_AUTH=1`` in the environment
  to activate it, so the demo and local development workflows do not
  break.
* **Constant-time** — the token comparison uses
  :func:`secrets.compare_digest` to avoid timing oracles.
* **Explicit-header** — the caller presents ``X-I3-User-Token: <token>``
  (matched against ``I3_USER_TOKENS`` — a JSON ``{user_id: token}``
  mapping) and/or ``X-I3-User-Id: <user_id>`` (matched against the path
  ``user_id``).

The two modes compose:

* If ``I3_USER_TOKENS`` is set, every request must present a valid token.
* If only ``X-I3-User-Id`` is required, the gate still prevents the
  trivial cross-user read by forcing the client to declare which user
  it is acting on behalf of (a reverse-proxy can then rewrite and
  validate the header upstream).

Both modes are opt-in; nothing in the default demo configuration changes.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Mapping

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

_ENABLE_ENV = "I3_REQUIRE_USER_AUTH"
_TOKENS_ENV = "I3_USER_TOKENS"  # JSON object {user_id: token}


def _auth_enabled() -> bool:
    """True when ``I3_REQUIRE_USER_AUTH=1`` is set in the environment."""
    return os.environ.get(_ENABLE_ENV, "").strip() == "1"


def _load_token_map() -> Mapping[str, str]:
    """Parse ``I3_USER_TOKENS`` as JSON; empty dict on any error.

    The error path is silent-by-design: a malformed env var should not
    crash the service, but should deny every request (every call to
    :func:`require_user_identity` below will then reject because the
    map does not contain the requested user_id).
    """
    raw = os.environ.get(_TOKENS_ENV, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.warning("auth.token_map_invalid_json")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("auth.token_map_not_object")
        return {}
    # Cast every value to str so a malformed entry cannot coerce a
    # numeric id into a string-typed compare_digest.
    return {str(k): str(v) for k, v in parsed.items()}


def require_user_identity(
    user_id: str,
    x_i3_user_id: str | None = Header(default=None, alias="X-I3-User-Id"),
    x_i3_user_token: str | None = Header(
        default=None, alias="X-I3-User-Token"
    ),
) -> None:
    """Dependency: enforce that the caller is acting for ``user_id``.

    The check is a no-op unless :envvar:`I3_REQUIRE_USER_AUTH` is set to
    ``1``; this keeps the zero-config demo workflow working while giving
    production deployments an opt-in caller-identity gate.

    When enabled:

    1. If :envvar:`I3_USER_TOKENS` is configured, the caller MUST present
       ``X-I3-User-Token`` whose value matches the server-side token for
       the path ``user_id`` (constant-time comparison).
    2. Otherwise, the caller MUST present ``X-I3-User-Id`` matching the
       path ``user_id`` — this allows a reverse proxy to attach the
       header after validating the session, without this module having
       to speak the operator's session format.

    Args:
        user_id: The path parameter ``user_id`` from the route.
        x_i3_user_id: The caller-declared user id from ``X-I3-User-Id``.
        x_i3_user_token: The caller-presented bearer token from
            ``X-I3-User-Token``.

    Raises:
        HTTPException: With status 401 when authentication fails.
    """
    if not _auth_enabled():
        return

    token_map = _load_token_map()
    if token_map:
        expected = token_map.get(user_id, "")
        if not expected or not x_i3_user_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "X-I3-User-Token"},
            )
        if not secrets.compare_digest(x_i3_user_token, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "X-I3-User-Token"},
            )
        return

    if not x_i3_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "X-I3-User-Id"},
        )
    if not secrets.compare_digest(x_i3_user_id, user_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "X-I3-User-Id"},
        )


async def require_user_identity_from_body(
    request: Request,
    x_i3_user_id: str | None = Header(default=None, alias="X-I3-User-Id"),
    x_i3_user_token: str | None = Header(
        default=None, alias="X-I3-User-Token"
    ),
) -> None:
    """Dependency for POST routes whose ``user_id`` lives in the body.

    The request body is read (tolerating non-JSON payloads silently),
    the ``user_id`` field is extracted if present, and the same check
    as :func:`require_user_identity` is applied.  The raw body bytes
    are stashed on ``request.state._i3_auth_body_cache`` so the route
    handler can re-read them without paying the cost twice.

    Args:
        request: The FastAPI request — used to read (and cache) the body.
        x_i3_user_id: Caller-declared user id.
        x_i3_user_token: Caller-presented bearer token.

    Raises:
        HTTPException: With status 401 when authentication fails.
    """
    if not _auth_enabled():
        return

    # Cache the body so the route handler can re-parse it without a
    # second ``await request.body()`` that Starlette would reject.
    raw = await request.body()
    request.state._i3_auth_body_cache = raw  # noqa: SLF001
    body_user_id: str | None = None
    if raw:
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("user_id")
            if isinstance(candidate, str):
                body_user_id = candidate

    if body_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "X-I3-User-Id"},
        )

    # Delegate to the path-variant gate using the resolved id.
    require_user_identity(
        user_id=body_user_id,
        x_i3_user_id=x_i3_user_id,
        x_i3_user_token=x_i3_user_token,
    )


__all__ = ["require_user_identity", "require_user_identity_from_body"]
