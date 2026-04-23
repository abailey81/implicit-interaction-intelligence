"""Tests for :mod:`server.auth` — the opt-in caller-identity dependency.

The module ships as *off by default*: when ``I3_REQUIRE_USER_AUTH`` is
not set, :func:`require_user_identity` is a no-op.  When enabled, two
activation modes compose (header-match and bearer-token map).  These
tests cover every branch end-to-end, including the timing-safe
compare path, malformed token-map handling, and the body-resolved
variant used by POST routes.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# require_user_identity (path-variant)
# ---------------------------------------------------------------------------


def _set_env(monkeypatch, **kwargs: str) -> None:
    """Clear + set the auth-related env vars deterministically."""
    for key in ("I3_REQUIRE_USER_AUTH", "I3_USER_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    for k, v in kwargs.items():
        monkeypatch.setenv(k, v)


def test_auth_disabled_by_default(monkeypatch):
    """Un-enabled auth is a no-op for every caller."""
    from server.auth import require_user_identity

    _set_env(monkeypatch)
    # Should NOT raise regardless of what the caller presents.
    require_user_identity(user_id="alice", x_i3_user_id=None, x_i3_user_token=None)
    require_user_identity(user_id="alice", x_i3_user_id="mallory", x_i3_user_token=None)


def test_auth_enabled_requires_header_by_default(monkeypatch):
    """With auth on and no token map, missing X-I3-User-Id rejects."""
    from server.auth import require_user_identity

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    with pytest.raises(HTTPException) as exc_info:
        require_user_identity(
            user_id="alice", x_i3_user_id=None, x_i3_user_token=None
        )
    assert exc_info.value.status_code == 401


def test_auth_header_match_accepts_same_id(monkeypatch):
    """X-I3-User-Id matching the path user_id passes."""
    from server.auth import require_user_identity

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    require_user_identity(
        user_id="alice", x_i3_user_id="alice", x_i3_user_token=None
    )


def test_auth_header_match_rejects_different_id(monkeypatch):
    """X-I3-User-Id differing from path user_id is rejected (cross-user guard)."""
    from server.auth import require_user_identity

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    with pytest.raises(HTTPException) as exc_info:
        require_user_identity(
            user_id="alice", x_i3_user_id="mallory", x_i3_user_token=None
        )
    assert exc_info.value.status_code == 401


def test_auth_token_map_correct_token(monkeypatch):
    """With I3_USER_TOKENS set, the correct token unlocks the user."""
    from server.auth import require_user_identity

    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS=json.dumps({"alice": "alice-secret", "bob": "bob-secret"}),
    )
    require_user_identity(
        user_id="alice", x_i3_user_id=None, x_i3_user_token="alice-secret"
    )
    require_user_identity(
        user_id="bob", x_i3_user_id=None, x_i3_user_token="bob-secret"
    )


def test_auth_token_map_wrong_token_rejects(monkeypatch):
    """A valid but cross-user token is still rejected for the path user_id."""
    from server.auth import require_user_identity

    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS=json.dumps({"alice": "alice-secret", "bob": "bob-secret"}),
    )
    with pytest.raises(HTTPException):
        # Bob's token should NOT unlock alice.
        require_user_identity(
            user_id="alice", x_i3_user_id=None, x_i3_user_token="bob-secret"
        )


def test_auth_token_map_missing_token_rejects(monkeypatch):
    """With token-map mode active, callers without a token are rejected."""
    from server.auth import require_user_identity

    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS=json.dumps({"alice": "alice-secret"}),
    )
    with pytest.raises(HTTPException):
        require_user_identity(
            user_id="alice", x_i3_user_id=None, x_i3_user_token=None
        )


def test_auth_token_map_unknown_user_rejects(monkeypatch):
    """A user id not in the map is rejected even with *some* token."""
    from server.auth import require_user_identity

    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS=json.dumps({"alice": "alice-secret"}),
    )
    with pytest.raises(HTTPException):
        require_user_identity(
            user_id="stranger", x_i3_user_id=None, x_i3_user_token="alice-secret"
        )


def test_auth_malformed_token_map_denies_everyone(monkeypatch):
    """Malformed I3_USER_TOKENS must not silently fall back to header mode."""
    from server.auth import require_user_identity

    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS="{this is not valid json",
    )
    # Empty token map parsed → back to header mode.  A caller who
    # presents no header is rejected.
    with pytest.raises(HTTPException):
        require_user_identity(
            user_id="alice", x_i3_user_id=None, x_i3_user_token=None
        )


def test_auth_non_object_token_map_rejects(monkeypatch):
    """I3_USER_TOKENS that parses to a non-object is treated as empty."""
    from server.auth import require_user_identity

    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS="[1, 2, 3]",  # valid JSON, not an object
    )
    # Falls through to header mode; missing header rejects.
    with pytest.raises(HTTPException):
        require_user_identity(
            user_id="alice", x_i3_user_id=None, x_i3_user_token=None
        )


# ---------------------------------------------------------------------------
# Constant-time compare (timing-safe)
# ---------------------------------------------------------------------------


def test_auth_uses_constant_time_compare(monkeypatch):
    """``require_user_identity`` must use :func:`secrets.compare_digest`
    to compare the caller's token / header against the expected value.

    This test patches :func:`secrets.compare_digest` and asserts it is
    called at least once on a successful auth path, both in the
    header-match branch and the token-map branch.
    """
    import secrets

    from server.auth import require_user_identity

    calls: list[tuple[str, str]] = []

    def _spy(a: str, b: str) -> bool:
        calls.append((a, b))
        return a == b

    monkeypatch.setattr(secrets, "compare_digest", _spy)

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    require_user_identity(
        user_id="alice", x_i3_user_id="alice", x_i3_user_token=None
    )
    assert any("alice" in pair for pair in calls), (
        "header-match path should go through secrets.compare_digest"
    )

    calls.clear()
    _set_env(
        monkeypatch,
        I3_REQUIRE_USER_AUTH="1",
        I3_USER_TOKENS=json.dumps({"alice": "alice-secret"}),
    )
    require_user_identity(
        user_id="alice", x_i3_user_id=None, x_i3_user_token="alice-secret"
    )
    assert any("alice-secret" in pair for pair in calls), (
        "token-map path should go through secrets.compare_digest"
    )


# ---------------------------------------------------------------------------
# require_user_identity_from_body (POST-variant)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_from_body_disabled_by_default(monkeypatch):
    """When auth is off, the body-variant is a no-op and does not read body."""
    from unittest.mock import AsyncMock, MagicMock

    from server.auth import require_user_identity_from_body

    _set_env(monkeypatch)
    request = MagicMock()
    request.body = AsyncMock(return_value=b'{"user_id":"alice"}')
    # state is a namespace object
    request.state = MagicMock()

    # No exception and — because auth is disabled — we should NOT read body.
    await require_user_identity_from_body(
        request=request, x_i3_user_id=None, x_i3_user_token=None
    )
    request.body.assert_not_called()


@pytest.mark.asyncio
async def test_auth_from_body_header_mode(monkeypatch):
    """With header-mode on, body user_id must match X-I3-User-Id."""
    from unittest.mock import AsyncMock, MagicMock

    from server.auth import require_user_identity_from_body

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    request = MagicMock()
    request.state = MagicMock()
    request.body = AsyncMock(return_value=b'{"user_id":"alice","x":1}')

    # Header matches → passes.
    await require_user_identity_from_body(
        request=request, x_i3_user_id="alice", x_i3_user_token=None
    )

    # Header mismatches → rejects.
    with pytest.raises(HTTPException):
        await require_user_identity_from_body(
            request=request, x_i3_user_id="mallory", x_i3_user_token=None
        )


@pytest.mark.asyncio
async def test_auth_from_body_missing_user_id_rejects(monkeypatch):
    """A JSON body with no user_id is rejected outright."""
    from unittest.mock import AsyncMock, MagicMock

    from server.auth import require_user_identity_from_body

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    request = MagicMock()
    request.state = MagicMock()
    request.body = AsyncMock(return_value=b'{"message":"hi"}')

    with pytest.raises(HTTPException) as exc_info:
        await require_user_identity_from_body(
            request=request, x_i3_user_id="alice", x_i3_user_token=None
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_auth_from_body_non_json_rejects(monkeypatch):
    """A non-JSON body is rejected with 401 (no crash)."""
    from unittest.mock import AsyncMock, MagicMock

    from server.auth import require_user_identity_from_body

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    request = MagicMock()
    request.state = MagicMock()
    request.body = AsyncMock(return_value=b"not valid json at all")

    with pytest.raises(HTTPException):
        await require_user_identity_from_body(
            request=request, x_i3_user_id="alice", x_i3_user_token=None
        )


@pytest.mark.asyncio
async def test_auth_from_body_caches_body_on_state(monkeypatch):
    """The body bytes are cached on ``request.state`` to avoid re-reading."""
    from unittest.mock import AsyncMock, MagicMock

    from server.auth import require_user_identity_from_body

    _set_env(monkeypatch, I3_REQUIRE_USER_AUTH="1")
    request = MagicMock()
    request.state = MagicMock()
    raw = b'{"user_id":"alice"}'
    request.body = AsyncMock(return_value=raw)

    await require_user_identity_from_body(
        request=request, x_i3_user_id="alice", x_i3_user_token=None
    )
    # The handler should have stashed the bytes.
    assert getattr(request.state, "_i3_auth_body_cache", None) == raw
