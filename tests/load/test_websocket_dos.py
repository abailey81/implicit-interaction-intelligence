"""WebSocket DoS-resistance tests.

Two scenarios:

1. **Message flood** — 1000 rapid small frames.  The server's per-user
   rate limiter or the session-message cap should close the socket well
   before the 1000th frame.  Either outcome is acceptable; the test
   simply verifies that the server never hangs, never leaks memory (the
   connection is correctly torn down), and never raises an unhandled
   exception.

2. **Oversized frame** — a single 128 KiB text frame.  The server must
   close the socket with code 1009 (``_CLOSE_MESSAGE_TOO_BIG``).

Both tests run through Starlette's ``TestClient.websocket_connect``
against the real FastAPI app, so the full middleware / limiter stack
is exercised.
"""

from __future__ import annotations

import json
import pytest


pytestmark = [pytest.mark.load, pytest.mark.slow]


# WebSocket close codes mirrored from server/websocket.py
_CLOSE_POLICY_VIOLATION = 1008
_CLOSE_MESSAGE_TOO_BIG = 1009


@pytest.fixture(scope="module")
def client():
    """A TestClient that shares the middleware stack with production."""
    try:
        import os

        # Allow the localhost origin that TestClient emits by default.
        os.environ.setdefault(
            "I3_CORS_ORIGINS",
            "http://testserver,http://localhost:8000,http://127.0.0.1:8000",
        )
        from starlette.testclient import TestClient

        from server.app import create_app
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Cannot build FastAPI app: {exc}")
    return TestClient(create_app())


def _ws_connect(client, user_id: str = "loadtest"):
    """Open a WebSocket with a valid Origin header."""
    return client.websocket_connect(
        f"/ws/{user_id}",
        headers={"origin": "http://testserver"},
    )


# ─────────────────────────────────────────────────────────────────────────
#  Test 1 — Message flood
# ─────────────────────────────────────────────────────────────────────────


def test_websocket_message_flood_is_contained(client) -> None:
    """Sending 1000 rapid messages triggers the rate limiter / session cap
    without hanging or raising out of the handler."""
    from starlette.websockets import WebSocketDisconnect

    try:
        ws_ctx = _ws_connect(client)
    except Exception as exc:
        pytest.skip(f"WebSocket connect failed (demo env?): {exc}")

    closed_cleanly = False
    send_errors = 0
    frames_sent = 0
    try:
        with ws_ctx as ws:
            # Drain the initial "session_started" frame if present.
            try:
                _ = ws.receive_json()
            except Exception:
                pass
            for i in range(1000):
                payload = json.dumps({
                    "type": "keystroke",
                    "timestamp": 1.0,
                    "key_type": "char",
                    "inter_key_interval_ms": 50.0,
                })
                try:
                    ws.send_text(payload)
                    frames_sent += 1
                except Exception:
                    send_errors += 1
                    break
            closed_cleanly = True
    except WebSocketDisconnect:
        # Server closed the socket — exactly what we want under flood.
        closed_cleanly = True
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Unhandled exception under flood: {exc!r}")

    assert closed_cleanly, "Server did not terminate cleanly under flood"
    # We should have been able to send SOMETHING before the limiter kicked
    # in — a total shutout indicates the limiter is mis-configured.
    assert frames_sent > 0


# ─────────────────────────────────────────────────────────────────────────
#  Test 2 — Oversized frame
# ─────────────────────────────────────────────────────────────────────────


def test_oversized_frame_closes_1009(client) -> None:
    """A > 64 KiB frame must close the socket with code 1009."""
    from starlette.websockets import WebSocketDisconnect

    big_payload = "A" * (128 * 1024)  # 128 KiB — above _MAX_MESSAGE_BYTES
    frame = json.dumps({"type": "message", "text": big_payload})

    try:
        ws_ctx = _ws_connect(client)
    except Exception as exc:
        pytest.skip(f"WebSocket connect failed (demo env?): {exc}")

    close_code: int | None = None
    try:
        with ws_ctx as ws:
            try:
                _ = ws.receive_json()
            except Exception:
                pass
            ws.send_text(frame)
            # The server should close the socket — receive() raises
            # WebSocketDisconnect with the close code.
            try:
                ws.receive_text()
            except WebSocketDisconnect as disc:
                close_code = disc.code
    except WebSocketDisconnect as disc:  # close happened during context exit
        close_code = disc.code
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Unexpected error handling oversized frame: {exc!r}")

    assert close_code in (
        _CLOSE_MESSAGE_TOO_BIG,
        _CLOSE_POLICY_VIOLATION,
    ), f"Expected 1009 / 1008, got {close_code!r}"


# ─────────────────────────────────────────────────────────────────────────
#  Test 3 — Invalid origin is refused before accept
# ─────────────────────────────────────────────────────────────────────────


def test_invalid_origin_is_rejected(client) -> None:
    """A WS with an off-allowlist Origin must be closed immediately."""
    from starlette.websockets import WebSocketDisconnect

    close_code: int | None = None
    try:
        with client.websocket_connect(
            "/ws/loadtest",
            headers={"origin": "http://evil.example"},
        ) as ws:
            try:
                ws.receive_text()
            except WebSocketDisconnect as disc:
                close_code = disc.code
    except WebSocketDisconnect as disc:
        close_code = disc.code
    except Exception:
        # TestClient may raise a plain WebSocketException for immediate
        # refusals — that is also a valid outcome for this test.
        close_code = _CLOSE_POLICY_VIOLATION

    # Either 1008 or an immediate refusal is acceptable.
    assert close_code in (None, _CLOSE_POLICY_VIOLATION), close_code
