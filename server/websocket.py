"""WebSocket handler for real-time interaction streaming.

Each connected client gets a persistent WebSocket that:

  - Buffers individual keystroke events for behavioral analysis
  - Processes completed messages through the full I3 pipeline
  - Streams back responses, state updates, and diary entries

Security controls enforced in this module (see SECURITY.md):

  - ``Origin`` header allow-list check (CORS does NOT cover WebSockets).
  - ``user_id`` regex validation (alphanumeric/underscore/dash, 1-64 chars).
  - Hard cap on inbound message size, measured BEFORE JSON decode.
  - Hard cap on messages-per-session (``_MAX_MESSAGES_PER_SESSION``).
  - Hard cap on session wall-clock duration (``_MAX_SESSION_SECONDS``).
  - Per-user rate limiter (600 msg / min) shared with REST middleware.
  - Keystroke-buffer length is bounded (``_MAX_KEYSTROKE_BUFFER``).
  - Bounded numeric coercion of every client-supplied number — rejects
    ``NaN``/``inf`` and absurd magnitudes that could poison downstream
    arithmetic or cause integer overflow.
  - Any JSON parse / validation error terminates the socket with code 1008.
  - Connection eviction is race-free: a stale handler can never wipe out
    the slot owned by a freshly installed socket for the same user_id.
  - ``end_session`` is idempotent — duplicate ``session_end`` from the
    client cannot crash the handler.

These mitigations eliminate the main DoS / resource-exhaustion vectors.
Authentication remains out of scope for the demo: any client that knows
a ``user_id`` may claim it. Production deployments must layer a JWT or
session-token check on top of the Origin allow-list.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from i3.interaction.types import KeystrokeEvent
from i3.pipeline.types import PipelineInput
from server.middleware import RateLimitMiddleware

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard security limits — tuned for the demo, enforced strictly.
# ---------------------------------------------------------------------------

_USER_ID_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""Allowed ``user_id`` pattern: alphanumeric / underscore / dash, 1-64 chars."""

_MAX_MESSAGE_BYTES: int = 64 * 1024         # 64 KiB per inbound frame
_MAX_MESSAGE_TEXT_CHARS: int = 8 * 1024     # 8 KiB of chat text per message
_MAX_MESSAGES_PER_SESSION: int = 1_000      # abort the session after 1000 msgs
_MAX_SESSION_SECONDS: int = 60 * 60         # 1 hour wall-clock
_MAX_KEYSTROKE_BUFFER: int = 2_000          # cap keystroke buffer growth

# SEC: Hard numeric ceilings on client-supplied integers/floats to prevent
# integer-overflow / DoS via absurd values being passed into downstream code.
_MAX_INT_FIELD: int = 1_000_000_000          # 1 billion — well below 2**31
_MAX_FLOAT_FIELD: float = 1e15               # generous but finite

# WebSocket close codes
_CLOSE_POLICY_VIOLATION = 1008
_CLOSE_MESSAGE_TOO_BIG = 1009
_CLOSE_GOING_AWAY = 1001

# Shared per-user sliding-window limiter (not tied to the REST middleware
# instance so a single process keeps a single limiter per scope).
_ws_rate_limiter = RateLimitMiddleware.ws_limiter()


# ---------------------------------------------------------------------------
# Connection bookkeeping
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active WebSocket connections and session bookkeeping."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}   # user_id -> ws
        self.sessions: dict[str, str] = {}                   # user_id -> session_id
        # SEC: Async lock guarding all mutations of active_connections /
        # sessions so concurrent connect()/disconnect() cannot race and
        # leave a stale or wrong-WebSocket entry behind.
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Accept the socket, register the user, and return a fresh session id."""
        await websocket.accept()
        prior: WebSocket | None
        async with self._lock:
            # SEC: Evict any prior connection for this user_id atomically and
            # immediately install the new one so the prior handler's finally
            # cleanup cannot wipe the *new* slot. We compare WebSocket
            # identity in disconnect() to prevent that race.
            prior = self.active_connections.get(user_id)
            self.active_connections[user_id] = websocket
            session_id = str(uuid.uuid4())
            self.sessions[user_id] = session_id

        if prior is not None:
            try:
                await prior.close(code=_CLOSE_GOING_AWAY)
            except Exception:  # pragma: no cover -- best effort cleanup
                pass

        logger.info("User %s connected, session %s", user_id, session_id)
        return session_id

    async def disconnect(self, user_id: str, websocket: WebSocket | None = None) -> None:
        """Remove *user_id* iff *websocket* is the currently registered socket.

        SEC: When *websocket* is provided, only remove the entry if the
        registered connection is the same object — this prevents an evicted
        old handler from clobbering the freshly-installed new connection.
        """
        async with self._lock:
            current = self.active_connections.get(user_id)
            if websocket is not None and current is not websocket:
                # The slot is owned by a newer connection — leave it alone.
                logger.debug(
                    "disconnect() for %s skipped: slot owned by newer socket",
                    user_id,
                )
                return
            self.active_connections.pop(user_id, None)
            self.sessions.pop(user_id, None)
        logger.info("User %s disconnected", user_id)

    async def send_json(self, user_id: str, data: dict[str, Any]) -> None:
        """Send a JSON payload to a connected user (no-op if absent)."""
        ws = self.active_connections.get(user_id)
        if ws is None:
            return
        try:
            await ws.send_json(data)
        except Exception:
            # SEC: Closed sockets / backpressure failures are logged and
            # swallowed; never propagate to the caller.
            logger.warning("Failed to send to %s, dropping message", user_id)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------


def _validate_user_id(user_id: str) -> bool:
    """Return ``True`` iff *user_id* matches :data:`_USER_ID_RE`."""
    return bool(_USER_ID_RE.match(user_id or ""))


def _validate_origin(websocket: WebSocket, allowed_origins: list[str]) -> bool:
    """Return ``True`` iff the WebSocket Origin header is in the allow-list.

    SEC: CORS policy does *not* apply to WebSocket upgrades — browsers
    happily send WS handshakes cross-origin. We must therefore enforce
    the same origin allow-list manually inside the handler. A missing
    Origin header (e.g., non-browser client) is rejected unless the
    server has been explicitly configured with the wildcard "*".
    """
    if not allowed_origins:
        return False
    if "*" in allowed_origins:
        return True
    origin = websocket.headers.get("origin")
    if not origin:
        return False
    # Exact-match allow-list (scheme + host + port). We deliberately do
    # not allow suffix / wildcard matching here.
    return origin in allowed_origins


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce *value* to an int bounded by ``_MAX_INT_FIELD``.

    SEC: Raises :class:`ValueError` on overflow / non-finite / non-numeric
    so the caller can close the socket cleanly.
    """
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non_integer: {value!r}") from exc
    if out < -_MAX_INT_FIELD or out > _MAX_INT_FIELD:
        raise ValueError(f"int_out_of_range: {out}")
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to a finite float bounded by ``_MAX_FLOAT_FIELD``.

    SEC: Rejects ``NaN``, ``+/-inf`` and absurd magnitudes that could
    poison downstream timestamp / interval arithmetic.
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non_float: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"non_finite_float: {out}")
    if out < -_MAX_FLOAT_FIELD or out > _MAX_FLOAT_FIELD:
        raise ValueError(f"float_out_of_range: {out}")
    return out


async def _recv_bounded(websocket: WebSocket) -> dict[str, Any]:
    """Receive one WebSocket frame, enforcing the size ceiling.

    Raises :class:`ValueError` if the frame exceeds ``_MAX_MESSAGE_BYTES``
    or cannot be parsed as JSON.  Callers should catch ``ValueError`` and
    close the socket with code 1009 (message too big) or 1008 (policy).
    """
    # SEC: Use the raw ASGI receive() so we can measure the payload BEFORE
    # decoding into a Python ``str``. ``websocket.receive_text()`` will
    # silently allocate the entire frame regardless of size, defeating
    # any post-hoc length check. Starlette's text frames arrive as
    # ``{"type": "websocket.receive", "text": str}``; binary frames as
    # ``{"type": "websocket.receive", "bytes": bytes}``. Either way we
    # check the byte length up front.
    message = await websocket.receive()
    msg_type = message.get("type")
    if msg_type == "websocket.disconnect":
        # Bubble up so the outer handler can run normal cleanup.
        raise WebSocketDisconnect(code=message.get("code", 1000))
    if msg_type != "websocket.receive":
        raise ValueError(f"unexpected_frame_type: {msg_type}")

    if "text" in message and message["text"] is not None:
        raw_text: str = message["text"]
        # SEC: bound by *bytes*, not chars — multi-byte characters cost
        # more on the wire and in downstream pipeline buffers.
        encoded = raw_text.encode("utf-8", errors="strict")
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ValueError("frame_too_large")
        payload_bytes = encoded
    elif "bytes" in message and message["bytes"] is not None:
        payload_bytes = message["bytes"]
        if len(payload_bytes) > _MAX_MESSAGE_BYTES:
            raise ValueError("frame_too_large")
    else:
        raise ValueError("empty_frame")

    try:
        data = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid_json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("frame_must_be_object")
    return data


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str) -> None:
    """Primary WebSocket endpoint -- one per user.

    The handler is wrapped in a single ``try`` so that *any* unexpected
    failure, validation error, or limit-exceeded condition results in a
    graceful close and pipeline cleanup.

    SEC NOTE: There is intentionally no authentication here — any client
    that knows a ``user_id`` may claim it. This is acceptable for the
    single-user demo scope; production deployments must layer a JWT or
    session-token check on top (see SECURITY.md).
    """
    # ------------------------------------------------------------------
    # SEC: Reject malformed user_ids BEFORE we accept the socket so the
    # handshake never completes for invalid ids.
    # ------------------------------------------------------------------
    if not _validate_user_id(user_id):
        await websocket.close(code=_CLOSE_POLICY_VIOLATION)
        logger.warning("Rejected WebSocket: invalid user_id=%r", user_id)
        return

    # ------------------------------------------------------------------
    # SEC: Origin header allow-list — CORS does *not* protect WebSockets.
    # We pull the same allow-list the REST CORS middleware uses, plus
    # the runtime override env var, so a misconfigured origin can never
    # bypass the WS path.
    # ------------------------------------------------------------------
    config = getattr(websocket.app.state, "config", None)
    allowed_origins: list[str] = []
    if config is not None:
        allowed_origins = list(getattr(config.server, "cors_origins", []) or [])
    env_origins = os.environ.get("I3_CORS_ORIGINS")
    if env_origins:
        allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    if not allowed_origins:
        allowed_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

    if not _validate_origin(websocket, allowed_origins):
        # SEC: 1008 (policy violation) — the spec does not define a 403
        # equivalent for WebSocket upgrade rejections, so we close
        # immediately with the closest applicable code and never accept.
        await websocket.close(code=_CLOSE_POLICY_VIOLATION)
        logger.warning(
            "Rejected WebSocket: origin=%r not in allow-list (user_id=%s)",
            websocket.headers.get("origin"),
            user_id,
        )
        return

    pipeline = websocket.app.state.pipeline
    session_id = await manager.connect(websocket, user_id)

    # SEC: Anything below this point must be safe to run in the finally
    # cleanup block, even if it failed half-way through.
    try:
        await pipeline.start_session(user_id)
    except Exception as exc:
        logger.error(
            "Failed to start pipeline session for %s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:  # pragma: no cover
            pass
        await manager.disconnect(user_id, websocket=websocket)
        return

    # Notify the client that the session is live
    await manager.send_json(user_id, {
        "type": "session_started",
        "session_id": session_id,
        "user_id": user_id,
    })

    keystroke_buffer: list[dict[str, Any]] = []
    messages_handled = 0
    session_start = time.monotonic()
    session_ended = False  # SEC: idempotency guard for end_session

    try:
        while True:
            # ------------------------------------------------------------
            # Wall-clock guard
            # ------------------------------------------------------------
            if time.monotonic() - session_start > _MAX_SESSION_SECONDS:
                logger.info(
                    "Closing WebSocket for %s: exceeded %ds session cap",
                    user_id,
                    _MAX_SESSION_SECONDS,
                )
                await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                break

            # ------------------------------------------------------------
            # Bounded receive (raises ValueError on violation)
            # ------------------------------------------------------------
            try:
                data = await _recv_bounded(websocket)
            except ValueError as exc:
                reason = str(exc)
                logger.warning(
                    "Closing WebSocket for %s: %s", user_id, reason
                )
                close_code = (
                    _CLOSE_MESSAGE_TOO_BIG
                    if reason == "frame_too_large"
                    else _CLOSE_POLICY_VIOLATION
                )
                await websocket.close(code=close_code)
                break

            # ------------------------------------------------------------
            # Per-user rate limit
            # ------------------------------------------------------------
            if not _ws_rate_limiter.allow(user_id):
                logger.warning("WebSocket rate limit exceeded for %s", user_id)
                await manager.send_json(
                    user_id,
                    {"type": "error", "code": 429, "detail": "rate_limited"},
                )
                await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                break

            msg_type = data.get("type")
            if not isinstance(msg_type, str):
                logger.warning("Closing WebSocket for %s: missing 'type'", user_id)
                await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                break

            # ----------------------------------------------------------
            # Keystroke -- buffer for later aggregation
            # ----------------------------------------------------------
            if msg_type == "keystroke":
                if len(keystroke_buffer) >= _MAX_KEYSTROKE_BUFFER:
                    # SEC: Drop oldest samples to bound memory growth.
                    keystroke_buffer = keystroke_buffer[-_MAX_KEYSTROKE_BUFFER // 2 :]
                try:
                    # SEC: Use bounded numeric coercion to reject NaN/inf
                    # and absurd magnitudes a hostile client could send.
                    ks_ts = _safe_float(data.get("timestamp", time.time()))
                    ks_key_type = str(data.get("key_type", "char"))[:16]
                    ks_iki = _safe_float(data.get("inter_key_interval_ms", 0))
                    if ks_iki < 0:
                        raise ValueError("negative_inter_key_interval")
                    ks = KeystrokeEvent(
                        timestamp=ks_ts,
                        key_type=ks_key_type,
                        inter_key_interval_ms=ks_iki,
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning("Invalid keystroke from %s: %s", user_id, exc)
                    await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                    break
                pipeline.monitor.process_keystroke(user_id, ks)
                # SEC: Only the validated numeric subset is buffered for
                # aggregation later — never the raw client dict.
                keystroke_buffer.append({"inter_key_interval_ms": ks_iki})

            # ----------------------------------------------------------
            # Full message -- run through the I3 pipeline
            # ----------------------------------------------------------
            elif msg_type == "message":
                text = data.get("text", "")
                if not isinstance(text, str):
                    await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                    break
                if len(text) > _MAX_MESSAGE_TEXT_CHARS:
                    logger.warning(
                        "Closing WebSocket for %s: message text %d > %d",
                        user_id,
                        len(text),
                        _MAX_MESSAGE_TEXT_CHARS,
                    )
                    await websocket.close(code=_CLOSE_MESSAGE_TOO_BIG)
                    break
                if not text.strip():
                    continue

                messages_handled += 1
                if messages_handled > _MAX_MESSAGES_PER_SESSION:
                    logger.info(
                        "Closing WebSocket for %s: %d messages > cap %d",
                        user_id,
                        messages_handled,
                        _MAX_MESSAGES_PER_SESSION,
                    )
                    await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                    break

                # SEC: Bounded coercion of every client-supplied number
                # so an attacker cannot inject NaN/inf or 2**62-sized
                # values into the pipeline.
                try:
                    msg_ts = _safe_float(data.get("timestamp", time.time()))
                    composition_ms = _safe_float(data.get("composition_time_ms", 0))
                    edit_count = _safe_int(data.get("edit_count", 0))
                    pause_ms = _safe_float(data.get("pause_before_send_ms", 0))
                    if composition_ms < 0 or pause_ms < 0 or edit_count < 0:
                        raise ValueError("negative_metric")
                except ValueError as exc:
                    logger.warning(
                        "Invalid numeric field in message from %s: %s",
                        user_id,
                        exc,
                    )
                    await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                    break

                pipeline_input = PipelineInput(
                    user_id=user_id,
                    session_id=session_id,
                    message_text=text,
                    timestamp=msg_ts,
                    composition_time_ms=composition_ms,
                    edit_count=edit_count,
                    pause_before_send_ms=pause_ms,
                    keystroke_timings=[
                        float(ks.get("inter_key_interval_ms", 0))
                        for ks in keystroke_buffer
                    ],
                )

                output = await pipeline.process_message(pipeline_input)

                # Main response
                await manager.send_json(user_id, {
                    "type": "response",
                    "text": output.response_text,
                    "route": output.route_chosen,
                    "latency_ms": round(output.latency_ms),
                    "timestamp": time.time(),
                })

                # Behavioural state update (used by the dashboard viz)
                await manager.send_json(user_id, {
                    "type": "state_update",
                    "user_state_embedding_2d": list(output.user_state_embedding_2d),
                    "adaptation": output.adaptation,
                    "engagement_score": output.engagement_score,
                    "deviation_from_baseline": output.deviation_from_baseline,
                    "routing_confidence": output.routing_confidence,
                    "messages_in_session": output.messages_in_session,
                    "baseline_established": output.baseline_established,
                })

                # Diary entry (only sent when the pipeline produces one)
                if output.diary_entry:
                    await manager.send_json(user_id, {
                        "type": "diary_entry",
                        "entry": output.diary_entry,
                    })

                # Done with this message's keystrokes
                keystroke_buffer.clear()

            # ----------------------------------------------------------
            # Explicit session end from the client
            # ----------------------------------------------------------
            elif msg_type == "session_end":
                # SEC: Idempotent — if the session has already been ended
                # we ignore the duplicate request rather than crash. We
                # then close the socket so further client traffic cannot
                # arrive against an ended pipeline session.
                if not session_ended:
                    session_ended = True
                    try:
                        diary = await pipeline.end_session(user_id, session_id)
                    except Exception as exc:
                        logger.warning(
                            "end_session failed for %s: %s", user_id, exc
                        )
                        diary = None
                    if diary:
                        await manager.send_json(user_id, {
                            "type": "diary_entry",
                            "entry": diary,
                        })
                try:
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                except Exception:  # pragma: no cover
                    pass
                break

            else:
                logger.warning(
                    "Closing WebSocket for %s: unknown type=%r", user_id, msg_type
                )
                await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                break

    except WebSocketDisconnect:
        logger.info("User %s WebSocket disconnected normally", user_id)
    except Exception as exc:
        # SEC: Never leak internal error details back to the client.
        logger.error("WebSocket error for %s: %s", user_id, exc, exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:  # pragma: no cover -- best effort cleanup
            pass
    finally:
        # SEC: Clear all per-connection state regardless of how we exited.
        keystroke_buffer.clear()
        if not session_ended:
            session_ended = True
            try:
                await pipeline.end_session(user_id, session_id)
            except Exception:  # pragma: no cover -- already in teardown
                pass
        # SEC: Pass the websocket identity so that an evicted-old-handler
        # cleanup can never wipe out the *new* connection's slot.
        await manager.disconnect(user_id, websocket=websocket)
