"""WebSocket handler for real-time interaction streaming.

Each connected client gets a persistent WebSocket that:

  - Buffers individual keystroke events for behavioral analysis
  - Processes completed messages through the full I3 pipeline
  - Streams back responses, state updates, and diary entries

Security controls enforced in this module (see SECURITY.md):

  - ``user_id`` regex validation (alphanumeric/underscore/dash, 1-64 chars).
  - Hard cap on inbound message size (``_MAX_MESSAGE_BYTES``).
  - Hard cap on messages-per-session (``_MAX_MESSAGES_PER_SESSION``).
  - Hard cap on session wall-clock duration (``_MAX_SESSION_SECONDS``).
  - Per-user rate limiter (600 msg / min) shared with REST middleware.
  - Keystroke-buffer length is bounded (``_MAX_KEYSTROKE_BUFFER``).
  - Any JSON parse / validation error terminates the socket with code 1008.

These mitigations eliminate the main DoS / resource-exhaustion vectors
without adding authentication (which is out of scope for the demo).
"""

from __future__ import annotations

import logging
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

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Accept the socket, register the user, and return a fresh session id."""
        await websocket.accept()
        # If the same user_id is already connected, close the old socket to
        # prevent connection-stacking DoS.
        prior = self.active_connections.get(user_id)
        if prior is not None:
            try:
                await prior.close(code=_CLOSE_GOING_AWAY)
            except Exception:  # pragma: no cover -- best effort cleanup
                pass
        self.active_connections[user_id] = websocket
        session_id = str(uuid.uuid4())
        self.sessions[user_id] = session_id
        logger.info("User %s connected, session %s", user_id, session_id)
        return session_id

    def disconnect(self, user_id: str) -> None:
        self.active_connections.pop(user_id, None)
        self.sessions.pop(user_id, None)
        logger.info("User %s disconnected", user_id)

    async def send_json(self, user_id: str, data: dict[str, Any]) -> None:
        """Send a JSON payload to a connected user (no-op if absent)."""
        ws = self.active_connections.get(user_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                logger.warning("Failed to send to %s, dropping message", user_id)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------


def _validate_user_id(user_id: str) -> bool:
    """Return ``True`` iff *user_id* matches :data:`_USER_ID_RE`."""
    return bool(_USER_ID_RE.match(user_id or ""))


async def _recv_bounded(websocket: WebSocket) -> dict[str, Any]:
    """Receive one WebSocket frame, enforcing the size ceiling.

    Raises :class:`ValueError` if the frame exceeds ``_MAX_MESSAGE_BYTES``
    or cannot be parsed as JSON.  Callers should catch ``ValueError`` and
    close the socket with code 1009 (message too big) or 1008 (policy).
    """
    # FastAPI's receive_text lets us measure bytes before parsing.
    raw = await websocket.receive_text()
    if len(raw.encode("utf-8", errors="ignore")) > _MAX_MESSAGE_BYTES:
        raise ValueError("frame_too_large")
    import json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
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
    """
    # ------------------------------------------------------------------
    # Reject malformed user_ids before we even accept the socket.
    # ------------------------------------------------------------------
    if not _validate_user_id(user_id):
        await websocket.close(code=_CLOSE_POLICY_VIOLATION)
        logger.warning("Rejected WebSocket: invalid user_id=%r", user_id)
        return

    pipeline = websocket.app.state.pipeline
    session_id = await manager.connect(websocket, user_id)

    # Start a pipeline session for this user
    await pipeline.start_session(user_id)

    # Notify the client that the session is live
    await manager.send_json(user_id, {
        "type": "session_started",
        "session_id": session_id,
        "user_id": user_id,
    })

    keystroke_buffer: list[dict[str, Any]] = []
    messages_handled = 0
    session_start = time.monotonic()

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
                    # Drop oldest samples to bound memory growth.
                    keystroke_buffer = keystroke_buffer[-_MAX_KEYSTROKE_BUFFER // 2 :]
                try:
                    ks = KeystrokeEvent(
                        timestamp=float(data.get("timestamp", time.time())),
                        key_type=str(data.get("key_type", "char"))[:16],
                        inter_key_interval_ms=float(data.get("inter_key_interval_ms", 0)),
                    )
                except (TypeError, ValueError):
                    logger.warning("Invalid keystroke from %s", user_id)
                    await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                    break
                pipeline.monitor.process_keystroke(user_id, ks)
                keystroke_buffer.append(data)

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

                pipeline_input = PipelineInput(
                    user_id=user_id,
                    session_id=session_id,
                    message_text=text,
                    timestamp=float(data.get("timestamp", time.time())),
                    composition_time_ms=float(data.get("composition_time_ms", 0)),
                    edit_count=int(data.get("edit_count", 0)),
                    pause_before_send_ms=float(data.get("pause_before_send_ms", 0)),
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
                diary = await pipeline.end_session(user_id, session_id)
                if diary:
                    await manager.send_json(user_id, {
                        "type": "diary_entry",
                        "entry": diary,
                    })

            else:
                logger.warning(
                    "Closing WebSocket for %s: unknown type=%r", user_id, msg_type
                )
                await websocket.close(code=_CLOSE_POLICY_VIOLATION)
                break

    except WebSocketDisconnect:
        logger.info("User %s WebSocket disconnected normally", user_id)
    except Exception as exc:
        # Never leak internal error details back to the client.
        logger.error("WebSocket error for %s: %s", user_id, exc, exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:  # pragma: no cover -- best effort cleanup
            pass
    finally:
        try:
            await pipeline.end_session(user_id, session_id)
        except Exception:  # pragma: no cover -- already in teardown
            pass
        manager.disconnect(user_id)
