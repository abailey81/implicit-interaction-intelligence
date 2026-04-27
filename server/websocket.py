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

from i3.explain.reasoning_trace import build_reasoning_trace
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
    """Manages active WebSocket connections and session bookkeeping.

    A single ``user_id`` may have *multiple* concurrent sockets open — the
    demo often runs in two browser windows side-by-side (desktop + mobile
    emulator), or a user may have two tabs pointed at the UI.  The old
    "evict the prior on every new connect" behaviour turned that into a
    reconnect war: the tabs fought over the slot, every connection got
    closed with 1001 within ~2 s, and the UI showed a constant "Reconnect"
    badge.  We now track every live socket per user and fan-out server
    messages to all of them.

    Each ``(user_id, websocket)`` pair still owns its own ``session_id``;
    multi-connection users get multiple simultaneous sessions, which the
    rest of the pipeline already tolerates.
    """

    def __init__(self) -> None:
        # user_id -> {websocket: session_id}
        self.active_connections: dict[str, dict[WebSocket, str]] = {}
        # SEC: Async lock guarding all mutations so concurrent
        # connect()/disconnect() never race.
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Accept the socket and register a fresh session id for it."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        async with self._lock:
            self.active_connections.setdefault(user_id, {})[websocket] = session_id
        logger.info(
            "User %s connected, session %s (active_sockets=%d)",
            user_id,
            session_id,
            len(self.active_connections.get(user_id, {})),
        )
        return session_id

    async def disconnect(self, user_id: str, websocket: WebSocket | None = None) -> None:
        """Remove *websocket* from *user_id*'s active-sockets map."""
        async with self._lock:
            bucket = self.active_connections.get(user_id)
            if not bucket:
                return
            if websocket is None:
                self.active_connections.pop(user_id, None)
            else:
                bucket.pop(websocket, None)
                if not bucket:
                    self.active_connections.pop(user_id, None)
        logger.info("User %s disconnected", user_id)

    async def send_json(self, user_id: str, data: dict[str, Any]) -> None:
        """Fan-out a JSON payload to every live socket for *user_id*.

        Closed / backpressure-errored sockets are logged and silently
        dropped — never propagated to the caller.
        """
        bucket = self.active_connections.get(user_id)
        if not bucket:
            return
        # Iterate over a snapshot so sends that fail + trigger concurrent
        # disconnect() from their handler cannot mutate the dict under us.
        for ws in list(bucket.keys()):
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
        # FIX: pass our already-generated session_id so the diary
        # ``sessions`` row matches the id used for subsequent
        # ``log_exchange`` calls.  Without this, every chat turn
        # silently raised sqlite3.IntegrityError (FOREIGN KEY).
        await pipeline.start_session(user_id, session_id=session_id)
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
    # Per-handler tracker for rising-edge biometric events so we can
    # emit ``biometric_event`` frames once per transition rather than
    # on every turn the user is in the same state.
    biometric_state_tracker: dict[str, str] = {}

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
                # SEC/BUG (B-1, 2026-04-23 audit): process_keystroke is
                # ``async def``.  The prior code dropped the returned
                # coroutine without awaiting it, so every keystroke event
                # from every WebSocket client was silently discarded and
                # the TCN encoder was fed zero-metric feature vectors.
                # The behavioural-baseline invariant depended on this
                # await actually happening.
                await pipeline.monitor.process_keystroke(user_id, ks)
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
                #
                # Accept metrics either at the top level of the frame
                # (legacy) or nested under ``composition_metrics`` —
                # the chat.js client + biometric/affect probes send the
                # nested form, so without this the authenticator sees
                # zero composition / edit / IKI on every turn and can
                # never distinguish typing patterns.
                comp_metrics = data.get("composition_metrics") or {}
                if not isinstance(comp_metrics, dict):
                    comp_metrics = {}

                def _pick_metric(key: str, default=0):
                    if key in data and data[key] is not None:
                        return data[key]
                    return comp_metrics.get(key, default)

                try:
                    msg_ts = _safe_float(data.get("timestamp", time.time()))
                    composition_ms = _safe_float(_pick_metric("composition_time_ms", 0))
                    edit_count = _safe_int(_pick_metric("edit_count", 0))
                    pause_ms = _safe_float(_pick_metric("pause_before_send_ms", 0))
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

                # Voice-prosody flagship feature: validate the optional
                # ``prosody_features`` field on the message frame.  The
                # JS-side ``VoiceProsodyMonitor`` ships eight numeric
                # scalars (pace, pitch mean/var, RMS energy mean/var,
                # voiced ratio, pause density, spectral centroid) plus
                # two metadata fields (samples_count, captured_seconds).
                # The audio buffer NEVER leaves the browser — only these
                # ten scalars do.  We validate via
                # :func:`i3.multimodal.validate_prosody_payload` which
                # rejects NaN/inf, missing keys, and non-dict payloads;
                # rejection degrades gracefully to the keystroke-only
                # path rather than closing the socket, since prosody is
                # a soft signal.
                from i3.multimodal.prosody import (
                    validate_gaze_payload,
                    validate_prosody_payload,
                )
                raw_prosody = data.get("prosody_features")
                validated_prosody = validate_prosody_payload(raw_prosody)
                if raw_prosody is not None and validated_prosody is None:
                    logger.debug(
                        "Rejected malformed prosody_features payload from %s",
                        user_id,
                    )
                prosody_features_dict: dict | None = (
                    validated_prosody.to_dict()
                    if validated_prosody is not None
                    else None
                )

                # Vision-gaze flagship: the JS side ships an aggregate
                # ``gaze_features`` dict on the message frame.  We
                # validate via :func:`validate_gaze_payload` (which
                # returns the 8-d numeric scalar dict on success);
                # rejection degrades to the camera-off path.  The
                # raw image NEVER reaches this code — only the
                # already-classified label + numeric scalars.
                raw_gaze = data.get("gaze_features")
                validated_gaze_scalars = validate_gaze_payload(raw_gaze)
                gaze_features_dict: dict | None = None
                if validated_gaze_scalars is not None and isinstance(raw_gaze, dict):
                    # Pass the original (unvalidated) dict through to
                    # the engine — the engine validates again and
                    # builds the JSON-safe output dict that lands on
                    # PipelineOutput.gaze.  Validation here is purely
                    # to gate whether the engine sees the payload at
                    # all.
                    gaze_features_dict = dict(raw_gaze)
                elif raw_gaze is not None:
                    logger.debug(
                        "Rejected malformed gaze_features payload from %s",
                        user_id,
                    )

                pipeline_input = PipelineInput(
                    user_id=user_id,
                    session_id=session_id,
                    message_text=text,
                    timestamp=msg_ts,
                    composition_time_ms=composition_ms,
                    edit_count=edit_count,
                    pause_before_send_ms=pause_ms,
                    # Prefer the per-event server-side buffer (sampled
                    # by the live JS client).  Fall back to a client-
                    # supplied keystroke_timings array on the message
                    # frame (used by Python probes that don't stream
                    # individual keystroke events).
                    keystroke_timings=(
                        [
                            float(ks.get("inter_key_interval_ms", 0))
                            for ks in keystroke_buffer
                        ]
                        if keystroke_buffer
                        else [
                            float(t)
                            for t in (
                                comp_metrics.get("keystroke_timings") or []
                            )
                            if isinstance(t, (int, float))
                        ]
                    ),
                    prosody_features=prosody_features_dict,
                    gaze_features=gaze_features_dict,
                )

                # ------------------------------------------------------
                # Streaming hook: if the pipeline picks the SLM path it
                # will invoke this callback once per decoded token delta.
                # Retrieval / tool / OOD paths never call it (there are
                # no intermediate tokens to emit), so streaming_started
                # remains False and the single-frame response below
                # still fires the usual {"type": "response"} behaviour.
                # ------------------------------------------------------
                streaming_state: dict[str, Any] = {"started": False}

                async def _on_token(delta: str) -> None:
                    if not isinstance(delta, str) or not delta:
                        return
                    streaming_state["started"] = True
                    await manager.send_json(
                        user_id,
                        {"type": "token", "delta": delta},
                    )

                output = await pipeline.process_message(
                    pipeline_input,
                    on_token=_on_token,
                )

                # Promote the StyleVector sub-axes to top-level keys so
                # both the state_update frame and the reasoning trace see
                # the same flattened shape the dashboard UI expects.
                adaptation_flat = dict(output.adaptation)
                style = adaptation_flat.get("style_mirror")
                if isinstance(style, dict):
                    for k in ("formality", "verbosity", "emotionality", "directness"):
                        if k in style and k not in adaptation_flat:
                            adaptation_flat[k] = style[k]

                # Convert the engine's AffectShift dataclass to a
                # plain dict for both the WS frame and the reasoning
                # trace.  ``None`` when the detector hasn't run yet
                # (legacy callers / first-turn flows).
                affect_shift_obj = getattr(output, "affect_shift", None)
                affect_shift_dict: dict | None = None
                if affect_shift_obj is not None:
                    try:
                        # Prefer the dataclass's own to_dict() so the
                        # serialisation contract lives next to the type.
                        affect_shift_dict = affect_shift_obj.to_dict()
                    except AttributeError:
                        # Defensive: a future change might pass a dict
                        # straight through.
                        if isinstance(affect_shift_obj, dict):
                            affect_shift_dict = dict(affect_shift_obj)

                # Live State Badge label + Accessibility Mode state —
                # both are already serialised to plain dicts inside
                # the engine helpers, so we just defensively coerce
                # them here.  ``None`` is a valid value (legacy
                # callers / classifier wasn't wired in).
                user_state_label_obj = getattr(output, "user_state_label", None)
                user_state_label_dict: dict | None = (
                    dict(user_state_label_obj)
                    if isinstance(user_state_label_obj, dict)
                    else None
                )
                accessibility_obj = getattr(output, "accessibility", None)
                accessibility_dict: dict | None = (
                    dict(accessibility_obj)
                    if isinstance(accessibility_obj, dict)
                    else None
                )

                # Biometric Identity Lock — keystroke-based continuous
                # auth result for this turn.  Already a plain dict
                # (the engine calls ``BiometricMatch.to_dict()``); we
                # coerce defensively.  ``None`` when the authenticator
                # isn't wired (legacy callers).
                biometric_obj = getattr(output, "biometric", None)
                biometric_dict: dict | None = (
                    dict(biometric_obj)
                    if isinstance(biometric_obj, dict)
                    else None
                )

                # Self-critique trace (Phase 7 HMI piece).  Populated
                # only on SLM-generation turns by the pipeline; the
                # engine itself nulls it for retrieval / tool / OOD
                # paths so we don't have to filter again here.
                critique_obj = getattr(output, "critique", None)
                critique_dict: dict | None = (
                    dict(critique_obj)
                    if isinstance(critique_obj, dict) and critique_obj
                    else None
                )

                # Co-reference resolution (multi-turn understanding).
                # ``None`` when no pronoun was detected on this turn or
                # when the user message had no compatible entity in
                # scope.  When non-``None`` the WS layer ships it on
                # the response/response_done frame so the chat UI can
                # render the ``coref · they → huawei`` chip.
                coref_obj = getattr(output, "coreference_resolution", None)
                coref_dict: dict | None = (
                    dict(coref_obj)
                    if isinstance(coref_obj, dict) and coref_obj
                    else None
                )

                # Voice-prosody multimodal fusion status.  Always a dict
                # when the engine ran (``prosody_active`` is False on
                # mic-off turns); ``None`` only on legacy callers with
                # no multimodal head wired in.  Drives the
                # ``voice prosody · active`` chat chip and the
                # reasoning-trace fusion sentence.
                multimodal_obj = getattr(output, "multimodal", None)
                multimodal_dict: dict | None = (
                    dict(multimodal_obj)
                    if isinstance(multimodal_obj, dict)
                    else None
                )

                # Vision-gaze classifier output (third multimodal
                # flagship — fine-tuned MobileNetV3 head).  Always
                # ``None`` when the camera was off this turn.
                gaze_obj = getattr(output, "gaze", None)
                gaze_dict_out: dict | None = (
                    dict(gaze_obj)
                    if isinstance(gaze_obj, dict)
                    else None
                )

                # Per-turn pipeline trace (Flow dashboard — third
                # flagship surface).  Always a dict when the engine
                # ran; ``None`` only on legacy callers / build-error
                # outputs.  Drives ``web/js/flow_dashboard.js`` which
                # animates each stage box pulsing in the order they
                # actually fired with real measurements.
                pipeline_trace_obj = getattr(output, "pipeline_trace", None)
                pipeline_trace_dict: dict | None = (
                    dict(pipeline_trace_obj)
                    if isinstance(pipeline_trace_obj, dict)
                    else None
                )
                # Detect rising-edge biometric transitions so a
                # one-shot ``biometric_event`` frame can fire.  We
                # stash the previous state on the WS handler scope so
                # the state machine sees registering→registered and
                # registered→drift transitions even across long
                # idle gaps.
                _prev_biometric_state = (
                    biometric_state_tracker.get(user_id, "")
                )
                _curr_biometric_state = (
                    str(biometric_dict.get("state", ""))
                    if biometric_dict is not None
                    else ""
                )
                _biometric_event: str | None = None
                if biometric_dict is not None:
                    if (
                        _prev_biometric_state in ("", "unregistered", "registering")
                        and _curr_biometric_state == "registered"
                    ):
                        _biometric_event = "registered"
                    elif (
                        _prev_biometric_state in ("registered", "verifying")
                        and biometric_dict.get("drift_alert")
                    ):
                        _biometric_event = "drift_alert"
                    elif (
                        _prev_biometric_state in ("registered", "verifying")
                        and _curr_biometric_state == "mismatch"
                    ):
                        _biometric_event = "mismatch"
                    if _curr_biometric_state:
                        biometric_state_tracker[user_id] = _curr_biometric_state

                # Iter 20 (2026-04-26): active-topic snapshot.
                # Surfaces the entity tracker's current top-of-stack
                # entity (preferring user-anchored topic) so the chat
                # tab can render a small "active topic" pill.  This
                # makes the conversation-context engine VISIBLE to
                # the recruiter — they see "we're talking about:
                # transformer" without opening the State tab.
                active_topic_dict: dict | None = None
                topic_history_list: list[dict] | None = None
                try:
                    snap = pipeline._entity_tracker.snapshot(
                        user_id, session_id,
                    )
                    if snap:
                        # Prefer the topmost user-anchored topic, else
                        # the topmost ORG/TOPIC, else the topmost frame.
                        anchored = next(
                            (f for f in snap if f.user_anchor_turn is not None
                             and f.kind in {"org", "topic"}),
                            None,
                        )
                        topic_kind = next(
                            (f for f in snap if f.kind in {"org", "topic"}),
                            None,
                        )
                        chosen = anchored or topic_kind or snap[0]
                        # Iter 26 (2026-04-26): topic history breadcrumb
                        # — pull up to 4 distinct USER-ANCHORED ORG/TOPIC
                        # entities from the stack (excluding the current
                        # active) so the chat hero shows "earlier: A,
                        # B, C".  Filtering to user-anchored prevents
                        # incidentally-mentioned places ("Cupertino"
                        # because the assistant said "Apple is
                        # headquartered in Cupertino") from polluting
                        # the breadcrumb.
                        seen_canon = {chosen.canonical}
                        history: list[dict] = []
                        for f in snap:
                            if len(history) >= 4:
                                break
                            if f.canonical in seen_canon:
                                continue
                            if f.kind not in {"org", "topic"}:
                                continue
                            if f.user_anchor_turn is None:
                                continue
                            seen_canon.add(f.canonical)
                            history.append({
                                "canonical": f.canonical,
                                "surface": f.text,
                                "kind": f.kind,
                                "user_anchored": True,
                            })
                        if history:
                            topic_history_list = history
                        active_topic_dict = {
                            "canonical": chosen.canonical,
                            "surface": chosen.text,
                            "kind": chosen.kind,
                            "user_anchored": chosen.user_anchor_turn is not None,
                            "last_turn_idx": chosen.last_turn_idx,
                            "stack_depth": len(snap),
                        }
                except Exception:
                    active_topic_dict = None

                # Build the visible reasoning trace.  This is the HMI
                # artefact: every implicit signal that shaped the
                # response gets surfaced as plain English on the chat
                # bubble.  Wrapped in try/except so a trace-builder
                # failure can never block the response.
                reasoning_trace: dict | None = None
                try:
                    composition_metrics = {
                        "composition_time_ms": composition_ms,
                        "edit_count": edit_count,
                        "pause_before_send_ms": pause_ms,
                        "keystroke_timings": [
                            float(ks.get("inter_key_interval_ms", 0))
                            for ks in keystroke_buffer
                        ],
                    }
                    # Multi-turn history: number of prior turn pairs
                    # the engine consumed for this response.  When 0
                    # (first turn) the trace falls back to its
                    # single-turn narrative; otherwise paragraph 4
                    # mentions the carried context.
                    try:
                        history_turns_used = int(
                            pipeline.get_last_history_turns_used(
                                user_id, session_id
                            )
                        )
                    except Exception:  # pragma: no cover - defensive
                        history_turns_used = 0
                    reasoning_trace = build_reasoning_trace(
                        keystroke_metrics=composition_metrics,
                        adaptation=adaptation_flat,
                        adaptation_changes=list(
                            getattr(output, "adaptation_changes", []) or []
                        ),
                        engagement_score=float(output.engagement_score),
                        deviation_from_baseline=float(
                            output.deviation_from_baseline
                        ),
                        messages_in_session=int(output.messages_in_session),
                        baseline_established=bool(output.baseline_established),
                        routing_confidence=output.routing_confidence,
                        response_path=getattr(
                            output, "response_path", "unknown"
                        ),
                        retrieval_score=float(
                            getattr(output, "retrieval_score", 0.0)
                        ),
                        user_message_preview=text[:80] if text else "",
                        response_preview=(output.response_text or "")[:80],
                        user_state_embedding_2d=tuple(
                            output.user_state_embedding_2d
                        ),
                        history_turns_used=history_turns_used,
                        affect_shift=affect_shift_dict,
                        user_state_label=user_state_label_dict,
                        accessibility=accessibility_dict,
                        biometric=getattr(
                            output, "biometric", None
                        ),
                        critique=critique_dict,
                        coreference_resolution=coref_dict,
                        personalisation=getattr(
                            output, "personalisation", None
                        ),
                        multimodal=multimodal_dict,
                        gaze=gaze_dict_out,
                        routing_decision=getattr(
                            output, "routing_decision", None
                        ),
                        privacy_budget=getattr(
                            output, "privacy_budget", None
                        ),
                        session_memory=getattr(
                            output, "session_memory", None
                        ),
                        explain_plan=getattr(
                            output, "explain_plan", None
                        ),
                    )
                except Exception:  # pragma: no cover - never block on trace
                    logger.warning(
                        "reasoning_trace failed for %s; shipping response "
                        "without it",
                        user_id,
                        exc_info=True,
                    )
                    reasoning_trace = None

                # Main response.  We also ship ``response_path`` (which
                # internal sub-stage of the hybrid stack answered) and
                # ``retrieval_score`` (cosine score of the top match) so
                # the UI can surface a confidence chip and light the
                # correct pipeline LED.
                if streaming_state["started"]:
                    # SLM path — the UI already rendered tokens one by
                    # one.  Send ``response_done`` so the streaming
                    # bubble can be finalised with the full text and
                    # the metadata chips.
                    response_frame: dict[str, Any] = {
                        "type": "response_done",
                        "text": output.response_text,
                        "route": output.route_chosen,
                        "latency_ms": round(output.latency_ms),
                        "timestamp": time.time(),
                        "response_path": getattr(output, "response_path", "unknown"),
                        "retrieval_score": getattr(output, "retrieval_score", 0.0),
                        "adaptation_changes": list(
                            getattr(output, "adaptation_changes", []) or []
                        ),
                        # Iteration 12 (2026-04-26): include the FULL
                        # adaptation snapshot (all 8 axes flattened) so
                        # the per-bubble Decision Trace expander in the
                        # chat tab can render bars without an extra
                        # round-trip.
                        "adaptation": adaptation_flat,
                        # Iteration 20: active conversation topic from
                        # the entity tracker (anchored topic wins).
                        "active_topic": active_topic_dict,
                        # Iteration 26: topic-history breadcrumb (last
                        # 4 distinct topics, excluding active).
                        "topic_history": topic_history_list,
                        "affect_shift": affect_shift_dict,
                        # Iter 51: safety caveat shipped as side-channel
                        # field (not inlined in response_text).  Frontend
                        # renders as "ⓘ moderation note" pill.
                        "safety_caveat": getattr(output, "safety_caveat", None),
                        "user_state_label": user_state_label_dict,
                        "accessibility": accessibility_dict,
                        "biometric": biometric_dict,
                        "critique": critique_dict,
                        "coreference_resolution": coref_dict,
                        "personalisation": getattr(
                            output, "personalisation", None
                        ),
                        "multimodal": multimodal_dict,
                        "gaze": gaze_dict_out,
                        "pipeline_trace": pipeline_trace_dict,
                        "routing_decision": getattr(
                            output, "routing_decision", None
                        ),
                        "privacy_budget": getattr(
                            output, "privacy_budget", None
                        ),
                        "safety": getattr(output, "safety", None),
                        "session_memory": getattr(output, "session_memory", None),
                        "explain_plan": getattr(output, "explain_plan", None),
                        # Iter 51: per-(user, session) stated facts so
                        # the Personal Facts dashboard tab can render
                        # live (wireFactsTab in huawei_tabs.js).
                        "personal_facts": getattr(output, "personal_facts", None),
                        # Iter 51: structured intent-parse result for
                        # turns where the engine ran the parser; None
                        # for normal chat.  Surfaced as a green chip.
                        "intent_result": getattr(output, "intent_result", None),
                    }
                    if reasoning_trace is not None:
                        response_frame["reasoning_trace"] = reasoning_trace
                    await manager.send_json(user_id, response_frame)
                else:
                    # Retrieval / tool / OOD — single-frame behaviour.
                    response_frame = {
                        "type": "response",
                        "text": output.response_text,
                        "route": output.route_chosen,
                        "latency_ms": round(output.latency_ms),
                        "timestamp": time.time(),
                        "response_path": getattr(output, "response_path", "unknown"),
                        "retrieval_score": getattr(output, "retrieval_score", 0.0),
                        "adaptation_changes": list(
                            getattr(output, "adaptation_changes", []) or []
                        ),
                        # Iteration 12 (2026-04-26): include the FULL
                        # adaptation snapshot (all 8 axes flattened) so
                        # the per-bubble Decision Trace expander in the
                        # chat tab can render bars without an extra
                        # round-trip.
                        "adaptation": adaptation_flat,
                        # Iteration 20: active conversation topic from
                        # the entity tracker (anchored topic wins).
                        "active_topic": active_topic_dict,
                        # Iteration 26: topic-history breadcrumb (last
                        # 4 distinct topics, excluding active).
                        "topic_history": topic_history_list,
                        "affect_shift": affect_shift_dict,
                        # Iter 51: safety caveat shipped as side-channel
                        # field (not inlined in response_text).  Frontend
                        # renders as "ⓘ moderation note" pill.
                        "safety_caveat": getattr(output, "safety_caveat", None),
                        "user_state_label": user_state_label_dict,
                        "accessibility": accessibility_dict,
                        "biometric": biometric_dict,
                        "critique": critique_dict,
                        "coreference_resolution": coref_dict,
                        "personalisation": getattr(
                            output, "personalisation", None
                        ),
                        "multimodal": multimodal_dict,
                        "gaze": gaze_dict_out,
                        "pipeline_trace": pipeline_trace_dict,
                        "routing_decision": getattr(
                            output, "routing_decision", None
                        ),
                        "privacy_budget": getattr(
                            output, "privacy_budget", None
                        ),
                        "safety": getattr(output, "safety", None),
                        "session_memory": getattr(output, "session_memory", None),
                        "explain_plan": getattr(output, "explain_plan", None),
                        # Iter 51: per-(user, session) stated facts so
                        # the Personal Facts dashboard tab can render
                        # live (wireFactsTab in huawei_tabs.js).
                        "personal_facts": getattr(output, "personal_facts", None),
                        # Iter 51: structured intent-parse result for
                        # turns where the engine ran the parser; None
                        # for normal chat.  Surfaced as a green chip.
                        "intent_result": getattr(output, "intent_result", None),
                    }
                    if reasoning_trace is not None:
                        response_frame["reasoning_trace"] = reasoning_trace
                    await manager.send_json(user_id, response_frame)

                # Biometric rising-edge event — fired once per
                # state transition (registered / drift_alert / mismatch)
                # so the front-end can play a one-shot animation
                # without re-triggering on every subsequent turn while
                # the state stays stable.
                if _biometric_event and biometric_dict is not None:
                    bio_evt_frame: dict[str, Any] = {
                        "type": "biometric_event",
                        "event": _biometric_event,
                        "timestamp": time.time(),
                    }
                    bio_evt_frame.update(biometric_dict)
                    await manager.send_json(user_id, bio_evt_frame)

                # Live State Badge frame — sent in addition to the
                # response so the badge updates even mid-conversation
                # (i.e. the same state classifier output gets surfaced
                # both as part of the chat message metadata chips AND
                # as the standalone nav-badge update).
                if user_state_label_dict is not None:
                    badge_frame: dict[str, Any] = {
                        "type": "state_badge",
                        "timestamp": time.time(),
                    }
                    badge_frame.update(user_state_label_dict)
                    await manager.send_json(user_id, badge_frame)

                # Accessibility-mode rising / falling edge frame.  Only
                # emitted on the actual transition turns so the UI
                # animation can play once and not re-trigger on every
                # subsequent turn while the mode stays active.
                if accessibility_dict is not None and (
                    accessibility_dict.get("activated_this_turn")
                    or accessibility_dict.get("deactivated_this_turn")
                ):
                    change_frame: dict[str, Any] = {
                        "type": "accessibility_change",
                        "timestamp": time.time(),
                    }
                    change_frame.update(accessibility_dict)
                    await manager.send_json(user_id, change_frame)

                # Behavioural state update (used by the dashboard viz).
                # The dashboard UI expects the four StyleVector sub-
                # dimensions (formality / verbosity / emotionality /
                # directness) at the *top level* of the ``adaptation``
                # object alongside ``cognitive_load`` / ``emotional_tone``.
                # ``adaptation_flat`` was already flattened above so the
                # reasoning trace and the dashboard observe the same
                # shape on every turn.
                await manager.send_json(user_id, {
                    "type": "state_update",
                    "user_state_embedding_2d": list(output.user_state_embedding_2d),
                    "adaptation": adaptation_flat,
                    "engagement_score": output.engagement_score,
                    "deviation_from_baseline": output.deviation_from_baseline,
                    "routing_confidence": output.routing_confidence,
                    "messages_in_session": output.messages_in_session,
                    "baseline_established": output.baseline_established,
                    "route_chosen": output.route_chosen,
                    "routing_decision": getattr(output, "routing_decision", None),
                    "privacy_budget": getattr(output, "privacy_budget", None),
                    "user_state_label": user_state_label_dict,
                    "accessibility": accessibility_dict,
                    "biometric": biometric_dict,
                    # Iter 20: active conversation topic (chat hero pill).
                    "active_topic": active_topic_dict,
                    # Iter 26: topic-history breadcrumb.
                    "topic_history": topic_history_list,
                    # Iter 51: side-channel safety caveat.
                    "safety_caveat": getattr(output, "safety_caveat", None),
                    # Iter 51: per-(user, session) stated facts for the
                    # Personal Facts dashboard tab — wireFactsTab in
                    # huawei_tabs.js listens for the i3:state_update
                    # CustomEvent and renders them.
                    "personal_facts": getattr(output, "personal_facts", None),
                    # Iter 51: structured intent-parse result if the
                    # engine ran the parser on this turn.
                    "intent_result": getattr(output, "intent_result", None),
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

            elif msg_type == "session_start":
                # SEC/UX: The browser's app.js sends ``{type: "session_start"}``
                # right after the socket opens as a handshake ack.  Treating
                # it as "unknown" closed every connection with 1008 the
                # instant the UI loaded, putting the client into an endless
                # reconnect loop (observed 2026-04-25 browser probe).  The
                # server-side session is already started on connect, so we
                # just acknowledge and continue.
                continue

            elif msg_type == "ping":
                # SEC/UX: Optional keepalive from the client -- respond so
                # proxies / load balancers don't time the socket out.
                await manager.send_json(
                    user_id, {"type": "pong", "timestamp": time.time()}
                )
                continue

            else:
                # SEC: Ignore unknown types instead of closing the socket.
                # Closing on an unknown type is a denial-of-service vector
                # for client-library version drift: a new UI field names
                # an unknown type once and the user can never chat again
                # because the server drops the socket before the message
                # round trip.  We log + drop the frame and keep the socket
                # open so the next valid frame still reaches the pipeline.
                logger.warning(
                    "Ignoring unknown WebSocket frame from %s: type=%r",
                    user_id,
                    msg_type,
                )
                continue

    except WebSocketDisconnect as _wd:
        logger.info(
            "User %s WebSocket disconnected normally (code=%r reason=%r)",
            user_id,
            getattr(_wd, "code", None),
            getattr(_wd, "reason", None),
        )
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
