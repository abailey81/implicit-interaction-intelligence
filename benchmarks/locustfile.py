"""Locust scenarios for the I3 HTTP + WebSocket surface.

Two user classes are defined:

* :class:`RestUser` -- hits the public REST endpoints
  (``/api/health``, ``/api/demo/seed``, ``/api/user/<id>/profile``).
* :class:`WsUser`   -- opens a WebSocket against ``/ws/<user_id>``,
  mimics the keystroke + message flow used by ``web/js/app.js``.

Ramp configuration (overridable via CLI flags):

* 10 users over 30 seconds
* steady for 5 minutes
* 30 second ramp-down

Example::

    locust -f benchmarks/locustfile.py \\
        --host http://localhost:8000 \\
        --users 10 --spawn-rate 1 --run-time 6m

A custom event ``ws_message_latency`` is fired on every WebSocket
round-trip so latency can be graphed in Locust's UI.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from typing import Any, Optional

try:  # pragma: no cover - optional dep
    from locust import HttpUser, LoadTestShape, between, events, task  # type: ignore
except Exception:  # noqa: BLE001
    # Locust may be absent in a minimal install; provide inert stand-ins
    # so static analysis still succeeds.
    HttpUser = object  # type: ignore[assignment,misc]
    LoadTestShape = object  # type: ignore[assignment,misc]

    def between(*_args: float, **_kw: float) -> Any:  # type: ignore[misc]
        return lambda *_a, **_k: None

    def task(*_args: Any, **_kw: Any) -> Any:  # type: ignore[misc]
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    class _Events:
        request = None

    events = _Events()  # type: ignore[assignment]


try:  # pragma: no cover
    from websocket import create_connection as _ws_connect  # type: ignore
except Exception:  # noqa: BLE001
    _ws_connect = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# REST scenario
# --------------------------------------------------------------------------- #


class RestUser(HttpUser):  # type: ignore[misc, valid-type]
    """User class that exercises the REST API."""

    wait_time = between(0.5, 2.0)

    user_id: str = ""

    def on_start(self) -> None:
        """Pick a stable synthetic user ID for the session."""
        self.user_id = f"locust-{uuid.uuid4().hex[:8]}"

    @task(3)
    def health(self) -> None:
        """Hit ``/api/health``.  Weight 3 (most frequent)."""
        self.client.get("/api/health", name="GET /api/health")

    @task(1)
    def seed_demo(self) -> None:
        """Hit ``/api/demo/seed``."""
        self.client.post(
            "/api/demo/seed",
            json={"user_id": self.user_id, "messages": 5},
            name="POST /api/demo/seed",
        )

    @task(2)
    def user_profile(self) -> None:
        """Hit ``/api/user/<user_id>/profile``."""
        self.client.get(
            f"/api/user/{self.user_id}/profile",
            name="GET /api/user/[id]/profile",
        )

    @task(1)
    def user_stats(self) -> None:
        """Hit ``/api/user/<user_id>/stats``."""
        self.client.get(
            f"/api/user/{self.user_id}/stats",
            name="GET /api/user/[id]/stats",
        )


# --------------------------------------------------------------------------- #
# WebSocket scenario
# --------------------------------------------------------------------------- #


def _fire_latency(name: str, elapsed_ms: float, exception: Optional[Exception]) -> None:
    """Emit a Locust request event for a WebSocket round-trip.

    Args:
        name: Logical request name (e.g. ``"ws.message"``).
        elapsed_ms: Round-trip duration in milliseconds.
        exception: Exception encountered, if any.
    """
    if events.request is None:  # pragma: no cover - locust absent
        return
    events.request.fire(
        request_type="WS",
        name=name,
        response_time=elapsed_ms,
        response_length=0,
        exception=exception,
        context={},
    )


class WsUser(HttpUser):  # type: ignore[misc, valid-type]
    """User class that simulates the chat WebSocket flow.

    Mirrors the client-side protocol implemented in ``web/js/app.js``:

    1. Connect to ``/ws/<user_id>``.
    2. Emit a few keystroke events (``type: "keystroke"``).
    3. Emit a message (``type: "message"``).
    4. Await the first ``type: "response"`` frame; measure latency.
    5. Sleep briefly, repeat.
    """

    wait_time = between(1.0, 3.0)

    user_id: str = ""
    _ws: Any = None

    def on_start(self) -> None:
        """Open a persistent WebSocket for the session."""
        self.user_id = f"locust-ws-{uuid.uuid4().hex[:8]}"
        if _ws_connect is None:
            return
        ws_url = self.host.replace("http://", "ws://").replace("https://", "wss://")
        try:
            self._ws = _ws_connect(f"{ws_url}/ws/{self.user_id}", timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            _fire_latency("ws.connect", 0.0, exc)
            self._ws = None

    def on_stop(self) -> None:
        """Close the WebSocket when the virtual user stops."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None

    @task
    def chat_turn(self) -> None:
        """Send a sequence of keystrokes followed by a final message."""
        if self._ws is None:
            return
        message = random.choice(
            [
                "Hello, how are you today?",
                "Can you summarise this paragraph for me?",
                "I'm feeling a bit anxious about work.",
                "Explain gradient descent in one sentence.",
            ]
        )
        try:
            # Simulate ~5 keystrokes.
            for ch in message[:5]:
                self._ws.send(
                    json.dumps(
                        {
                            "type": "keystroke",
                            "char": ch,
                            "timestamp_ms": int(time.time() * 1000),
                        }
                    )
                )
                time.sleep(0.04 + random.random() * 0.04)

            # Send the full message and time the first response frame.
            t0 = time.perf_counter()
            self._ws.send(
                json.dumps(
                    {
                        "type": "message",
                        "text": message,
                        "timestamp_ms": int(time.time() * 1000),
                    }
                )
            )
            # Drain frames until we see a response.
            deadline = time.time() + 10.0
            while time.time() < deadline:
                raw = self._ws.recv()
                try:
                    frame = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                if frame.get("type") == "response":
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    _fire_latency("ws.message", elapsed_ms, None)
                    break
            else:
                _fire_latency("ws.message", 10_000.0, TimeoutError("no response"))
        except Exception as exc:  # noqa: BLE001
            _fire_latency("ws.message", 0.0, exc)


# --------------------------------------------------------------------------- #
# Load shape: ramp up -> steady -> ramp down
# --------------------------------------------------------------------------- #


class RampSteadyRampShape(LoadTestShape):  # type: ignore[misc, valid-type]
    """10-user ramp up over 30s, steady for 5 min, ramp down for 30s."""

    stages: list[dict[str, float]] = [
        {"duration": 30.0, "users": 10, "spawn_rate": 1.0},
        {"duration": 30.0 + 300.0, "users": 10, "spawn_rate": 1.0},
        {"duration": 30.0 + 300.0 + 30.0, "users": 0, "spawn_rate": 1.0},
    ]

    def tick(self) -> Optional[tuple[int, float]]:
        """Return ``(target_users, spawn_rate)`` for the current run time."""
        run_time = self.get_run_time()  # type: ignore[attr-defined]
        for stage in self.stages:
            if run_time < stage["duration"]:
                return int(stage["users"]), float(stage["spawn_rate"])
        return None
