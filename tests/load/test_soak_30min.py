"""30-minute WebSocket soak test.

Sends one chat message every 10 seconds for 30 minutes (180 messages) to
verify:

* No latency regression across three 10-minute windows.
* Memory footprint (via :mod:`psutil`) does not grow by more than 50 MB.
* No error frames received from the server.
* No backpressure / disconnects.

Skipped by default because it takes 30 minutes. Run with::

    pytest tests/load/test_soak_30min.py -m slow
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.load]


# ---------------------------------------------------------------------------
# Soak-test configuration
# ---------------------------------------------------------------------------

_DEFAULT_HOST: str = os.environ.get("I3_SOAK_HOST", "127.0.0.1")
_DEFAULT_PORT: int = int(os.environ.get("I3_SOAK_PORT", "8000"))
_DEFAULT_USER_ID: str = os.environ.get("I3_SOAK_USER_ID", "soak_user")

_TOTAL_DURATION_S: int = 30 * 60          # 30 minutes
_INTERVAL_S: float = 10.0                 # one message every 10 seconds
_EXPECTED_MESSAGES: int = _TOTAL_DURATION_S // int(_INTERVAL_S)  # 180
_WINDOW_S: int = 10 * 60                  # 10-minute reporting windows
_MAX_MEMORY_GROWTH_MB: float = 50.0
_RECV_TIMEOUT_S: float = 6.0              # must be < _INTERVAL_S


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return ``True`` iff a TCP connect to *host:port* completes.

    The soak test auto-skips when the server is unreachable, so devs can
    run the full suite locally without spinning up an I3 process.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values* (linear interpolation).

    Args:
        values: Sorted or unsorted list of floats.
        pct: Percentile in [0, 100].
    """
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    idx = (pct / 100.0) * (len(data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(data) - 1)
    weight = idx - lower
    return float(data[lower] * (1.0 - weight) + data[upper] * weight)


@dataclass
class WindowStats:
    """Latency statistics for one 10-minute reporting window."""

    index: int
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0

    def summary(self) -> dict[str, Any]:
        """Return P50/P95/P99/mean as a JSON-friendly dict."""
        return {
            "window": self.index,
            "n": len(self.latencies_ms),
            "errors": self.errors,
            "p50_ms": round(_percentile(self.latencies_ms, 50), 2),
            "p95_ms": round(_percentile(self.latencies_ms, 95), 2),
            "p99_ms": round(_percentile(self.latencies_ms, 99), 2),
            "mean_ms": round(
                sum(self.latencies_ms) / max(len(self.latencies_ms), 1), 2
            ),
        }


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------


def _make_message(index: int) -> dict[str, Any]:
    """Build a synthetic chat message for the soak loop."""
    return {
        "type": "message",
        "text": (
            f"soak message {index} — a steady-cadence probe that exercises "
            "the TCN + user model + bandit + cloud fallback path."
        ),
        "timestamp": time.time(),
        "composition_time_ms": 3200.0,
        "edit_count": 1,
        "pause_before_send_ms": 400.0,
    }


async def _soak_run(url: str, duration_s: int = _TOTAL_DURATION_S) -> dict[str, Any]:
    """Drive the soak loop and return aggregate statistics.

    Args:
        url: WebSocket URL of the running I3 server.
        duration_s: Total runtime in seconds; defaults to 30 minutes.

    Returns:
        Dict with per-window summaries and global checks.
    """
    try:
        import websockets
        from websockets.exceptions import ConnectionClosed, WebSocketException
    except ImportError:
        pytest.skip("websockets package not installed")

    try:
        import psutil
    except ImportError:
        pytest.skip("psutil package not installed")

    origin_parts = url.replace("ws://", "http://").replace("wss://", "https://").split("/")
    origin = "/".join(origin_parts[:3])

    process = psutil.Process(os.getpid())
    baseline_mb = process.memory_info().rss / (1024.0 * 1024.0)

    windows: list[WindowStats] = [WindowStats(index=i) for i in range(3)]
    disconnects = 0
    errors: list[dict[str, Any]] = []

    start = time.monotonic()
    messages_sent = 0
    async with websockets.connect(
        url, origin=origin, open_timeout=10.0, ping_interval=20
    ) as ws:
        while time.monotonic() - start < duration_s:
            elapsed = time.monotonic() - start
            window_idx = min(int(elapsed // _WINDOW_S), len(windows) - 1)
            window = windows[window_idx]

            payload = _make_message(messages_sent)
            send_t = time.perf_counter()
            try:
                await ws.send(json.dumps(payload))
            except (ConnectionClosed, WebSocketException) as exc:
                errors.append({"phase": "send", "err": type(exc).__name__})
                window.errors += 1
                disconnects += 1
                break

            # Wait for the "response" frame specifically. state_update and
            # diary_entry frames are recorded but not counted toward the
            # message-level latency window.
            try:
                while True:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=_RECV_TIMEOUT_S
                    )
                    frame = json.loads(raw)
                    if frame.get("type") == "response":
                        latency_ms = (time.perf_counter() - send_t) * 1000.0
                        window.latencies_ms.append(latency_ms)
                        break
                    if frame.get("type") == "error":
                        window.errors += 1
                        errors.append({"phase": "recv", "err": frame})
                        break
            except asyncio.TimeoutError:
                window.errors += 1
                errors.append({"phase": "recv_timeout", "idx": messages_sent})
            except (ConnectionClosed, WebSocketException) as exc:
                errors.append({"phase": "recv_ws", "err": type(exc).__name__})
                window.errors += 1
                disconnects += 1
                break

            messages_sent += 1

            # Wait until the next 10-second boundary, preserving cadence.
            next_tick = start + (messages_sent * _INTERVAL_S)
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    final_mb = process.memory_info().rss / (1024.0 * 1024.0)
    growth_mb = final_mb - baseline_mb

    return {
        "messages_sent": messages_sent,
        "disconnects": disconnects,
        "errors": errors,
        "windows": [w.summary() for w in windows],
        "memory_baseline_mb": round(baseline_mb, 2),
        "memory_final_mb": round(final_mb, 2),
        "memory_growth_mb": round(growth_mb, 2),
    }


def test_soak_30min() -> None:
    """Run a 30-minute steady-cadence soak and assert the health invariants.

    Auto-skips when the target server is not reachable.
    """
    host = _DEFAULT_HOST
    port = _DEFAULT_PORT
    if not _server_reachable(host, port):
        pytest.skip(f"I3 server not reachable at {host}:{port}")

    url = f"ws://{host}:{port}/ws/{_DEFAULT_USER_ID}"
    # SEC: explicit origin match for the server-side allow-list.
    parsed = urlparse(url)
    assert parsed.scheme == "ws"

    stats = asyncio.run(_soak_run(url))

    # --- Invariants ----------------------------------------------------
    assert stats["disconnects"] == 0, (
        f"backpressure / disconnect detected: {stats['disconnects']}"
    )
    assert stats["messages_sent"] >= int(0.95 * _EXPECTED_MESSAGES), (
        f"sent only {stats['messages_sent']} of {_EXPECTED_MESSAGES} messages"
    )
    assert stats["memory_growth_mb"] <= _MAX_MEMORY_GROWTH_MB, (
        f"memory grew by {stats['memory_growth_mb']} MB "
        f"(cap {_MAX_MEMORY_GROWTH_MB} MB)"
    )
    # No error frames at all.
    assert not stats["errors"], f"unexpected errors during soak: {stats['errors']!r}"

    # Latency regression across windows — P95 in window 3 must not exceed
    # 2.0 * window 1 P95, and no window may be missing observations.
    w1, w2, w3 = stats["windows"]
    assert w1["n"] >= 50, f"window 1 underfilled: {w1}"
    assert w2["n"] >= 50, f"window 2 underfilled: {w2}"
    assert w3["n"] >= 50, f"window 3 underfilled: {w3}"
    if w1["p95_ms"] > 0:
        ratio = w3["p95_ms"] / w1["p95_ms"]
        assert ratio <= 2.0, (
            f"P95 regression across windows: w1={w1['p95_ms']}ms "
            f"w3={w3['p95_ms']}ms ratio={ratio:.2f}"
        )

    # Emit a human-readable summary to pytest's captured stdout so operators
    # can copy-paste it into the demo dry-run log.
    print(json.dumps(stats, indent=2))  # noqa: T201
