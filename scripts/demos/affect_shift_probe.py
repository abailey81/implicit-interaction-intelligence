"""Affect-shift probe — verifies the Huawei HMI Lab pitch piece end-to-end.

Sends six messages over a single WebSocket session against a running
I3 server with deliberately escalating keystroke metrics:

* Turns 1–3: ``composition_time_ms=1200``, ``edit_count=0``,
  ``IKI=100ms`` (calm baseline).
* Turns 4–6: ``composition_time_ms=3500``, ``edit_count=4``,
  ``IKI=180ms`` (rushed / frustrated).

Expectations:

* Turns 1–3 produce ``affect_shift.detected = False``.
* The first rushed turn that satisfies the detector's window
  conditions (typically turn 4 or 5) flips ``detected = True`` with
  ``direction = "rising_load"``.
* When ``detected`` and not debounced, the server appends the
  detector's polite check-in to ``response_text``.
* The reasoning trace's first paragraph contains
  ``"Affect-shift detected this turn"`` on the shift turn.

Run with the server already up::

    python scripts/demos/affect_shift_probe.py

Optional env vars:

* ``I3_PROBE_HOST`` (default ``127.0.0.1``)
* ``I3_PROBE_PORT`` (default ``8000``)
* ``I3_PROBE_USER_ID`` (default ``affect_probe``)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time

import websockets

# Force stdout/stderr to UTF-8 so the σ character in the reasoning
# trace doesn't crash the probe on Windows consoles whose default
# code page (cp1251 / cp1252) can't encode it.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )


HOST = os.environ.get("I3_PROBE_HOST", "127.0.0.1")
PORT = int(os.environ.get("I3_PROBE_PORT", "8000"))
USER_ID = os.environ.get("I3_PROBE_USER_ID", "affect_probe")

CALM_MESSAGES = [
    "Hi, just saying hello to start.",
    "How are things on your side today?",
    "Cool. Tell me a quick fact about edge AI.",
]
RUSHED_MESSAGES = [
    "wait actually never mind that one",
    "argh I keep typing this wrong sorry",
    "ok this is getting frustrating, hmm",
]


async def _drain_until(
    ws: websockets.WebSocketClientProtocol,
    *,
    expect_types: set[str],
    timeout: float = 25.0,
) -> dict:
    """Read frames until one of *expect_types* arrives or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        try:
            frame = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(frame, dict) and frame.get("type") in expect_types:
            return frame
    raise TimeoutError(f"never saw frame of type {expect_types}")


async def _send_turn(
    ws: websockets.WebSocketClientProtocol,
    *,
    text: str,
    composition_time_ms: float,
    edit_count: int,
    iki_mean_ms: float,
    n_keystrokes: int = 12,
) -> None:
    """Stream synthetic keystrokes then send the message frame."""
    # 1. Stream keystroke events with the requested mean IKI so the
    #    server-side InteractionMonitor sees a real timing signal.
    for i in range(n_keystrokes):
        await ws.send(json.dumps({
            "type": "keystroke",
            "timestamp": time.time(),
            "key_type": "char",
            "inter_key_interval_ms": float(iki_mean_ms),
        }))
    # 2. Send the message itself (server reads composition fields
    #    from the top level of the frame).
    await ws.send(json.dumps({
        "type": "message",
        "text": text,
        "timestamp": time.time(),
        "composition_time_ms": float(composition_time_ms),
        "edit_count": int(edit_count),
        "pause_before_send_ms": float(iki_mean_ms * 1.5),
    }))


def _summarise_shift(frame: dict) -> str:
    sh = frame.get("affect_shift") or {}
    if not sh:
        return "affect_shift=<absent>"
    if not sh.get("detected"):
        return (
            "affect_shift={detected: false, "
            f"direction: {sh.get('direction')!s}, "
            f"magnitude: {float(sh.get('magnitude', 0.0)):.2f}, "
            f"iki%: {float(sh.get('iki_delta_pct', 0.0)):+.0f}, "
            f"edit%: {float(sh.get('edit_delta_pct', 0.0)):+.0f}}}"
        )
    return (
        "affect_shift={detected: TRUE, "
        f"direction: {sh.get('direction')!s}, "
        f"magnitude: {float(sh.get('magnitude', 0.0)):.2f}, "
        f"iki%: {float(sh.get('iki_delta_pct', 0.0)):+.0f}, "
        f"edit%: {float(sh.get('edit_delta_pct', 0.0)):+.0f}, "
        f"suggestion: {(sh.get('suggestion') or '')[:80]!r}}}"
    )


async def main() -> int:
    url = f"ws://{HOST}:{PORT}/ws/{USER_ID}"
    headers = [("Origin", f"http://{HOST}:{PORT}")]
    print(f"connecting to {url} ...")

    # The websockets API renamed the header kwarg between releases;
    # try the newer form first, fall back to the legacy one.
    try:
        ws_cm = websockets.connect(url, additional_headers=headers)
    except TypeError:
        ws_cm = websockets.connect(url, extra_headers=headers)

    async with ws_cm as ws:
        # Server greeting
        greet = await _drain_until(ws, expect_types={"session_started"})
        session_id = greet.get("session_id")
        print(f"session started: {session_id}")

        turns: list[dict] = []
        for idx, text in enumerate(CALM_MESSAGES, start=1):
            print(f"\n--- TURN {idx} (calm) ---")
            await _send_turn(
                ws,
                text=text,
                composition_time_ms=1200.0,
                edit_count=0,
                iki_mean_ms=100.0,
            )
            frame = await _drain_until(
                ws, expect_types={"response", "response_done"}
            )
            print(f"  user>      {text!r}")
            ai = (frame.get("text") or "")
            print(f"  ai>        {ai[:160]!r}{'…' if len(ai) > 160 else ''}")
            print(f"  {_summarise_shift(frame)}")
            rt = frame.get("reasoning_trace") or {}
            paras = rt.get("narrative_paragraphs") or []
            if paras:
                print(f"  trace[0]>  {paras[0][:160]!r}")
            turns.append(frame)

        for idx, text in enumerate(RUSHED_MESSAGES, start=4):
            print(f"\n--- TURN {idx} (rushed) ---")
            await _send_turn(
                ws,
                text=text,
                composition_time_ms=3500.0,
                edit_count=4,
                iki_mean_ms=180.0,
            )
            frame = await _drain_until(
                ws, expect_types={"response", "response_done"}
            )
            print(f"  user>      {text!r}")
            ai = (frame.get("text") or "")
            print(f"  ai>        {ai[:240]!r}{'…' if len(ai) > 240 else ''}")
            print(f"  {_summarise_shift(frame)}")
            rt = frame.get("reasoning_trace") or {}
            paras = rt.get("narrative_paragraphs") or []
            if paras:
                print(f"  trace[0]>  {paras[0][:240]!r}")
            turns.append(frame)

        # End the session cleanly
        try:
            await ws.send(json.dumps({"type": "session_end"}))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    calm_ok = all(
        not ((t.get("affect_shift") or {}).get("detected"))
        for t in turns[:3]
    )
    rushed_any = any(
        (t.get("affect_shift") or {}).get("detected")
        for t in turns[3:]
    )
    rushed_dirs = [
        (t.get("affect_shift") or {}).get("direction")
        for t in turns[3:]
    ]
    suggestion_appended = any(
        (t.get("affect_shift") or {}).get("suggestion")
        and (t.get("affect_shift") or {}).get("suggestion") in (t.get("text") or "")
        for t in turns[3:]
    )
    rushed_trace_has_phrase = any(
        any(
            "Affect-shift detected" in p
            for p in (t.get("reasoning_trace") or {}).get("narrative_paragraphs", [])
        )
        for t in turns[3:]
    )

    print("\n=== verification ===")
    print(f"calm turns 1-3 had no shift           : {calm_ok}")
    print(f"rushed turns 4-6 produced ANY shift   : {rushed_any}")
    print(f"rushed-turn directions                : {rushed_dirs}")
    print(f"suggestion appended to a response     : {suggestion_appended}")
    print(f"reasoning-trace para1 cites the shift : {rushed_trace_has_phrase}")

    ok = calm_ok and rushed_any and suggestion_appended and rushed_trace_has_phrase
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)
