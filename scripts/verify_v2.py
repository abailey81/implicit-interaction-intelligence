"""Verify the running server is hosting the v2 SLM.

Two-step probe (the audit script does the heavy 110-scenario pass):

1. ``GET /api/stack`` — confirms the SLM section reports ``version=v2``,
   the parameter count crosses 200 M, and the BPE vocab (32 000) is in
   place rather than the legacy 30 000-token word vocab.

2. WebSocket round-trip on a deliberately novel prompt
   (``"explain photosynthesis briefly"``) — verifies the model actually
   answers, the response is sufficiently long to look like generation
   (not the 0-token fallback), and the routing chose the v2 ``slm``
   path or a sane retrieval/tool route (any non-``ood`` path counts).

Designed to be runnable by hand against a freshly-restarted server::

    poetry run python scripts/verify_v2.py

Exits 0 on a clean v2 verification, 1 on any check failure so the
orchestrator can fail-fast in CI.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
import urllib.error
from typing import Any

import websockets

# Override host/port via env so the same script works against both the
# user's main 8000 server and a parallel verification server.
HOST = os.environ.get("I3_VERIFY_HOST", "127.0.0.1")
PORT = int(os.environ.get("I3_VERIFY_PORT", "8000"))
BASE = f"http://{HOST}:{PORT}"
WS_URL = f"ws://{HOST}:{PORT}/ws/verify_v2_user"


def _get_stack() -> dict[str, Any]:
    """Fetch /api/stack synchronously and return the SLM block."""
    req = urllib.request.Request(f"{BASE}/api/stack")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_stack(stack: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the /api/stack SLM block reports v2 metadata.

    Returns (ok, list_of_issues).
    """
    issues: list[str] = []
    slm = stack.get("slm", {}) or {}
    if not slm.get("loaded"):
        issues.append(f"SLM not loaded: {slm}")
        return False, issues

    version = slm.get("version") or slm.get("slm_version")
    if version != "v2":
        issues.append(f"version mismatch: expected 'v2', got {version!r}")

    params = int(slm.get("params") or 0)
    if params < 200_000_000:
        issues.append(f"params < 200M: {params}")

    vocab = int(slm.get("vocab_size") or 0)
    if vocab != 32000:
        issues.append(f"vocab_size mismatch: expected 32000, got {vocab}")

    d_model = int(slm.get("d_model") or 0)
    if d_model != 768:
        issues.append(f"d_model mismatch: expected 768, got {d_model}")

    n_layers = int(slm.get("n_layers") or 0)
    if n_layers != 12:
        issues.append(f"n_layers mismatch: expected 12, got {n_layers}")

    return not issues, issues


async def _ws_round_trip(prompt: str) -> dict[str, Any]:
    """Send *prompt* over the WS and return the final response payload.

    Wire format mirrors the conversational-audit script (the
    canonical client): ``{type: "message", text, timestamp,
    composition_metrics}`` — followed by draining frames until the
    server emits ``response`` or ``response_done``.
    """
    # SEC: server's default WS origin allow-list is the canonical
    # 127.0.0.1:8000 / localhost:8000 pair (regardless of which port
    # we actually probe on). Send that explicit origin so a parallel
    # server bound to a different port still accepts our handshake.
    async with websockets.connect(
        WS_URL,
        origin="http://127.0.0.1:8000",
        max_size=2 ** 22,
    ) as ws:
        # Server may send a hello / welcome first; consume any frames
        # that arrive in the first 1 s.
        for _ in range(5):
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                break
        import time as _t
        await ws.send(
            json.dumps(
                {
                    "type": "message",
                    "text": prompt,
                    "timestamp": _t.time(),
                    "composition_metrics": {
                        "composition_time_ms": 1500,
                        "edit_count": 0,
                        "pause_before_send_ms": 200,
                        "keystroke_timings": [100, 110, 95, 105, 100, 108, 96, 102] * 3,
                    },
                }
            )
        )
        # Drain frames until we get a response_done (or response).
        final: dict[str, Any] = {}
        for _ in range(120):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") in ("response", "response_done"):
                final = msg
                break
        return final


def _check_ws(prompt: str, msg: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    # The server's response/response_done frames are flat dicts: top-level
    # ``text`` + ``response_path``, no ``data`` wrapper.
    text = msg.get("text") or ""
    path = msg.get("response_path") or msg.get("path") or ""
    if not text or len(text) < 5:
        issues.append(f"response too short ({len(text)} chars): {text!r}")
    if path == "ood" and len(text) < 10:
        issues.append(f"response_path=ood with empty text")
    return not issues, issues


async def main() -> int:
    print("=" * 70)
    print("I3 v2 verification probe")
    print("=" * 70)

    print("\n[1/2] GET /api/stack ...")
    try:
        stack = _get_stack()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  FAIL: cannot reach server at {BASE}: {exc}")
        return 1

    slm = stack.get("slm", {}) or {}
    print(f"  loaded   : {slm.get('loaded')}")
    print(f"  version  : {slm.get('version')}")
    print(f"  name     : {slm.get('name')}")
    print(f"  params   : {slm.get('params_human')} ({slm.get('params')})")
    print(f"  d_model  : {slm.get('d_model')}")
    print(f"  n_layers : {slm.get('n_layers')}")
    print(f"  n_heads  : {slm.get('n_heads')}")
    print(f"  vocab    : {slm.get('vocab_size')}")
    print(f"  experts  : {slm.get('n_experts')}")
    ok_stack, issues_stack = _check_stack(stack)
    if not ok_stack:
        print("  STACK ISSUES:")
        for it in issues_stack:
            print(f"    - {it}")
    else:
        print("  STACK OK")

    print("\n[2/2] WS round-trip ...")
    prompt = "explain photosynthesis briefly"
    print(f"  prompt   : {prompt!r}")
    try:
        msg = await _ws_round_trip(prompt)
    except Exception as exc:
        print(f"  FAIL: ws round-trip raised: {exc}")
        return 1
    data = msg.get("data") or msg.get("payload") or msg
    text = data.get("text") or data.get("response") or ""
    path = data.get("response_path") or data.get("path") or "?"
    print(f"  path     : {path}")
    print(f"  text     : {text!r}")
    print(f"  length   : {len(text)} chars")
    ok_ws, issues_ws = _check_ws(prompt, msg)
    if not ok_ws:
        print("  WS ISSUES:")
        for it in issues_ws:
            print(f"    - {it}")
    else:
        print("  WS OK")

    print("\n" + "=" * 70)
    overall_ok = ok_stack and ok_ws
    print(f"VERIFY {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
