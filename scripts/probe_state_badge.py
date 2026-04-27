"""End-to-end probe for the Live State Badge + Accessibility Mode features.

Drives the full FastAPI app through Starlette's TestClient WebSocket
support (no real network, no real uvicorn — keeps the harness's "agent
must not kill uvicorn" invariant intact).  Walks 13 turns through the
pattern:

    Turn  1-3:  calm typing      → expect state = "calm" or "warming up",
                                   accessibility.active = False
    Turn  4-6:  rushed typing    → state should drift toward "stressed"
                                   by turn 5-6
    Turn  7-9:  keep rushed      → accessibility.activated_this_turn=True
                                   on turn 7-9
    Turn 10-13: back to calm     → accessibility.deactivated_this_turn=True
                                   after 4 consecutive calm turns

Reports each turn's state badge label, confidence, accessibility
flags, and the reasoning-trace para 2 + 3 sentences that mention the
new signals.  Also confirms ``conf 1.00`` on a deterministic demo
prompt and exercises the ``POST /api/accessibility/{user_id}/toggle``
endpoint.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any


def _fmt_label(label: dict | None) -> str:
    if not label:
        return "—"
    state = label.get("state", "?")
    conf = label.get("confidence", 0.0)
    secondary = label.get("secondary_state")
    parts = [f"{state} {conf:.2f}"]
    if secondary:
        parts.append(f"[2nd: {secondary}]")
    sigs = label.get("contributing_signals") or []
    if sigs:
        parts.append("(" + ", ".join(sigs[:3]) + ")")
    return " ".join(parts)


def _fmt_access(state: dict | None) -> str:
    if not state:
        return "—"
    bits = [f"active={state.get('active')}"]
    if state.get("activated_this_turn"):
        bits.append("RISING")
    if state.get("deactivated_this_turn"):
        bits.append("FALLING")
    if state.get("reason"):
        bits.append(f"reason='{state['reason']}'")
    return " ".join(bits)


def _walk_turn(
    websocket: Any,
    *,
    text: str,
    composition_ms: float,
    edit_count: int,
    pause_ms: float,
    iki_mean: float,
    iki_std: float,
    n_keystrokes: int = 20,
) -> dict[str, Any]:
    """Send one turn over the WS and collect the response + metadata."""
    # Stream a few keystroke frames so the IKI buffer is non-trivial.
    for i in range(n_keystrokes):
        # Inject jitter so iki_std is non-zero.
        jitter = ((i % 5) - 2) * (iki_std / 4.0)
        websocket.send_json({
            "type": "keystroke",
            "timestamp": time.time(),
            "key_type": "char",
            "inter_key_interval_ms": max(0.0, iki_mean + jitter),
        })
    # Send the message frame.
    websocket.send_json({
        "type": "message",
        "text": text,
        "timestamp": time.time(),
        "composition_time_ms": composition_ms,
        "edit_count": edit_count,
        "pause_before_send_ms": pause_ms,
    })

    # Drain frames until we see the response (or response_done) plus
    # the trailing state_update + state_badge / accessibility_change.
    out: dict[str, Any] = {
        "response": None,
        "state_update": None,
        "state_badge": None,
        "accessibility_change": None,
        "tokens": [],
    }
    deadline = time.time() + 25.0
    while time.time() < deadline:
        try:
            frame = websocket.receive_json(mode="text")
        except Exception:
            break
        ftype = frame.get("type")
        if ftype == "token":
            out["tokens"].append(frame.get("delta", ""))
        elif ftype in ("response", "response_done"):
            out["response"] = frame
        elif ftype == "state_update":
            out["state_update"] = frame
            # state_update is the *last* frame for a turn.
            break
        elif ftype == "state_badge":
            out["state_badge"] = frame
        elif ftype == "accessibility_change":
            out["accessibility_change"] = frame
        elif ftype == "session_started":
            # First-frame-after-connect; ignore.
            continue
        elif ftype == "diary_entry":
            continue
        else:
            # Unknown — record and continue.
            out.setdefault("other", []).append(frame)
    return out


def main() -> int:
    """Run the probe."""
    # Lazy import — env must be configured (CORS allow-list etc.).
    import logging
    import os

    os.environ.setdefault("I3_CORS_ORIGINS", "http://localhost:8000")
    os.environ.setdefault("I3_DISABLE_ADMIN", "1")
    # Quiet the loud bootstrap logs so the probe transcript is readable.
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.ERROR)
    logging.getLogger("opentelemetry").setLevel(logging.ERROR)

    from fastapi.testclient import TestClient

    from server.app import create_app

    print("=" * 78)
    print("Live State Badge + Accessibility Mode probe")
    print("=" * 78)
    print("[boot] creating app + lifespan (loads SLM + retrieval, ~30-60 s)...")
    sys.stdout.flush()

    app = create_app()
    transcript: list[dict[str, Any]] = []

    with TestClient(app) as client:
        print("[boot] TestClient ready, opening WS...")
        sys.stdout.flush()
        with client.websocket_connect(
            "/ws/probe_user",
            headers={"origin": "http://localhost:8000"},
        ) as ws:
            print("[boot] WS connected")
            sys.stdout.flush()
            # Drain initial session_started frame.
            try:
                first = ws.receive_json(mode="text")
                if first.get("type") != "session_started":
                    print(f"[warn] unexpected first frame: {first}")
                else:
                    session_id = first.get("session_id", "")
                    print(f"[init] session_id={session_id}")
            except Exception as e:
                print(f"[error] failed to receive session_started: {e}")
                return 1

            # ---------- Phase 1: calm turns 1-3 ------------------------
            print("\n--- Phase 1: calm typing (turns 1-3) ---")
            for turn in range(1, 4):
                r = _walk_turn(
                    ws,
                    text="hello, can you tell me a fun fact about cats?",
                    composition_ms=1200,
                    edit_count=0,
                    pause_ms=200,
                    iki_mean=100.0,
                    iki_std=10.0,
                )
                resp = r.get("response") or {}
                label = resp.get("user_state_label")
                access = resp.get("accessibility")
                badge = r.get("state_badge")
                print(
                    f"T{turn}: badge={_fmt_label(label)} | "
                    f"access={_fmt_access(access)} | "
                    f"path={resp.get('response_path')} "
                    f"score={resp.get('retrieval_score', 0):.2f}"
                )
                _print_trace_para(resp)
                transcript.append({
                    "turn": turn, "phase": "calm-1",
                    "state_label": label, "accessibility": access,
                    "response_path": resp.get("response_path"),
                    "retrieval_score": resp.get("retrieval_score"),
                })

            # ---------- Phase 2: rushed turns 4-6 ----------------------
            print("\n--- Phase 2: rushed typing (turns 4-6) ---")
            for turn in range(4, 7):
                r = _walk_turn(
                    ws,
                    text="wait i meant the fast moving onesss",
                    composition_ms=3500,
                    edit_count=4,
                    pause_ms=180,
                    iki_mean=180.0,
                    iki_std=40.0,
                )
                resp = r.get("response") or {}
                label = resp.get("user_state_label")
                access = resp.get("accessibility")
                print(
                    f"T{turn}: badge={_fmt_label(label)} | "
                    f"access={_fmt_access(access)} | "
                    f"path={resp.get('response_path')} "
                    f"score={resp.get('retrieval_score', 0):.2f}"
                )
                _print_trace_para(resp)
                transcript.append({
                    "turn": turn, "phase": "rushed-1",
                    "state_label": label, "accessibility": access,
                    "response_path": resp.get("response_path"),
                    "retrieval_score": resp.get("retrieval_score"),
                })

            # ---------- Phase 3: more rushed turns 7-9 -----------------
            print("\n--- Phase 3: more rushed typing (turns 7-9) ---")
            for turn in range(7, 10):
                r = _walk_turn(
                    ws,
                    text="ugh sorry typo too many edts in arow",
                    composition_ms=3500,
                    edit_count=5,
                    pause_ms=180,
                    iki_mean=185.0,
                    iki_std=45.0,
                )
                resp = r.get("response") or {}
                label = resp.get("user_state_label")
                access = resp.get("accessibility")
                print(
                    f"T{turn}: badge={_fmt_label(label)} | "
                    f"access={_fmt_access(access)} | "
                    f"path={resp.get('response_path')}"
                )
                _print_trace_para(resp)
                transcript.append({
                    "turn": turn, "phase": "rushed-2",
                    "state_label": label, "accessibility": access,
                    "response_path": resp.get("response_path"),
                    "retrieval_score": resp.get("retrieval_score"),
                })

            # ---------- Phase 4: calm recovery turns 10-13 -------------
            print("\n--- Phase 4: calm recovery (turns 10-13) ---")
            for turn in range(10, 14):
                r = _walk_turn(
                    ws,
                    text="thanks, that was helpful! anything else?",
                    composition_ms=1200,
                    edit_count=0,
                    pause_ms=200,
                    iki_mean=100.0,
                    iki_std=10.0,
                )
                resp = r.get("response") or {}
                label = resp.get("user_state_label")
                access = resp.get("accessibility")
                print(
                    f"T{turn}: badge={_fmt_label(label)} | "
                    f"access={_fmt_access(access)} | "
                    f"path={resp.get('response_path')}"
                )
                _print_trace_para(resp)
                transcript.append({
                    "turn": turn, "phase": "calm-recovery",
                    "state_label": label, "accessibility": access,
                    "response_path": resp.get("response_path"),
                    "retrieval_score": resp.get("retrieval_score"),
                })

            # ---------- Demo retrieval check ---------------------------
            print("\n--- Demo retrieval prompt (expect conf ~1.00) ---")
            r = _walk_turn(
                ws,
                text="hi",
                composition_ms=600,
                edit_count=0,
                pause_ms=200,
                iki_mean=100.0,
                iki_std=10.0,
            )
            resp = r.get("response") or {}
            score = float(resp.get("retrieval_score") or 0.0)
            print(f"  path={resp.get('response_path')} retrieval_score={score:.2f}")

        # ---------- Manual toggle endpoint -----------------------------
        print("\n--- Manual toggle endpoint ---")
        rsp = client.post(
            "/api/accessibility/probe_user/toggle",
            json={"session_id": "manual_probe", "force": True},
        )
        print(f"  POST force=True → {rsp.status_code}: {rsp.json()}")
        rsp_body = rsp.json()
        print(f"  active={rsp_body.get('active')} font_scale={rsp_body.get('font_scale')}")
        rsp = client.post(
            "/api/accessibility/probe_user/toggle",
            json={"session_id": "manual_probe", "force": False},
        )
        print(f"  POST force=False → {rsp.status_code}: active={rsp.json().get('active')}")

    print("\n" + "=" * 78)
    print("Probe complete.")
    print("=" * 78)
    # Dump the transcript JSON for the report.
    print("\nTranscript (JSON):")
    print(json.dumps(transcript, indent=2, default=str))
    return 0


def _print_trace_para(resp: dict | None) -> None:
    """Print the para 2 + para 3 of the reasoning trace, when present."""
    if not resp:
        return
    trace = resp.get("reasoning_trace") or {}
    paras = trace.get("narrative_paragraphs") or []
    if len(paras) >= 2:
        p2 = paras[1].strip()
        # Only print the new state-classifier sentence (the part after
        # the existing baseline / engagement description).  If the
        # classifier didn't fire we still print para2 so the demo
        # diagnostics are complete.
        idx = p2.find("The state classifier")
        if idx >= 0:
            print(f"     para2: …{p2[idx:idx + 220]}")
    if len(paras) >= 3:
        p3 = paras[2].strip()
        idx = p3.find("Accessibility mode is active")
        if idx >= 0:
            print(f"     para3: …{p3[idx:idx + 220]}")


if __name__ == "__main__":
    sys.exit(main())
