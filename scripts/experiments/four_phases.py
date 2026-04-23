"""Scenario runner that drives the 4-phase live demo against a running I3 server.

The script walks through the exact messages scripted in
``docs/slides/demo-script.md`` with four different keystroke cadences, one per
phase:

* **Phase 1 — Cold Start**: ~60 WPM, neutral.
* **Phase 2 — Energetic**:  ~100 WPM, bursty.
* **Phase 3 — Fatigue**:    ~30 WPM, long pauses.
* **Phase 4 — Accessibility**: ~15 WPM, many corrections.

Usage::

    python scripts/run_four_phases.py \
        --url ws://127.0.0.1:8000/ws/demo_user

A transcript of every inbound / outbound frame is saved to
``reports/four_phase_transcript_<ts>.json`` for post-hoc inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError as exc:  # pragma: no cover - import-time guard
    raise SystemExit(
        "The 'websockets' package is required. Install with: pip install websockets"
    ) from exc


logger = logging.getLogger("i3.demo.four_phase")


# ---------------------------------------------------------------------------
# ANSI colour helpers (no external dependency on rich/colorama)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_COLOURS: dict[str, str] = {
    "phase1": "\033[36m",  # cyan
    "phase2": "\033[33m",  # yellow
    "phase3": "\033[35m",  # magenta
    "phase4": "\033[31m",  # red
    "ok": "\033[32m",      # green
    "warn": "\033[33m",    # yellow
}


def _banner(tag: str, title: str) -> None:
    """Print a bold coloured banner to stdout."""
    colour = _COLOURS.get(tag, "")
    bar = "=" * 72
    print(f"\n{colour}{_BOLD}{bar}")  # noqa: T201
    print(f"{title}")  # noqa: T201
    print(f"{bar}{_RESET}\n")  # noqa: T201


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseMessage:
    """One scripted message in a phase.

    Attributes:
        text: The message body to send.
        composition_time_ms: Simulated composition duration (ms).
        edit_count: Number of simulated backspaces.
        pause_before_send_ms: Pre-send hesitation (ms).
    """

    text: str
    composition_time_ms: float
    edit_count: int
    pause_before_send_ms: float


@dataclass
class PhaseSpec:
    """A phase in the 4-phase demo.

    Attributes:
        tag: Colour tag for banner output.
        title: Human-readable phase title.
        wpm: Target words-per-minute for keystroke pacing.
        messages: Ordered list of :class:`PhaseMessage` entries.
        wait_between_messages_s: Pause between messages within the phase.
        snapshot: Expected dashboard snapshot summary to display on exit.
    """

    tag: str
    title: str
    wpm: int
    messages: list[PhaseMessage]
    wait_between_messages_s: float
    snapshot: str


def _build_phases() -> list[PhaseSpec]:
    """Return the canonical 4-phase demo plan."""
    return [
        PhaseSpec(
            tag="phase1",
            title="Phase 1  —  Cold Start   (neutral, ~60 WPM)",
            wpm=60,
            wait_between_messages_s=4.0,
            messages=[
                PhaseMessage(
                    "Hi, I'm interested in learning about temporal convolutional "
                    "networks. Can you give me an overview?",
                    composition_time_ms=4200.0,
                    edit_count=1,
                    pause_before_send_ms=450.0,
                ),
                PhaseMessage(
                    "What's the difference between dilated convolutions and standard "
                    "convolutions, and why would you use dilated ones in a TCN?",
                    composition_time_ms=7100.0,
                    edit_count=2,
                    pause_before_send_ms=620.0,
                ),
            ],
            snapshot=(
                "cognitive_load~0.25  formality~0.60  warmth~0.50  "
                "accessibility~0.05"
            ),
        ),
        PhaseSpec(
            tag="phase2",
            title="Phase 2  —  Energetic    (casual, ~100 WPM)",
            wpm=100,
            wait_between_messages_s=2.5,
            messages=[
                PhaseMessage(
                    "Ok this is super cool!! Can you just give me the gist of how "
                    "attention works?? Like the 30-second version!",
                    composition_time_ms=2100.0,
                    edit_count=0,
                    pause_before_send_ms=180.0,
                ),
                PhaseMessage(
                    "Sweet! And is attention the same as self-attention? Or different??",
                    composition_time_ms=1500.0,
                    edit_count=0,
                    pause_before_send_ms=160.0,
                ),
            ],
            snapshot=(
                "cognitive_load~0.25  formality~0.35  verbosity~0.30  "
                "warmth~0.65"
            ),
        ),
        PhaseSpec(
            tag="phase3",
            title="Phase 3  —  Fatigue     (slow, ~30 WPM) <<< KEY MOMENT >>>",
            wpm=30,
            wait_between_messages_s=6.0,
            messages=[
                PhaseMessage(
                    "I want to understand how backprop works through time but I'm "
                    "having trouble keeping it straight in my head",
                    composition_time_ms=12500.0,
                    edit_count=4,
                    pause_before_send_ms=1300.0,
                ),
                PhaseMessage(
                    "can you maybe give me a simple analogy",
                    composition_time_ms=6800.0,
                    edit_count=2,
                    pause_before_send_ms=1600.0,
                ),
            ],
            snapshot=(
                "cognitive_load~0.80  formality~0.45  verbosity~0.40  "
                "warmth~0.75"
            ),
        ),
        PhaseSpec(
            tag="phase4",
            title="Phase 4  —  Accessibility  (very slow, ~15 WPM)",
            wpm=15,
            wait_between_messages_s=7.0,
            messages=[
                PhaseMessage(
                    "what time is it in tokyo",
                    composition_time_ms=9500.0,
                    edit_count=4,
                    pause_before_send_ms=2100.0,
                ),
                PhaseMessage(
                    "actually where is the closest coffee shop",
                    composition_time_ms=14200.0,
                    edit_count=10,
                    pause_before_send_ms=2800.0,
                ),
            ],
            snapshot=(
                "cognitive_load~0.50  verbosity~0.20  warmth~0.70  "
                "accessibility~0.70"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# WebSocket driver
# ---------------------------------------------------------------------------


@dataclass
class Transcript:
    """In-memory transcript of everything sent/received during a run."""

    started_at: str
    url: str
    phases: list[dict[str, Any]] = field(default_factory=list)
    inbound: list[dict[str, Any]] = field(default_factory=list)


def _wpm_to_iki_ms(wpm: int) -> float:
    """Convert a words-per-minute figure to per-character inter-key interval.

    Uses the canonical 5-characters-per-word assumption.
    """
    if wpm <= 0:
        return 1000.0
    chars_per_second = (wpm * 5.0) / 60.0
    return 1000.0 / chars_per_second


async def _send_keystrokes(
    ws: Any, text: str, wpm: int, transcript: Transcript
) -> None:
    """Stream keystroke events for *text* at the configured cadence.

    Emits one ``{"type": "keystroke", ...}`` frame per character and
    records them on the transcript so the resulting JSON can be
    diff'd against the server-side pipeline logs.
    """
    iki_ms = _wpm_to_iki_ms(wpm)
    for ch in text:
        frame: dict[str, Any] = {
            "type": "keystroke",
            "timestamp": time.time(),
            "key_type": "char",
            "inter_key_interval_ms": iki_ms,
        }
        await ws.send(json.dumps(frame))
        transcript.inbound.append({"dir": "out", **frame, "char": ch})
        await asyncio.sleep(iki_ms / 1000.0)


async def _receiver(ws: Any, transcript: Transcript, stop: asyncio.Event) -> None:
    """Background task that records every server frame into the transcript."""
    try:
        while not stop.is_set():
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            transcript.inbound.append({"dir": "in", **payload})
    except asyncio.TimeoutError:
        return
    except (ConnectionClosed, WebSocketException) as exc:
        logger.warning("receiver.ws_closed", extra={"err": type(exc).__name__})


async def _drain_inbound(ws: Any, timeout: float, transcript: Transcript) -> None:
    """Drain server frames for up to *timeout* seconds."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            return
        except (ConnectionClosed, WebSocketException) as exc:
            logger.warning(
                "drain.ws_closed", extra={"err": type(exc).__name__}
            )
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        transcript.inbound.append({"dir": "in", **payload})


async def _run_phase(
    ws: Any, phase: PhaseSpec, transcript: Transcript
) -> None:
    """Execute a single phase: banner, keystrokes, messages, snapshot."""
    _banner(phase.tag, f"{phase.title}")
    phase_log: dict[str, Any] = {
        "tag": phase.tag,
        "title": phase.title,
        "wpm": phase.wpm,
        "messages": [],
        "started_at": time.time(),
    }

    for idx, message in enumerate(phase.messages):
        print(f"  > typing message {idx + 1}/{len(phase.messages)}: {message.text}")  # noqa: T201
        await _send_keystrokes(ws, message.text, phase.wpm, transcript)
        frame: dict[str, Any] = {
            "type": "message",
            "text": message.text,
            "timestamp": time.time(),
            "composition_time_ms": message.composition_time_ms,
            "edit_count": message.edit_count,
            "pause_before_send_ms": message.pause_before_send_ms,
        }
        await ws.send(json.dumps(frame))
        transcript.inbound.append({"dir": "out", **frame})
        phase_log["messages"].append(frame)

        # Let the server process and push response + state_update back.
        await _drain_inbound(ws, timeout=phase.wait_between_messages_s, transcript=transcript)

    phase_log["ended_at"] = time.time()
    transcript.phases.append(phase_log)
    print(
        f"  {_COLOURS['ok']}[expected dashboard] {phase.snapshot}{_RESET}"
    )


async def _run(url: str, output_dir: Path) -> int:
    """Drive every phase against *url* and persist the transcript."""
    transcript = Transcript(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        url=url,
    )
    origin = url.replace("ws://", "http://").replace("wss://", "https://")
    origin = "/".join(origin.split("/")[:3])
    try:
        async with websockets.connect(
            url,
            origin=origin,
            open_timeout=10.0,
            ping_interval=20,
            max_size=128 * 1024,
        ) as ws:
            for phase in _build_phases():
                await _run_phase(ws, phase, transcript)
            # Explicit session end so the server flushes the diary row.
            await ws.send(json.dumps({"type": "session_end"}))
            transcript.inbound.append({"dir": "out", "type": "session_end"})
            await _drain_inbound(ws, timeout=2.0, transcript=transcript)
    except (ConnectionClosed, WebSocketException, OSError) as exc:
        logger.error(
            "run.ws_failure",
            extra={"err": type(exc).__name__, "detail": str(exc)},
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_path = output_dir / f"four_phase_transcript_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "url": transcript.url,
                "started_at": transcript.started_at,
                "phases": transcript.phases,
                "inbound": transcript.inbound,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _banner("ok", f"Transcript written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runner."""
    parser = argparse.ArgumentParser(
        description="Run the 4-phase I3 demo against a live server."
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/demo_user",
        help="WebSocket URL of the running I3 server.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for the transcript JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG / INFO / WARNING / ERROR).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point — returns process exit code."""
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(_run(url=args.url, output_dir=Path(args.output_dir)))


if __name__ == "__main__":
    sys.exit(main())
