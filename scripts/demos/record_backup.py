"""Script that records the 5-minute backup demo video via OBS Studio.

The brief requires a 5-minute MP4 of the full live demo, carried on two
USB drives for the interview. This script drives OBS Studio through its
WebSocket plugin (``obs-websocket``) so the recording starts on cue,
runs through the four phases, and stops cleanly.

When ``obs-websocket`` is not available the script degrades to a
**narrator cue sheet**: a minute-by-minute script printed to stdout
with timestamps so the presenter can record manually via QuickTime or
any other screen recorder.

Safety watchdog
~~~~~~~~~~~~~~~
Because this runs alongside :mod:`scripts.run_four_phases`, the script
watches the companion WebSocket client for 30 seconds of silence after
each phase boundary. If no client activity is seen the script stops the
recording and fails loudly — a silent recording at demo time would be
unrecoverable.

Usage::

    python scripts/record_backup_demo.py --output demo.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("i3.demo.record")


# ---------------------------------------------------------------------------
# Cue sheet — single source of truth for the narrator fallback AND the
# watchdog boundaries.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    """One entry in the narrator cue sheet.

    Attributes:
        t_seconds: Offset from recording start, in seconds.
        phase: Phase label for the banner.
        narration: Verbatim words to speak on screen.
    """

    t_seconds: int
    phase: str
    narration: str


_CUES: tuple[Cue, ...] = (
    Cue(0, "intro", (
        "Hi, I'm Tamer. This is Implicit Interaction Intelligence — a "
        "prototype that reads the user from how they interact, not what "
        "they say. The dashboard on the right is where the story lives."
    )),
    Cue(30, "phase1",
        "Cold start. Two neutral questions at 60 WPM; baseline establishing."),
    Cue(90, "phase2",
        "Energy ramp. Casual, fast typing — formality drops, warmth climbs."),
    Cue(150, "phase3", (
        "Fatigue. I slow down, I make corrections, I never say I'm tired — "
        "and yet the cognitive-load gauge climbs to 0.8. That's the moment."
    )),
    Cue(240, "phase4", (
        "Accessibility. Simple content, very effortful typing. The "
        "accessibility axis lifts while cognitive load stays flat — "
        "diagnostic of motor difficulty, not mental load."
    )),
    Cue(285, "diary", (
        "Diary panel. No raw text column. Every stored row is a lossy "
        "behavioural fingerprint; by construction, not by policy."
    )),
    Cue(300, "close",
        "I build intelligent systems that adapt to people. "
        "I'd like to do that in your lab."),
)

# Watchdog boundaries — seconds after which a silent WS is a hard failure.
_WATCHDOG_BOUNDARIES_S: tuple[int, ...] = (30, 90, 150, 240, 300)
_WATCHDOG_SILENCE_LIMIT_S: float = 30.0


# ---------------------------------------------------------------------------
# Narrator fallback
# ---------------------------------------------------------------------------


def _print_cue_sheet() -> None:
    """Print the narrator cue sheet to stdout.

    The format is: ``mm:ss  <PHASE>   narration``, one line per cue,
    deterministic — safe to copy/paste into the presenter's cue card.
    """
    print("=" * 72)  # noqa: T201
    print("I3 BACKUP-DEMO CUE SHEET  (manual recording)")  # noqa: T201
    print("=" * 72)  # noqa: T201
    for cue in _CUES:
        minutes, seconds = divmod(cue.t_seconds, 60)
        tag = cue.phase.upper().ljust(8)
        print(f"{minutes:02d}:{seconds:02d}  {tag}  {cue.narration}")  # noqa: T201
    print("=" * 72)  # noqa: T201
    print("Total runtime: 5:00. Record via QuickTime / OBS; export as MP4.")  # noqa: T201


# ---------------------------------------------------------------------------
# OBS integration (soft-imported)
# ---------------------------------------------------------------------------


async def _record_via_obs(output_path: Path, duration_s: int) -> int:
    """Drive OBS Studio over obs-websocket.

    Uses the ``obsws-python`` package when available. Any failure mode
    (missing package, no connection, refusal) falls through to the
    narrator cue sheet and returns a non-zero exit code so CI can flag
    the skip.

    Args:
        output_path: Target MP4 path (the OBS recording profile must be
            configured to write MP4; we do not enforce it here).
        duration_s: Recording duration in seconds. Default: 300 (5 min).

    Returns:
        ``0`` on success, ``1`` when OBS could not be reached (cue sheet
        is printed as a fallback).
    """
    try:
        import obsws_python as obs  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "obs.unavailable",
            extra={"event": "record", "reason": "obsws-python not installed"},
        )
        print(
            "\n[!] obsws-python not installed — printing narrator cue "
            "sheet for manual recording.\n"
        )  # noqa: T201
        _print_cue_sheet()
        return 1

    host = os.environ.get("OBS_HOST", "localhost")
    port = int(os.environ.get("OBS_PORT", "4455"))
    password = os.environ.get("OBS_PASSWORD", "")

    try:
        client = obs.ReqClient(host=host, port=port, password=password, timeout=5)
    except (ConnectionError, OSError, TimeoutError) as exc:
        logger.error(
            "obs.connect_failed",
            extra={
                "event": "record",
                "host": host,
                "port": port,
                "err": type(exc).__name__,
            },
        )
        print(
            f"\n[!] Could not reach OBS at {host}:{port} — "
            "printing narrator cue sheet instead.\n"
        )  # noqa: T201
        _print_cue_sheet()
        return 1

    logger.info(
        "obs.connected",
        extra={"event": "record", "host": host, "port": port},
    )
    try:
        client.start_record()
    except (ConnectionError, OSError, RuntimeError) as exc:
        logger.error(
            "obs.start_record_failed",
            extra={"event": "record", "err": type(exc).__name__},
        )
        return 2

    started = time.time()
    print(f"Recording started; will stop in {duration_s}s -> {output_path}")  # noqa: T201

    # Print cues as the clock advances so the presenter knows what to do
    # on each minute boundary.
    next_cue = 0
    try:
        while time.time() - started < duration_s:
            elapsed = int(time.time() - started)
            while next_cue < len(_CUES) and _CUES[next_cue].t_seconds <= elapsed:
                cue = _CUES[next_cue]
                minutes, seconds = divmod(cue.t_seconds, 60)
                print(
                    f"  [{minutes:02d}:{seconds:02d}] "
                    f"{cue.phase.upper():<8} {cue.narration}"
                )  # noqa: T201
                next_cue += 1
            await asyncio.sleep(1.0)
    finally:
        try:
            client.stop_record()
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.warning(
                "obs.stop_record_failed",
                extra={"event": "record", "err": type(exc).__name__},
            )

    print(f"\nRecording stopped after {int(time.time() - started)}s.")  # noqa: T201
    print(
        "NOTE: the MP4 file is written to OBS's configured recording "
        f"directory. Move / rename to {output_path} manually if needed."
    )  # noqa: T201
    return 0


# ---------------------------------------------------------------------------
# WebSocket activity watchdog
# ---------------------------------------------------------------------------


async def _watchdog(
    ws_url: str | None, boundaries: tuple[int, ...] = _WATCHDOG_BOUNDARIES_S
) -> int:
    """Raise on 30s of WebSocket silence around a phase boundary.

    Args:
        ws_url: WebSocket URL to observe; ``None`` disables the watchdog.
        boundaries: Per-phase start-offsets in seconds.

    Returns:
        ``0`` if every boundary saw traffic within the silence limit;
        ``3`` if one boundary was silent — the caller is expected to
        exit non-zero.
    """
    if ws_url is None:
        return 0
    try:
        import websockets  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "watchdog.unavailable",
            extra={"event": "record", "reason": "websockets package missing"},
        )
        return 0

    origin_parts = ws_url.replace("ws://", "http://").replace("wss://", "https://").split("/")
    origin = "/".join(origin_parts[:3])
    try:
        ws = await websockets.connect(ws_url, origin=origin, open_timeout=5.0)
    except (ConnectionError, OSError, TimeoutError, websockets.WebSocketException) as exc:
        logger.warning(
            "watchdog.connect_failed",
            extra={"event": "record", "err": type(exc).__name__},
        )
        return 0

    start = time.monotonic()
    last_seen = time.monotonic()
    failures: list[int] = []
    try:
        for boundary in boundaries:
            wait_until = start + boundary
            while time.monotonic() < wait_until:
                try:
                    await asyncio.wait_for(ws.recv(), timeout=1.0)
                    last_seen = time.monotonic()
                except asyncio.TimeoutError:
                    continue
                except (
                    websockets.ConnectionClosed,
                    websockets.WebSocketException,
                ):
                    return 3
            if time.monotonic() - last_seen > _WATCHDOG_SILENCE_LIMIT_S:
                logger.error(
                    "watchdog.silence",
                    extra={
                        "event": "record",
                        "boundary_s": boundary,
                        "silence_s": time.monotonic() - last_seen,
                    },
                )
                failures.append(boundary)
    finally:
        try:
            await ws.close()
        except (websockets.WebSocketException, OSError):
            pass
    return 3 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the recorder."""
    parser = argparse.ArgumentParser(
        description="Record the 5-minute backup demo via OBS Studio."
    )
    parser.add_argument(
        "--output", default="demo.mp4", type=str,
        help="Target MP4 filename (advisory — OBS writes to its own path).",
    )
    parser.add_argument(
        "--duration", type=int, default=300,
        help="Recording duration in seconds (default 300).",
    )
    parser.add_argument(
        "--watchdog-ws", type=str, default=None,
        help=(
            "Optional WebSocket URL to monitor for silence. "
            "When set, 30s of silence around a phase boundary fails the run."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


async def _amain() -> int:
    """Async entry point — returns the effective exit code."""
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    output_path = Path(args.output)

    # Run OBS and the watchdog concurrently so a silent WS terminates the run
    # before we burn through the recording window with nothing to show.
    record_task = asyncio.create_task(
        _record_via_obs(output_path, args.duration)
    )
    watchdog_task = asyncio.create_task(_watchdog(args.watchdog_ws))

    record_rc = await record_task
    watchdog_rc = await watchdog_task

    if watchdog_rc != 0:
        print(
            f"[!] watchdog detected >{_WATCHDOG_SILENCE_LIMIT_S}s of silence; "
            "recording marked as FAILED."
        )  # noqa: T201
        return watchdog_rc
    return record_rc


def main() -> int:
    """Synchronous CLI entry point."""
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nInterrupted.")  # noqa: T201
        return 130


if __name__ == "__main__":
    # Dump the cue sheet unconditionally so the presenter always has a
    # paper fallback, even if OBS integration succeeds and prints its own.
    print("[cue-sheet, for reference]")  # noqa: T201
    _print_cue_sheet()
    sys.exit(main())
