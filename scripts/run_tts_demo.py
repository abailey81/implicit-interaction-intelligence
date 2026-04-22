"""CLI: render the 8 archetype AdaptationVectors through the TTS engine.

Usage::

    python scripts/run_tts_demo.py [--out reports/tts_demo] [--backend pyttsx3]

The script iterates the archetypes in
:data:`server.routes_tts.ARCHETYPES`, derives TTS parameters, calls the
engine, and writes one WAV per archetype into ``<out>/<name>.wav`` —
plus a ``README.md`` with a summary table of the derived parameters.

The script uses whatever TTS backend is available on the host; if only
the Web Speech API backend is installed (which returns a directive
rather than audio) the script logs a warning and still writes the
parameter table so the research note has something to cite.
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
from pathlib import Path
from typing import Iterable

from i3.adaptation.types import AdaptationVector
from i3.tts import (
    TTSEngine,
    TTSOutput,
    TTSParams,
    derive_tts_params,
    explain_params,
    list_backend_statuses,
)
from server.routes_tts import ARCHETYPES, CANONICAL_PREVIEW_PHRASE

logger = logging.getLogger("i3.tts.demo")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argv slice (defaults to :data:`sys.argv[1:]`).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Render the 8 archetype AdaptationVectors as WAVs "
            "(+ a parameter table)."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/tts_demo"),
        help="Output directory (default: reports/tts_demo).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional TTS backend hint (pyttsx3 | piper | kokoro | web_speech_api).",
    )
    parser.add_argument(
        "--phrase",
        type=str,
        default=CANONICAL_PREVIEW_PHRASE,
        help="Phrase to synthesise for each archetype.",
    )
    return parser.parse_args(argv)


def _write_wav(path: Path, output: TTSOutput) -> bool:
    """Write *output* as a WAV file when audio is present.

    Args:
        path: Destination WAV path.
        output: :class:`TTSOutput` returned by the engine.

    Returns:
        ``True`` if audio bytes were written, ``False`` if only a
        directive was returned (no server-side audio available).
    """
    if output.audio_wav_base64 is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(output.audio_wav_base64))
    return True


def _render_table(rows: Iterable[tuple[str, TTSParams, str]]) -> str:
    """Render a Markdown table of the derived params.

    Args:
        rows: Iterable of ``(archetype_name, params, explanation)``.

    Returns:
        Markdown-formatted table as a string.
    """
    lines = [
        "| archetype | rate (wpm) | pitch (cents) | pause (ms) | volume (dB) | enunciation |",
        "|-----------|-----------:|--------------:|-----------:|------------:|-------------|",
    ]
    for name, params, _ in rows:
        lines.append(
            f"| `{name}` | {params.rate_wpm} | "
            f"{params.pitch_cents:+.1f} | {params.pause_ms_between_sentences} | "
            f"{params.volume_db:+.1f} | {params.enunciation} |"
        )
    return "\n".join(lines)


def _write_readme(
    out_dir: Path,
    rows: list[tuple[str, TTSParams, str]],
    backend_name: str,
    phrase: str,
    audio_written: int,
) -> None:
    """Write ``README.md`` summarising the demo output.

    Args:
        out_dir: Destination directory.
        rows: Archetype rows (name, params, explanation).
        backend_name: Backend actually used.
        phrase: Synthesised phrase.
        audio_written: Count of WAVs written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _render_table(rows)
    explanations = "\n".join(
        f"- `{name}` — {explanation}" for name, _, explanation in rows
    )
    body = (
        "# TTS archetype demo\n\n"
        f"**Backend used:** `{backend_name}`  \n"
        f"**WAVs written:** {audio_written} / {len(rows)}  \n"
        f"**Phrase:** \"{phrase}\"\n\n"
        "## Derived parameters\n\n"
        f"{table}\n\n"
        "## Explanations\n\n"
        f"{explanations}\n\n"
        "## How these were produced\n\n"
        "Each row is the output of "
        "`i3.tts.conditioning.derive_tts_params(archetype)` passed to "
        "`i3.tts.TTSEngine.speak(...)`.  The archetypes live in "
        "`server.routes_tts.ARCHETYPES`; each is an AdaptationVector "
        "that isolates one dimension so the effect of that dimension "
        "on prosody can be heard in isolation.\n\n"
        "See [`docs/research/adaptive_tts.md`](../../docs/research/adaptive_tts.md) "
        "for the full derivation and citations.\n"
    )
    (out_dir / "README.md").write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argv (defaults to :data:`sys.argv[1:]`).

    Returns:
        ``0`` on success, ``2`` when no TTS backend is installed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    out_dir: Path = args.out

    engine = TTSEngine(allow_web_speech=True)
    statuses = list_backend_statuses()
    logger.info("TTS backend availability:")
    for st in statuses:
        logger.info("  %-16s %s", st.name, "available" if st.available else "missing")

    rows: list[tuple[str, TTSParams, str]] = []
    audio_written = 0
    backend_name = args.backend or "auto"

    for archetype_name, adaptation in ARCHETYPES.items():
        try:
            params = derive_tts_params(adaptation)
        except ValueError as exc:
            logger.error("archetype %s failed: %s", archetype_name, exc)
            continue
        explanation = explain_params(params, adaptation)

        try:
            output = engine.speak(
                args.phrase, params, backend_hint=args.backend
            )
        except RuntimeError as exc:
            logger.warning(
                "skipping %s — backend error: %s", archetype_name, type(exc).__name__
            )
            rows.append((archetype_name, params, explanation))
            continue
        except ValueError as exc:
            logger.warning("skipping %s — %s", archetype_name, exc)
            rows.append((archetype_name, params, explanation))
            continue

        wrote = _write_wav(out_dir / f"{archetype_name}.wav", output)
        if wrote:
            audio_written += 1
            backend_name = output.backend_name
            logger.info(
                "wrote %s (%d ms, %s)",
                out_dir / f"{archetype_name}.wav",
                output.duration_ms,
                output.backend_name,
            )
        else:
            logger.info(
                "%s — directive-only path (%s); no WAV written",
                archetype_name,
                output.backend_name,
            )
        rows.append((archetype_name, params, explanation))

    if not rows:
        logger.error("no archetypes processed — aborting")
        return 2

    _write_readme(
        out_dir=out_dir,
        rows=rows,
        backend_name=backend_name,
        phrase=args.phrase,
        audio_written=audio_written,
    )
    logger.info("wrote %s/README.md (%d archetypes)", out_dir, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
