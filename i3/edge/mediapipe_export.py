"""Google MediaPipe packaging for I3 on-device models.

`MediaPipe <https://developers.google.com/mediapipe>`_ is Google's
on-device ML framework. Its Tasks API wraps a TFLite model with a
schema (input / output metadata, pre- and post-processing graph) so
Android / iOS / web clients can consume it with a single API call.
For an I3 user-state encoder or sequence classifier, wrapping a
pre-existing ``.tflite`` as a MediaPipe task gives a consistent,
well-documented deployment surface on Android.

This module is intentionally a **thin packaging stub** — the actual
MediaPipe Model Maker CLI + ``bundler`` C++ tool is not available as a
Python API in all environments, so the stub documents the required
steps and writes a ``task_info.json`` placeholder next to the input
``.tflite``. Real packaging is a TODO.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def wrap_for_mediapipe(tflite_path: Path, out_path: Path) -> Path:
    """Package an existing ``.tflite`` as a MediaPipe task bundle.

    This is a stub that:

    1. Validates that ``tflite_path`` exists and is non-empty.
    2. Copies it to ``out_path`` (or ``out_path/model.tflite`` when
       ``out_path`` is a directory).
    3. Writes a sibling ``task_info.json`` with placeholder metadata
       that a future implementation must replace with a real
       MediaPipe Tasks bundle (see the TODOs below).

    TODOs for a full implementation:

    * Run ``mediapipe_model_maker`` on the input to derive the task
      graph (``model_task.binarypb``).
    * Pack the TFLite + task graph + metadata file into a single
      ``.task`` archive as produced by MediaPipe's ``bundler`` tool.
    * Emit a ``task_metadata.pbtxt`` with the correct input /
      output tensor names and pre-processing options.
    * Add an Android / iOS / web consumer example in the docs.

    Args:
        tflite_path: Path to an existing ``.tflite`` model.
        out_path: Destination. If it ends in ``.task`` or
            ``.tflite`` it is treated as a file; otherwise it is
            treated as a directory and ``model.tflite`` is placed
            inside it.

    Returns:
        The resolved output path.

    Raises:
        FileNotFoundError: If ``tflite_path`` does not exist.
        RuntimeError: If ``tflite_path`` is empty.
    """
    tflite_path = Path(tflite_path)
    if not tflite_path.exists():
        raise FileNotFoundError(
            f"Input .tflite not found: {tflite_path}"
        )
    if tflite_path.stat().st_size == 0:
        raise RuntimeError(
            f"Input .tflite is empty: {tflite_path}"
        )

    out_path = Path(out_path)
    if out_path.suffix in (".task", ".tflite"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dest = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        dest = out_path / "model.tflite"

    shutil.copy2(tflite_path, dest)
    info_path = dest.with_name("task_info.json")
    info: dict[str, Any] = {
        "runtime": "mediapipe",
        "source_tflite": str(tflite_path.name),
        "packaged_at": dest.name,
        "status": "stub",
        "todo": [
            "Invoke mediapipe_model_maker to derive the task graph.",
            "Pack TFLite + task graph into a .task archive.",
            "Emit task_metadata.pbtxt with input/output tensor names.",
        ],
    }
    info_path.write_text(json.dumps(info, indent=2))
    logger.info(
        "MediaPipe stub bundle written to %s (metadata: %s)",
        dest,
        info_path,
    )
    return dest.resolve()


__all__ = ["wrap_for_mediapipe"]
