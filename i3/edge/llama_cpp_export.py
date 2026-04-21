"""llama.cpp / GGUF exporter for the Adaptive SLM.

`llama.cpp <https://github.com/ggerganov/llama.cpp>`_ is the reference
SOTA C++ LLM runtime for CPU inference. Its native container format is
``GGUF`` (GPT-Generated Unified Format) — a versioned FlatBuffer-like
binary that supersedes the older ``GGML`` format. As of 2026, GGUF is
the dominant quantised on-device LLM format: the ecosystem around it
(``llama-cpp-python``, ``ollama``, ``lm-studio``, ``koboldcpp``) makes
it the easiest way to ship a small transformer to a mainstream
developer laptop.

Export pipeline for arbitrary transformers is a **two-step** process:

1. PyTorch checkpoint → HuggingFace Transformers format (via
   ``save_pretrained`` — produces ``config.json`` + ``pytorch_model.bin``
   + tokenizer files).
2. HuggingFace format → GGUF via ``llama.cpp/convert_hf_to_gguf.py``
   and then ``llama.cpp/quantize`` for the chosen bit-width.

For the I3 Adaptive SLM, step 1 requires mapping the custom architecture
onto one of llama.cpp's supported model families (``llama``,
``qwen2``, ``phi3``, ``gpt2``, etc.). This module wraps both steps and
exposes a single ``convert_slm_to_gguf`` entry point.

Soft-imports ``llama-cpp-python``. If missing, every function raises
:class:`RuntimeError` with the install hint ``pip install
llama-cpp-python``. The actual GGUF conversion can also be driven via a
shell-out to ``llama.cpp/convert_hf_to_gguf.py`` when the C++ binaries
are on ``PATH``.

Usage::

    import torch
    from pathlib import Path
    from i3.slm.model import AdaptiveSLM
    from i3.edge.llama_cpp_export import convert_slm_to_gguf

    slm = AdaptiveSLM().eval()
    gguf = convert_slm_to_gguf(
        pytorch_model=slm,
        out_path=Path("exports/llama_cpp/slm-q4_k_m.gguf"),
        quantisation="Q4_K_M",
    )
"""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404 - used deliberately for llama.cpp convert script
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import llama_cpp as _llama_cpp_module  # type: ignore[import-not-found]

    _LLAMA_CPP_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _llama_cpp_module = None  # type: ignore[assignment]
    _LLAMA_CPP_AVAILABLE = False
    logger.info(
        "llama-cpp-python not installed; GGUF export will be unavailable. "
        "Install with: pip install llama-cpp-python"
    )


SUPPORTED_QUANTISATIONS: Final[list[str]] = [
    "F16",
    "Q8_0",
    "Q5_K_M",
    "Q4_K_M",
    "Q4_0",
    "Q3_K_S",
    "Q2_K",
]
"""Quantisation labels supported by the llama.cpp ``quantize`` binary.

Ordered from highest fidelity (``F16``) to most aggressive
compression (``Q2_K``). ``Q4_K_M`` is the recommended production
default in the llama.cpp ecosystem — it offers the best
quality/size trade-off for 7B-class models at ~4.8 bits per weight.
"""


_INSTALL_HINT: str = (
    "llama-cpp-python is required to export a GGUF file. Install with:\n\n"
    "    pip install llama-cpp-python\n\n"
    "For conversion from arbitrary PyTorch checkpoints you also need the "
    "llama.cpp repo cloned locally (for convert_hf_to_gguf.py + the "
    "quantize binary). See https://github.com/ggerganov/llama.cpp."
)


def _require_llama_cpp() -> ModuleType:
    """Return the ``llama_cpp`` module or raise.

    Returns:
        The imported ``llama_cpp`` module.

    Raises:
        RuntimeError: If ``llama-cpp-python`` is not installed.
    """
    if not _LLAMA_CPP_AVAILABLE or _llama_cpp_module is None:
        raise RuntimeError(_INSTALL_HINT)
    return _llama_cpp_module


def _save_as_hf(pytorch_model: Any, hf_dir: Path) -> Path:
    """Persist a PyTorch model as a HuggingFace-format directory.

    If the model exposes ``save_pretrained`` (duck-typed HF transformer)
    we call that directly; otherwise we persist the raw ``state_dict``
    to ``pytorch_model.bin`` and emit a minimal ``config.json`` so that
    ``convert_hf_to_gguf.py`` can later pick it up.

    Args:
        pytorch_model: The PyTorch module to persist.
        hf_dir: Destination directory; created if absent.

    Returns:
        The directory path, resolved.

    Raises:
        RuntimeError: If persistence fails.
    """
    import json

    import torch

    hf_dir = Path(hf_dir)
    hf_dir.mkdir(parents=True, exist_ok=True)

    save_fn = getattr(pytorch_model, "save_pretrained", None)
    if callable(save_fn):
        save_fn(str(hf_dir))
        return hf_dir.resolve()

    state = getattr(pytorch_model, "state_dict", None)
    if not callable(state):
        raise RuntimeError(
            "convert_slm_to_gguf expects either save_pretrained() or a "
            "torch.nn.Module with state_dict()."
        )
    torch.save(state(), hf_dir / "pytorch_model.bin")
    (hf_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["I3AdaptiveSLM"],
                "model_type": "llama",  # closest llama.cpp supported family
                "note": "Auto-generated by i3.edge.llama_cpp_export",
            },
            indent=2,
        )
    )
    return hf_dir.resolve()


def _run_convert_hf_to_gguf(hf_dir: Path, fp16_gguf: Path) -> None:
    """Shell out to ``convert_hf_to_gguf.py`` to emit an FP16 GGUF.

    Args:
        hf_dir: A HuggingFace-style model directory.
        fp16_gguf: Destination ``.gguf`` path for the FP16 intermediate.

    Raises:
        RuntimeError: If the converter script cannot be located on
            ``PATH`` or the subprocess exits non-zero.
    """
    converter = shutil.which("convert_hf_to_gguf.py") or shutil.which(
        "convert-hf-to-gguf"
    )
    if converter is None:
        raise RuntimeError(
            "convert_hf_to_gguf.py not found on PATH. Clone "
            "https://github.com/ggerganov/llama.cpp and add its root "
            "(or a python launcher wrapping convert_hf_to_gguf.py) to PATH."
        )
    cmd: list[str] = [
        converter,
        str(hf_dir),
        "--outfile",
        str(fp16_gguf),
        "--outtype",
        "f16",
    ]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(  # nosec B603 - args controlled
        cmd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "convert_hf_to_gguf.py failed (exit "
            f"{proc.returncode}):\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def _run_llama_cpp_quantize(
    fp16_gguf: Path, quantised_gguf: Path, quantisation: str
) -> None:
    """Shell out to the llama.cpp ``quantize`` binary.

    Args:
        fp16_gguf: Input FP16 GGUF file.
        quantised_gguf: Output quantised GGUF file.
        quantisation: One of :data:`SUPPORTED_QUANTISATIONS`.

    Raises:
        RuntimeError: If the quantize binary is missing or fails.
    """
    quantize = shutil.which("llama-quantize") or shutil.which(
        "quantize"
    )
    if quantize is None:
        raise RuntimeError(
            "llama.cpp quantize binary not found on PATH. Build "
            "llama.cpp and add its build directory (or install "
            "`llama-cpp` via your package manager) to PATH."
        )
    cmd: list[str] = [
        quantize,
        str(fp16_gguf),
        str(quantised_gguf),
        quantisation,
    ]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(  # nosec B603 - args controlled
        cmd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama.cpp quantize failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_slm_to_gguf(
    pytorch_model: Any,
    out_path: Path,
    quantisation: str = "Q4_K_M",
) -> Path:
    """Convert a PyTorch SLM to a quantised GGUF file.

    The two-step PyTorch → HuggingFace → GGUF pipeline is:

    1. Persist the PyTorch module as a HuggingFace directory.
    2. Invoke ``convert_hf_to_gguf.py`` to emit an FP16 GGUF.
    3. Invoke ``llama.cpp/quantize`` to re-compress at the requested
       bit-width.

    For the I3 Adaptive SLM this assumes the model has been mapped
    onto one of llama.cpp's supported architectures (``llama``,
    ``qwen2``, ``phi3`` etc.) via ``save_pretrained``. If it has not,
    this function falls back to writing a minimal ``config.json``
    with ``model_type: llama`` and lets the converter do its best —
    the caller should verify correctness before shipping.

    Args:
        pytorch_model: The SLM instance to export.
        out_path: Destination path for the final quantised ``.gguf``.
        quantisation: A label from :data:`SUPPORTED_QUANTISATIONS`;
            defaults to ``"Q4_K_M"``.

    Returns:
        The resolved path of the final GGUF.

    Raises:
        RuntimeError: If ``llama-cpp-python`` is missing, if the shell
            tools are unavailable, or if any sub-step fails.
        ValueError: If ``quantisation`` is not a supported label.
    """
    _require_llama_cpp()
    if quantisation not in SUPPORTED_QUANTISATIONS:
        raise ValueError(
            f"Unsupported quantisation {quantisation!r}; "
            f"expected one of {SUPPORTED_QUANTISATIONS}."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="i3-gguf-") as tmp:
        tmp_path = Path(tmp)
        hf_dir = _save_as_hf(pytorch_model, tmp_path / "hf")
        fp16 = tmp_path / "model.f16.gguf"
        _run_convert_hf_to_gguf(hf_dir, fp16)
        _run_llama_cpp_quantize(fp16, out_path, quantisation)

    logger.info(
        "Wrote GGUF (%s) to %s", quantisation, out_path
    )
    return out_path.resolve()


__all__ = [
    "SUPPORTED_QUANTISATIONS",
    "convert_slm_to_gguf",
]
