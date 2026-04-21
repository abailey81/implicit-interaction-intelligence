"""Smoke tests for each alternative edge runtime exporter.

Each test:

1. Imports the exporter module (must always succeed — modules
   soft-import their backend).
2. Skips gracefully if the backing library is not installed.
3. Calls the primary export function with a deliberately wrong or
   empty input and asserts a clear :class:`RuntimeError` /
   :class:`FileNotFoundError` / :class:`ValueError` is raised —
   catching the case where a function silently swallows errors.

These tests are cheap, deterministic and safe to run on any host.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _has(mod: str) -> bool:
    """Return True if ``mod`` is importable.

    Args:
        mod: A dotted module name.
    """
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# MLX
# ---------------------------------------------------------------------------


def test_mlx_module_imports() -> None:
    """The MLX module must always import even without mlx installed."""
    from i3.edge import mlx_export  # noqa: F401


def test_mlx_smoke_missing_file() -> None:
    """MLX inference helper raises on a non-existent file."""
    from i3.edge.mlx_export import mlx_inference_smoke

    if not _has("mlx.core"):
        with pytest.raises(RuntimeError, match="mlx"):
            mlx_inference_smoke(Path("/does/not/exist.npz"), [0.0])
        return
    with pytest.raises(RuntimeError, match="not found|empty|No 2-D"):
        mlx_inference_smoke(Path("/does/not/exist.npz"), [0.0])


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------


def test_llama_cpp_module_imports() -> None:
    from i3.edge import llama_cpp_export  # noqa: F401


def test_llama_cpp_bad_quant_label(tmp_path: Path) -> None:
    """A bogus quantisation label must raise ValueError (or
    RuntimeError when llama.cpp is missing)."""
    from i3.edge.llama_cpp_export import convert_slm_to_gguf

    class _Dummy:
        def state_dict(self) -> dict[str, object]:
            return {}

    out = tmp_path / "bad.gguf"
    if not _has("llama_cpp"):
        with pytest.raises(RuntimeError, match="llama"):
            convert_slm_to_gguf(_Dummy(), out, quantisation="NOT_A_QUANT")
    else:
        with pytest.raises(ValueError, match="Unsupported quantisation"):
            convert_slm_to_gguf(_Dummy(), out, quantisation="NOT_A_QUANT")


# ---------------------------------------------------------------------------
# TVM
# ---------------------------------------------------------------------------


def test_tvm_module_imports() -> None:
    from i3.edge import tvm_export  # noqa: F401


def test_tvm_missing_onnx(tmp_path: Path) -> None:
    from i3.edge.tvm_export import compile_tcn_to_tvm

    missing = tmp_path / "nope.onnx"
    if not _has("tvm"):
        with pytest.raises(RuntimeError, match="TVM"):
            compile_tcn_to_tvm(missing, out_dir=tmp_path / "out")
        return
    with pytest.raises(FileNotFoundError):
        compile_tcn_to_tvm(missing, out_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# IREE
# ---------------------------------------------------------------------------


def test_iree_module_imports() -> None:
    from i3.edge import iree_export  # noqa: F401


def test_iree_bad_backend(tmp_path: Path) -> None:
    from i3.edge.iree_export import compile_to_iree

    if not _has("iree.compiler"):
        with pytest.raises(RuntimeError, match="IREE"):
            compile_to_iree(
                tmp_path / "x.onnx", backend="vmvx"  # type: ignore[arg-type]
            )
        return
    with pytest.raises(ValueError, match="Unsupported IREE backend"):
        compile_to_iree(
            tmp_path / "x.onnx",
            backend="not-a-backend",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Core ML
# ---------------------------------------------------------------------------


def test_coreml_module_imports() -> None:
    from i3.edge import coreml_export  # noqa: F401


def test_coreml_missing_tools(tmp_path: Path) -> None:
    from i3.edge.coreml_export import convert_tcn_to_coreml

    class _Dummy:
        def eval(self) -> "_Dummy":
            return self

    if not _has("coremltools"):
        with pytest.raises(RuntimeError, match="coremltools"):
            convert_tcn_to_coreml(_Dummy(), tmp_path / "out.mlpackage")


# ---------------------------------------------------------------------------
# TensorRT-LLM
# ---------------------------------------------------------------------------


def test_tensorrt_llm_module_imports() -> None:
    from i3.edge import tensorrt_llm_export  # noqa: F401


def test_tensorrt_llm_refuses_without_gpu(tmp_path: Path) -> None:
    """Must refuse to build on a non-NVIDIA host with a clear error."""
    from i3.edge.tensorrt_llm_export import convert_slm_to_trtllm

    # Both the 'no library' and 'no GPU' branches raise RuntimeError
    # — we just want to make sure the function does not silently
    # proceed.
    with pytest.raises(RuntimeError):
        convert_slm_to_trtllm(tmp_path / "x.onnx", tmp_path / "out")


# ---------------------------------------------------------------------------
# OpenVINO
# ---------------------------------------------------------------------------


def test_openvino_module_imports() -> None:
    from i3.edge import openvino_export  # noqa: F401


def test_openvino_bad_precision(tmp_path: Path) -> None:
    from i3.edge.openvino_export import convert_tcn_to_openvino

    if not _has("openvino"):
        with pytest.raises(RuntimeError, match="OpenVINO"):
            convert_tcn_to_openvino(
                tmp_path / "x.onnx",
                tmp_path / "ov",
                precision="INT2",  # type: ignore[arg-type]
            )
        return
    with pytest.raises(ValueError, match="Unknown precision"):
        convert_tcn_to_openvino(
            tmp_path / "x.onnx",
            tmp_path / "ov",
            precision="INT2",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------


def test_mediapipe_module_imports() -> None:
    from i3.edge import mediapipe_export  # noqa: F401


def test_mediapipe_missing_tflite(tmp_path: Path) -> None:
    from i3.edge.mediapipe_export import wrap_for_mediapipe

    with pytest.raises(FileNotFoundError):
        wrap_for_mediapipe(
            tmp_path / "not.tflite", tmp_path / "out"
        )


def test_mediapipe_empty_tflite(tmp_path: Path) -> None:
    from i3.edge.mediapipe_export import wrap_for_mediapipe

    empty = tmp_path / "empty.tflite"
    empty.touch()
    with pytest.raises(RuntimeError, match="empty"):
        wrap_for_mediapipe(empty, tmp_path / "out")
