"""Single CLI entry point that drives every alternative edge exporter.

Iterates over the set of I3 alternative edge runtimes and writes each
artefact into ``exports/{runtime}/``. Runtimes whose backing library is
not installed are skipped with a clear log line — the script is always
safe to run end-to-end in CI.

Usage::

    python scripts/export_all_runtimes.py \\
        --checkpoint checkpoints/slm-base.pt \\
        --onnx exports/onnx/tcn.onnx \\
        --out-root exports

The script expects:

* A PyTorch checkpoint usable by the TCN + SLM constructors (optional
  for runtimes that only need the ONNX graph).
* An ONNX graph of the TCN encoder (for TVM / IREE / OpenVINO).

On an Apple Silicon Mac with ``mlx`` + ``coremltools`` installed the
run covers MLX, Core ML and the MediaPipe stub; on an Intel Meteor
Lake laptop with ``openvino`` it adds the NPU-targeted OpenVINO IR.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("export_all_runtimes")


def _load_pytorch_model(checkpoint: Path | None) -> tuple[Any, Any] | None:
    """Best-effort load of ``(tcn, slm)`` PyTorch models.

    Returns ``None`` if the project-internal imports or checkpoint
    loading raise — callers should then skip the PyTorch-rooted
    exporters.

    Args:
        checkpoint: Optional path to a serialised checkpoint.

    Returns:
        ``(tcn, slm)`` modules in ``eval`` mode, or ``None``.
    """
    try:
        import torch

        from i3.encoder.tcn import TemporalConvNet
        from i3.slm.model import AdaptiveSLM
    except ImportError as exc:
        logger.warning(
            "Could not import i3 model classes (%s); skipping "
            "PyTorch-rooted exports.",
            exc,
        )
        return None

    tcn = TemporalConvNet().eval()
    slm = AdaptiveSLM().eval()
    if checkpoint is not None and checkpoint.exists():
        try:
            state = torch.load(
                str(checkpoint), map_location="cpu", weights_only=True
            )
            if isinstance(state, dict) and "tcn" in state:
                tcn.load_state_dict(state["tcn"])
            if isinstance(state, dict) and "slm" in state:
                slm.load_state_dict(state["slm"])
            logger.info("Loaded checkpoint %s", checkpoint)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Checkpoint %s did not load cleanly (%s); continuing "
                "with randomly-initialised weights.",
                checkpoint,
                exc,
            )
    return tcn, slm


def _try(runtime: str, fn: Callable[[], Any]) -> bool:
    """Run ``fn`` and log the outcome; return True on success.

    Args:
        runtime: A human-readable runtime name for log messages.
        fn: A no-argument callable that performs the export.

    Returns:
        True if ``fn`` succeeded, False if it raised a RuntimeError
        (missing lib, unsupported host) or FileNotFoundError.
    """
    logger.info("=== %s ===", runtime)
    try:
        fn()
    except RuntimeError as exc:
        logger.warning("[%s] skipped: %s", runtime, exc)
        return False
    except FileNotFoundError as exc:
        logger.warning("[%s] missing input: %s", runtime, exc)
        return False
    except ValueError as exc:
        logger.error("[%s] invalid argument: %s", runtime, exc)
        return False
    logger.info("[%s] OK", runtime)
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argv list (for testing).

    Returns:
        Process exit code (always 0 — per-runtime failures are logged
        but never block the overall run).
    """
    parser = argparse.ArgumentParser(
        description="Export I3 models via every alternative edge runtime."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PyTorch checkpoint to load into TCN / SLM.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="ONNX file for TVM / IREE / OpenVINO targets.",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=None,
        help="TFLite file for the MediaPipe stub bundle.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("exports"),
        help="Root directory under which per-runtime artefacts go.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    models = _load_pytorch_model(args.checkpoint)
    tcn: Any = None
    slm: Any = None
    if models is not None:
        tcn, slm = models

    # MLX
    if tcn is not None and slm is not None:

        def _mlx() -> None:
            from i3.edge.mlx_export import (
                convert_slm_to_mlx,
                convert_tcn_to_mlx,
            )

            convert_tcn_to_mlx(tcn, out_root / "mlx" / "tcn.npz")
            convert_slm_to_mlx(slm, out_root / "mlx" / "slm.npz")

        _try("mlx", _mlx)

    # llama.cpp GGUF
    if slm is not None:

        def _gguf() -> None:
            from i3.edge.llama_cpp_export import convert_slm_to_gguf

            convert_slm_to_gguf(
                slm,
                out_root / "llama_cpp" / "slm-q4_k_m.gguf",
                quantisation="Q4_K_M",
            )

        _try("llama.cpp", _gguf)

    # TVM
    if args.onnx is not None:

        def _tvm() -> None:
            from i3.edge.tvm_export import compile_tcn_to_tvm

            compile_tcn_to_tvm(
                args.onnx,
                target="llvm -mcpu=cortex-a76",
                out_dir=out_root / "tvm" / "cortex-a76",
            )

        _try("tvm", _tvm)

    # IREE
    if args.onnx is not None:

        def _iree() -> None:
            from i3.edge.iree_export import compile_to_iree

            compile_to_iree(
                args.onnx,
                backend="vmvx",
                out_path=out_root / "iree" / "tcn-vmvx.vmfb",
            )

        _try("iree", _iree)

    # Core ML
    if tcn is not None:

        def _coreml() -> None:
            from i3.edge.coreml_export import convert_tcn_to_coreml

            convert_tcn_to_coreml(
                tcn, out_root / "coreml" / "tcn.mlpackage"
            )

        _try("coreml", _coreml)

    # TensorRT-LLM
    if args.onnx is not None:

        def _trtllm() -> None:
            from i3.edge.tensorrt_llm_export import convert_slm_to_trtllm

            convert_slm_to_trtllm(args.onnx, out_root / "tensorrt_llm")

        _try("tensorrt_llm", _trtllm)

    # OpenVINO
    if args.onnx is not None:

        def _ov() -> None:
            from i3.edge.openvino_export import convert_tcn_to_openvino

            convert_tcn_to_openvino(
                args.onnx,
                out_root / "openvino",
                precision="INT8",
            )

        _try("openvino", _ov)

    # MediaPipe
    if args.tflite is not None:

        def _mp() -> None:
            from i3.edge.mediapipe_export import wrap_for_mediapipe

            wrap_for_mediapipe(
                args.tflite, out_root / "mediapipe" / "tcn.task"
            )

        _try("mediapipe", _mp)

    logger.info("Done. Artefacts under %s", out_root.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
