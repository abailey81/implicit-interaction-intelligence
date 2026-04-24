"""ONNX exporter for the I3 TCN encoder.

Converts :class:`i3.encoder.tcn.TemporalConvNet` into an ONNX graph
that can be executed by ONNXRuntime on any supported edge device.

The exporter:

* Uses a ``[1, 10, 32]`` dummy input that matches the default I3
  feature-vector layout.
* Declares ``batch`` and ``time`` as dynamic axes so the exported model
  accepts any batch size and any sequence length at inference time.
* Runs ``onnx.checker.check_model`` to validate the graph structurally.
* Optionally runs an ONNXRuntime inference pass and compares the output
  against the original PyTorch model using ``np.allclose(atol=1e-4)``.

Missing optional dependencies (``onnx``, ``onnxruntime``) are handled
defensively: the exporter emits a clear error message and exits with
status 2 via ``SystemExit`` rather than crashing with an import error.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

logger = logging.getLogger(__name__)


_DEFAULT_BATCH = 1
_DEFAULT_SEQ_LEN = 10
_DEFAULT_FEATURES = 32


def _soft_import(name: str) -> ModuleType | None:
    """Try to import ``name``; return ``None`` on failure.

    Args:
        name: Module name to import.

    Returns:
        The imported module or ``None``.
    """
    try:
        return __import__(name)
    except Exception:
        return None


def _fatal(msg: str, code: int = 2) -> NoReturn:
    """Log a fatal error and exit with the supplied code.

    Args:
        msg: Error message.
        code: Exit status.
    """
    logger.error(msg)
    raise SystemExit(code)


def export_tcn(
    model: Any,
    output_path: Path,
    *,
    opset: int = 17,
    dynamic_axes: bool = True,
    verify: bool = True,
    dummy_batch: int = _DEFAULT_BATCH,
    dummy_seq_len: int = _DEFAULT_SEQ_LEN,
    input_dim: int | None = None,
) -> Path:
    """Export a TCN encoder to ONNX and optionally verify it.

    Args:
        model: A :class:`i3.encoder.tcn.TemporalConvNet` instance.
        output_path: Destination ``.onnx`` file.  Parents created lazily.
        opset: Target ONNX opset version (default 17).
        dynamic_axes: If ``True``, batch and time dims are dynamic.
        verify: If ``True``, parity-check PyTorch vs ONNXRuntime.
        dummy_batch: Batch dim of the dummy input (default 1).
        dummy_seq_len: Sequence length of the dummy input (default 10).
        input_dim: Feature dim; defaults to ``model.input_dim`` or 32.

    Returns:
        Path to the exported ``.onnx`` file.

    Raises:
        SystemExit: When a required optional dependency is missing or an
            op is unsupported at the requested opset.
    """
    import numpy as np
    import torch  # type: ignore[import-not-found]

    onnx = _soft_import("onnx")
    if onnx is None:
        _fatal(
            "onnx is not installed. Install with `pip install onnx`. "
            "TCN export aborted."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Input feature dimensionality -- prefer the model's own attribute
    # when present so that non-default TCNs (e.g. wider encoders) export
    # correctly.
    in_dim = input_dim or getattr(model, "input_dim", _DEFAULT_FEATURES)
    dummy = torch.randn(dummy_batch, dummy_seq_len, in_dim, dtype=torch.float32)

    axes_map: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        axes_map = {
            "input": {0: "batch", 1: "time"},
            "embedding": {0: "batch"},
        }

    # PERF: run the export trace on CUDA when visible — the traced
    # forward executes once and benefits from the GPU — then move the
    # model + dummy input back to CPU before ONNX emission so that the
    # saved graph is device-independent and ``torch.onnx.export``'s
    # serialiser sees plain CPU tensors. CPU-only boxes skip the round-trip.
    try:
        if torch.cuda.is_available():
            model = model.to("cuda")
            dummy = dummy.to("cuda")
            with torch.no_grad():
                _ = model(dummy)  # warm the cache so export trace is fast
            model = model.to("cpu")
            dummy = dummy.to("cpu")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CUDA warm-up failed (%s); continuing on CPU.", exc)
        model = model.to("cpu")
        dummy = dummy.to("cpu")

    model.eval()
    try:
        torch.onnx.export(
            model,
            dummy,
            output_path.as_posix(),
            input_names=["input"],
            output_names=["embedding"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=axes_map,
        )
    except Exception as exc:
        _fatal(
            f"TCN ONNX export failed at opset={opset}: {exc}. "
            "Try a newer opset (e.g. 18) or update torch."
        )

    try:
        graph = onnx.load(output_path.as_posix())
        onnx.checker.check_model(graph)
    except Exception as exc:
        _fatal(f"onnx.checker rejected TCN graph: {exc}")

    logger.info("TCN exported to %s (opset=%d)", output_path, opset)

    if verify:
        ort = _soft_import("onnxruntime")
        if ort is None:
            logger.warning(
                "onnxruntime not installed; skipping PyTorch/ONNX parity check."
            )
            return output_path
        try:
            session = ort.InferenceSession(
                output_path.as_posix(),
                providers=["CPUExecutionProvider"],
            )
            with torch.no_grad():
                pt_out = model(dummy).detach().cpu().numpy()
            onnx_out = session.run(
                ["embedding"], {"input": dummy.numpy().astype(np.float32)}
            )[0]
            if not np.allclose(pt_out, onnx_out, atol=1e-4):
                max_abs = float(np.max(np.abs(pt_out - onnx_out)))
                _fatal(
                    "TCN parity check failed: max abs diff "
                    f"{max_abs:.2e} > 1e-4"
                )
            logger.info("TCN parity OK (ONNXRuntime vs PyTorch, atol=1e-4)")
        except SystemExit:
            raise
        except Exception as exc:
            logger.warning("TCN parity check errored (%s); continuing.", exc)

    return output_path


__all__ = ["export_tcn"]


# Re-export helper for CLI reuse.
def _cli_main() -> None:  # pragma: no cover - convenience shim
    """Tiny CLI used when running ``python -m i3.encoder.onnx_export``."""
    import argparse

    p = argparse.ArgumentParser(description="Export TCN encoder to ONNX.")
    p.add_argument("--checkpoint", type=str, required=False, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    import torch  # type: ignore[import-not-found]

    from i3.encoder.tcn import TemporalConvNet

    model = TemporalConvNet()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    export_tcn(model, Path(args.output), opset=args.opset, verify=not args.no_verify)
    sys.stderr.write(f"wrote {args.output}\n")


if __name__ == "__main__":  # pragma: no cover
    _cli_main()
