"""Intel OpenVINO exporter for the I3 TCN encoder.

`OpenVINO <https://docs.openvino.ai/>`_ is Intel's cross-hardware
inference toolkit. It targets Intel CPUs (with AVX-512 / AMX
acceleration on Sapphire Rapids and newer), Intel integrated GPUs,
and — most relevant for a 2026 laptop-class edge story — the Intel
**NPU** on Meteor Lake / Lunar Lake SoCs. The NPU gives ~11 TOPS of
INT8 compute at ~1 W: exactly the envelope I3's TCN encoder needs.

Pipeline::

    +-----------+  onnx    +------+   ov.convert_model
    | PyTorch   | -------> | ONNX | ------------------+
    +-----------+          +------+                   v
                                           +------------------+
                                           |   ov.Model (IR)  |
                                           +------------------+
                                                       |
                                                 ov.save_model
                                                       |
                                                       v
                                           +------------------+
                                           |  model.xml + bin |
                                           | OpenVINO IR v11  |
                                           +------------------+

Soft-imports ``openvino``. When absent, functions raise
:class:`RuntimeError` with the install hint ``pip install openvino``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import openvino as _ov  # type: ignore[import-not-found]

    _OPENVINO_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _ov = None  # type: ignore[assignment]
    _OPENVINO_AVAILABLE = False
    logger.info(
        "openvino not installed; OpenVINO export will be unavailable. "
        "Install with: pip install openvino"
    )


Precision = Literal["FP32", "FP16", "INT8"]

_INSTALL_HINT: str = (
    "Intel OpenVINO is required to export to IR. Install with:\n\n"
    "    pip install openvino\n\n"
    "Use `pip install openvino-dev[onnx]` if you also need the Model "
    "Optimizer CLI tools. "
    "See https://docs.openvino.ai/."
)


def _require_openvino() -> ModuleType:
    """Return the ``openvino`` module or raise.

    Raises:
        RuntimeError: If OpenVINO is not installed.
    """
    if not _OPENVINO_AVAILABLE or _ov is None:
        raise RuntimeError(_INSTALL_HINT)
    return _ov


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_tcn_to_openvino(
    onnx_path: Path,
    out_dir: Path,
    precision: Precision = "INT8",
) -> Path:
    """Convert an ONNX TCN to OpenVINO IR (``.xml`` + ``.bin``).

    Args:
        onnx_path: Path to the ONNX file to convert.
        out_dir: Destination directory. ``model.xml`` and ``model.bin``
            are written there.
        precision: Target numeric precision. ``"INT8"`` triggers NNCF
            post-training quantisation before ``ov.save_model``;
            ``"FP16"`` applies ``compress_to_fp16=True`` on save;
            ``"FP32"`` leaves weights unchanged.

    Returns:
        The resolved path of ``out_dir``.

    Raises:
        RuntimeError: If OpenVINO is not installed or the conversion /
            NNCF step fails.
        FileNotFoundError: If ``onnx_path`` does not exist.
        ValueError: If ``precision`` is unknown.
    """
    ov = _require_openvino()

    if precision not in ("FP32", "FP16", "INT8"):
        raise ValueError(
            f"Unknown precision {precision!r}; expected FP32, FP16 or INT8."
        )

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_model = getattr(ov, "convert_model", None)
    save_model = getattr(ov, "save_model", None)
    if convert_model is None or save_model is None:
        raise RuntimeError(
            "openvino is installed but ov.convert_model / ov.save_model "
            "is missing. Please upgrade: pip install -U 'openvino>=2024.0'"
        )

    try:
        model = convert_model(str(onnx_path))
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "ov.convert_model failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if precision == "INT8":
        try:
            import nncf  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "INT8 quantisation requires NNCF. "
                "Install with: pip install nncf"
            ) from exc
        # Post-training quantisation needs a representative calibration
        # dataset; callers should replace this with a real generator.
        logger.info(
            "Applying NNCF post-training quantisation (INT8). "
            "Replace the synthetic calibration loader with a real "
            "dataset before shipping."
        )
        try:
            import numpy as np

            def _calib():
                for _ in range(8):
                    yield np.zeros((1, 100, 32), dtype=np.float32)

            model = nncf.quantize(
                model, nncf.Dataset(_calib()), subset_size=8
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                "NNCF INT8 quantisation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    compress_to_fp16 = precision == "FP16"
    xml_path = out_dir / "model.xml"
    try:
        save_model(model, str(xml_path), compress_to_fp16=compress_to_fp16)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        raise RuntimeError(
            "ov.save_model failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    logger.info(
        "Wrote OpenVINO IR (%s) to %s", precision, out_dir
    )
    return out_dir.resolve()


__all__ = ["convert_tcn_to_openvino"]
