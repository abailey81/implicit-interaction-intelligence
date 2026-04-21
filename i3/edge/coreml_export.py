"""Apple Core ML exporter for the I3 TCN encoder.

`Core ML <https://developer.apple.com/documentation/coreml>`_ is
Apple's on-device inference framework. It targets the full Apple
chipset fleet — the CPU, the integrated GPU, and the Neural Engine
(ANE) — with a single compute-unit dispatch knob. The I3 AI Glasses
scenario (iOS 17+) ships via Core ML: the Neural Engine exposes dense
INT8 / FP16 matmul throughput that no other Apple runtime can match.

Pipeline::

    +-----------+   coremltools.converters.convert
    | PyTorch   | ----------------------------------+
    +-----------+                                   v
                                       +---------------------+
                                       |  ct.models.MLModel  |
                                       +---------------------+
                                                   |
                                             .save(...)
                                                   |
                                                   v
                                       +---------------------+
                                       |   Model.mlpackage   |
                                       | (ML Program binary) |
                                       +---------------------+

Soft-imports ``coremltools``. When absent the functions raise
:class:`RuntimeError` with the install hint ``pip install coremltools``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import coremltools as _ct  # type: ignore[import-not-found]

    _COREML_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _ct = None  # type: ignore[assignment]
    _COREML_AVAILABLE = False
    logger.info(
        "coremltools not installed; Core ML export will be unavailable. "
        "Install with: pip install coremltools"
    )


ComputeUnits = Literal[
    "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"
]

_INSTALL_HINT: str = (
    "Apple coremltools is required to export a .mlpackage. Install with:\n\n"
    "    pip install coremltools\n\n"
    "Core ML targets require macOS (coremltools can still run on Linux "
    "for conversion, but validation needs a Mac). "
    "See https://apple.github.io/coremltools/."
)

# iOS 17 = Core ML 7 = the first Apple OS revision with full ML Program
# (mlprogram) support for transformer-class ops and INT4 palettisation.
_MIN_DEPLOYMENT_TARGET: str = "iOS17"


def _require_coreml() -> ModuleType:
    """Return the ``coremltools`` module or raise.

    Raises:
        RuntimeError: If ``coremltools`` is not installed.
    """
    if not _COREML_AVAILABLE or _ct is None:
        raise RuntimeError(_INSTALL_HINT)
    return _ct


def _resolve_compute_units(
    ct: ModuleType, compute_units: ComputeUnits
) -> Any:
    """Map a string compute-unit label to the coremltools enum.

    Args:
        ct: The imported ``coremltools`` module.
        compute_units: One of the :data:`ComputeUnits` values.

    Returns:
        The matching ``ct.ComputeUnit.*`` enum member.

    Raises:
        ValueError: If the label is unknown.
    """
    enum = getattr(ct, "ComputeUnit", None)
    if enum is None:
        raise RuntimeError(
            "coremltools is installed but ct.ComputeUnit is missing; "
            "please upgrade coremltools."
        )
    mapping: dict[str, str] = {
        "cpuOnly": "CPU_ONLY",
        "cpuAndGPU": "CPU_AND_GPU",
        "cpuAndNeuralEngine": "CPU_AND_NE",
        "all": "ALL",
    }
    key = mapping.get(compute_units)
    if key is None:
        raise ValueError(
            f"Unknown compute_units {compute_units!r}; expected one of "
            f"{list(mapping)}."
        )
    member = getattr(enum, key, None)
    if member is None:
        raise RuntimeError(
            f"coremltools.ComputeUnit has no member {key!r} — upgrade "
            "coremltools."
        )
    return member


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_tcn_to_coreml(
    pytorch_model: Any,
    out_path: Path,
    compute_units: ComputeUnits = "cpuAndNeuralEngine",
) -> Path:
    """Convert the TCN encoder to an ``.mlpackage`` Core ML bundle.

    The conversion traces the PyTorch model using ``torch.jit.trace``
    (implicitly, via ``coremltools.converters.convert``) and lowers it
    to Core ML's ML Program IR. The resulting ``.mlpackage`` is the
    directory-style container that Xcode and Instruments understand.

    Args:
        pytorch_model: A trained ``TemporalConvNet`` in ``eval`` mode.
        out_path: Destination path for the ``.mlpackage`` directory.
        compute_units: Which Apple compute units to allow. Defaults to
            ``"cpuAndNeuralEngine"`` — this is the right choice for
            an AI Glasses form factor where GPU use would shorten
            battery life.

    Returns:
        The resolved path of the written ``.mlpackage``.

    Raises:
        RuntimeError: If ``coremltools`` is not installed or
            conversion fails.
        ValueError: If ``compute_units`` is not recognised.
    """
    ct = _require_coreml()

    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pytorch_model = pytorch_model.eval()

    # Use a representative example input; coremltools needs
    # either a traced model or explicit input specs.
    example = torch.zeros(1, 100, 32)
    try:
        traced = torch.jit.trace(pytorch_model, example, strict=False)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(
            "torch.jit.trace failed before Core ML conversion: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    ios_target = getattr(ct.target, _MIN_DEPLOYMENT_TARGET, None)
    if ios_target is None:
        raise RuntimeError(
            "coremltools.target has no iOS17 member; upgrade "
            "coremltools to >= 7.0."
        )

    compute = _resolve_compute_units(ct, compute_units)

    try:
        mlmodel = ct.converters.convert(
            traced,
            inputs=[
                ct.TensorType(name="input", shape=example.shape),
            ],
            compute_units=compute,
            minimum_deployment_target=ios_target,
            convert_to="mlprogram",
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "coremltools.converters.convert failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        mlmodel.save(str(out_path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"MLModel.save failed writing to {out_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    logger.info("Wrote Core ML bundle to %s", out_path)
    return out_path.resolve()


__all__ = ["convert_tcn_to_coreml"]
