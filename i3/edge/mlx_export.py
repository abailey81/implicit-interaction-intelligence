"""Apple MLX exporter for I3 TCN encoder and Adaptive SLM.

`MLX <https://github.com/ml-explore/mlx>`_ is Apple's native array
framework for Apple Silicon. It uses a unified memory model so tensors
do not need to be copied between the CPU and the integrated GPU and it
ships first-class Python bindings. For an I3 developer running the
local-laptop demo on an M-series Mac, MLX is the fastest path from a
PyTorch checkpoint to a native-speed on-device forward pass — no
Core ML conversion, no ONNX round-trip.

Pipeline::

    +------------------+   state_dict()
    | PyTorch nn.Module | ----------------+
    +------------------+                  v
                             +------------------------+
                             | dict[str, np.ndarray]  |
                             +------------------------+
                                        |
                     mlx.utils.tree_unflatten + mx.array
                                        |
                                        v
                             +------------------------+
                             | dict[str, mx.array]    |
                             +------------------------+
                                        |
                                 mx.save(...)
                                        |
                                        v
                             +------------------------+
                             |   model.npz  (on-disk) |
                             | MLX native serialised  |
                             +------------------------+

Soft-imports ``mlx``. If the package is missing every function raises
:class:`RuntimeError` with a clear install hint: ``pip install mlx``.
MLX only ships wheels for macOS on arm64; on any other platform the
soft-import will silently fail and calls raise at runtime.

Usage::

    import torch
    from pathlib import Path
    from i3.encoder.tcn import TemporalConvNet
    from i3.edge.mlx_export import convert_tcn_to_mlx, mlx_inference_smoke

    encoder = TemporalConvNet().eval()
    out = convert_tcn_to_mlx(encoder, Path("exports/mlx/tcn.npz"))
    mlx_inference_smoke(out, feature_vector=[0.0] * 32)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import mlx.core as _mx  # type: ignore[import-not-found]
    import mlx.utils as _mx_utils  # type: ignore[import-not-found]

    _MLX_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _mx = None  # type: ignore[assignment]
    _mx_utils = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False
    logger.info(
        "mlx not installed; MLX export will be unavailable. "
        "Install with: pip install mlx  (Apple Silicon only)"
    )


_INSTALL_HINT: str = (
    "Apple MLX is required to export a .npz MLX container. Install on "
    "an Apple Silicon Mac with:\n\n    pip install mlx\n\n"
    "MLX only provides wheels for macOS/arm64; other platforms are "
    "unsupported. See https://ml-explore.github.io/mlx/."
)


def _require_mlx() -> tuple[ModuleType, ModuleType]:
    """Return the ``(mlx.core, mlx.utils)`` modules or raise.

    Returns:
        A tuple ``(mx, mx_utils)`` with the two required MLX modules.

    Raises:
        RuntimeError: If MLX is not installed.
    """
    if not _MLX_AVAILABLE or _mx is None or _mx_utils is None:
        raise RuntimeError(_INSTALL_HINT)
    return _mx, _mx_utils


def _torch_state_to_numpy(model: Any) -> dict[str, Any]:
    """Convert a PyTorch ``state_dict`` to a nested numpy dict.

    Args:
        model: A ``torch.nn.Module`` with a ``state_dict`` method.

    Returns:
        A flat ``{parameter_name: numpy.ndarray}`` mapping suitable for
        re-materialising into ``mx.array`` tensors.

    Raises:
        RuntimeError: If ``model`` does not expose ``state_dict``.
    """
    import numpy as np  # local import to avoid hard dep at module load

    state = getattr(model, "state_dict", None)
    if not callable(state):
        raise RuntimeError(
            "convert_*_to_mlx expects a torch.nn.Module with state_dict()."
        )
    out: dict[str, Any] = {}
    for key, tensor in state().items():
        # Detach, move to CPU, convert to numpy. .numpy() enforces CPU
        # tensor — MLX will copy into unified memory on load.
        out[key] = np.asarray(tensor.detach().cpu().numpy())
    return out


def _write_mlx_npz(
    weights: dict[str, Any], out_path: Path
) -> Path:
    """Serialise the weight dict to an MLX ``.npz`` file.

    Args:
        weights: A flat mapping of parameter names to numpy arrays.
        out_path: Destination path; parents are created if needed.

    Returns:
        The resolved absolute path of the written file.

    Raises:
        RuntimeError: If MLX is not installed, or if serialisation fails.
    """
    mx, mx_utils = _require_mlx()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # mlx.utils.tree_flatten / tree_unflatten gives us a deterministic
    # flat representation that mx.save can persist.
    mlx_weights = {k: mx.array(v) for k, v in weights.items()}
    try:
        flat = mx_utils.tree_flatten(mlx_weights)
    except AttributeError as exc:
        raise RuntimeError(
            "mlx.utils.tree_flatten missing; please upgrade mlx."
        ) from exc

    try:
        mx.save(str(out_path), dict(flat))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"mx.save failed writing to {out_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    logger.info("Wrote MLX container to %s", out_path)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_tcn_to_mlx(pytorch_model: Any, out_path: Path) -> Path:
    """Convert the TCN encoder to MLX's native ``.npz`` format.

    Args:
        pytorch_model: A trained ``TemporalConvNet`` instance in
            ``eval`` mode.
        out_path: Destination path for the MLX container.

    Returns:
        The resolved path of the written file.

    Raises:
        RuntimeError: If MLX is not installed or the underlying
            conversion fails.
    """
    _require_mlx()
    weights = _torch_state_to_numpy(pytorch_model)
    return _write_mlx_npz(weights, Path(out_path))


def convert_slm_to_mlx(pytorch_model: Any, out_path: Path) -> Path:
    """Convert the Adaptive SLM to MLX's native ``.npz`` format.

    The conversion is parametric and works for any ``torch.nn.Module``
    whose ``state_dict`` entries map 1:1 to the target MLX module. If
    the consumer uses a custom MLX network, it should rely on
    :func:`mlx.utils.tree_unflatten` on load to reshape.

    Args:
        pytorch_model: A trained ``AdaptiveSLM`` instance in ``eval``
            mode.
        out_path: Destination path for the MLX container.

    Returns:
        The resolved path of the written file.

    Raises:
        RuntimeError: If MLX is not installed or the underlying
            conversion fails.
    """
    _require_mlx()
    weights = _torch_state_to_numpy(pytorch_model)
    return _write_mlx_npz(weights, Path(out_path))


def mlx_inference_smoke(
    mlx_path: Path, feature_vector: Sequence[float]
) -> Any:
    """Load an MLX container and run a single forward-compatible pass.

    This is intentionally minimal — it loads the weights, takes the
    input vector, multiplies by the first available 2-D weight matrix
    (the closest thing to a forward pass that does not assume a
    particular architecture) and returns the resulting ``mx.array``.
    The caller can assert non-NaN / non-zero to sanity-check that the
    weights materialised correctly in unified memory.

    Args:
        mlx_path: Path to a ``.npz`` file written by one of the
            ``convert_*_to_mlx`` helpers.
        feature_vector: A single feature vector (Python floats). Its
            length must match the first dimension of the first weight
            matrix encountered.

    Returns:
        An ``mx.array`` holding the smoke-test output.

    Raises:
        RuntimeError: If MLX is not installed or the file is empty /
            contains no 2-D weight tensor.
    """
    mx, _ = _require_mlx()
    mlx_path = Path(mlx_path)
    if not mlx_path.exists():
        raise RuntimeError(f"MLX container not found: {mlx_path}")

    try:
        loaded = mx.load(str(mlx_path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"mx.load failed for {mlx_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not loaded:
        raise RuntimeError(f"MLX container {mlx_path} is empty.")

    x = mx.array(list(feature_vector))
    for _name, tensor in loaded.items():
        shape = getattr(tensor, "shape", ())
        if len(shape) == 2 and shape[1] == x.shape[0]:
            return tensor @ x
    raise RuntimeError(
        "No 2-D weight whose trailing dim matches the feature vector "
        f"length ({x.shape[0]}) was found in {mlx_path}."
    )


__all__ = [
    "convert_slm_to_mlx",
    "convert_tcn_to_mlx",
    "mlx_inference_smoke",
]
