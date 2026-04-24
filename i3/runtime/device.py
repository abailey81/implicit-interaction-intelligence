"""Device selection, CUDA fast-path toggles, and autocast helpers.

These three concerns live together because they are almost always used
together:

* :func:`pick_device` decides *where* a tensor will live (CUDA > MPS > CPU
  with an override for operator choice).
* :func:`enable_cuda_optimizations` flips on the CUDA/cuDNN tuning knobs
  (cuDNN benchmark, TF32 matmul) that only make sense once we know CUDA is
  visible.  It is a safe no-op on CPU-only builds.
* :func:`autocast_context` yields the right ``torch.amp.autocast`` context
  manager for the active device, or a ``nullcontext`` on CPU where mixed
  precision would just slow things down.

Keeping them colocated means callers import *one* module to get the full
GPU fast-path story — and the CPU fall-backs are exercised by every unit
test, so the module works on machines that never see CUDA.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import ContextManager

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _mps_available() -> bool:
    """Return ``True`` when the Apple Silicon MPS backend is usable."""
    # ``torch.backends.mps`` is absent on pre-1.12 builds and on Linux /
    # Windows wheels. ``is_available`` and ``is_built`` can both exist but
    # report False on hardware that does not support it.
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    if is_available is None:
        return False
    try:
        return bool(is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available :class:`torch.device`.

    Selection order, highest priority first:

    1. An explicit override in ``prefer`` that is neither empty nor the
       special string ``"auto"``. Recognised values are ``"cuda"``,
       ``"cuda:<N>"``, ``"mps"``, and ``"cpu"``. If CUDA is requested but
       unavailable we fall through to auto-detect rather than crashing —
       this keeps CPU-only CI green when someone hard-codes ``"cuda"``.
    2. CUDA, when :func:`torch.cuda.is_available` returns ``True``.
    3. Apple Silicon MPS, when :mod:`torch.backends.mps` is available.
    4. CPU as the always-available fallback.

    Parameters
    ----------
    prefer : str, optional
        User / CLI override. ``None`` and ``"auto"`` both trigger
        auto-detection.

    Returns
    -------
    torch.device
        The selected device.
    """
    override = (prefer or "").strip().lower()
    if override and override != "auto":
        if override.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(override)
            logger.warning(
                "pick_device: requested %r but CUDA is not available; "
                "falling back to auto-detect.",
                override,
            )
        elif override == "mps":
            if _mps_available():
                return torch.device("mps")
            logger.warning(
                "pick_device: requested 'mps' but backend is not available; "
                "falling back to auto-detect."
            )
        elif override == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(
                "pick_device: unknown device override %r; auto-detecting.",
                override,
            )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# CUDA fast-path toggles
# ---------------------------------------------------------------------------


_cuda_optimizations_applied = False


def enable_cuda_optimizations() -> bool:
    """Turn on cuDNN benchmark and TF32 fast matmul when CUDA is visible.

    Idempotent — repeated calls are no-ops after the first successful pass.

    Returns
    -------
    bool
        ``True`` when the CUDA toggles were applied, ``False`` when CUDA
        is unavailable (in which case the function is a safe no-op).
    """
    global _cuda_optimizations_applied
    if _cuda_optimizations_applied:
        return torch.cuda.is_available()

    if not torch.cuda.is_available():
        # Still mark as "applied" so we do not re-check on every call; the
        # return value tells the caller whether the GPU path is live.
        _cuda_optimizations_applied = True
        return False

    # cuDNN benchmark auto-tunes convolution algorithms the first time a
    # given shape is seen and caches the winner.  Small one-time cost,
    # large steady-state speed-up on the TCN encoder.
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not set cudnn.benchmark", exc_info=True)

    # TF32 matmul is the RTX-30/40-series fast path for fp32 GEMMs — up to
    # 8x over strict IEEE fp32 with negligible accuracy impact for the
    # model sizes we train here.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not set float32 matmul precision", exc_info=True)

    # Explicit TF32 toggles for older torch builds where
    # set_float32_matmul_precision does not cover cuDNN.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not toggle TF32 flags", exc_info=True)

    logger.info(
        "CUDA optimizations enabled: cudnn.benchmark=True, TF32 matmul=high"
    )
    _cuda_optimizations_applied = True
    return True


# ---------------------------------------------------------------------------
# Autocast helper
# ---------------------------------------------------------------------------


def autocast_context(
    device: torch.device | str | None,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
) -> ContextManager[None]:
    """Return a mixed-precision ``autocast`` context for *device*.

    * CUDA -> ``torch.amp.autocast(device_type="cuda", dtype=fp16)``
      (fp16 is the RTX 40-series fast path; bf16 is chosen on older GPUs
      that report native bf16 support).
    * MPS  -> ``torch.amp.autocast(device_type="mps")`` if the running
      torch build supports it, else :class:`contextlib.nullcontext`.
    * CPU / unknown -> :class:`contextlib.nullcontext` — CPU autocast
      exists but is rarely a win for our workloads and would change
      numerics; keeping it off preserves CPU behaviour bit-for-bit.

    Parameters
    ----------
    device : torch.device | str | None
        Target device. Strings and ``None`` are accepted for ergonomics.
    dtype : torch.dtype, optional
        Override autocast dtype. Defaults to ``torch.float16`` on CUDA,
        otherwise the torch default.
    enabled : bool
        Short-circuit to ``nullcontext`` when ``False``.  Lets callers
        gate AMP on a config flag without importing ``contextlib``.

    Returns
    -------
    ContextManager[None]
        The autocast context manager (or a ``nullcontext`` no-op).
    """
    if not enabled:
        return nullcontext()

    if device is None:
        return nullcontext()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    dtype_ = dtype
    if dev.type == "cuda":
        if dtype_ is None:
            # bf16 when the GPU advertises it natively (A100/H100/RTX 40-
            # series via CUDA 11.5+), fp16 otherwise.  bf16 has a wider
            # exponent range and avoids most overflow issues without a
            # GradScaler — but fp16 is the safer default on Ampere and
            # earlier.
            try:
                if torch.cuda.is_bf16_supported():
                    dtype_ = torch.bfloat16
                else:
                    dtype_ = torch.float16
            except Exception:
                dtype_ = torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype_)
    if dev.type == "mps":
        # MPS autocast is supported from torch 2.x; fall back to a no-op
        # if the running build does not expose it.
        try:
            return torch.amp.autocast(device_type="mps")
        except (RuntimeError, TypeError, ValueError):  # pragma: no cover
            return nullcontext()
    return nullcontext()


__all__ = [
    "autocast_context",
    "enable_cuda_optimizations",
    "pick_device",
]
