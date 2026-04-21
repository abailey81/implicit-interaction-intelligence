"""ExecuTorch export hook stubs for Implicit Interaction Intelligence (I³).

This module shows **where** the ExecuTorch export pipeline would hook
into I³'s encoder and Adaptive SLM. It deliberately does **not** call
ExecuTorch: the import is soft, so this module is safe to import even
when ExecuTorch is not installed, and every exported function raises
:class:`NotImplementedError` with a clear explanation.

ExecuTorch is Meta/PyTorch's native on-device inference runtime (50 KB
base footprint) supporting ARM Ethos-U NPU, Qualcomm Hexagon, and
MediaTek APU backends — the same backend interface a future Da Vinci
Lite backend would target.

The export pipeline has four stages:

    1. ``torch.export(model, example_inputs)`` produces an
       ``ExportedProgram`` — a static, FX-style graph with pytree-shaped
       inputs and outputs.
    2. ``to_edge(exported, compile_config)`` normalises the graph to
       ExecuTorch's core op set.
    3. ``to_executorch(edge_program, backend_config)`` lowers to the
       target backend (Ethos-U, Hexagon, APU, or — hypothetically —
       Da Vinci Lite).
    4. ``program.save("model.pte")`` writes the portable executable
       artefact.

See :doc:`docs/huawei/kirin_deployment.md` for the engineering context
and :doc:`docs/huawei/harmony_hmaf_integration.md` for the HMAF
consumption side.

Example:
    Sketched usage — does not run until ExecuTorch and a Da Vinci
    backend are installed::

        import torch
        from i3.encoder.tcn import TCNEncoder
        from i3.huawei.executorch_hooks import export_encoder_to_pte

        model = TCNEncoder(...).eval()
        example = torch.randn(1, 10, 32)
        export_encoder_to_pte(
            model=model,
            example_inputs=(example,),
            output_path="build/i3_encoder.pte",
            backend="executorch.backends.arm.ethos_u",  # placeholder
            quantize_int8=True,
        )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    # Soft import — ExecuTorch is a large optional dependency. Nothing
    # in this module is safe to call without it, but importing the
    # module itself must never fail.
    import executorch  # type: ignore[import-not-found]  # noqa: F401

    _EXECUTORCH_AVAILABLE: bool = True
except ImportError:
    _EXECUTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend identifiers
# ---------------------------------------------------------------------------


ARM_ETHOS_U_BACKEND: str = "executorch.backends.arm.ethos_u"
"""ARM Ethos-U backend identifier.

Used as a proxy target for Kirin-class NPUs until a Huawei-provided
Da Vinci Lite backend is available.
"""


QUALCOMM_HEXAGON_BACKEND: str = "executorch.backends.qualcomm.hexagon"
"""Qualcomm Hexagon DSP backend identifier (for comparison benchmarking)."""


DAVINCI_LITE_BACKEND: str = "executorch.backends.huawei.davinci_lite"
"""**Hypothetical** Huawei Da Vinci Lite backend identifier.

As of writing, no public ExecuTorch backend targets HiSilicon's
Da Vinci NPU. This identifier is a placeholder for when one exists —
typically via a Huawei-contributed PR to the ExecuTorch repository or
a vendored backend in a Huawei SDK.
"""


# ---------------------------------------------------------------------------
# Hook: encoder export
# ---------------------------------------------------------------------------


def export_encoder_to_pte(
    model: Any,
    example_inputs: tuple[Any, ...],
    output_path: str | Path,
    backend: str = ARM_ETHOS_U_BACKEND,
    quantize_int8: bool = True,
) -> Path:
    """**Stub:** export the TCN encoder to an ExecuTorch ``.pte`` artefact.

    This is the on-device deployment hook for the TCN encoder
    (50 KB INT8). The encoder's static graph — no dynamic shapes,
    no control flow, no string ops — makes it an especially clean
    target for ExecuTorch export.

    Args:
        model: The :class:`torch.nn.Module` encoder to export. Must be
            in ``.eval()`` mode.
        example_inputs: Representative input tuple, used by
            ``torch.export`` to specialise the graph. Shape must match
            the production input; for the I³ encoder this is typically
            ``(torch.randn(1, 10, 32),)``.
        output_path: Destination path for the ``.pte`` artefact.
        backend: ExecuTorch backend identifier. Defaults to ARM Ethos-U;
            swap for :data:`DAVINCI_LITE_BACKEND` once a Huawei backend
            is available.
        quantize_int8: If true, apply INT8 dynamic quantisation before
            export. Encoder is small enough that the memory savings
            matter less than the latency win on NPU.

    Returns:
        The absolute path to the written ``.pte`` file.

    Raises:
        NotImplementedError: Always — this is a stub. Install ExecuTorch
            and a Da Vinci backend (or the Ethos-U proxy) and replace
            the body with the four-stage pipeline documented in the
            module docstring.

    Pipeline sketch (pseudocode)::

        from torch.export import export
        from executorch.exir import to_edge
        from executorch.backends.{backend} import BackendPartitioner

        # 1. torch.export
        exported = export(model, example_inputs)

        # 2. to_edge
        edge = to_edge(exported)

        # 3. (optional) INT8 quantise
        if quantize_int8:
            edge = edge.run_pass(QuantizePass(dtype="int8"))

        # 4. to_executorch → .pte
        program = edge.to_executorch(
            backend_config={"partitioner": BackendPartitioner()}
        )
        Path(output_path).write_bytes(program.buffer)
    """
    _require_executorch("export_encoder_to_pte")
    raise NotImplementedError(
        "export_encoder_to_pte is a stub. Implement the four-stage "
        "pipeline described in this function's docstring once "
        f"ExecuTorch and the {backend!r} backend are installed. "
        f"Target output path: {output_path}. Quantise INT8: {quantize_int8}."
    )


# ---------------------------------------------------------------------------
# Hook: SLM export
# ---------------------------------------------------------------------------


def export_slm_to_pte(
    model: Any,
    example_inputs: tuple[Any, ...],
    output_path: str | Path,
    backend: str = ARM_ETHOS_U_BACKEND,
    quantize_regime: str = "int8_dynamic",
) -> Path:
    """**Stub:** export the Adaptive SLM to an ExecuTorch ``.pte`` artefact.

    The Adaptive SLM is the larger half of I³ (~6.3 M parameters,
    ~6.3 MB at INT8). Its export is more involved than the encoder's
    because of the KV-cache and the per-layer cross-attention to the
    conditioning tokens — both of which require careful handling at
    the ``torch.export`` boundary.

    Args:
        model: The :class:`torch.nn.Module` SLM to export. Must expose
            a forward signature compatible with static export (no
            Python-level control flow inside the forward pass).
        example_inputs: Representative input tuple. For the I³ SLM,
            typically ``(token_ids, conditioning_tokens, causal_mask)``.
        output_path: Destination path for the ``.pte`` artefact.
        backend: ExecuTorch backend identifier (see constants above).
        quantize_regime: One of:

            * ``"int8_dynamic"`` — current I³ default (~2.2× FP32 speed).
            * ``"int4_weight_only"`` — torchao INT4 weight-only
              (~1.73× over FP32 speedup, 65 % less memory).
            * ``"int4_weight_int8_act"`` — ExecuTorch-recommended regime
              for small-model LLM inference on edge NPUs.

    Returns:
        The absolute path to the written ``.pte`` file.

    Raises:
        NotImplementedError: Always — this is a stub.
        ValueError: If ``quantize_regime`` is not one of the three
            supported identifiers.

    Notes:
        The SLM's cross-attention sub-layer (the novel conditioning
        path) must be exported as a separate sub-graph or as a fused
        op; ``torch.export`` handles both, but backend op-coverage
        varies. On an ARM Ethos-U backend the cross-attention softmax
        typically falls back to CPU — this costs <5 % of total
        latency in our profiling.
    """
    valid_regimes = {
        "int8_dynamic",
        "int4_weight_only",
        "int4_weight_int8_act",
    }
    if quantize_regime not in valid_regimes:
        raise ValueError(
            f"quantize_regime must be one of {sorted(valid_regimes)}; "
            f"got {quantize_regime!r}."
        )

    _require_executorch("export_slm_to_pte")
    raise NotImplementedError(
        "export_slm_to_pte is a stub. See the module docstring for the "
        "four-stage pipeline and this function's docstring for SLM-"
        f"specific export caveats. Target: {output_path}, backend: "
        f"{backend!r}, quantize_regime: {quantize_regime!r}."
    )


# ---------------------------------------------------------------------------
# Hook: conditioning projector export
# ---------------------------------------------------------------------------


def export_conditioning_projector_to_pte(
    model: Any,
    example_inputs: tuple[Any, ...],
    output_path: str | Path,
    backend: str = ARM_ETHOS_U_BACKEND,
) -> Path:
    """**Stub:** export the cross-attention ``ConditioningProjector`` to ``.pte``.

    The conditioning projector is a ~25 KB MLP. It produces the
    4 conditioning tokens consumed by the SLM's per-layer
    cross-attention. Exporting it separately lets the HMAF runtime
    cache the 4 conditioning tokens across multiple SLM calls within
    the same session — a meaningful optimisation when the user state
    doesn't change between consecutive messages.

    Args:
        model: The projector :class:`torch.nn.Module`.
        example_inputs: Typically
            ``(torch.randn(1, 72),)`` — concatenation of the 8-dim
            ``AdaptationVector`` and the 64-dim user-state embedding.
        output_path: Destination path for the ``.pte`` artefact.
        backend: ExecuTorch backend identifier.

    Returns:
        The absolute path to the written ``.pte`` file.

    Raises:
        NotImplementedError: Always — this is a stub.
    """
    _require_executorch("export_conditioning_projector_to_pte")
    raise NotImplementedError(
        "export_conditioning_projector_to_pte is a stub. The projector "
        "is the smallest of the three I³ sub-models and by far the "
        "easiest to export: a dense MLP with no dynamic shapes. "
        f"Target: {output_path}, backend: {backend!r}."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_executorch(caller_name: str) -> None:
    """Raise :class:`NotImplementedError` if ExecuTorch is not installed.

    Args:
        caller_name: The name of the calling function — included in the
            error message so users can tell which hook to implement.

    Raises:
        NotImplementedError: If ExecuTorch is unavailable.
    """
    if not _EXECUTORCH_AVAILABLE:
        raise NotImplementedError(
            f"{caller_name} requires ExecuTorch. Install with "
            "`pip install executorch` (and a backend package such as "
            "`executorch-arm-ethos-u`) before calling this hook. See "
            "docs/huawei/kirin_deployment.md for the full export pipeline."
        )


def is_executorch_available() -> bool:
    """Return True if ExecuTorch is importable in the current environment.

    Useful for CI matrices and conditional integration tests.

    Returns:
        Boolean indicating ExecuTorch availability.
    """
    return _EXECUTORCH_AVAILABLE


__all__ = [
    "ARM_ETHOS_U_BACKEND",
    "DAVINCI_LITE_BACKEND",
    "QUALCOMM_HEXAGON_BACKEND",
    "export_conditioning_projector_to_pte",
    "export_encoder_to_pte",
    "export_slm_to_pte",
    "is_executorch_available",
]
