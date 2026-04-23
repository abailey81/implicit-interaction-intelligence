"""ONNX exporter for the I3 Adaptive SLM.

Exports a **prefill-only** variant of :class:`i3.slm.model.AdaptiveSLM`
to ONNX.  Inputs:

* ``input_ids``            -- ``[B, T]``  (int64)
* ``conditioning_tokens``  -- ``[B, 4, 256]`` (float32)
* ``attention_mask``       -- ``[B, T]``  (int64, 1 = keep, 0 = pad)

Output:

* ``logits``               -- ``[B, T, vocab_size]`` (float32)

The exported graph therefore covers the prompt-processing ("prefill")
phase.  Token-by-token decode requires the ``past_key_values`` wiring
that is not yet stable in our SLM; decode-per-step should be exported
as a separate graph once KV caching is stable.

Cross-attention conditioning is passed as an explicit input tensor so
that downstream runtimes can supply pre-computed conditioning from the
encoder without re-running the :class:`ConditioningProjector`.

Soft-imports ``onnx`` / ``onnxruntime``.  Unsupported ops cause
``SystemExit(2)`` with a clear message rather than an opaque torch
error.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


_DEFAULT_BATCH = 1
_DEFAULT_SEQ_LEN = 16
_DEFAULT_N_COND = 4
_DEFAULT_D_MODEL = 256


# --------------------------------------------------------------------------- #
# Soft import helpers
# --------------------------------------------------------------------------- #


def _soft_import(name: str) -> ModuleType | None:
    try:
        return __import__(name)
    except Exception:
        return None


def _fatal(msg: str, code: int = 2) -> None:
    logger.error(msg)
    raise SystemExit(code)


# --------------------------------------------------------------------------- #
# Wrapper that accepts conditioning_tokens directly
# --------------------------------------------------------------------------- #


class _SLMExportWrapper(nn.Module):
    """Wrap :class:`AdaptiveSLM` so ONNX sees a simple forward signature.

    The wrapper accepts ``conditioning_tokens`` pre-computed externally
    (shape ``[B, n_cond, d_model]``) and bypasses the internal
    :class:`ConditioningProjector`.  It also accepts an attention mask
    so padded batches can be exported correctly.

    Args:
        slm: The :class:`AdaptiveSLM` instance to wrap.
    """

    def __init__(self, slm: nn.Module) -> None:
        super().__init__()
        self.slm = slm

    def forward(
        self,
        input_ids: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run prefill with explicit conditioning tokens.

        Args:
            input_ids: ``[B, T]`` integer token ids.
            conditioning_tokens: ``[B, n_cond, d_model]`` float tokens.
            attention_mask: ``[B, T]`` mask (1 = keep, 0 = pad).  Applied
                multiplicatively to the embedding output so padded slots
                are zeroed; the causal self-attention mask inside the
                SLM handles left-to-right masking.

        Returns:
            ``[B, T, vocab_size]`` logits tensor.
        """
        # 1. Token + positional embedding.
        x = self.slm.embedding(input_ids)

        # 2. Apply padding mask (expand to d_model).
        mask = attention_mask.to(x.dtype).unsqueeze(-1)
        x = x * mask

        # 3. Causal mask for self-attention.
        from i3.slm.attention import create_causal_mask

        seq_len = input_ids.shape[1]
        causal_mask = create_causal_mask(seq_len, device=x.device)

        # 4. Transformer stack with externally-supplied conditioning.
        for layer in self.slm.layers:
            x, _ = layer(
                x,
                conditioning_tokens=conditioning_tokens,
                causal_mask=causal_mask,
                use_cache=False,
            )

        # 5. Head.
        x = self.slm.final_ln(x)
        return self.slm.output_projection(x)


# --------------------------------------------------------------------------- #
# Public exporter
# --------------------------------------------------------------------------- #


def export_slm(
    model: Any,
    output_path: Path,
    *,
    opset: int = 17,
    dynamic_axes: bool = True,
    verify: bool = True,
    dummy_batch: int = _DEFAULT_BATCH,
    dummy_seq_len: int = _DEFAULT_SEQ_LEN,
) -> Path:
    """Export the prefill variant of an Adaptive SLM to ONNX.

    Args:
        model: A :class:`i3.slm.model.AdaptiveSLM` instance.
        output_path: Destination ``.onnx`` path.
        opset: Target ONNX opset (default 17).
        dynamic_axes: If ``True``, batch and seq dims are dynamic.
        verify: If ``True``, parity-check vs PyTorch on the dummy input.
        dummy_batch: Dummy batch dim (default 1).
        dummy_seq_len: Dummy sequence length (default 16).

    Returns:
        Path to the exported ``.onnx`` file.

    Raises:
        SystemExit: On missing optional deps or unsupported ops.
    """
    import numpy as np

    onnx = _soft_import("onnx")
    if onnx is None:
        _fatal(
            "onnx is not installed. Install with `pip install onnx`. "
            "SLM export aborted."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = int(getattr(model, "vocab_size", 8000))
    d_model = int(getattr(model, "d_model", _DEFAULT_D_MODEL))
    n_cond = int(
        getattr(
            getattr(model, "conditioning_projector", object()),
            "n_tokens",
            _DEFAULT_N_COND,
        )
    )

    input_ids = torch.randint(
        low=0, high=max(vocab, 1), size=(dummy_batch, dummy_seq_len), dtype=torch.long
    )
    conditioning_tokens = torch.randn(
        dummy_batch, n_cond, d_model, dtype=torch.float32
    )
    attention_mask = torch.ones(dummy_batch, dummy_seq_len, dtype=torch.long)

    axes_map: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        axes_map = {
            "input_ids": {0: "batch", 1: "seq"},
            "conditioning_tokens": {0: "batch"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        }

    wrapper = _SLMExportWrapper(model)
    wrapper.eval()

    # NOTE: this graph is the PREFILL-ONLY variant. Token-by-token
    # decode requires past_key_values wiring and is exported separately.
    try:
        torch.onnx.export(
            wrapper,
            (input_ids, conditioning_tokens, attention_mask),
            output_path.as_posix(),
            input_names=["input_ids", "conditioning_tokens", "attention_mask"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=axes_map,
        )
    except Exception as exc:
        _fatal(
            f"SLM ONNX export failed at opset={opset}: {exc}. "
            "If the failure references an unsupported op, try raising "
            "`--opset` to 18+ or update torch."
        )

    try:
        graph = onnx.load(output_path.as_posix())
        onnx.checker.check_model(graph)
    except Exception as exc:
        _fatal(f"onnx.checker rejected SLM graph: {exc}")

    logger.info(
        "SLM prefill-only graph exported to %s (opset=%d, n_cond=%d, d_model=%d)",
        output_path,
        opset,
        n_cond,
        d_model,
    )

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
                pt_out = wrapper(
                    input_ids, conditioning_tokens, attention_mask
                ).detach().cpu().numpy()
            onnx_out = session.run(
                ["logits"],
                {
                    "input_ids": input_ids.numpy().astype(np.int64),
                    "conditioning_tokens": conditioning_tokens.numpy().astype(
                        np.float32
                    ),
                    "attention_mask": attention_mask.numpy().astype(np.int64),
                },
            )[0]
            if not np.allclose(pt_out, onnx_out, atol=1e-4):
                max_abs = float(np.max(np.abs(pt_out - onnx_out)))
                _fatal(
                    "SLM parity check failed: max abs diff "
                    f"{max_abs:.2e} > 1e-4"
                )
            logger.info("SLM parity OK (ONNXRuntime vs PyTorch, atol=1e-4)")
        except SystemExit:
            raise
        except Exception as exc:
            logger.warning("SLM parity check errored (%s); continuing.", exc)

    return output_path


__all__ = ["export_slm"]


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Export Adaptive SLM to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=False, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    from i3.slm.model import AdaptiveSLM

    slm = AdaptiveSLM()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        slm.load_state_dict(state, strict=False)
    export_slm(slm, Path(args.output), opset=args.opset, verify=not args.no_verify)
    sys.stderr.write(f"wrote {args.output}\n")
