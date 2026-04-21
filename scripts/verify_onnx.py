"""Verify ONNX exports against their PyTorch originals.

Runs ONNXRuntime inference on a random dummy input and compares the
output tensor element-wise against the PyTorch model using
``np.allclose(atol=1e-4)``.  Exits with a non-zero status on mismatch
so the script can be wired into CI.

Usage::

    python scripts/verify_onnx.py --encoder exports/encoder.onnx \\
                                  --slm     exports/slm.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("i3.scripts.verify_onnx")

_TOL = 1e-4


def _verify_tcn(onnx_path: Path) -> bool:
    """Verify a TCN encoder ONNX export.

    Args:
        onnx_path: Path to the encoder ``.onnx`` file.

    Returns:
        ``True`` on success, ``False`` on mismatch or import failure.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.error("onnxruntime / torch missing: %s", exc)
        return False

    from i3.encoder.tcn import TemporalConvNet

    model = TemporalConvNet().eval()
    dummy = torch.randn(1, 10, getattr(model, "input_dim", 32))
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    onnx_out = session.run(
        ["embedding"], {"input": dummy.numpy().astype(np.float32)}
    )[0]

    max_abs = float(np.max(np.abs(pt_out - onnx_out)))
    ok = np.allclose(pt_out, onnx_out, atol=_TOL)
    logger.info(
        "TCN parity: max_abs_diff=%.2e tol=%.0e => %s",
        max_abs,
        _TOL,
        "OK" if ok else "FAIL",
    )
    return bool(ok)


def _verify_slm(onnx_path: Path) -> bool:
    """Verify an Adaptive SLM ONNX export.

    Args:
        onnx_path: Path to the SLM ``.onnx`` file.

    Returns:
        ``True`` on success, ``False`` on mismatch or import failure.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.error("onnxruntime / torch missing: %s", exc)
        return False

    from i3.slm.model import AdaptiveSLM
    from i3.slm.onnx_export import _SLMExportWrapper

    slm = AdaptiveSLM().eval()
    wrapper = _SLMExportWrapper(slm).eval()
    vocab = slm.vocab_size
    d_model = slm.d_model
    n_cond = slm.conditioning_projector.n_tokens

    input_ids = torch.randint(0, max(vocab, 1), (1, 16), dtype=torch.long)
    cond = torch.randn(1, n_cond, d_model)
    mask = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        pt_out = wrapper(input_ids, cond, mask).cpu().numpy()

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )
    onnx_out = session.run(
        ["logits"],
        {
            "input_ids": input_ids.numpy().astype(np.int64),
            "conditioning_tokens": cond.numpy().astype(np.float32),
            "attention_mask": mask.numpy().astype(np.int64),
        },
    )[0]

    max_abs = float(np.max(np.abs(pt_out - onnx_out)))
    ok = np.allclose(pt_out, onnx_out, atol=_TOL)
    logger.info(
        "SLM parity: max_abs_diff=%.2e tol=%.0e => %s",
        max_abs,
        _TOL,
        "OK" if ok else "FAIL",
    )
    return bool(ok)


def main() -> int:
    """Parse CLI args and run the verifiers.

    Returns:
        0 on full success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Verify ONNX exports vs PyTorch.")
    parser.add_argument("--encoder", type=Path, default=None)
    parser.add_argument("--slm", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    results: list[bool] = []
    if args.encoder is not None:
        if not args.encoder.exists():
            logger.error("encoder ONNX not found: %s", args.encoder)
            results.append(False)
        else:
            results.append(_verify_tcn(args.encoder))
    if args.slm is not None:
        if not args.slm.exists():
            logger.error("slm ONNX not found: %s", args.slm)
            results.append(False)
        else:
            results.append(_verify_slm(args.slm))

    if not results:
        logger.error("nothing to verify; supply --encoder and/or --slm")
        return 1
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
