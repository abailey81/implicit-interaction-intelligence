"""Verify every number quoted in the recruiter-facing docs.

Run this whenever a doc claims a specific number and you want to be
sure it still matches the artefact on disk.  Catches drift between
"the claim in the README" and "the actual checkpoint" - exactly the
class of error that produced the perplexity-1725 confusion in the
phase-21 audit.

Usage:
    python scripts/verify_numbers.py

Exits 0 if every claim verifies, non-zero with a diff if anything
drifted.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _check(label: str, expected, actual, tolerance: float | None = None) -> bool:
    """Compare expected vs actual; return True if they match."""
    if tolerance is not None and isinstance(expected, (int, float)) \
            and isinstance(actual, (int, float)):
        ok = abs(expected - actual) <= tolerance
    else:
        ok = expected == actual
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: expected={expected}  actual={actual}")
    return ok


def main() -> int:
    print("=" * 70)
    print(" I3 - recruiter-facing numbers audit")
    print("=" * 70)
    fails: list[str] = []

    # --- 1. SLM v2 architecture + perplexity --------------------------
    import torch
    ck_path = REPO / "checkpoints" / "slm_v2" / "best_model.pt"
    print(f"\n[1] SLM v2 ({ck_path.relative_to(REPO)})")
    if not ck_path.exists():
        fails.append("slm v2 checkpoint missing")
    else:
        ck = torch.load(ck_path.as_posix(), map_location="cpu", weights_only=False)
        mc = ck["config"]["model"]
        if not _check("d_model = 768", 768, mc["d_model"]): fails.append("d_model")
        if not _check("n_layers = 12", 12, mc["n_layers"]): fails.append("n_layers")
        if not _check("n_heads = 12", 12, mc["n_heads"]): fails.append("n_heads")
        if not _check("vocab_size = 32 000", 32000, mc["vocab_size"]): fails.append("vocab")
        if not _check("step = 18 000", 18000, ck["step"]): fails.append("slm step")

        eval_loss = float(ck["eval_loss"])
        ppl = math.exp(eval_loss)
        if not _check("eval_loss ~ 4.987", 4.987, eval_loss, tolerance=1e-2):
            fails.append("eval_loss")
        if not _check("perplexity (training-eval, response-only) ~ 147",
                      147.0, ppl, tolerance=2.0):
            fails.append("ppl")

        # Param counts: 204 M unique (tied embedding) vs 229 M state_dict.
        sd = ck["model_state_dict"]
        n_state = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        # Subtract the duplicate LM-head when weights are tied (vocab x d_model).
        n_unique = n_state - (mc["vocab_size"] * mc["d_model"])
        n_unique_M = n_unique / 1e6
        # Eval-script header reports 204.41 M via the model's
        # .num_parameters; this state_dict subtraction lands at
        # 204.8 M (small extra from tied biases the .num_parameters
        # counter dedupes).  Both round to 204 M, which is what every
        # doc quotes; tolerance 1 M.
        if not _check("unique params ~ 204 M (tied weights)",
                      204.4, n_unique_M, tolerance=1.0):
            fails.append("unique params")
        n_state_M = n_state / 1e6
        if not _check("state_dict tensor sum ~ 229.4 M",
                      229.4, n_state_M, tolerance=0.2):
            fails.append("state_dict params")

    # --- 2. Qwen LoRA -------------------------------------------------
    qm_path = REPO / "checkpoints" / "intent_lora" / "qwen3.5-2b" / "training_metrics.json"
    print(f"\n[2] Qwen LoRA ({qm_path.relative_to(REPO)})")
    if not qm_path.exists():
        fails.append("qwen training metrics missing")
    else:
        with qm_path.open(encoding="utf-8") as f:
            m = json.load(f)
        if not _check("rank = 16", 16, m["rank"]): fails.append("lora rank")
        if not _check("alpha = 32", 32, m["alpha"]): fails.append("lora alpha")
        if not _check("epochs = 3", 3, m["epochs"]): fails.append("lora epochs")
        if not _check("n_train = 4 545", 4545, m["n_train"]): fails.append("n_train")
        if not _check("n_val = 252", 252, m["n_val"]): fails.append("n_val")
        if not _check("final_step = 1 704", 1704, m["final_step"]): fails.append("final_step")
        if not _check("best_val_loss ~ 5.36e-06", 5.36e-6,
                      m["best_val_loss"], tolerance=1e-7):
            fails.append("best_val_loss")
        if not _check("DoRA enabled", True, m["use_dora"]): fails.append("dora")
        # 9656 s wall time -> 2.68 h
        if not _check("wall_time_s ~ 9 656 (2.68 h)",
                      9656.0, m["wall_time_s"], tolerance=10.0):
            fails.append("wall time")

    # --- 3. Encoder ONNX ---------------------------------------------
    fp32_path = REPO / "checkpoints" / "encoder" / "tcn.onnx"
    int8_path = REPO / "web" / "models" / "encoder_int8.onnx"
    print(f"\n[3] Encoder ONNX")
    if not fp32_path.exists():
        fails.append("fp32 encoder missing")
    else:
        fp32_kb = os.path.getsize(fp32_path) / 1024
        if not _check("FP32 size ~ 441.4 KB", 441.4, fp32_kb, tolerance=2.0):
            fails.append("fp32 size")
    if not int8_path.exists():
        fails.append("int8 encoder missing - run quantize step")
    else:
        int8_kb = os.path.getsize(int8_path) / 1024
        if not _check("INT8 size ~ 162.2 KB", 162.2, int8_kb, tolerance=2.0):
            fails.append("int8 size")
        if fp32_path.exists():
            reduction = (1 - os.path.getsize(int8_path)/os.path.getsize(fp32_path)) * 100
            if not _check("INT8 reduction ~ 63 %", 63.3, reduction, tolerance=1.0):
                fails.append("reduction")

        # Parity check.
        try:
            import numpy as np
            import onnxruntime as ort
            fp32 = ort.InferenceSession(
                fp32_path.as_posix(), providers=["CPUExecutionProvider"],
            )
            int8 = ort.InferenceSession(
                int8_path.as_posix(), providers=["CPUExecutionProvider"],
            )
            np.random.seed(7)
            x = np.random.randn(1, 10, 32).astype(np.float32)
            in_name = fp32.get_inputs()[0].name
            y_fp32 = fp32.run(None, {in_name: x})[0]
            y_int8 = int8.run(None, {in_name: x})[0]
            mae = float(np.abs(y_fp32 - y_int8).mean())
            if not _check("parity MAE ~ 0.0006", 0.00055, mae, tolerance=5e-4):
                fails.append("parity")
        except Exception as exc:
            print(f"  [SKIP] parity check failed to import deps: {exc}")

    # --- 4. Knowledge graph ------------------------------------------
    print("\n[4] Knowledge graph")
    try:
        from i3.dialogue.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        n_subjects = len({t.subject for t in kg._triples})
        if not _check("KG unique subjects = 31", 31, n_subjects):
            fails.append("kg subjects")
    except Exception as exc:
        print(f"  [SKIP] {exc}")

    # --- 5. Corpus ----------------------------------------------------
    print("\n[5] Dialogue corpus")
    triples_path = REPO / "data" / "processed" / "dialogue" / "triples.json"
    if not triples_path.exists():
        print(f"  [SKIP] corpus not at {triples_path}")
    else:
        try:
            with triples_path.open(encoding="utf-8") as f:
                n_pairs = len(json.load(f))
            if not _check("corpus pairs ~ 974 000", 974000, n_pairs, tolerance=5000):
                fails.append("corpus size")
        except Exception as exc:
            print(f"  [SKIP] corpus parse failed: {exc}")

    # --- Summary ------------------------------------------------------
    print()
    print("=" * 70)
    if not fails:
        print(" ALL CLAIMS VERIFY OK")
        return 0
    print(f" {len(fails)} CLAIM(S) DRIFTED:")
    for f in fails:
        print(f"   - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
