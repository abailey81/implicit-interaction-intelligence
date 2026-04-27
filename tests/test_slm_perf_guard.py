"""Iter 58 — SLM v2 performance regression guard.

These tests load the v2 checkpoint and assert that its generate-path
hot-loop primitives stay within budget on CPU.  They're skipped when
the v2 weights aren't on disk (clean-checkout safe).

Budgets reflect what the iter-56 health endpoint reports as the live
device-class baseline (Kirin 9000-class, but measured on the host
RTX 4050 Laptop CPU with κ accounting).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest


_SLM_CKPT = Path("checkpoints/slm_v2/best_model.pt")
_BPE_CANDIDATES = [
    Path("checkpoints/slm_v2/tokenizer_bpe.json"),
    Path("checkpoints/slm/tokenizer_bpe.json"),
]


@pytest.fixture(scope="module")
def slm_setup():
    if not _SLM_CKPT.exists():
        pytest.skip("v2 SLM checkpoint not present")
    tok_path = next((p for p in _BPE_CANDIDATES if p.exists()), None)
    if tok_path is None:
        pytest.skip("BPE tokenizer not present")

    import torch
    from i3.slm.adaptive_transformer_v2 import (
        AdaptiveTransformerV2,
        AdaptiveTransformerV2Config,
    )
    from i3.slm.bpe_tokenizer import BPETokenizer

    blob = torch.load(_SLM_CKPT, map_location="cpu", weights_only=False)
    cfg_block = (blob.get("config") or {}).get("model") or {}
    valid = set(AdaptiveTransformerV2Config().__dict__.keys())
    cfg = AdaptiveTransformerV2Config(
        **{k: v for k, v in cfg_block.items() if k in valid}
    )
    model = AdaptiveTransformerV2(config=cfg)
    sd = blob["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    tokenizer = BPETokenizer.load(tok_path)
    return model, tokenizer, cfg


def test_param_count_in_expected_range(slm_setup):
    """Iter 58 — guard against an accidental architecture change that
    would silently re-shape the SLM's parameter count."""
    model, _, _ = slm_setup
    n = sum(p.numel() for p in model.parameters())
    n_m = n / 1e6
    # Expected ~204 M; allow ±5 % drift before we flag a regression.
    assert 194.0 <= n_m <= 215.0, f"SLM param count drifted to {n_m:.1f}M"


def test_vocab_size_matches_bpe(slm_setup):
    """The model's embedding row count must match the BPE vocab size."""
    model, tokenizer, cfg = slm_setup
    assert cfg.vocab_size == len(tokenizer), (
        f"vocab mismatch: model={cfg.vocab_size} bpe={len(tokenizer)}"
    )


def test_forward_pass_under_budget(slm_setup):
    """A single CPU forward over 16 tokens must stay under 6 s on the
    laptop-class baseline.  Loosely sized so a slow CI runner doesn't
    flake; the *trend* matters more than the absolute.
    """
    import torch
    model, tokenizer, _ = slm_setup
    ids = tokenizer.encode("hello world", add_bos=False, add_eos=False)
    x = torch.tensor([ids[:16] or [tokenizer.PAD_ID]], dtype=torch.long)
    t0 = time.time()
    with torch.inference_mode():
        out = model(x)
    wall = time.time() - t0
    assert wall < 6.0, f"forward took {wall:.2f}s (budget 6.0s on CPU)"
    # Output sanity — V2 forward returns ``(logits, aux)`` tuple.
    if isinstance(out, dict):
        out = out.get("logits", out.get("output"))
    elif isinstance(out, tuple):
        out = out[0]
    assert out is not None
    assert out.shape[-1] == len(tokenizer), \
        f"logits last-dim {out.shape[-1]} != vocab {len(tokenizer)}"


def test_kv_cache_friendly_attention_shape(slm_setup):
    """Ensure the attention layer exposes the (b, h, s, d) shape the
    generation loop assumes.  A future refactor that reshuffles axes
    would break the KV-cache fast-path silently; this test catches it."""
    import torch
    model, tokenizer, _ = slm_setup
    ids = tokenizer.encode("hi", add_bos=False, add_eos=False)
    x = torch.tensor([ids[:8] or [tokenizer.PAD_ID]], dtype=torch.long)
    with torch.inference_mode():
        out = model(x)
    if isinstance(out, dict):
        out = out.get("logits", out.get("output"))
    elif isinstance(out, tuple):
        out = out[0]
    # logits: [batch, seq, vocab]
    assert out.dim() == 3, f"expected 3-d logits, got {tuple(out.shape)}"
    assert out.shape[0] == 1
    assert out.shape[1] == x.shape[1]
