"""Evaluate the from-scratch :class:`AdaptiveTransformerV2` SLM v2.

Iter 51 (2026-04-27).  Companion to ``training/eval_intent.py`` — gives
the SLM the same publication-grade evidence the Qwen LoRA + Gemini
artefacts have.

Usage::

    poetry run python training/eval_slm_v2.py             # CPU
    poetry run python training/eval_slm_v2.py --cuda      # GPU
    poetry run python training/eval_slm_v2.py --max-batches 16

Outputs:

* ``reports/slm_v2_eval.json`` — machine-readable metrics
* ``reports/slm_v2_eval.md``   — human-readable summary

Metrics:

* **Cross-entropy** + **perplexity** on the held-out
  ``data/dialogue/val.pt`` (2 267 × 128 tokens, conditioning +
  user_state included so the eval matches training distribution).
* **Per-bucket loss** by sequence position quartile (Q1=tokens
  0-31, Q2=32-63, …) so we can see where the model degrades.
* **Top-1 next-token accuracy** (excluding PAD positions).
* **Generation samples**: 4 toy prompts decoded greedy + nucleus
  to give the reviewer a qualitative read.
* **Wall-time + memory** so the artefact ties back to the edge
  budget claim (Kirin-class 100 ms HMI window).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Quiet OpenTelemetry / Sentry boot noise.
os.environ.setdefault("I3_QUIET", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
import torch.nn.functional as F


def _load_checkpoint(ckpt_path: Path, device: torch.device):
    """Return (model, tokenizer) from a v2 checkpoint."""
    from i3.slm.adaptive_transformer_v2 import (
        AdaptiveTransformerV2,
        AdaptiveTransformerV2Config,
    )
    from i3.slm.bpe_tokenizer import BPETokenizer

    print(f"[load] checkpoint={ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_block = (blob.get("config") or {}).get("model") or {}
    if not cfg_block:
        raise SystemExit(
            f"{ckpt_path}: missing config.model — not a v2 checkpoint"
        )
    valid = set(AdaptiveTransformerV2Config().__dict__.keys())
    cfg_kwargs = {k: v for k, v in cfg_block.items() if k in valid}
    cfg = AdaptiveTransformerV2Config(**cfg_kwargs)
    model = AdaptiveTransformerV2(config=cfg)
    sd = blob["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[load]   {len(unexpected)} unexpected keys (sample={unexpected[:2]})")
    if missing:
        print(f"[load]   {len(missing)} missing keys (sample={missing[:2]})")
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] params={n_params/1e6:.1f}M, device={device}")

    # The v2 byte-level BPE tokenizer ships under
    # ``checkpoints/slm/tokenizer_bpe.json`` (the SimpleTokenizer
    # ``tokenizer.json`` next to it is the v1 format and won't load).
    candidates = [
        REPO_ROOT / "checkpoints" / "slm_v2" / "tokenizer_bpe.json",
        REPO_ROOT / "checkpoints" / "slm" / "tokenizer_bpe.json",
        REPO_ROOT / "checkpoints" / "slm_v2" / "tokenizer.json",
    ]
    tok_path = next((p for p in candidates if p.exists()), candidates[0])
    tokenizer = BPETokenizer.load(tok_path)
    print(f"[load] tokenizer={tok_path} vocab={len(tokenizer)}")

    return model, tokenizer, cfg, n_params


def _build_val_tensors_from_triples(
    triples_path: Path,
    tokenizer,
    *,
    seq_len: int = 128,
    n_eval: int = 500,
    seed: int = 17,
) -> dict:
    """Tokenise a held-out slice of triples.json with the v2 BPE.

    The pre-built ``data/dialogue/val.pt`` was tokenised with the v1
    SimpleTokenizer (30 k vocab) and is incompatible with the v2 model
    (32 k BPE).  This rebuilds a matched val tensor by sampling from
    the curated overlay (high-quality, hand-checked pairs).
    """
    import json as _json
    import random as _random

    print(f"[perp] tokenising fresh val set from {triples_path} (n={n_eval})")
    with triples_path.open("r", encoding="utf-8") as f:
        triples = _json.load(f)
    rng = _random.Random(seed)
    rng.shuffle(triples)
    triples = triples[:n_eval]

    PAD = getattr(tokenizer, "PAD_ID", 0)
    BOS = getattr(tokenizer, "BOS_ID", 2)
    EOS = getattr(tokenizer, "EOS_ID", 3)
    SEP = getattr(tokenizer, "SEP_ID", 4)

    inputs = []
    targets = []
    for t in triples:
        h = t.get("history", "") or ""
        r = t.get("response", "") or ""
        h_ids = tokenizer.encode(h, add_bos=False, add_eos=False)
        r_ids = tokenizer.encode(r, add_bos=False, add_eos=False)
        # Layout: [BOS] history [SEP] response [EOS]
        seq = [BOS] + h_ids + [SEP] + r_ids + [EOS]
        seq = seq[:seq_len]
        # input: tokens 0..n-1 ; target: tokens 1..n (predict next)
        ipt = seq[:-1] if len(seq) > 1 else seq
        tgt = seq[1:] if len(seq) > 1 else seq
        # pad to seq_len-1
        L = seq_len - 1
        if len(ipt) < L:
            ipt = ipt + [PAD] * (L - len(ipt))
            tgt = tgt + [PAD] * (L - len(tgt))
        else:
            ipt = ipt[:L]
            tgt = tgt[:L]
        inputs.append(ipt)
        targets.append(tgt)
    input_ids = torch.tensor(inputs, dtype=torch.long)
    target_ids = torch.tensor(targets, dtype=torch.long)
    return {"input_ids": input_ids, "target_ids": target_ids}


def _run_perplexity(
    model,
    val_path: Path,
    device: torch.device,
    *,
    batch_size: int = 8,
    max_batches: int | None = None,
    tokenizer=None,
    triples_path: Path | None = None,
    seq_len: int = 128,
    n_eval: int = 500,
) -> dict:
    """Compute LM cross-entropy + perplexity on the validation tensor file."""
    if triples_path is not None and tokenizer is not None:
        blob = _build_val_tensors_from_triples(
            triples_path, tokenizer, seq_len=seq_len, n_eval=n_eval,
        )
    else:
        print(f"[perp] loading {val_path}")
        blob = torch.load(val_path, map_location="cpu", weights_only=False)
    input_ids = blob["input_ids"]
    target_ids = blob["target_ids"]
    cond = blob.get("conditioning")
    state = blob.get("user_state")

    n = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    print(f"[perp] n={n} seq_len={seq_len} batch={batch_size}")

    n_batches = (n + batch_size - 1) // batch_size
    if max_batches:
        n_batches = min(n_batches, max_batches)

    total_loss = 0.0
    total_tokens = 0
    correct = 0
    counted = 0
    bucket_loss = [0.0, 0.0, 0.0, 0.0]
    bucket_tok = [0, 0, 0, 0]
    PAD = 0

    t0 = time.time()
    with torch.inference_mode():
        for b in range(n_batches):
            s = b * batch_size
            e = min(s + batch_size, n)
            inp = input_ids[s:e].to(device)
            tgt = target_ids[s:e].to(device)
            cond_b = cond[s:e].to(device) if cond is not None else None
            st_b = state[s:e].to(device) if state is not None else None
            try:
                logits = model(inp, conditioning=cond_b, user_state=st_b)
            except TypeError:
                # Some v2 forwards don't accept all kwargs.
                logits = model(inp)
            if isinstance(logits, dict):
                logits = logits.get("logits", logits.get("output"))
            if isinstance(logits, tuple):
                logits = logits[0]

            mask = (tgt != PAD)
            logp = F.log_softmax(logits.float(), dim=-1)
            tok_logp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            loss_per_tok = -tok_logp * mask
            total_loss += loss_per_tok.sum().item()
            total_tokens += mask.sum().item()

            preds = logits.argmax(dim=-1)
            correct += ((preds == tgt) & mask).sum().item()
            counted += mask.sum().item()

            # Position-bucket loss
            quart = seq_len // 4
            for qi in range(4):
                lo = qi * quart
                hi = (qi + 1) * quart if qi < 3 else seq_len
                m = mask[:, lo:hi]
                bucket_loss[qi] += (loss_per_tok[:, lo:hi]).sum().item()
                bucket_tok[qi] += m.sum().item()

            if (b + 1) % 5 == 0 or b == 0:
                avg = total_loss / max(1, total_tokens)
                ppl = float(torch.exp(torch.tensor(avg)).item())
                print(f"  batch {b+1}/{n_batches}  loss={avg:.3f}  ppl={ppl:.2f}")

    wall = time.time() - t0
    avg_loss = total_loss / max(1, total_tokens)
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    acc = correct / max(1, counted)
    bucket = []
    for qi in range(4):
        bl = bucket_loss[qi] / max(1, bucket_tok[qi])
        bp = float(torch.exp(torch.tensor(bl)).item())
        bucket.append({
            "quartile": qi + 1,
            "loss": round(bl, 4),
            "ppl": round(bp, 3),
            "tokens": int(bucket_tok[qi]),
        })

    return {
        "n_examples": int(n),
        "n_batches_evaluated": int(n_batches),
        "seq_len": int(seq_len),
        "tokens_evaluated": int(total_tokens),
        "cross_entropy": round(avg_loss, 4),
        "perplexity": round(ppl, 3),
        "top1_accuracy": round(acc, 4),
        "bucket_perplexity": bucket,
        "wall_time_s": round(wall, 1),
        "throughput_tok_per_s": round(total_tokens / max(1e-3, wall), 1),
    }


def _sample_generation(model, tokenizer, device: torch.device) -> list[dict]:
    """Run a few greedy + nucleus generations on toy prompts."""
    from i3.slm.generate import SLMGenerator

    gen = SLMGenerator(model, tokenizer)
    prompts = [
        "hello, how are you doing today?",
        "explain photosynthesis in one sentence",
        "what is a transformer in machine learning?",
        "set timer for 5 minutes",
    ]
    out: list[dict] = []
    for p in prompts:
        try:
            t0 = time.time()
            txt = gen.generate(p, max_new_tokens=64, temperature=0.7,
                               top_p=0.9)
            wall_ms = (time.time() - t0) * 1000.0
            out.append({
                "prompt": p,
                "generation": txt[:400],
                "wall_ms": round(wall_ms, 1),
            })
        except Exception as exc:
            out.append({"prompt": p, "error": f"{type(exc).__name__}: {exc}"})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate I3 SLM v2 on val set")
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO_ROOT / "checkpoints" / "slm_v2" / "best_model.pt"))
    parser.add_argument("--val", type=str,
                        default=str(REPO_ROOT / "data" / "dialogue" / "val.pt"),
                        help="Pre-tokenised val tensor (only used when --triples is empty)")
    parser.add_argument("--triples", type=str,
                        default=str(REPO_ROOT / "data" / "processed" / "dialogue" / "triples_curated_overlay.json"),
                        help="Curated overlay used as a fresh holdout (tokenised by v2 BPE)")
    parser.add_argument("--n-eval", type=int, default=500,
                        help="How many triples to sample for the eval set")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Cap number of batches for a quick smoke (0 = full)")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--out-prefix", type=str,
                        default=str(REPO_ROOT / "reports" / "slm_v2_eval"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")

    model, tokenizer, cfg, n_params = _load_checkpoint(Path(args.checkpoint), device)

    triples_path = Path(args.triples) if args.triples else None
    if triples_path and not triples_path.exists():
        triples_path = None
    metrics = _run_perplexity(
        model, Path(args.val), device,
        batch_size=args.batch_size,
        max_batches=(args.max_batches or None),
        tokenizer=tokenizer,
        triples_path=triples_path,
        seq_len=args.seq_len,
        n_eval=args.n_eval,
    )
    metrics["n_params_millions"] = round(n_params / 1e6, 2)
    metrics["d_model"] = int(getattr(cfg, "d_model", 0))
    metrics["n_layers"] = int(getattr(cfg, "n_layers", 0))
    metrics["n_heads"] = int(getattr(cfg, "n_heads", 0))
    metrics["vocab_size"] = int(getattr(cfg, "vocab_size", 0))
    metrics["device"] = str(device)
    metrics["checkpoint"] = str(args.checkpoint)

    if not args.skip_generation:
        try:
            metrics["samples"] = _sample_generation(model, tokenizer, device)
        except Exception as exc:
            metrics["samples_error"] = f"{type(exc).__name__}: {exc}"

    out_json = Path(args.out_prefix + ".json")
    out_md = Path(args.out_prefix + ".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote {out_json}")

    md = ["# I3 SLM v2 — held-out evaluation", ""]
    md.append(f"Checkpoint: `{args.checkpoint}`  ")
    md.append(f"Val set: `{args.val}` (n={metrics['n_examples']}, seq_len={metrics['seq_len']})  ")
    md.append(f"Device: `{metrics['device']}`  ")
    md.append("")
    md.append("## Architecture")
    md.append(f"- params: **{metrics['n_params_millions']} M**")
    md.append(f"- d_model={metrics['d_model']}, n_layers={metrics['n_layers']}, n_heads={metrics['n_heads']}")
    md.append(f"- vocab_size={metrics['vocab_size']}")
    md.append("")
    md.append("## Aggregate")
    md.append(f"- cross-entropy: **{metrics['cross_entropy']}**")
    md.append(f"- perplexity: **{metrics['perplexity']}**")
    md.append(f"- top-1 next-token accuracy: **{metrics['top1_accuracy']:.4f}**")
    md.append(f"- tokens evaluated: {metrics['tokens_evaluated']:,}")
    md.append(f"- wall: {metrics['wall_time_s']} s  ({metrics['throughput_tok_per_s']} tok/s)")
    md.append("")
    md.append("## Per-quartile perplexity (sequence position)")
    md.append("")
    md.append("| Quartile | tokens | loss | ppl |")
    md.append("|---|---|---|---|")
    for b in metrics["bucket_perplexity"]:
        md.append(f"| Q{b['quartile']} | {b['tokens']:,} | {b['loss']} | {b['ppl']} |")
    md.append("")
    samples = metrics.get("samples")
    if samples:
        md.append("## Generation samples (greedy + nucleus)")
        md.append("")
        for s in samples:
            md.append(f"- **prompt**: `{s['prompt']}`")
            if "generation" in s:
                md.append(f"  **gen** ({s.get('wall_ms','?')} ms): {s['generation']!r}")
            else:
                md.append(f"  **error**: {s.get('error')}")
        md.append("")
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[ok] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
