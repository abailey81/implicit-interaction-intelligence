"""Training loop for AdaptiveTransformerV2 (400M-target from-scratch SLM).

This is the v2 trainer: it's *additive* next to :mod:`i3.slm.train` and
targets the RTX 4050 Laptop GPU (6.4 GB VRAM, ~5 GB usable). It keeps
the from-scratch, no-HuggingFace rule of the Huawei HMI portfolio piece
and introduces only a single new CUDA-kernel dependency
(``bitsandbytes``) for 8-bit AdamW optimiser state.

Key v2 capabilities
-------------------
* **bf16 mixed precision** via ``torch.amp.autocast("cuda",
  dtype=torch.bfloat16)``. Ada Lovelace (SM 8.9) supports bf16
  natively; we do *not* fall back to fp16 because bf16 has a wider
  exponent range which matters for from-scratch training where the
  loss can briefly explode during warmup.

* **8-bit AdamW** (``bnb.optim.AdamW8bit``) to halve optimiser-state
  VRAM. Falls back to ``torch.optim.AdamW`` with a clear note if
  bitsandbytes fails to import on Windows — the memory ceiling then
  drops from ~400M to ~300M.

* **Gradient checkpointing** wrapped around every transformer block.
  We call ``torch.utils.checkpoint.checkpoint(...)`` over the v2
  block iteration loop, honoured by a ``use_grad_checkpointing``
  config flag.

* **Gradient accumulation** via ``--grad-accum-steps`` so the
  effective batch can be raised without increasing per-step VRAM.

* **Cosine LR schedule** with 2 % warmup, peak LR 3e-4.

* **Weight decay 0.1** applied to all *non*-bias, *non*-norm params,
  matching the GPT-family recipe.

* **Gradient clipping** at norm 1.0.

* **Loss** is next-token cross-entropy computed *only on the response
  portion* — the history tokens are masked out via ``ignore_index``.

* **Aux losses**: ``moe_load_balance`` + ``act_ponder`` from the v2
  model, each weighted at 0.01.

* **Multi-task heads** (typing biometrics / affect / reading level)
  ride on the TCN user-state embedding. They're off by default (MTL
  weight 0) because the dialogue corpus has no user-state labels;
  when a batch ships user-state metadata the heads compose into the
  total loss with weight ``--mtl-weight``.

* **Checkpointing** matches the v1 layout (``model_state_dict``,
  ``optimizer_state_dict``, …) so downstream loaders don't branch on
  version, plus extra keys for the v2-specific LR schedule and
  eval loss.

* **Early-stop on repeated OOM** — two consecutive OOMs abort the run
  so the GPU doesn't silently hang.

Usage
-----
Minimal smoke run::

    poetry run python -m i3.slm.train_v2 \\
        --corpus data/processed/dialogue/triples.json \\
        --tokenizer checkpoints/slm/tokenizer_bpe.json \\
        --out checkpoints/slm_v2 \\
        --max-steps 20 --batch-size 2 --max-seq-len 256

Full run (~3 epochs on 974k pairs)::

    poetry run python -m i3.slm.train_v2 \\
        --corpus data/processed/dialogue/triples.json \\
        --tokenizer checkpoints/slm/tokenizer_bpe.json \\
        --out checkpoints/slm_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch.utils.data import DataLoader, Dataset

from i3.slm.adaptive_transformer_v2 import (
    AdaptiveTransformerV2,
    AdaptiveTransformerV2Config,
)
from i3.slm.bpe_tokenizer import BPETokenizer

logger = logging.getLogger("train_v2")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SLMDialogueDataset(Dataset):
    """Dialogue pairs from ``triples.json`` tokenised as ``history [SEP] response [EOS]``.

    The dataset streams the raw JSON once at construction time, tokenises
    lazily per ``__getitem__`` (BPE encoding is cheap enough that
    pre-tokenising the whole 974 k-pair corpus into a single tensor would
    cost ~8 GB of disk + RAM — streaming keeps the hot path small).

    Each item returns four fixed-length tensors:

    * ``input_ids``      ``[max_seq_len]`` with PAD filling the tail
    * ``target_ids``     ``[max_seq_len]`` — the *shift* is applied in
      the trainer, not here, so both tensors are aligned.
    * ``attention_mask`` ``[max_seq_len]`` 1 on real tokens, 0 on PAD
    * ``response_mask``  ``[max_seq_len]`` 1 on response tokens only;
      the loss is zeroed on every other position via ``ignore_index``.
    """

    def __init__(
        self,
        pairs: list[dict[str, str]],
        tokenizer: BPETokenizer,
        max_seq_len: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.pad_id = tokenizer.PAD_ID
        self.sep_id = tokenizer.SEP_ID
        self.eos_id = tokenizer.EOS_ID
        self.bos_id = tokenizer.BOS_ID

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        history = pair.get("history", "") or ""
        response = pair.get("response", "") or ""

        # Budget: leave room for [BOS] history [SEP] response [EOS].
        # We cap history at ⅔ of the budget and response at ⅓ so the
        # response is almost always fully present (the loss is
        # computed on the response, so truncating there would waste
        # the training signal).
        history_budget = max(8, (self.max_seq_len * 2) // 3 - 2)
        response_budget = max(8, self.max_seq_len - history_budget - 3)

        h_ids = self.tokenizer.encode(history, max_length=history_budget)
        r_ids = self.tokenizer.encode(response, max_length=response_budget)

        # Assemble: [BOS] h... [SEP] r... [EOS]
        seq: list[int] = [self.bos_id] + list(h_ids) + [self.sep_id] + list(r_ids) + [self.eos_id]
        # Response portion indices: everything strictly after the [SEP]
        # and up to and including [EOS].  The trainer uses this to mask
        # the loss on the history tokens.
        resp_start = 1 + len(h_ids) + 1  # after [BOS] + history + [SEP]
        resp_end = resp_start + len(r_ids) + 1  # include [EOS]

        # Truncate to max_seq_len (belt + braces — budgets above almost
        # always keep us inside).
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
            resp_end = min(resp_end, self.max_seq_len)

        input_ids = torch.full((self.max_seq_len,), self.pad_id, dtype=torch.long)
        input_ids[: len(seq)] = torch.tensor(seq, dtype=torch.long)

        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[: len(seq)] = 1

        response_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        if resp_end > resp_start:
            response_mask[resp_start:resp_end] = 1

        # Targets are the same as inputs — the trainer handles the
        # causal shift (logits[:, :-1] predicts target_ids[:, 1:]).
        target_ids = input_ids.clone()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }


def load_triples(path: Path) -> list[dict[str, str]]:
    """Load ``triples.json`` with explicit UTF-8 (Windows defaults to cp1251)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of pairs")
    return data


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


class CosineWarmupLR:
    """Cosine decay after a linear warmup (identical semantics to v1)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(int(warmup_steps), 1)
        self.max_steps = max(int(max_steps), self.warmup_steps + 1)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self._step = 0
        self._last_lr = [base_lr]

    def step(self) -> None:
        self._step += 1
        lr = self._compute_lr(self._step)
        self._last_lr = [lr]
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _compute_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps
        progress = min(
            (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._step = int(sd.get("step", 0))
        self.warmup_steps = int(sd.get("warmup_steps", self.warmup_steps))
        self.max_steps = int(sd.get("max_steps", self.max_steps))
        self.base_lr = float(sd.get("base_lr", self.base_lr))
        self.min_lr = float(sd.get("min_lr", self.min_lr))


# ---------------------------------------------------------------------------
# Gradient-checkpointed v2 wrapper
# ---------------------------------------------------------------------------


class _GradCkptTransformerV2(AdaptiveTransformerV2):
    """v2 model with ``torch.utils.checkpoint`` applied to every block.

    We subclass rather than monkey-patch because the v2 forward is tightly
    coupled to the ACT halting state — the cleanest seam is to swap the
    ``layer(...)`` call inside the per-layer loop for the checkpointed
    version, which we do by overriding ``forward`` and copy-pasting the
    minimal outer machinery. Any changes upstream in
    :class:`AdaptiveTransformerV2.forward` would need to be mirrored
    here; the alternative (inserting hooks via ``nn.Module.register``)
    would be more fragile.
    """

    def __init__(
        self,
        config: AdaptiveTransformerV2Config,
        use_grad_checkpointing: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.use_grad_checkpointing = bool(use_grad_checkpointing)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        adaptation_vector: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        # Fall back to the plain implementation when checkpointing is
        # off (or in eval mode where it would waste compute).
        if not self.use_grad_checkpointing or not self.training or use_cache:
            return super().forward(
                input_ids,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                use_cache=use_cache,
            )

        # --- mirror v2 forward prologue exactly -------------------
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq_len], got {tuple(input_ids.shape)}"
            )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len == 0:
            empty_logits = torch.zeros(batch_size, 0, self.vocab_size, device=device)
            self.aux_losses = {
                "moe_load_balance": torch.zeros((), device=device),
                "act_ponder": torch.zeros((), device=device),
            }
            return empty_logits, {}

        if adaptation_vector is None:
            adaptation_vector = torch.zeros(batch_size, self.adaptation_dim, device=device)
            if self.adaptation_dim > 0:
                adaptation_vector[:, 0] = 0.5
            if self.adaptation_dim > 5:
                adaptation_vector[:, 5] = 0.5
        if user_state is None:
            user_state = torch.zeros(batch_size, self.conditioning_dim, device=device)

        x: torch.Tensor = self.embedding(input_ids)
        cond_tokens: torch.Tensor = self.conditioning_projector(adaptation_vector, user_state)

        from i3.slm.attention import create_causal_mask

        causal_mask: torch.Tensor = create_causal_mask(seq_len, device=device)

        p_cum = torch.zeros(batch_size, seq_len, device=device)
        halted = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        h_freeze = torch.zeros_like(x)
        ponder_total = torch.zeros((), device=device)

        layer_info: dict[str, dict[str, torch.Tensor]] = {}
        self.act.reset()

        # A closure over each layer so ``checkpoint`` can re-enter it.
        for i, layer in enumerate(self.layers):

            def _run_layer(
                x_in: torch.Tensor,
                cond_in: torch.Tensor,
                adapt_in: torch.Tensor,
                mask_in: torch.Tensor,
                _layer: nn.Module = layer,
            ) -> torch.Tensor:
                out, _info = _layer(
                    x_in,
                    conditioning_tokens=cond_in,
                    adaptation=adapt_in,
                    causal_mask=mask_in,
                    use_cache=False,
                )
                # NOTE: ``torch.utils.checkpoint`` requires a tensor
                # (or tuple of tensors) return value to preserve
                # autograd hooks.  We drop ``attn_info`` here because
                # those tensors are ``.detach()``ed inside the block
                # anyway, so they wouldn't contribute to the
                # gradient — if we ever need them at train time we
                # can thread them through via a module-level buffer.
                return out

            x_new = grad_ckpt(
                _run_layer,
                x,
                cond_tokens,
                adaptation_vector,
                causal_mask,
                use_reentrant=False,
            )

            p_cum, halt_now, _rem = self.act(
                h=x_new, p_cum=p_cum, adaptation=adaptation_vector
            )
            ponder_total = ponder_total + self.act.ponder_loss

            halt_now_mask = halt_now.unsqueeze(-1)
            halted_mask = halted.unsqueeze(-1)
            h_freeze = torch.where(halt_now_mask, x_new, h_freeze)
            x = torch.where(halted_mask, h_freeze, x_new)
            halted = halted | halt_now

            layer_info[f"layer_{i}"] = {}  # attention maps dropped under ckpt

        x = self.final_ln(x)
        logits = self.output_projection(x)

        # Recompute aux losses from live gate weights (same recipe as v2).
        from i3.slm.moe_ffn import MoEFeedForward

        moe_loss = torch.zeros((), device=device)
        if self.layers:
            terms = []
            for layer in self.layers:
                gate_logits = layer.moe_ffn.gate(adaptation_vector)
                gate_w = torch.softmax(gate_logits, dim=-1)
                terms.append(MoEFeedForward.load_balance_loss(gate_w))
            moe_loss = torch.stack(terms).mean()
        ponder_loss = ponder_total / max(len(self.layers), 1)

        self.aux_losses = {"moe_load_balance": moe_loss, "act_ponder": ponder_loss}
        layer_info["act"] = {
            "p_cum_final": p_cum.detach(),
            "halted_mask": halted.detach(),
        }
        return logits, layer_info


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


@dataclass
class TrainingV2Config:
    """Runtime knobs for the v2 trainer.

    Only the fields that are *not* in :class:`AdaptiveTransformerV2Config`
    live here — model shape comes from that dataclass, training schedule
    comes from this one.
    """

    batch_size: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip_norm: float = 1.0
    n_epochs: int = 3
    save_every_steps: int = 2000
    eval_every_steps: int = 1000
    mtl_weight: float = 0.0
    moe_aux_weight: float = 0.01
    act_ponder_weight: float = 0.01
    seed: int = 17
    eval_pairs: int = 200
    log_every: int = 50
    num_workers: int = 2
    # Deterministic sub-sampling of the dialogue corpus. When
    # ``corpus_subset_size > 0`` the trainer samples this many pairs
    # from the full ``triples.json`` using ``random.Random(corpus_subset_seed).sample``.
    # The 200-pair held-out eval slice is carved *after* the subset so it
    # remains reproducible for a given (size, seed) pair.
    corpus_subset_size: int = 0
    corpus_subset_seed: int = 17


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to load YAML %s: %s", path, exc)
        return {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="i3.slm.train_v2",
        description="Train AdaptiveTransformerV2 on the dialogue corpus.",
    )
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/slm_v2"))
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    # Shape overrides — pulled from the YAML by default.
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--n-experts", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    # Training-schedule overrides.
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--grad-clip-norm", type=float, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--save-every-steps", type=int, default=None)
    p.add_argument("--eval-every-steps", type=int, default=None)
    p.add_argument("--mtl-weight", type=float, default=None)
    p.add_argument("--moe-aux-weight", type=float, default=None)
    p.add_argument("--act-ponder-weight", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument(
        "--corpus-subset-size",
        type=int,
        default=None,
        help=(
            "If >0, deterministically sample this many pairs from the full "
            "corpus via random.Random(seed).sample. Overrides YAML "
            "training_v2.corpus_subset_size."
        ),
    )
    p.add_argument(
        "--corpus-subset-seed",
        type=int,
        default=None,
        help="Seed for the corpus subset sampler (overrides YAML training_v2.corpus_subset_seed).",
    )
    p.add_argument(
        "--no-grad-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster but uses more VRAM).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device auto-detection (e.g. 'cuda', 'cpu').",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Mixed-precision dtype for autocast (default bfloat16).",
    )
    return p


def _override(default: Any, cli_value: Any) -> Any:
    return default if cli_value is None else cli_value


def resolve_configs(
    args: argparse.Namespace,
) -> tuple[AdaptiveTransformerV2Config, TrainingV2Config]:
    """Merge YAML defaults with CLI overrides.

    CLI wins when provided; otherwise we fall back to the
    ``slm_v2:`` / ``training_v2:`` sections of ``configs/default.yaml``;
    otherwise we use the dataclass defaults.
    """
    yaml_cfg = _load_yaml_config(args.config)
    slm_v2_yaml = yaml_cfg.get("slm_v2", {}) or {}
    tr_v2_yaml = yaml_cfg.get("training_v2", {}) or {}

    model_cfg = AdaptiveTransformerV2Config(
        vocab_size=int(slm_v2_yaml.get("vocab_size", 32000)),
        d_model=int(_override(slm_v2_yaml.get("d_model", 960), args.d_model)),
        n_layers=int(_override(slm_v2_yaml.get("n_layers", 16), args.n_layers)),
        n_heads=int(_override(slm_v2_yaml.get("n_heads", 12), args.n_heads)),
        d_ff=int(_override(slm_v2_yaml.get("d_ff", 3840), args.d_ff)),
        n_experts=int(_override(slm_v2_yaml.get("n_experts", 2), args.n_experts)),
        max_seq_len=int(_override(slm_v2_yaml.get("max_seq_len", 1024), args.max_seq_len)),
        dropout=float(slm_v2_yaml.get("dropout", 0.1)),
        adaptation_dim=int(slm_v2_yaml.get("adaptation_dim", 8)),
        conditioning_dim=int(slm_v2_yaml.get("conditioning_dim", 64)),
        n_cross_heads=int(slm_v2_yaml.get("n_cross_heads", 4)),
    )
    train_cfg = TrainingV2Config(
        batch_size=int(_override(tr_v2_yaml.get("batch_size", 4), args.batch_size)),
        grad_accum_steps=int(
            _override(tr_v2_yaml.get("grad_accum_steps", 8), args.grad_accum_steps)
        ),
        learning_rate=float(_override(tr_v2_yaml.get("learning_rate", 3.0e-4), args.lr)),
        weight_decay=float(_override(tr_v2_yaml.get("weight_decay", 0.1), args.weight_decay)),
        warmup_ratio=float(
            _override(tr_v2_yaml.get("warmup_ratio", 0.02), args.warmup_ratio)
        ),
        grad_clip_norm=float(
            _override(tr_v2_yaml.get("grad_clip_norm", 1.0), args.grad_clip_norm)
        ),
        n_epochs=int(_override(tr_v2_yaml.get("n_epochs", 3), args.n_epochs)),
        save_every_steps=int(
            _override(tr_v2_yaml.get("save_every_steps", 2000), args.save_every_steps)
        ),
        eval_every_steps=int(
            _override(tr_v2_yaml.get("eval_every_steps", 1000), args.eval_every_steps)
        ),
        mtl_weight=float(_override(tr_v2_yaml.get("mtl_weight", 0.0), args.mtl_weight)),
        moe_aux_weight=float(
            _override(tr_v2_yaml.get("moe_aux_weight", 0.01), args.moe_aux_weight)
        ),
        act_ponder_weight=float(
            _override(tr_v2_yaml.get("act_ponder_weight", 0.01), args.act_ponder_weight)
        ),
        seed=int(_override(tr_v2_yaml.get("seed", 17), args.seed)),
        num_workers=int(_override(2, args.num_workers)),
        log_every=int(_override(50, args.log_every)),
        corpus_subset_size=int(
            _override(tr_v2_yaml.get("corpus_subset_size", 0), args.corpus_subset_size)
        ),
        corpus_subset_seed=int(
            _override(tr_v2_yaml.get("corpus_subset_seed", 17), args.corpus_subset_seed)
        ),
    )
    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# Param groups + optimizer
# ---------------------------------------------------------------------------


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Standard 2-group split: weight-decay on all non-bias/non-norm params.

    The GPT-style recipe: decay matrices, do not decay biases or LayerNorm
    gamma/beta. We also skip embeddings from decay, following Chinchilla
    and the PaLM recipe — tying the output head to the embeddings means
    the same tensor would otherwise get pulled both by the LM loss and
    the decay term and the decay dominates at low-frequency tokens.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    no_decay_names = ("bias",)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n in name for n in no_decay_names):
            no_decay_params.append(p)
        elif "norm" in name.lower() or "ln" in name.lower():
            no_decay_params.append(p)
        elif "embedding" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, str]:
    """Try 8-bit AdamW from bitsandbytes; fall back to fp32 AdamW on failure.

    Returns the optimizer and a short tag ("adamw8bit" / "adamw_fp32") so
    the caller can surface it in the smoke-test report.
    """
    groups = make_param_groups(model, weight_decay)
    try:
        import bitsandbytes as bnb  # noqa: WPS433 - conditional import on purpose

        opt = bnb.optim.AdamW8bit(
            groups,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        return opt, f"adamw8bit(bnb={bnb.__version__})"
    except Exception as exc:
        logger.warning(
            "bitsandbytes unavailable (%s). Falling back to torch.optim.AdamW. "
            "The VRAM ceiling drops from 400M to ~300M parameters — consider "
            "shrinking d_model if you hit OOM.",
            exc,
        )
        opt = torch.optim.AdamW(
            groups,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        return opt, "adamw_fp32(torch.optim)"


# ---------------------------------------------------------------------------
# VRAM telemetry
# ---------------------------------------------------------------------------


def _vram_free_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / (1024**3)


def _vram_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def _vram_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SLMTrainerV2:
    """Full training loop for AdaptiveTransformerV2 with bf16 + 8-bit AdamW."""

    def __init__(
        self,
        model_cfg: AdaptiveTransformerV2Config,
        train_cfg: TrainingV2Config,
        tokenizer: BPETokenizer,
        device: torch.device,
        out_dir: Path,
        tokenizer_path: Path,
        corpus_path: Path,
        use_grad_checkpointing: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.tokenizer = tokenizer
        self.device = device
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = tokenizer_path
        self.corpus_path = corpus_path
        self.amp_dtype = amp_dtype

        # --- model ------------------------------------------------------
        self.model: _GradCkptTransformerV2 = _GradCkptTransformerV2(
            config=model_cfg,
            use_grad_checkpointing=use_grad_checkpointing,
        ).to(device)

        logger.info(
            "model: %s, params=%s, size(fp32)=%.1fMB",
            type(self.model).__name__,
            f"{self.model.num_parameters:,}",
            self.model.size_mb,
        )

        # --- optimizer --------------------------------------------------
        self.optimizer, self.opt_tag = build_optimizer(
            self.model, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
        )
        logger.info("optimizer: %s, wd=%s", self.opt_tag, train_cfg.weight_decay)

        # The scheduler is reset in :meth:`train` once we know
        # ``max_steps`` (which depends on dataset size and grad accum).
        self.lr_sched: CosineWarmupLR | None = None

        # --- optional MTL heads ----------------------------------------
        self.mtl_heads: dict[str, nn.Module] | None = None
        if train_cfg.mtl_weight > 0:
            from i3.slm.multi_task_heads import (
                AffectHead,
                ReadingLevelHead,
                TypingBiometricsHead,
            )

            self.mtl_heads = {
                "typing": TypingBiometricsHead(embedding_dim=model_cfg.conditioning_dim).to(
                    device
                ),
                "affect": AffectHead(embedding_dim=model_cfg.conditioning_dim).to(device),
                "reading": ReadingLevelHead(embedding_dim=model_cfg.conditioning_dim).to(device),
            }
            logger.info("MTL heads enabled (weight=%s)", train_cfg.mtl_weight)

        # --- bookkeeping -----------------------------------------------
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.oom_streak = 0
        self.max_oom_streak = 2

    # ------------------------------------------------------------------
    # forward helpers
    # ------------------------------------------------------------------

    def _batch_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (scalar_loss, metrics_dict)."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        target_ids = batch["target_ids"].to(self.device, non_blocking=True)
        response_mask = batch["response_mask"].to(self.device, non_blocking=True)

        # bf16 autocast (we skip GradScaler — bf16 doesn't underflow).
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            logits, _info = self.model(input_ids, use_cache=False)

            # Causal shift: logits[t] predicts target[t+1].
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous().clone()
            shift_resp_mask = response_mask[:, 1:].contiguous()

            # Mask non-response positions with ignore_index.  This zeros
            # the loss on history tokens (we only care about generating
            # the response) *and* on PAD (response_mask is 0 there).
            shift_targets[shift_resp_mask == 0] = -100

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
            )

            aux = getattr(self.model, "aux_losses", {}) or {}
            moe_aux = aux.get(
                "moe_load_balance", torch.zeros((), device=self.device)
            )
            act_aux = aux.get("act_ponder", torch.zeros((), device=self.device))

            total = (
                ce_loss
                + self.train_cfg.moe_aux_weight * moe_aux
                + self.train_cfg.act_ponder_weight * act_aux
            )

            # Optional MTL.  The dialogue corpus has no user-state
            # metadata, so we only branch here when a batch provides
            # ``user_state`` + the relevant labels.
            if self.mtl_heads and "user_state" in batch:
                us = batch["user_state"].to(self.device, non_blocking=True)
                mtl_total = torch.zeros((), device=self.device)
                if "typing_label" in batch:
                    logits_t = self.mtl_heads["typing"](us)
                    mtl_total = mtl_total + F.cross_entropy(
                        logits_t, batch["typing_label"].to(self.device)
                    )
                if "affect_label" in batch:
                    logits_a = self.mtl_heads["affect"](us)
                    mtl_total = mtl_total + F.cross_entropy(
                        logits_a, batch["affect_label"].to(self.device)
                    )
                if "reading_label" in batch:
                    pred_r = self.mtl_heads["reading"](us)
                    mtl_total = mtl_total + F.mse_loss(
                        pred_r, batch["reading_label"].to(self.device).float()
                    )
                total = total + self.train_cfg.mtl_weight * mtl_total

        metrics = {
            "ce_loss": float(ce_loss.detach().item()),
            "moe_aux": float(moe_aux.detach().item()) if moe_aux.requires_grad or True else 0.0,
            "act_aux": float(act_aux.detach().item()) if act_aux.requires_grad or True else 0.0,
            "total_loss": float(total.detach().item()),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses: list[float] = []
        seen_pairs = 0
        target_pairs = self.train_cfg.eval_pairs
        for batch in eval_loader:
            loss, _m = self._batch_loss(batch)
            losses.append(float(loss.item()))
            seen_pairs += batch["input_ids"].size(0)
            if seen_pairs >= target_pairs:
                break
        self.model.train()
        if not losses:
            return {"eval_loss": float("nan"), "perplexity": float("nan")}
        mean = sum(losses) / len(losses)
        ppl = math.exp(min(mean, 20.0))
        return {"eval_loss": mean, "perplexity": ppl}

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def _save(self, filename: str, eval_loss: float | None) -> Path:
        path = self.out_dir / filename
        ckpt: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                self.lr_sched.state_dict() if self.lr_sched is not None else {}
            ),
            "step": self.global_step,
            "config": {
                "model": asdict(self.model_cfg),
                "training": asdict(self.train_cfg),
                "optimizer": self.opt_tag,
                "amp_dtype": str(self.amp_dtype).split(".")[-1],
            },
            "tokenizer_path": str(self.tokenizer_path),
            "corpus_path": str(self.corpus_path),
            "eval_loss": eval_loss if eval_loss is not None else float("inf"),
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(ckpt, path)
        logger.info("checkpoint saved: %s (eval_loss=%s)", path, eval_loss)
        return path

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        self.model.train()

        # --- compute total optimiser steps ----------------------------
        steps_per_epoch = max(len(train_loader) // self.train_cfg.grad_accum_steps, 1)
        total_optim_steps = steps_per_epoch * self.train_cfg.n_epochs
        if max_steps is not None:
            total_optim_steps = min(total_optim_steps, int(max_steps))
        warmup_steps = max(int(total_optim_steps * self.train_cfg.warmup_ratio), 1)
        self.lr_sched = CosineWarmupLR(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            max_steps=total_optim_steps,
            base_lr=self.train_cfg.learning_rate,
        )
        logger.info(
            "training: steps_per_epoch=%d, total=%d, warmup=%d, effective_batch=%d",
            steps_per_epoch,
            total_optim_steps,
            warmup_steps,
            self.train_cfg.batch_size * self.train_cfg.grad_accum_steps,
        )

        # --- pre-loop telemetry ---------------------------------------
        if self.device.type == "cuda":
            free_gb = _vram_free_gb()
            logger.info("pre-train VRAM free: %.2f GB", free_gb or 0.0)
            torch.cuda.reset_peak_memory_stats()

        t_start = time.time()
        samples_seen = 0
        train_loss_log: list[tuple[int, float]] = []
        step_times: list[float] = []
        peak_vram_warned = False
        first_fwd_done = False

        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.train_cfg.n_epochs):
            for micro_step, batch in enumerate(train_loader):
                if self.global_step >= total_optim_steps:
                    break
                t_step = time.time()
                try:
                    loss, metrics = self._batch_loss(batch)
                    # Scale by accum so the gradient magnitude matches
                    # a single-batch step at the effective size.
                    (loss / self.train_cfg.grad_accum_steps).backward()
                    first_fwd_done = True
                    self.oom_streak = 0
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    self.oom_streak += 1
                    logger.error(
                        "OOM at step %d (streak=%d): %s",
                        self.global_step,
                        self.oom_streak,
                        exc,
                    )
                    if not first_fwd_done:
                        raise RuntimeError(
                            "OOM at step 0 — reduce batch_size or d_model"
                        ) from exc
                    if self.oom_streak >= self.max_oom_streak:
                        logger.error(
                            "early-stop: %d consecutive OOMs",
                            self.oom_streak,
                        )
                        raise
                    continue

                # Optimiser step on the last micro-batch of the group.
                is_accum_boundary = (
                    (micro_step + 1) % self.train_cfg.grad_accum_steps == 0
                )
                if is_accum_boundary:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.train_cfg.grad_clip_norm,
                    ).item()
                    self.optimizer.step()
                    self.lr_sched.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    step_times.append(time.time() - t_step)
                    samples_seen += (
                        self.train_cfg.batch_size * self.train_cfg.grad_accum_steps
                    )
                    train_loss_log.append((self.global_step, metrics["total_loss"]))

                    # Peak VRAM warning.
                    if (
                        self.device.type == "cuda"
                        and not peak_vram_warned
                        and self.global_step <= 50
                    ):
                        peak = _vram_peak_gb()
                        if peak > 5.5:
                            logger.warning(
                                "peak VRAM %.2f GB > 5.5 GB budget — "
                                "consider shrinking batch_size or d_model",
                                peak,
                            )
                            peak_vram_warned = True

                    # Logging.
                    if self.global_step % self.train_cfg.log_every == 0 or (
                        self.global_step <= 20
                    ):
                        recent = (
                            sum(t for t in step_times[-self.train_cfg.log_every:])
                            / min(len(step_times), self.train_cfg.log_every)
                        )
                        sps = (self.train_cfg.batch_size * self.train_cfg.grad_accum_steps) / max(
                            recent, 1e-6
                        )
                        vram_alloc = _vram_allocated_gb()
                        vram_peak = _vram_peak_gb()
                        logger.info(
                            "step=%d ep=%d  loss=%.4f  ce=%.4f  moe=%.4f  act=%.4f  "
                            "lr=%.2e  |g|=%.3f  sps=%.1f  vram=%.2fGB(peak=%.2fGB)",
                            self.global_step,
                            epoch,
                            metrics["total_loss"],
                            metrics["ce_loss"],
                            metrics["moe_aux"],
                            metrics["act_aux"],
                            self.lr_sched.get_last_lr()[0],
                            grad_norm,
                            sps,
                            vram_alloc,
                            vram_peak,
                        )

                    # Eval.
                    if self.global_step % self.train_cfg.eval_every_steps == 0:
                        eval_metrics = self.evaluate(eval_loader)
                        logger.info(
                            "  eval: loss=%.4f  ppl=%.2f  (best=%.4f)",
                            eval_metrics["eval_loss"],
                            eval_metrics["perplexity"],
                            self.best_eval_loss,
                        )
                        if eval_metrics["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self._save("best_model.pt", self.best_eval_loss)

                    # Periodic checkpoint.
                    if self.global_step % self.train_cfg.save_every_steps == 0:
                        self._save(
                            f"step_{self.global_step}.pt",
                            eval_loss=None,
                        )

            if self.global_step >= total_optim_steps:
                break

        total_time = time.time() - t_start
        final_path = self._save("final_model.pt", eval_loss=None)
        logger.info(
            "training complete: %d steps in %.1f min  best_eval=%.4f  final=%s",
            self.global_step,
            total_time / 60.0,
            self.best_eval_loss,
            final_path,
        )
        return {
            "final_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "total_time_s": total_time,
            "train_losses": train_loss_log,
        }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)

    model_cfg, train_cfg = resolve_configs(args)

    # Seed.
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    device = _pick_device(args.device)
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]

    logger.info("device=%s  amp_dtype=%s  cuda=%s", device, amp_dtype, torch.cuda.is_available())
    if device.type == "cuda":
        logger.info(
            "gpu=%s  capability=%s  vram_free=%.2fGB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
            _vram_free_gb() or 0.0,
        )

    # --- tokenizer -----------------------------------------------------
    tokenizer = BPETokenizer.load(args.tokenizer)
    logger.info("tokenizer: %d tokens, %d merges", len(tokenizer), len(tokenizer.merges))
    # Reconcile vocab_size with tokenizer — the model's vocab must >= tokenizer.
    if model_cfg.vocab_size < len(tokenizer):
        raise ValueError(
            f"model vocab_size={model_cfg.vocab_size} < tokenizer size={len(tokenizer)}"
        )

    # --- corpus --------------------------------------------------------
    pairs = load_triples(args.corpus)
    logger.info("corpus: %d pairs from %s", len(pairs), args.corpus)

    # Deterministic sub-sampling. The YAML ``training_v2.corpus_subset_size``
    # (or the ``--corpus-subset-size`` CLI flag) selects a fixed-size slice
    # of the full triples corpus *before* the eval holdout is carved.
    # ``random.Random(seed).sample`` gives a deterministic draw for a given
    # (size, seed) pair so the same 300 k pairs come out on every run.
    subset_size = int(train_cfg.corpus_subset_size)
    if subset_size > 0 and subset_size < len(pairs):
        subset_rng = random.Random(train_cfg.corpus_subset_seed)
        pairs = subset_rng.sample(pairs, k=subset_size)
        logger.info(
            "corpus subset: %d pairs (seed=%d)",
            len(pairs),
            train_cfg.corpus_subset_seed,
        )
    elif subset_size > 0:
        logger.info(
            "corpus subset: requested %d but corpus has only %d pairs — using full corpus",
            subset_size,
            len(pairs),
        )
    else:
        logger.info("corpus subset: disabled (corpus_subset_size=0) — using full corpus")

    # Shuffle the (possibly subsampled) pool and hold out a small eval
    # slice from *after* the subset so both the subset and the eval set
    # are reproducible for a given (corpus_subset_size, corpus_subset_seed, seed).
    rng = random.Random(train_cfg.seed)
    rng.shuffle(pairs)
    eval_n = min(2000, max(200, len(pairs) // 100))
    eval_pairs = pairs[:eval_n]
    train_pairs = pairs[eval_n:]
    logger.info(
        "split: %d train pairs, %d eval pairs (eval held out after subset)",
        len(train_pairs),
        len(eval_pairs),
    )

    max_seq_len = args.max_seq_len or model_cfg.max_seq_len
    train_ds = SLMDialogueDataset(train_pairs, tokenizer, max_seq_len)
    eval_ds = SLMDialogueDataset(eval_pairs, tokenizer, max_seq_len)

    # Windows note: persistent_workers avoids repeatedly paying the
    # spawn cost of the CPython 3.12 worker bootstrap.
    dl_kwargs = dict(
        num_workers=train_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if train_cfg.num_workers > 0:
        dl_kwargs["persistent_workers"] = True  # type: ignore[assignment]

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        **dl_kwargs,  # type: ignore[arg-type]
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=min(train_cfg.num_workers, 1),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # --- trainer -------------------------------------------------------
    trainer = SLMTrainerV2(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        device=device,
        out_dir=args.out,
        tokenizer_path=args.tokenizer,
        corpus_path=args.corpus,
        use_grad_checkpointing=not args.no_grad_checkpointing,
        amp_dtype=amp_dtype,
    )
    trainer.train(train_loader, eval_loader, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
