"""Adaptive Transformer v2 -- MoE FFN + per-layer cross-attention + ACT halting.

A drop-in upgraded variant of the I3 decoder.  Composes the four new
from-scratch components:

* :class:`~i3.slm.cross_attention.MultiHeadCrossAttention` --
  re-used per layer so that adaptation conditioning is injected at
  **every** residual update (v1 already does this; v2 makes the
  per-layer guarantee explicit).
* :class:`~i3.slm.moe_ffn.MoEFeedForward` -- replaces the plain FFN,
  with a softmax gate over ``n_experts`` conditioned on the
  :class:`AdaptationVector`.
* :class:`~i3.slm.act_halting.ACTController` -- an adaptive-depth
  halting mechanism conditioned on the same adaptation vector.  Halted
  tokens are *frozen* via a halting mask; unhalted tokens continue
  through the remaining layers.
* Reuses the existing :class:`~i3.slm.cross_attention.ConditioningProjector`
  and :class:`~i3.slm.embeddings.TransformerEmbedding` so the input /
  output contract of :class:`AdaptiveSLM` is preserved.

Design constraints
------------------
* **Additive**.  ``transformer.py``, ``cross_attention.py`` and
  ``model.py`` are untouched; v2 lives alongside v1.
* **From-scratch**.  No HuggingFace ``transformers``, no pretrained
  weights, no external libraries beyond ``torch``.
* **Shape-compatible**.  :meth:`AdaptiveTransformerV2.forward` returns
  the same ``(logits, layer_info)`` pair as
  :meth:`AdaptiveSLM.forward`, so the rest of the pipeline is
  oblivious to which version is wired in.
* **Cheap halting**.  For v2 we do *not* implement per-token dynamic
  skipping at the CUDA-kernel level.  We track a halting mask and
  zero-weight halted-token deltas, which is a faithful implementation
  of the ACT *semantics* without rewriting attention.

Auxiliary losses
----------------
After every forward pass, ``self.aux_losses`` is populated with::

    {
        "moe_load_balance": scalar tensor,   # Shazeer 2017
        "act_ponder":       scalar tensor,   # Graves 2016
    }

Add both to the main cross-entropy with small coefficients, e.g.::

    loss = ce_loss
    loss = loss + 0.01 * aux_losses["moe_load_balance"]
    loss = loss + 0.01 * aux_losses["act_ponder"]

References
----------
* Shazeer et al. (2017) -- *Outrageously Large Neural Networks.*
* Fedus et al. (2022)   -- *Switch Transformers.*
* Graves (2016)         -- *Adaptive Computation Time for Recurrent
  Neural Networks.*
* Xiong et al. (2020)   -- *On Layer Normalization in the Transformer
  Architecture.*  (Pre-LN rationale, carried over from v1.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveTransformerV2Config:
    """Architecture hyperparameters for :class:`AdaptiveTransformerV2`.

    Deliberately a plain ``@dataclass`` rather than a pydantic / OmegaConf
    schema so the v2 architecture stays dependency-free.
    """

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = 30000
    n_experts: int = 4
    max_seq_len: int = 1024
    dropout: float = 0.1
    adaptation_dim: int = 8
    conditioning_dim: int = 64
    n_cross_heads: int = 4
    n_conditioning_tokens: int = 4
    padding_idx: int = 0
    tie_weights: bool = True
    # ACT controller knobs
    ponder_cost: float = 0.01
    halt_threshold: float = 0.99


# ---------------------------------------------------------------------------
# Per-layer block
# ---------------------------------------------------------------------------

class _AdaptiveTransformerBlockV2(nn.Module):
    """Pre-LN block: self-attn -> cross-attn -> MoE FFN.

    Identical structure to :class:`AdaptiveTransformerBlock` from v1
    except the FFN sub-layer is a
    :class:`~i3.slm.moe_ffn.MoEFeedForward`.  The block is shape-preserving:
    ``(batch, seq, d_model) -> (batch, seq, d_model)``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int,
        adaptation_dim: int,
        n_cross_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Delayed imports keep sibling-module import cycles safe.
        from i3.slm.attention import MultiHeadSelfAttention
        from i3.slm.cross_attention import MultiHeadCrossAttention
        from i3.slm.moe_ffn import MoEFeedForward

        self.self_attn: MultiHeadSelfAttention = MultiHeadSelfAttention(
            d_model, n_heads, dropout
        )
        self.cross_attn: MultiHeadCrossAttention = MultiHeadCrossAttention(
            d_model, n_cross_heads, dropout
        )
        self.moe_ffn: MoEFeedForward = MoEFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            n_experts=n_experts,
            adaptation_dim=adaptation_dim,
            dropout=dropout,
        )

        self.ln1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.ln2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.ln3: nn.LayerNorm = nn.LayerNorm(d_model)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        adaptation: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one v2 block.

        Returns
        -------
        x_out : torch.Tensor
            ``[batch, seq, d_model]``.
        attention_info : dict[str, torch.Tensor]
            Keys ``"self_attn"``, ``"cross_attn"`` and ``"moe_gate"``.
        """
        # ----- self-attention -----
        h = self.ln1(x)
        h, self_attn_weights = self.self_attn(
            h, mask=causal_mask, use_cache=use_cache
        )
        x = x + self.dropout(h)

        # ----- cross-attention on conditioning tokens -----
        h = self.ln2(x)
        h, cross_attn_weights = self.cross_attn(
            query=h,
            key=conditioning_tokens,
            value=conditioning_tokens,
        )
        x = x + self.dropout(h)

        # ----- MoE feed-forward (adaptation-gated) -----
        h = self.ln3(x)
        h = self.moe_ffn(h, adaptation=adaptation)
        x = x + self.dropout(h)

        attention_info: dict[str, torch.Tensor] = {
            "self_attn": self_attn_weights,
            "cross_attn": cross_attn_weights,
        }
        if self.moe_ffn.last_gate_weights is not None:
            attention_info["moe_gate"] = self.moe_ffn.last_gate_weights
        return x, attention_info

    def clear_cache(self) -> None:
        """Delegate to the self-attention KV cache."""
        self.self_attn.clear_cache()


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AdaptiveTransformerV2(nn.Module):
    """Upgraded Adaptive SLM with MoE FFN, per-layer conditioning and ACT halting.

    Forward signature matches :meth:`~i3.slm.model.AdaptiveSLM.forward` so
    the rest of the I3 pipeline can pass either v1 or v2 without code
    changes::

        logits, layer_info = model(input_ids, adaptation_vector, user_state)

    After ``forward`` returns, :attr:`aux_losses` is populated with the
    scalar auxiliary losses -- the caller is expected to read and add
    them to the main cross-entropy loss.

    Parameters
    ----------
    config : AdaptiveTransformerV2Config | None
        Configuration dataclass.  If ``None``, sensible defaults are used
        and individual ``**kwargs`` can override them.
    **overrides
        Field-by-field overrides applied on top of ``config`` (or the
        default config if ``config`` is ``None``).
    """

    def __init__(
        self,
        config: AdaptiveTransformerV2Config | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()

        if config is None:
            config = AdaptiveTransformerV2Config()
        if overrides:
            # Build a new config with the overrides applied so the caller
            # can mix-and-match without having to import the dataclass.
            cfg_dict = {**config.__dict__, **overrides}
            config = AdaptiveTransformerV2Config(**cfg_dict)
        self.config: AdaptiveTransformerV2Config = config

        # Delayed imports for the same reason as in v1.
        from i3.slm.act_halting import ACTController
        from i3.slm.cross_attention import ConditioningProjector
        from i3.slm.embeddings import TransformerEmbedding

        # ----- convenience attrs mirroring v1 -----
        self.d_model: int = config.d_model
        self.vocab_size: int = config.vocab_size
        self.n_heads: int = config.n_heads
        self.n_layers: int = config.n_layers
        self.d_ff: int = config.d_ff
        self.max_seq_len: int = config.max_seq_len
        self.dropout: float = config.dropout
        self.adaptation_dim: int = config.adaptation_dim
        self.conditioning_dim: int = config.conditioning_dim

        # ----- token + positional embedding -----
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            padding_idx=config.padding_idx,
        )

        # ----- conditioning projector (adaptation + user state -> cond tokens) -----
        self.conditioning_projector = ConditioningProjector(
            adaptation_dim=config.adaptation_dim,
            user_state_dim=config.conditioning_dim,
            d_model=config.d_model,
            n_tokens=config.n_conditioning_tokens,
            dropout=config.dropout,
        )

        # ----- transformer blocks (Pre-LN + MoE FFN) -----
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                _AdaptiveTransformerBlockV2(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    n_experts=config.n_experts,
                    adaptation_dim=config.adaptation_dim,
                    n_cross_heads=config.n_cross_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # ----- ACT halting controller (shared across layers) -----
        # One controller reused for every layer is intentional: the head
        # parameters are shared so the halting policy is a single learnable
        # function of (h, adaptation), not a per-depth policy.
        self.act = ACTController(
            d_model=config.d_model,
            adaptation_dim=config.adaptation_dim,
            ponder_cost=config.ponder_cost,
            halt_threshold=config.halt_threshold,
            max_layers=config.n_layers,
        )

        # ----- output head -----
        self.final_ln: nn.LayerNorm = nn.LayerNorm(config.d_model)
        self.output_projection: nn.Linear = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # ----- init + optional weight tying (mirror v1's approach) -----
        self.apply(self._init_weights)
        self.tie_weights_flag: bool = config.tie_weights
        if config.tie_weights:
            self.output_projection.weight = (
                self.embedding.token_embedding.embedding.weight
            )

        # ----- aux losses container, populated each forward -----
        self.aux_losses: dict[str, torch.Tensor] = {}

        # Unique-param count (dedupe tied weights).
        seen: set[int] = set()
        unique_params = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                unique_params += p.numel()
        self._n_params: int = unique_params

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Match v1's init policy: Xavier for Linear, N(0, 0.02) for embeddings."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        adaptation_vector: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Run the v2 decoder.

        Matches the v1 signature exactly so the caller cannot tell which
        version is in use.

        Returns
        -------
        logits : torch.Tensor
            ``[batch, seq, vocab_size]``.
        layer_info : dict
            Per-layer attention info (``self_attn``, ``cross_attn``,
            ``moe_gate``) plus an ``act`` sub-dict holding the final
            ``p_cum``, the halt mask and ponder stats.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq_len], got "
                f"{tuple(input_ids.shape)}"
            )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len == 0:
            empty_logits = torch.zeros(
                batch_size, 0, self.vocab_size, device=device
            )
            self.aux_losses = {
                "moe_load_balance": torch.zeros((), device=device),
                "act_ponder": torch.zeros((), device=device),
            }
            return empty_logits, {}

        # ----- default conditioning -----
        if adaptation_vector is None:
            adaptation_vector = torch.zeros(
                batch_size, self.adaptation_dim, device=device
            )
            if self.adaptation_dim > 0:
                adaptation_vector[:, 0] = 0.5  # cognitive_load
            if self.adaptation_dim > 5:
                adaptation_vector[:, 5] = 0.5  # emotional_tone
        if user_state is None:
            user_state = torch.zeros(
                batch_size, self.conditioning_dim, device=device
            )

        # ----- embed tokens -----
        x: torch.Tensor = self.embedding(input_ids)  # [B, S, D]

        # ----- conditioning tokens (used per layer via cross-attention) -----
        cond_tokens: torch.Tensor = self.conditioning_projector(
            adaptation_vector, user_state
        )

        # ----- causal self-attention mask -----
        from i3.slm.attention import create_causal_mask
        causal_mask: torch.Tensor = create_causal_mask(seq_len, device=device)

        # ----- ACT state -----
        # p_cum  : cumulative halting probability per token ([B, S]).
        # halted : boolean mask of tokens that have crossed halt_threshold
        #          on some earlier layer ([B, S]).
        # h_freeze : the frozen representation snapshot taken at the moment
        #            each token first halts ([B, S, D]).  For tokens that
        #            never halt it stays zero and the final layer output
        #            is used instead.
        #
        # The "cheap" ACT semantics used here (per the v2 spec): once a
        # token is halted, subsequent layers do not update it -- its
        # representation is frozen to whatever it was at halt time.  We
        # implement this by routing the layer output through a
        # ``torch.where`` on the halt mask: halted tokens keep
        # ``h_freeze``, still-active tokens take ``x_new``.
        p_cum: torch.Tensor = torch.zeros(batch_size, seq_len, device=device)
        halted: torch.Tensor = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        h_freeze: torch.Tensor = torch.zeros_like(x)
        ponder_total: torch.Tensor = torch.zeros((), device=device)

        # ----- per-layer loop -----
        layer_info: dict[str, dict[str, torch.Tensor]] = {}
        moe_gates: list[torch.Tensor] = []
        self.act.reset()

        for i, layer in enumerate(self.layers):
            x_new, attn_info = layer(
                x,
                conditioning_tokens=cond_tokens,
                adaptation=adaptation_vector,
                causal_mask=causal_mask,
                use_cache=use_cache,
            )

            # Ask the ACT head for halting probabilities at this depth.
            # Halted tokens are already clipped by ACTController (p_new
            # is forced to zero for tokens with p_cum >= halt_threshold),
            # so p_cum never grows past threshold for them.
            p_cum, halt_now, _remainder = self.act(
                h=x_new,
                p_cum=p_cum,
                adaptation=adaptation_vector,
            )
            ponder_total = ponder_total + self.act.ponder_loss

            # Freeze halted tokens: tokens that just halted snapshot x_new;
            # tokens halted on an earlier layer keep their previous freeze;
            # still-active tokens use the fresh layer output x_new.
            halt_now_mask: torch.Tensor = halt_now.unsqueeze(-1)  # [B, S, 1]
            halted_mask: torch.Tensor = halted.unsqueeze(-1)
            # Update h_freeze: take x_new for tokens halting *this* layer,
            # otherwise keep whatever was there.
            h_freeze = torch.where(halt_now_mask, x_new, h_freeze)
            # Compose the new residual stream:
            #   - previously halted tokens -> h_freeze (unchanged)
            #   - tokens halting now      -> x_new (equivalently h_freeze)
            #   - still active            -> x_new
            x = torch.where(halted_mask, h_freeze, x_new)

            # Mark tokens that halted at this layer.
            halted = halted | halt_now

            layer_info[f"layer_{i}"] = attn_info
            if "moe_gate" in attn_info:
                moe_gates.append(attn_info["moe_gate"])

        # Final-depth fallback: tokens that never crossed halt_threshold
        # simply use the last layer's x (already their current value).

        # ----- output head -----
        x = self.final_ln(x)
        logits: torch.Tensor = self.output_projection(x)

        # ----- auxiliary losses -----
        # MoE load balance: average across layers so a single scalar goes
        # into the training loop.  Note: we re-compute the loss from the
        # layer's live (grad-enabled) gate weights rather than the
        # ``.detach()``ed ``last_gate_weights`` cache so gradients flow
        # back to the gate's Linear layer.
        from i3.slm.moe_ffn import MoEFeedForward
        moe_loss = torch.zeros((), device=device)
        if self.layers:
            load_balance_terms = []
            for layer in self.layers:
                gate_logits = layer.moe_ffn.gate(adaptation_vector)
                gate_weights = torch.softmax(gate_logits, dim=-1)
                load_balance_terms.append(
                    MoEFeedForward.load_balance_loss(gate_weights)
                )
            moe_loss = torch.stack(load_balance_terms).mean()

        # ACT ponder: average across layers.
        ponder_loss: torch.Tensor = ponder_total / max(len(self.layers), 1)

        self.aux_losses = {
            "moe_load_balance": moe_loss,
            "act_ponder": ponder_loss,
        }

        layer_info["act"] = {
            "p_cum_final": p_cum.detach(),
            "halted_mask": halted.detach(),
        }

        return logits, layer_info

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        adaptation_vector: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward and return (logits, per_layer_self_attention_weights).

        Mirrors :meth:`AdaptiveSLM.forward_with_attention` so the v2 model
        is a drop-in replacement for the State-tab attention extractor at
        ``server/routes.py:_compute_attention_cpu``.

        Returns
        -------
        logits : torch.Tensor
            ``[batch, seq, vocab_size]``.
        per_layer : list[torch.Tensor]
            One tensor per layer of shape ``[batch, n_heads, seq, seq]``
            holding the self-attention weights.  Cross-attention and MoE
            gate weights are dropped — the State-tab heatmap only renders
            self-attention.
        """
        logits, layer_info = self.forward(
            input_ids,
            adaptation_vector=adaptation_vector,
            user_state=user_state,
            use_cache=False,
        )
        per_layer: list[torch.Tensor] = []
        for i in range(len(self.layers)):
            layer_d = layer_info.get(f"layer_{i}") or {}
            attn = layer_d.get("self_attn")
            if attn is not None:
                per_layer.append(attn)
        return logits, per_layer

    def clear_cache(self) -> None:
        """Clear KV caches in all layers (autoregressive inference)."""
        for layer in self.layers:
            layer.clear_cache()
        self.act.reset()

    @property
    def num_parameters(self) -> int:
        """Total parameters with tied weights counted once."""
        return self._n_params

    @property
    def size_mb(self) -> float:
        """Approximate model size in MB assuming fp32 storage."""
        return self._n_params * 4 / (1024 * 1024)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}("
            f"vocab={self.vocab_size}, d_model={self.d_model}, "
            f"layers={len(self.layers)}, n_experts={self.config.n_experts}, "
            f"params={self._n_params:,} ({self.size_mb:.1f} MB)"
            f")"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Tiny config for fast smoke.
    tiny_cfg = AdaptiveTransformerV2Config(
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        vocab_size=1000,
        n_experts=2,
        max_seq_len=64,
        dropout=0.1,
        adaptation_dim=8,
        conditioning_dim=64,
        n_cross_heads=2,
    )
    torch.manual_seed(0)
    model = AdaptiveTransformerV2(config=tiny_cfg)
    print(f"[v2] model: {model}")

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tiny_cfg.vocab_size, (batch_size, seq_len))
    adaptation = torch.rand(batch_size, tiny_cfg.adaptation_dim)
    user_state = torch.randn(batch_size, tiny_cfg.conditioning_dim)

    logits, layer_info = model(
        input_ids=input_ids,
        adaptation_vector=adaptation,
        user_state=user_state,
    )
    print(f"[v2] logits shape: {tuple(logits.shape)}")
    print(f"[v2] aux_losses:")
    for k, v in model.aux_losses.items():
        print(f"    {k}: {v.item():.6f}")
    print(
        f"[v2] halted tokens (of {batch_size * seq_len}): "
        f"{int(layer_info['act']['halted_mask'].sum().item())}"
    )

    # Target-size param count for the 300M config.
    big_cfg = AdaptiveTransformerV2Config(
        d_model=960,
        n_layers=16,
        n_heads=12,
        d_ff=3840,
        vocab_size=32000,
        n_experts=4,
        max_seq_len=1024,
        dropout=0.1,
        adaptation_dim=8,
        conditioning_dim=64,
        n_cross_heads=4,
    )
    big = AdaptiveTransformerV2(config=big_cfg)
    print(f"[v2] 300M-target config param count: {big.num_parameters:,} "
          f"({big.size_mb:.1f} MB fp32)")
