"""
Full Adaptive Small Language Model assembly for the I3 system.

Built entirely from scratch in PyTorch -- no HuggingFace. Composes the token
embedding, positional encoding, conditioning projector, and stacked
AdaptiveTransformerBlocks into a single causal language model conditioned on
real-time user state and adaptation signals.

Module:
    AdaptiveSLM -- complete causal language model (~6-8 M parameters) that
                   generates text conditioned on an AdaptationVector (8-dim)
                   and UserStateEmbedding (64-dim).

Architecture overview::

    Token IDs [batch, seq_len]
        --> TransformerEmbedding  (token lookup + sinusoidal position)
        --> [batch, seq_len, d_model]

    AdaptationVector [batch, 8]  }
                                 }--> ConditioningProjector --> [batch, n_cond, d_model]
    UserStateEmbedding [batch, 64] }

    For each of n_layers AdaptiveTransformerBlocks:
        x = SelfAttention(x) + CrossAttention(x, conditioning) + FFN(x)

    x --> LayerNorm --> Linear(d_model, vocab_size) --> logits [batch, seq_len, vocab_size]

    The output projection weights are tied with the token embedding weights
    (Press & Wolf, 2017, "Using the Output Embedding to Improve Language Models").

Default configuration targets ~6-8 M parameters:
    vocab_size=8000, d_model=256, n_heads=4, n_layers=4, d_ff=512,
    max_seq_len=256, conditioning_dim=64, adaptation_dim=8,
    n_cross_heads=2, n_conditioning_tokens=4
"""

from __future__ import annotations

import torch
import torch.nn as nn

from i3.slm.transformer import AdaptiveTransformerBlock


class AdaptiveSLM(nn.Module):
    """Adaptive Small Language Model with cross-attention conditioning.

    A causal language model (~6-8 M parameters) that generates text
    conditioned on:

    - **AdaptationVector** (8-dim): cognitive load, verbosity, technicality,
      formality, reading level, emotional tone, urgency, accessibility.
    - **UserStateEmbedding** (64-dim): the user's current interaction state
      produced by the upstream TCN encoder.

    The conditioning mechanism uses cross-attention at every transformer
    layer, allowing the model to continuously modulate token probabilities
    based on the desired adaptation.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (default 8000).
    d_model : int
        Model / embedding dimension (default 256).
    n_heads : int
        Number of self-attention heads per layer (default 4).
    n_layers : int
        Number of stacked AdaptiveTransformerBlocks (default 4).
    d_ff : int
        Inner dimension of the position-wise FFN (default 512).
    max_seq_len : int
        Maximum sequence length for positional encoding (default 256).
    conditioning_dim : int
        Dimensionality of the UserStateEmbedding input (default 64).
    adaptation_dim : int
        Dimensionality of the AdaptationVector input (default 8).
    n_cross_heads : int
        Number of cross-attention heads per layer (default 2).
    n_conditioning_tokens : int
        Number of synthetic conditioning tokens produced by the
        ConditioningProjector (default 4).
    dropout : float
        Dropout probability throughout the model (default 0.1).
    tie_weights : bool
        If ``True``, tie the output projection weights to the token
        embedding weights (default ``True``).
    padding_idx : int
        Index of the ``[PAD]`` token (default 0).

    Attributes
    ----------
    embedding : TransformerEmbedding
        Token + sinusoidal positional embedding layer.
    conditioning_projector : ConditioningProjector
        Projects (AdaptationVector, UserStateEmbedding) into conditioning
        tokens of shape ``[batch, n_conditioning_tokens, d_model]``.
    layers : nn.ModuleList[AdaptiveTransformerBlock]
        Stack of Pre-LN transformer blocks with cross-attention.
    final_ln : nn.LayerNorm
        Final layer normalisation before the output projection.
    output_projection : nn.Linear
        Projects from ``d_model`` to ``vocab_size`` (optionally weight-tied).
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        conditioning_dim: int = 64,
        adaptation_dim: int = 8,
        n_cross_heads: int = 2,
        n_conditioning_tokens: int = 4,
        dropout: float = 0.1,
        tie_weights: bool = True,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        # Delayed imports to avoid circular dependencies at module-load time.
        from i3.slm.cross_attention import ConditioningProjector
        from i3.slm.embeddings import TransformerEmbedding

        self.d_model: int = d_model
        self.vocab_size: int = vocab_size

        # ----- Token + Positional Embedding ------------------------------------
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        # ----- Conditioning Projector ------------------------------------------
        self.conditioning_projector = ConditioningProjector(
            adaptation_dim=adaptation_dim,
            user_state_dim=conditioning_dim,
            d_model=d_model,
            n_tokens=n_conditioning_tokens,
            dropout=dropout,
        )

        # ----- Transformer Blocks (Pre-LN with cross-attention) ----------------
        self.layers = nn.ModuleList([
            AdaptiveTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                n_cross_heads=n_cross_heads,
            )
            for _ in range(n_layers)
        ])

        # ----- Output head ------------------------------------------------------
        self.final_ln = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # ----- Weight initialisation --------------------------------------------
        # SEC: init MUST happen BEFORE weight tying. If tying is done first
        # and then `self.apply(_init_weights)` runs, Xavier-init on the
        # output_projection will silently re-initialise the (shared) embedding
        # weight, destroying the GPT-2 style N(0, 0.02) embedding init.
        # Apply Xavier uniform to all Linear layers and normal init to all
        # Embedding layers, THEN tie weights so the embedding init wins.
        self.apply(self._init_weights)

        # ----- Weight tying (Press & Wolf, 2017) --------------------------------
        # The output projection shares its weight matrix with the token
        # embedding, reducing parameter count and improving generalisation.
        # SEC: After tying, output_projection.weight IS embedding.weight
        # (same Parameter object, identical data_ptr). state_dict round-trip
        # preserves this because PyTorch saves the underlying tensor once
        # and load_state_dict assigns by name (the Linear's weight will be
        # restored, then the tied alias points to the loaded tensor).
        self.tie_weights: bool = tie_weights
        if tie_weights:
            self.output_projection.weight = (
                self.embedding.token_embedding.embedding.weight
            )

        # ----- Parameter census -------------------------------------------------
        # SEC: When weights are tied, sum(p.numel()) double-counts the shared
        # tensor (PyTorch yields the same Parameter twice through different
        # module attribute paths). Deduplicate by id() to get the true count.
        seen: set[int] = set()
        unique_params = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                unique_params += p.numel()
        self._n_params: int = unique_params

    # ----- initialisation -------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialise weights for all sub-modules.

        Strategy:
            - ``nn.Linear``: Xavier uniform (maintains variance across depth
              for both forward and backward passes).
            - ``nn.Embedding``: Normal(0, 0.02) following GPT-2 convention.
            - ``nn.LayerNorm``: PyTorch defaults (weight=1, bias=0) are fine.

        Note: Some sub-modules (e.g. ``MultiHeadSelfAttention``) apply their
        own initialisation in ``__init__``.  This global pass provides a
        consistent baseline; the per-module init may override it.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----- forward --------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        adaptation_vector: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Run the full model: embed, condition, transform, project.

        Parameters
        ----------
        input_ids : torch.Tensor
            Integer token IDs of shape ``[batch, seq_len]``.
        adaptation_vector : torch.Tensor, optional
            Adaptation signal of shape ``[batch, 8]``.  If ``None``, a
            neutral default is used (cognitive_load=0.5,
            emotional_tone=0.5, all others 0).
        user_state : torch.Tensor, optional
            User state embedding of shape ``[batch, 64]`` from the
            upstream TCN encoder.  If ``None``, zeros are used.
        use_cache : bool, optional
            If ``True``, enable KV caching in self-attention for
            efficient token-by-token autoregressive inference
            (default ``False``).

        Returns
        -------
        logits : torch.Tensor
            ``[batch, seq_len, vocab_size]`` -- raw (unnormalised) scores
            over the vocabulary for each position.  Apply ``softmax`` or
            pass directly to ``F.cross_entropy`` for training.
        layer_info : dict[str, dict[str, torch.Tensor]]
            Per-layer attention weights keyed by ``"layer_0"``, ...,
            ``"layer_{n-1}"``.  Each entry contains ``"self_attn"`` and
            ``"cross_attn"`` weight tensors for interpretability /
            visualisation.
        """
        # SEC: Validate input shape — must be 2D [batch, seq_len].
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}"
            )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # SEC: Empty-sequence guard — return well-shaped empty logits rather
        # than crash inside the embedding lookup or attention.
        if seq_len == 0:
            empty_logits = torch.zeros(
                batch_size, 0, self.vocab_size, device=device
            )
            return empty_logits, {}

        # ----- Default conditioning if not provided -----------------------------
        # SEC: Use adaptation/conditioning dims captured from the actual
        # ConditioningProjector instance, not hard-coded literals — keeps
        # forward() consistent with the configured model.
        if adaptation_vector is None:
            adapt_dim = self.conditioning_projector.adaptation_dim
            adaptation_vector = torch.zeros(batch_size, adapt_dim, device=device)
            # Neutral midpoints for the two semantically-meaningful dims
            # (cognitive_load and emotional_tone). Other dims stay at 0.
            if adapt_dim > 0:
                adaptation_vector[:, 0] = 0.5   # cognitive_load
            if adapt_dim > 5:
                adaptation_vector[:, 5] = 0.5   # emotional_tone

        if user_state is None:
            user_state_dim = self.conditioning_projector.user_state_dim
            user_state = torch.zeros(batch_size, user_state_dim, device=device)

        # ----- Embed tokens (lookup + sinusoidal position) ----------------------
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]

        # ----- Project conditioning into d_model-dimensional tokens -------------
        cond_tokens = self.conditioning_projector(
            adaptation_vector, user_state
        )  # [batch, n_conditioning_tokens, d_model]

        # ----- Causal mask for autoregressive self-attention --------------------
        from i3.slm.attention import create_causal_mask
        causal_mask = create_causal_mask(
            seq_len, device=device
        )  # [1, 1, seq_len, seq_len]

        # ----- Pass through stacked transformer layers --------------------------
        layer_info: dict[str, dict[str, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            x, attn_info = layer(
                x,
                conditioning_tokens=cond_tokens,
                causal_mask=causal_mask,
                use_cache=use_cache,
            )
            layer_info[f"layer_{i}"] = attn_info

        # ----- Final layer norm + output projection -----------------------------
        x = self.final_ln(x)                        # [batch, seq_len, d_model]
        logits = self.output_projection(x)           # [batch, seq_len, vocab_size]

        return logits, layer_info

    # ----- cache management -----------------------------------------------------

    def clear_cache(self) -> None:
        """Clear KV caches in all transformer layers.

        Must be called between sequences during autoregressive inference.
        """
        for layer in self.layers:
            layer.clear_cache()

    # ----- parameter utilities --------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of parameters (including tied weights counted once)."""
        return self._n_params

    @property
    def size_mb(self) -> float:
        """Approximate model size in megabytes assuming FP32 storage.

        .. math::

            \\text{size\\_mb} = \\frac{\\text{num\\_parameters} \\times 4}{1024^2}
        """
        return self._n_params * 4 / (1024 * 1024)

    # ----- repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab={self.vocab_size}, d_model={self.d_model}, "
            f"layers={len(self.layers)}, "
            f"params={self._n_params:,} ({self.size_mb:.1f} MB)"
            f")"
        )
