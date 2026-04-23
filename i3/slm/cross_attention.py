"""
Cross-Attention and Conditioning Projector for Implicit Interaction Intelligence (I3).

This module implements the novel conditioning mechanism that allows a Small Language
Model to modulate its text generation based on real-time user state signals. Rather
than fine-tuning or prompt engineering, we inject user context directly into the
transformer's attention mechanism via learned conditioning tokens.

The core insight: by projecting the AdaptationVector (8-dim behavioral targets) and
UserStateEmbedding (64-dim temporal context from the TCN encoder) into a small set
of conditioning tokens, we create a differentiable interface between the perception
pipeline and the language generation pipeline. The transformer can then learn to
attend to these tokens at every layer and position, enabling continuous, fine-grained
modulation of style, complexity, and tone throughout the generated response.

Architecture Overview:
    ConditioningProjector:
        [AdaptationVector(8) || UserStateEmbedding(64)] -> 72-dim
        -> MLP with GELU activation
        -> Reshape to [batch, n_tokens, d_model]
        -> LayerNorm
        -> conditioning tokens (used as K, V in cross-attention)

    MultiHeadCrossAttention:
        Q: text sequence embeddings [batch, seq_len, d_model]
        K, V: conditioning tokens   [batch, cond_len, d_model]
        -> Scaled dot-product attention
        -> Output: [batch, seq_len, d_model]

This is inserted into each transformer block as an additional sub-layer between
self-attention and the feed-forward network, following the standard pre-norm
residual pattern.

Author: Implicit Interaction Intelligence (I3) Project
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention for conditioning text generation on external signals.

    Unlike self-attention where Q, K, V all derive from the same sequence, cross-
    attention decouples the query source from the key-value source. Here:

        - Query comes from the main text sequence (decoder hidden states)
        - Key and Value come from conditioning tokens (projected user signals)

    This allows every token position in the generated sequence to independently
    attend to the user's cognitive state and desired adaptation parameters,
    enabling the model to continuously modulate its output style, complexity,
    and emotional tone at a sub-token granularity.

    The attention computation follows the standard scaled dot-product formulation:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    with separate learned projections for Q (from text), K and V (from conditioning),
    and a final output projection.

    Architecture:
        Query:  [batch, seq_len, d_model]  -- from text sequence hidden states
        Key:    [batch, cond_len, d_model]  -- from conditioning tokens
        Value:  [batch, cond_len, d_model]  -- from conditioning tokens

        Per-head reshaping:
            Q: [batch, n_heads, seq_len, d_k]
            K: [batch, n_heads, cond_len, d_k]
            V: [batch, n_heads, cond_len, d_k]

        Scores:       [batch, n_heads, seq_len, cond_len]
        Attn weights: [batch, n_heads, seq_len, cond_len]  (after softmax + dropout)
        Attn output:  [batch, n_heads, seq_len, d_k]       (after matmul with V)

        Final output: [batch, seq_len, d_model]  (after concat + W_o projection)

    Args:
        d_model: Dimensionality of the model (must be divisible by n_heads).
        n_heads: Number of attention heads for multi-head attention.
        dropout: Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.d_k: int = d_model // n_heads

        # Separate projections for query (text) vs key/value (conditioning)
        # No bias following modern transformer conventions (GPT-2, LLaMA)
        self.W_q: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_k: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_v: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.W_o: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout: nn.Dropout = nn.Dropout(dropout)

        # Initialize weights with Xavier uniform for stable training
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization to all projection matrices.

        Xavier initialization maintains variance across layers by scaling weights
        according to fan-in and fan-out, which is critical for stable gradient
        flow through the cross-attention layers during early training.
        """
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head cross-attention between text sequence and conditioning.

        The query derives from the text decoder's hidden states, while key and
        value derive from the conditioning tokens produced by ConditioningProjector.
        No causal mask is applied because the conditioning tokens are not sequential
        -- every text position should be able to attend to all conditioning tokens.

        Args:
            query: Text sequence hidden states.
                   Shape: [batch, seq_len, d_model]
            key:   Conditioning tokens (from ConditioningProjector).
                   Shape: [batch, cond_len, d_model]
            value: Conditioning tokens (same source as key).
                   Shape: [batch, cond_len, d_model]

        Returns:
            output: Cross-attended text representations, same shape as query.
                    Shape: [batch, seq_len, d_model]
            attn_weights: Attention weight distribution over conditioning tokens.
                          Shape: [batch, n_heads, seq_len, cond_len]
                          Useful for interpretability -- shows which conditioning
                          tokens each text position attends to most strongly.
        """
        batch_size: int = query.size(0)
        seq_len: int = query.size(1)
        cond_len: int = key.size(1)

        # Project and reshape into multi-head format
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k]
        #                            -> [batch, n_heads, seq_len, d_k]
        Q: torch.Tensor = (
            self.W_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # [batch, n_heads, seq_len, d_k]

        K: torch.Tensor = (
            self.W_k(key)
            .view(batch_size, cond_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # [batch, n_heads, cond_len, d_k]

        V: torch.Tensor = (
            self.W_v(value)
            .view(batch_size, cond_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )  # [batch, n_heads, cond_len, d_k]

        # Scaled dot-product attention
        # [batch, n_heads, seq_len, d_k] x [batch, n_heads, d_k, cond_len]
        # -> [batch, n_heads, seq_len, cond_len]
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )

        # Softmax over conditioning dimension (no causal mask needed)
        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # [batch, n_heads, seq_len, cond_len] x [batch, n_heads, cond_len, d_k]
        # -> [batch, n_heads, seq_len, d_k]
        attn_output: torch.Tensor = torch.matmul(attn_weights, V)

        # Concatenate heads and apply output projection
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k]
        #                                 -> [batch, seq_len, d_model]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection to mix information across heads
        output: torch.Tensor = self.W_o(attn_output)  # [batch, seq_len, d_model]

        return output, attn_weights


class ConditioningProjector(nn.Module):
    """
    Projects adaptation signals and user state into conditioning tokens for
    cross-attention in the transformer decoder.

    This is the key novel component of the I3 architecture: it creates a
    differentiable bridge between the perception pipeline (which produces
    the AdaptationVector and UserStateEmbedding) and the language generation
    pipeline (the transformer decoder).

    The AdaptationVector encodes 8 behavioral targets:
        [0] cognitive_load   -- target complexity level (0=simple, 1=complex)
        [1] formality        -- register formality (0=casual, 1=formal)
        [2] verbosity        -- response length preference (0=terse, 1=verbose)
        [3] emotionality     -- emotional expressiveness (0=neutral, 1=expressive)
        [4] directness       -- communication directness (0=indirect, 1=direct)
        [5] emotional_tone   -- valence (0=negative, 0.5=neutral, 1=positive)
        [6] accessibility    -- simplification level (0=technical, 1=accessible)
        [7] reserved         -- reserved for future use

    The UserStateEmbedding (64-dim) is the output of the Temporal Convolutional
    Network (TCN) encoder, which compresses a sliding window of raw sensor
    features into a dense temporal representation capturing the user's recent
    behavioral trajectory.

    Architecture:
        Input:  [adaptation_vector(8) || user_state_embedding(64)] = 72-dim
        Layer 1: Linear(72, d_model * n_tokens) + GELU activation + Dropout
        Layer 2: Linear(d_model * n_tokens, d_model * n_tokens)
        Reshape: [batch, n_tokens, d_model]
        LayerNorm: per-token normalization for stable cross-attention

    The two-layer MLP with GELU provides sufficient capacity to learn non-linear
    mappings from the compact 72-dim signal space to the high-dimensional token
    space the transformer operates in. LayerNorm at the output ensures the
    conditioning tokens have similar magnitude to the text embeddings, preventing
    attention score imbalance.

    Args:
        adaptation_dim: Dimensionality of the AdaptationVector (default: 8).
        user_state_dim: Dimensionality of the UserStateEmbedding (default: 64).
        d_model: Transformer model dimensionality (default: 256).
        n_tokens: Number of conditioning tokens to generate (default: 4).
                  More tokens give the model more "slots" to decompose the
                  conditioning signal, at the cost of slightly increased
                  cross-attention compute.
        dropout: Dropout probability in the projection MLP.
    """

    def __init__(
        self,
        adaptation_dim: int = 8,
        user_state_dim: int = 64,
        d_model: int = 256,
        n_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.adaptation_dim: int = adaptation_dim
        self.user_state_dim: int = user_state_dim
        self.n_tokens: int = n_tokens
        self.d_model: int = d_model

        input_dim: int = adaptation_dim + user_state_dim  # 72

        # Two-layer MLP: project from compact signal space to token space
        # GELU chosen over ReLU for smoother gradients in the conditioning path
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, d_model * n_tokens),      # 72 -> 1024
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * n_tokens, d_model * n_tokens),  # 1024 -> 1024
        )

        # LayerNorm ensures conditioning tokens have similar magnitude to
        # text embeddings, preventing attention score imbalance
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization to linear layers, zero-init biases.

        Xavier uniform for weights ensures stable forward/backward pass variance.
        Zero-initialized biases mean the projector starts near the origin,
        producing near-zero conditioning tokens before training -- this is
        desirable because it means the model initially behaves like an
        unconditioned language model and gradually learns to use the conditioning.
        """
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        adaptation_vector: torch.Tensor,
        user_state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project adaptation signals and user state into conditioning tokens.

        The adaptation vector and user state embedding are concatenated and
        passed through a two-layer MLP, then reshaped into a sequence of
        conditioning tokens that serve as key-value pairs in the transformer's
        cross-attention layers.

        Args:
            adaptation_vector: Behavioral adaptation targets.
                               Shape: [batch, adaptation_dim]  (default [batch, 8])
            user_state_embedding: Temporal user state from TCN encoder.
                                  Shape: [batch, user_state_dim]  (default [batch, 64])

        Returns:
            conditioning_tokens: Sequence of conditioning tokens for cross-attention.
                                 Shape: [batch, n_tokens, d_model]  (default [batch, 4, 256])
                                 These are used as both K and V in MultiHeadCrossAttention.
        """
        # SEC: Validate input shapes — fail-fast on misuse rather than
        # producing nonsense conditioning tokens.
        if adaptation_vector.dim() != 2:
            raise ValueError(
                f"adaptation_vector must be 2D [batch, adaptation_dim], "
                f"got shape {tuple(adaptation_vector.shape)}"
            )
        if user_state_embedding.dim() != 2:
            raise ValueError(
                f"user_state_embedding must be 2D [batch, user_state_dim], "
                f"got shape {tuple(user_state_embedding.shape)}"
            )
        if adaptation_vector.size(-1) != self.adaptation_dim:
            raise ValueError(
                f"adaptation_vector last dim {adaptation_vector.size(-1)} "
                f"!= adaptation_dim {self.adaptation_dim}"
            )
        if user_state_embedding.size(-1) != self.user_state_dim:
            raise ValueError(
                f"user_state_embedding last dim {user_state_embedding.size(-1)} "
                f"!= user_state_dim {self.user_state_dim}"
            )
        if adaptation_vector.size(0) != user_state_embedding.size(0):
            raise ValueError(
                f"batch size mismatch: adaptation_vector={adaptation_vector.size(0)}, "
                f"user_state_embedding={user_state_embedding.size(0)}"
            )

        # Concatenate adaptation targets with user state context
        # [batch, 8] || [batch, 64] -> [batch, 72]
        combined: torch.Tensor = torch.cat(
            [adaptation_vector, user_state_embedding], dim=-1
        )

        # Project through MLP to conditioning token space
        # [batch, 72] -> [batch, n_tokens * d_model]  (e.g., [batch, 1024])
        projected: torch.Tensor = self.projection(combined)

        # Reshape flat projection into a sequence of conditioning tokens
        # [batch, n_tokens * d_model] -> [batch, n_tokens, d_model]
        # e.g., [batch, 1024] -> [batch, 4, 256]
        tokens: torch.Tensor = projected.view(-1, self.n_tokens, self.d_model)

        # Normalize each token independently to match text embedding magnitudes
        # [batch, n_tokens, d_model] -> [batch, n_tokens, d_model]
        tokens = self.layer_norm(tokens)

        return tokens

    @staticmethod
    def create_default_conditioning(
        batch_size: int = 1,
        adaptation_dim: int = 8,
        user_state_dim: int = 64,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create default (neutral) conditioning vectors for inference without
        a live perception pipeline.

        Produces a "neutral" adaptation vector with moderate cognitive load
        and neutral emotional tone, plus a zero user state embedding (no
        temporal context). Useful for:
            - Testing the model without sensor input
            - Baseline generation for comparison
            - Fallback when the perception pipeline is unavailable

        Args:
            batch_size: Number of sequences in the batch.
            adaptation_dim: Dimensionality of the AdaptationVector.
            user_state_dim: Dimensionality of the UserStateEmbedding.
            device: Target device for the tensors (CPU/CUDA).

        Returns:
            adaptation: Neutral adaptation vector.
                        Shape: [batch_size, adaptation_dim]
                        Values: all zeros except cognitive_load=0.5,
                                emotional_tone=0.5
            user_state: Zero user state embedding (no temporal context).
                        Shape: [batch_size, user_state_dim]
        """
        # Start with zeros (neutral baseline for all dimensions)
        adaptation: torch.Tensor = torch.zeros(
            batch_size, adaptation_dim, device=device
        )
        # Set cognitive_load and emotional_tone to neutral midpoint
        adaptation[:, 0] = 0.5  # cognitive_load: moderate complexity
        adaptation[:, 5] = 0.5  # emotional_tone: neutral valence

        # Zero embedding indicates no temporal user state information
        user_state: torch.Tensor = torch.zeros(
            batch_size, user_state_dim, device=device
        )

        return adaptation, user_state
