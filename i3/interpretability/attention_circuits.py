"""Circuit-style analysis of I³'s cross-attention conditioning heads.

This module treats each (layer, head) pair in the stacked cross-attention
as a candidate "conditioning circuit" and asks two mechanical questions:

1. **Who attends?** For every head, what fraction of its attention mass
   lands on conditioning tokens versus on an (absent) baseline? Every
   cross-attention head in I³ attends only to conditioning tokens by
   construction, so the mass fraction is ``1.0`` by definition; the
   diagnostic value comes from tracking *which* conditioning token each
   head prefers and how concentrated that attention is (entropy).
2. **Who specialises?** A head is a "conditioning specialist" if it
   reliably concentrates its attention on a small number of conditioning
   tokens — i.e. it has low entropy on a majority of the input tokens.

The API is a thin dataclass wrapper around the tensors returned by
:class:`~i3.interpretability.attention_extractor.CrossAttentionExtractor`;
it does not retrain anything and has no side effects on the model.

References
----------
- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B.,
  et al. (2021). *A Mathematical Framework for Transformer Circuits.*
  Anthropic transformer-circuits thread.
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N.,
  Henighan, T., et al. (2022). *In-context Learning and Induction
  Heads.* arXiv:2209.11895.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from i3.interpretability.attention_extractor import CrossAttentionExtractor


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class HeadSpec:
    """Identifier and summary for a single attention head.

    Attributes:
        layer: Zero-based layer index.
        head: Zero-based head index within the layer.
        mean_max_weight: Mean over output positions of the max attention
            weight on any conditioning token. Higher = more focused.
        mean_entropy: Mean Shannon entropy (nats) of the head's
            attention distribution over conditioning tokens. Lower =
            more specialised.
        preferred_cond_token: Zero-based index of the conditioning token
            that received the highest *average* attention from this
            head.
    """

    layer: int
    head: int
    mean_max_weight: float
    mean_entropy: float
    preferred_cond_token: int


@dataclass
class AttentionPattern:
    """Structured cross-attention pattern for a single prompt.

    Attributes:
        per_layer: List of numpy arrays, one per layer, each of shape
            ``[heads, seq_len, n_cond]`` holding the float attention
            weights.
        per_head_entropy: Array of shape ``[n_layers, n_heads]``
            containing the mean entropy (over output positions) of each
            head.
        per_token_conditioning_focus: Array of shape ``[seq_len]``
            containing the average-across-layers-and-heads max
            attention weight placed on any conditioning token for each
            output position.
        n_cond: Number of conditioning tokens.
        tokens: Optional list of decoded token strings, length
            ``seq_len``. Kept for downstream plotting.
    """

    per_layer: list[np.ndarray]
    per_head_entropy: np.ndarray
    per_token_conditioning_focus: np.ndarray
    n_cond: int
    tokens: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        """Number of layers in the pattern."""
        return len(self.per_layer)

    @property
    def n_heads(self) -> int:
        """Heads per layer (assumed constant)."""
        if not self.per_layer:
            return 0
        return int(self.per_layer[0].shape[0])

    @property
    def seq_len(self) -> int:
        """Number of output positions in the prompt."""
        if not self.per_layer:
            return 0
        return int(self.per_layer[0].shape[1])


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------


def extract_attention_patterns(
    model: nn.Module,
    prompt: torch.Tensor,
    conditioning_vector: torch.Tensor,
    user_state: Optional[torch.Tensor] = None,
    max_tokens: int = 32,
    decoded_tokens: Optional[Sequence[str]] = None,
) -> AttentionPattern:
    """Run the model and extract a structured cross-attention pattern.

    Args:
        model: :class:`AdaptiveSLM` (or stub) exposing the standard
            forward signature.
        prompt: Integer token ids of shape ``[seq_len]`` or
            ``[1, seq_len]``.
        conditioning_vector: Float adaptation vector of shape
            ``[adaptation_dim]`` or ``[1, adaptation_dim]``.
        user_state: Optional user-state embedding. If ``None`` a zero
            tensor is supplied using the model's declared
            ``user_state_dim``.
        max_tokens: Prompt is truncated to at most this many tokens to
            keep extraction bounded.
        decoded_tokens: Optional list of decoded token strings for
            display purposes.

    Returns:
        A populated :class:`AttentionPattern`.

    Raises:
        ValueError: If the prompt is empty or the extractor returned
            no attention maps.
    """
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    if prompt.dim() != 2:
        raise ValueError(
            f"prompt must be 1-D or 2-D, got shape {tuple(prompt.shape)}"
        )

    if conditioning_vector.dim() == 1:
        conditioning_vector = conditioning_vector.unsqueeze(0)

    # Truncate for bounded extraction cost.
    if prompt.size(1) > max_tokens:
        prompt = prompt[:, :max_tokens]

    if prompt.numel() == 0:
        raise ValueError("prompt must have at least one token")

    if user_state is None:
        projector = getattr(model, "conditioning_projector", None)
        user_dim = getattr(projector, "user_state_dim", 64)
        user_state = torch.zeros(conditioning_vector.size(0), user_dim)
    elif user_state.dim() == 1:
        user_state = user_state.unsqueeze(0)

    with CrossAttentionExtractor(model, squeeze_batch=True) as extractor:
        with torch.no_grad():
            model(prompt, conditioning_vector, user_state)
        maps = extractor.get_attention_maps()

    if not maps:
        raise ValueError("extractor returned no attention maps")

    per_layer_np: list[np.ndarray] = []
    for m in maps:
        if m.numel() == 0:
            # Layer did not fire (should not happen on a normal forward,
            # but the extractor guarantees None-free entries).
            continue
        if m.dim() == 4 and m.size(0) == 1:
            m = m.squeeze(0)
        if m.dim() != 3:
            raise ValueError(
                "expected [heads, seq_len, n_cond] attention tensor, got "
                f"shape {tuple(m.shape)}"
            )
        per_layer_np.append(m.float().cpu().numpy())

    if not per_layer_np:
        raise ValueError("no valid attention maps captured")

    n_heads = per_layer_np[0].shape[0]
    seq_len = per_layer_np[0].shape[1]
    n_cond = per_layer_np[0].shape[2]

    entropy_grid = np.zeros((len(per_layer_np), n_heads), dtype=np.float32)
    for li, layer in enumerate(per_layer_np):
        # Shannon entropy over conditioning dim, averaged over seq_len.
        eps = 1e-9
        ent = -(layer * np.log(layer + eps)).sum(axis=-1)  # [heads, seq_len]
        entropy_grid[li] = ent.mean(axis=1)

    # Per-token focus: mean across layers & heads of the MAX weight on any
    # conditioning token at that position.
    focus = np.zeros(seq_len, dtype=np.float32)
    total = 0
    for layer in per_layer_np:
        focus += layer.max(axis=-1).sum(axis=0)  # sum over heads -> [seq_len]
        total += layer.shape[0]
    if total:
        focus = focus / float(total)

    tokens_list = list(decoded_tokens) if decoded_tokens is not None else []

    return AttentionPattern(
        per_layer=per_layer_np,
        per_head_entropy=entropy_grid,
        per_token_conditioning_focus=focus,
        n_cond=int(n_cond),
        tokens=tokens_list,
    )


# ---------------------------------------------------------------------------
# Circuit identification.
# ---------------------------------------------------------------------------


def identify_conditioning_specialists(
    pattern: AttentionPattern,
    threshold: float = 0.6,
    majority_fraction: float = 0.5,
) -> list[HeadSpec]:
    """Return heads that concentrate attention on conditioning tokens.

    A head is classified as a "conditioning specialist" when its
    per-position max attention weight exceeds ``threshold`` on more than
    ``majority_fraction`` of the output positions.

    Args:
        pattern: An :class:`AttentionPattern` produced by
            :func:`extract_attention_patterns`.
        threshold: Minimum max-weight to count an output position as
            "focused" on a conditioning token.
        majority_fraction: Fraction of output positions that must be
            focused for the head to qualify.

    Returns:
        List of :class:`HeadSpec` entries for each qualifying head,
        sorted by descending ``mean_max_weight``.
    """
    specs: list[HeadSpec] = []
    for li, layer in enumerate(pattern.per_layer):
        # layer shape: [heads, seq_len, n_cond]
        max_weights = layer.max(axis=-1)  # [heads, seq_len]
        preferred = layer.mean(axis=1).argmax(axis=-1)  # [heads]
        eps = 1e-9
        entropy = -(layer * np.log(layer + eps)).sum(axis=-1).mean(axis=-1)
        focused_frac = (max_weights > threshold).mean(axis=-1)
        mean_mw = max_weights.mean(axis=-1)
        for h in range(layer.shape[0]):
            if focused_frac[h] > majority_fraction:
                specs.append(
                    HeadSpec(
                        layer=int(li),
                        head=int(h),
                        mean_max_weight=float(mean_mw[h]),
                        mean_entropy=float(entropy[h]),
                        preferred_cond_token=int(preferred[h]),
                    )
                )
    specs.sort(key=lambda s: s.mean_max_weight, reverse=True)
    return specs


# ---------------------------------------------------------------------------
# Narrative summary.
# ---------------------------------------------------------------------------


def summarise_circuit(
    pattern: AttentionPattern,
    threshold: float = 0.6,
) -> str:
    """Produce a one-paragraph natural-language summary of the circuit.

    Args:
        pattern: An :class:`AttentionPattern`.
        threshold: Passed through to
            :func:`identify_conditioning_specialists`.

    Returns:
        A single paragraph describing how many heads specialise, which
        layers they live in, their mean attention entropy, and which
        conditioning token they most commonly prefer.
    """
    if pattern.n_layers == 0:
        return "No attention pattern captured; summary unavailable."

    specialists = identify_conditioning_specialists(pattern, threshold=threshold)
    total_heads = pattern.n_layers * pattern.n_heads

    if not specialists:
        mean_focus = float(pattern.per_token_conditioning_focus.mean())
        return (
            f"Across {pattern.n_layers} layers and {pattern.n_heads} heads per "
            f"layer ({total_heads} cross-attention heads total), no head passed "
            f"the specialisation threshold of {threshold:.2f}. The average "
            f"per-token conditioning focus was {mean_focus:.3f}, suggesting "
            f"that conditioning information is distributed across many heads "
            f"rather than concentrated in a small circuit. This is the expected "
            f"profile for an untrained or lightly-trained model, where the "
            f"ConditioningProjector tokens have not yet differentiated."
        )

    by_layer: dict[int, int] = {}
    for sp in specialists:
        by_layer[sp.layer] = by_layer.get(sp.layer, 0) + 1
    dominant_layer = max(by_layer.items(), key=lambda kv: kv[1])
    top = specialists[0]
    mean_entropy = float(np.mean([sp.mean_entropy for sp in specialists]))

    return (
        f"Of {total_heads} cross-attention heads, {len(specialists)} meet the "
        f"conditioning-specialist threshold (max attention weight > "
        f"{threshold:.2f} on more than half of the output positions). "
        f"Specialisation is concentrated in layer {dominant_layer[0]} "
        f"({dominant_layer[1]} heads) and the most focused head is layer "
        f"{top.layer} head {top.head} with mean max weight "
        f"{top.mean_max_weight:.3f} and mean entropy {top.mean_entropy:.3f} "
        f"nats, preferring conditioning token {top.preferred_cond_token}. "
        f"The specialist cohort has mean entropy {mean_entropy:.3f} nats, "
        f"indicating {'sharp' if mean_entropy < 0.8 else 'moderate'} "
        f"selectivity over the conditioning axis."
    )


__all__ = [
    "AttentionPattern",
    "HeadSpec",
    "extract_attention_patterns",
    "identify_conditioning_specialists",
    "summarise_circuit",
]
