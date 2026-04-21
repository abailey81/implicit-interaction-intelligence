"""JSON-serialisable cross-attention heatmap for front-end visualisation.

Produces the minimal representation needed by a browser-side heatmap
component: the tokens actually emitted by the SLM and, for each token, a
nested list of attention weights over the 4 conditioning tokens, keyed by
(layer, head).

The conditioning tokens in the I^3 architecture have the following
semantic interpretation by position, as documented in
``docs/ARCHITECTURE.md`` §8:

    Token 0 -- adaptation-driven (cognitive load, formality axes)
    Token 1 -- adaptation-driven (verbosity, directness axes)
    Token 2 -- user-state driven (short-term temporal context)
    Token 3 -- user-state driven (long-term drift)

These labels are purely descriptive: the ConditioningProjector is a
learned MLP and each token's semantics are emergent.

References
----------
- Vig, J. (2019). *A Multiscale Visualisation of Attention in the
  Transformer Model*.  ACL demo track.  (BertViz, the canonical
  attention-visualisation schema this module produces JSON compatible
  with.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Canonical conditioning-token labels used in the exported JSON.  These
# match ``n_conditioning_tokens=4`` in the default AdaptiveSLM config.
DEFAULT_CONDITIONING_LABELS: list[str] = [
    "cond_0_adaptation",
    "cond_1_style",
    "cond_2_user_state_short",
    "cond_3_user_state_long",
]


@dataclass
class TokenHeatmapPayload:
    """Structured representation returned by :meth:`TokenHeatmap.build`.

    Attributes:
        tokens: Decoded string tokens, length ``seq_len``.
        conditioning_labels: Labels for the conditioning axis, length
            ``n_cond``.
        layers: Per-layer per-head attention maps.  Shape
            ``[n_layers][n_heads][seq_len][n_cond]`` as nested lists of
            Python floats.  Exactly this structure is what a d3.js
            heatmap component consumes.
        shape: Triple ``(n_layers, n_heads, seq_len, n_cond)`` for quick
            shape inspection without having to walk the nested list.
    """

    tokens: list[str]
    conditioning_labels: list[str]
    layers: list[list[list[list[float]]]]
    shape: tuple[int, int, int, int]


class TokenHeatmap:
    """Builds JSON-serialisable cross-attention heatmaps.

    Parameters
    ----------
    conditioning_labels : list[str], optional
        Labels applied to the conditioning-token axis of the heatmap.
        Must match the number of conditioning tokens produced by the
        :class:`~i3.slm.cross_attention.ConditioningProjector` (4 in
        the default config).  Defaults to
        :data:`DEFAULT_CONDITIONING_LABELS`.

    Example
    -------
    >>> heatmap = TokenHeatmap()
    >>> payload = heatmap.build(tokens, attention_maps)
    >>> import json; json.dumps(payload.to_dict())
    """

    def __init__(
        self, conditioning_labels: Optional[list[str]] = None
    ) -> None:
        self.conditioning_labels: list[str] = (
            list(conditioning_labels)
            if conditioning_labels is not None
            else list(DEFAULT_CONDITIONING_LABELS)
        )

    def build(
        self,
        tokens: list[str],
        attention_maps: list[torch.Tensor],
    ) -> TokenHeatmapPayload:
        """Construct a :class:`TokenHeatmapPayload` from raw inputs.

        Args:
            tokens: Decoded string tokens produced by the SLM, in
                generation order.
            attention_maps: One attention tensor per layer.  Each must
                have shape ``[heads, seq_len, n_cond]`` OR
                ``[batch, heads, seq_len, n_cond]`` (batch is squeezed
                to 1 for heatmap export).

        Returns:
            :class:`TokenHeatmapPayload` whose :meth:`to_dict` produces
            a plain Python dict safe to pass to ``json.dumps``.

        Raises:
            ValueError: If any attention tensor has an unexpected rank
                or if its seq_len does not match ``len(tokens)``.
        """
        if not isinstance(tokens, list) or not all(
            isinstance(t, str) for t in tokens
        ):
            raise TypeError("tokens must be a list[str]")
        if not isinstance(attention_maps, list):
            raise TypeError("attention_maps must be a list[torch.Tensor]")
        # Empty cases -> empty payload, shape zeros.  Safer than raising
        # for the front-end which can then render a "no attention" stub.
        if not attention_maps:
            return TokenHeatmapPayload(
                tokens=list(tokens),
                conditioning_labels=list(self.conditioning_labels),
                layers=[],
                shape=(0, 0, len(tokens), len(self.conditioning_labels)),
            )

        # Normalise every layer's tensor to [heads, seq_len, n_cond].
        n_layers = len(attention_maps)
        per_layer: list[torch.Tensor] = []
        for i, t in enumerate(attention_maps):
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"attention_maps[{i}] must be a torch.Tensor, got "
                    f"{type(t).__name__}"
                )
            if t.dim() == 4 and t.size(0) == 1:
                t = t.squeeze(0)
            if t.dim() != 3:
                raise ValueError(
                    "each attention tensor must be 3-D "
                    "[heads, seq_len, n_cond], got shape "
                    f"{tuple(t.shape)} at layer {i}"
                )
            per_layer.append(t.detach().cpu())

        n_heads = per_layer[0].size(0)
        seq_len_attn = per_layer[0].size(1)
        n_cond = per_layer[0].size(2)

        if seq_len_attn != len(tokens):
            # Tokens may include BOS/EOS/specials that were stripped from
            # the display list, or generation may have stopped partway.
            # Pad or truncate the token list to the attention seq_len so
            # the front-end sees a consistent rectangular matrix.
            if seq_len_attn > len(tokens):
                tokens = list(tokens) + [
                    "<?>"
                ] * (seq_len_attn - len(tokens))
            else:
                tokens = list(tokens)[:seq_len_attn]

        # Truncate/extend conditioning labels to match n_cond.
        labels = list(self.conditioning_labels)
        if len(labels) < n_cond:
            labels += [f"cond_{i}" for i in range(len(labels), n_cond)]
        elif len(labels) > n_cond:
            labels = labels[:n_cond]

        # Convert to nested list-of-floats for JSON.
        nested: list[list[list[list[float]]]] = []
        for t in per_layer:
            layer_lst: list[list[list[float]]] = []
            for h in range(n_heads):
                head_lst: list[list[float]] = []
                for s in range(seq_len_attn):
                    head_lst.append(
                        [float(x) for x in t[h, s, :n_cond].tolist()]
                    )
                layer_lst.append(head_lst)
            nested.append(layer_lst)

        return TokenHeatmapPayload(
            tokens=list(tokens),
            conditioning_labels=labels,
            layers=nested,
            shape=(n_layers, n_heads, seq_len_attn, n_cond),
        )

    @staticmethod
    def to_dict(payload: TokenHeatmapPayload) -> dict:
        """Serialise a payload into a plain dict for ``json.dumps``.

        Args:
            payload: The payload produced by :meth:`build`.

        Returns:
            Dict with keys ``tokens``, ``conditioning_labels``,
            ``layers`` and ``shape`` -- all JSON-safe.
        """
        return {
            "tokens": list(payload.tokens),
            "conditioning_labels": list(payload.conditioning_labels),
            "layers": payload.layers,
            "shape": list(payload.shape),
        }


__all__ = [
    "DEFAULT_CONDITIONING_LABELS",
    "TokenHeatmap",
    "TokenHeatmapPayload",
]
