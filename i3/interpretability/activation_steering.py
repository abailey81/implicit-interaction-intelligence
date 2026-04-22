"""Activation-steering interventions via SAE feature directions.

This module implements the ``ActivationAddition`` (ActAdd) family of
inference-time interventions introduced by Turner et al. (2023) and
later generalised by the representation-engineering literature (Zou et
al., 2023). Given a direction in residual space and a target layer,
:class:`ActivationSteerer` adds a scalar multiple of that direction to
the residual stream at each forward pass, implemented as a forward hook
on the corresponding :class:`AdaptiveTransformerBlock`.

Typical usage with a trained SAE::

    steerer = ActivationSteerer(model, saes_by_layer={2: sae_layer2})
    output = steerer.steer(
        model,
        prompt="Explain gradient descent.",
        feature_idx=17,
        magnitude=2.0,
        target_layer=2,
    )

The steering direction is taken from the decoder column of the named
feature (see :func:`i3.interpretability.sae_analysis.feature_steering_vector`).
Because the direction lives in the same residual-stream basis as the
hidden state, adding a scalar multiple is dimensionally coherent and
does not require a learned projection.

References
----------
- Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., &
  MacDiarmid, M. (2023). *Activation Addition: Steering Language Models
  Without Optimisation.* arXiv:2308.10248.
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., et al.
  (2023). *Representation Engineering: A Top-Down Approach to AI
  Transparency.* arXiv:2310.01405.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from i3.interpretability.sparse_autoencoder import SparseAutoencoder


# ---------------------------------------------------------------------------
# SteeringResult.
# ---------------------------------------------------------------------------


@dataclass
class SteeringResult:
    """Outcome of a single :meth:`ActivationSteerer.steer` invocation.

    Attributes:
        text: The decoded model output, or a placeholder if the caller
            supplied no tokenizer.
        baseline_logits_last: Last-position logits from the un-steered
            forward pass.
        steered_logits_last: Last-position logits from the steered
            forward pass.
        feature_idx: Feature whose decoder column was used.
        magnitude: The scalar multiplier applied to the direction.
        target_layer: Layer index the hook was attached to.
    """

    text: str
    baseline_logits_last: torch.Tensor
    steered_logits_last: torch.Tensor
    feature_idx: int
    magnitude: float
    target_layer: int

    @property
    def logit_shift_l2(self) -> float:
        """L2 distance between baseline and steered last-position logits."""
        return float(
            torch.norm(
                self.steered_logits_last - self.baseline_logits_last
            ).item()
        )


# ---------------------------------------------------------------------------
# ActivationSteerer.
# ---------------------------------------------------------------------------


class ActivationSteerer:
    """Steer generation by adding an SAE feature direction to the residual.

    Parameters
    ----------
    model : nn.Module
        A loaded :class:`~i3.slm.model.AdaptiveSLM` (or stub with a
        ``layers`` :class:`nn.ModuleList`).
    saes_by_layer : dict[int, SparseAutoencoder]
        Mapping of ``layer_index -> trained SAE``. The steerer looks up
        the decoder column of the requested feature from the SAE whose
        key matches ``target_layer``.

    Notes
    -----
    The hook is attached as a post-forward hook on
    ``model.layers[target_layer]`` and replaces the block's output
    tensor with ``output + magnitude * direction``. The hook is always
    removed on context exit, so mis-scaled interventions cannot persist
    across calls.
    """

    def __init__(
        self,
        model: nn.Module,
        saes_by_layer: dict[int, SparseAutoencoder],
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module, got {type(model).__name__}"
            )
        layers = getattr(model, "layers", None)
        if layers is None:
            raise AttributeError("model has no 'layers' attribute")
        if not saes_by_layer:
            raise ValueError("saes_by_layer must not be empty")
        for li, sae in saes_by_layer.items():
            if not 0 <= li < len(layers):
                raise ValueError(
                    f"layer index {li} out of range [0, {len(layers)})"
                )
            if not isinstance(sae, SparseAutoencoder):
                raise TypeError(
                    f"saes_by_layer[{li}] must be SparseAutoencoder, "
                    f"got {type(sae).__name__}"
                )
        self._model = model
        self._saes_by_layer: dict[int, SparseAutoencoder] = dict(saes_by_layer)

    # ------------------------------------------------------------------
    # Core steering call.
    # ------------------------------------------------------------------

    def steer(
        self,
        model: nn.Module,
        prompt: str | torch.Tensor,
        feature_idx: int,
        magnitude: float = 1.0,
        target_layer: int = 2,
        tokenizer: Optional[object] = None,
        adaptation_vector: Optional[torch.Tensor] = None,
        user_state: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
    ) -> SteeringResult:
        """Steer generation along the named feature direction.

        Args:
            model: Same model that was passed to ``__init__``. Kept in
                the method signature for symmetry with Turner 2023's
                APIs.
            prompt: Either a natural-language string (requires a
                ``tokenizer``) or a 2-D tensor of token IDs of shape
                ``[1, seq_len]``.
            feature_idx: Zero-based index of the feature whose decoder
                column is used as the steering direction.
            magnitude: Scalar multiplier on the direction. Positive
                values push the residual toward the feature; negative
                values push away.
            target_layer: Index of the :class:`AdaptiveTransformerBlock`
                to steer.
            tokenizer: Optional tokenizer with ``encode(text, ...) ->
                list[int]`` and ``decode(list[int]) -> str``. Required
                when ``prompt`` is a string.
            adaptation_vector: Optional ``[1, 8]`` tensor. Defaults to
                the neutral midpoint used by :meth:`AdaptiveSLM.forward`.
            user_state: Optional ``[1, 64]`` tensor. Defaults to zeros.
            max_new_tokens: Upper bound on generated tokens when a
                tokenizer is supplied. Ignored otherwise.

        Returns:
            :class:`SteeringResult` summarising the intervention.

        Raises:
            ValueError: On invalid ``target_layer``, ``feature_idx``,
                or prompt types.
        """
        if model is not self._model:
            raise ValueError(
                "steer(model=...) must equal the model passed to __init__"
            )
        if target_layer not in self._saes_by_layer:
            raise ValueError(
                f"no SAE registered for layer {target_layer}; "
                f"available: {sorted(self._saes_by_layer.keys())}"
            )
        sae = self._saes_by_layer[target_layer]
        if not 0 <= feature_idx < sae.d_dict:
            raise ValueError(
                f"feature_idx {feature_idx} out of range [0, {sae.d_dict})"
            )

        layers = getattr(model, "layers")
        if not 0 <= target_layer < len(layers):
            raise ValueError(
                f"target_layer {target_layer} out of range "
                f"[0, {len(layers)})"
            )

        input_ids = self._prepare_input_ids(prompt, tokenizer)

        direction = sae.decoder.weight.detach()[:, feature_idx].clone()
        # Unit-normalise for scale invariance; the SAE trainer already
        # enforces near-unit norms, but we re-normalise defensively.
        norm = direction.norm()
        if float(norm.item()) > 0.0:
            direction = direction / norm

        # Baseline forward pass (no hook).
        model.eval()
        with torch.no_grad():
            baseline_out = model(
                input_ids,
                adaptation_vector,
                user_state,
            )
        baseline_logits = (
            baseline_out[0] if isinstance(baseline_out, tuple) else baseline_out
        )
        baseline_last = baseline_logits[0, -1, :].detach().clone()

        # Steered forward pass (hook on the target layer).
        handle = layers[target_layer].register_forward_hook(
            _make_steering_hook(direction, float(magnitude))
        )
        try:
            with torch.no_grad():
                steered_out = model(
                    input_ids,
                    adaptation_vector,
                    user_state,
                )
        finally:
            try:
                handle.remove()
            except RuntimeError:  # pragma: no cover - defensive
                pass

        steered_logits = (
            steered_out[0] if isinstance(steered_out, tuple) else steered_out
        )
        steered_last = steered_logits[0, -1, :].detach().clone()

        # Optional greedy-decode when a tokenizer is supplied.
        text = _maybe_decode(
            model=model,
            input_ids=input_ids,
            steered_last_logits=steered_last,
            tokenizer=tokenizer,
            adaptation_vector=adaptation_vector,
            user_state=user_state,
            direction=direction,
            magnitude=float(magnitude),
            target_layer=target_layer,
            max_new_tokens=max_new_tokens,
        )

        return SteeringResult(
            text=text,
            baseline_logits_last=baseline_last,
            steered_logits_last=steered_last,
            feature_idx=feature_idx,
            magnitude=float(magnitude),
            target_layer=int(target_layer),
        )

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_input_ids(
        prompt: str | torch.Tensor,
        tokenizer: Optional[object],
    ) -> torch.Tensor:
        """Normalise the supplied prompt into a 2-D id tensor."""
        if isinstance(prompt, torch.Tensor):
            if prompt.dim() == 1:
                return prompt.unsqueeze(0)
            if prompt.dim() == 2:
                return prompt
            raise ValueError(
                "prompt tensor must be 1-D or 2-D, got shape "
                f"{tuple(prompt.shape)}"
            )
        if not isinstance(prompt, str):
            raise ValueError(
                "prompt must be str or torch.Tensor, got "
                f"{type(prompt).__name__}"
            )
        if tokenizer is None:
            raise ValueError(
                "string prompts require a tokenizer (none supplied)"
            )
        encode_fn = getattr(tokenizer, "encode", None)
        if encode_fn is None:
            raise ValueError("tokenizer has no .encode() method")
        ids = encode_fn(prompt)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


# ---------------------------------------------------------------------------
# Module-level hook factory and decode helper.
# ---------------------------------------------------------------------------


def _make_steering_hook(
    direction: torch.Tensor,
    magnitude: float,
) -> object:
    """Return a forward hook that adds ``magnitude * direction`` to the output.

    Args:
        direction: 1-D tensor of length ``d_model``.
        magnitude: Scalar multiplier.

    Returns:
        A hook callable usable with :meth:`nn.Module.register_forward_hook`.
    """

    delta = (direction * magnitude).detach()

    def _hook(_m: nn.Module, _inp: tuple, output: object) -> object:
        if isinstance(output, tuple):
            if not output:
                return output
            head = output[0]
            if isinstance(head, torch.Tensor):
                new_head = head + delta.to(
                    dtype=head.dtype, device=head.device
                )
                return (new_head, *output[1:])
            return output
        if isinstance(output, torch.Tensor):
            return output + delta.to(dtype=output.dtype, device=output.device)
        return output

    return _hook


def _maybe_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    steered_last_logits: torch.Tensor,
    tokenizer: Optional[object],
    adaptation_vector: Optional[torch.Tensor],
    user_state: Optional[torch.Tensor],
    direction: torch.Tensor,
    magnitude: float,
    target_layer: int,
    max_new_tokens: int,
) -> str:
    """Greedy-decode the steered continuation if a tokenizer is provided.

    When the caller does not pass a tokenizer we return a deterministic
    placeholder string so that downstream logging has something to print
    without requiring tokenizer round-tripping.
    """
    if tokenizer is None:
        return f"<steered: feature at layer {target_layer}>"

    decode_fn = getattr(tokenizer, "decode", None)
    if decode_fn is None:
        return f"<steered: feature at layer {target_layer}>"

    layers = getattr(model, "layers")
    ids = input_ids.clone()
    handle = layers[target_layer].register_forward_hook(
        _make_steering_hook(direction, magnitude)
    )
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(ids, adaptation_vector, user_state)
                logits = out[0] if isinstance(out, tuple) else out
                next_id = int(logits[0, -1, :].argmax().item())
                ids = torch.cat(
                    [ids, torch.tensor([[next_id]], dtype=ids.dtype)],
                    dim=-1,
                )
                if hasattr(tokenizer, "EOS_ID") and next_id == getattr(
                    tokenizer, "EOS_ID"
                ):
                    break
    finally:
        try:
            handle.remove()
        except RuntimeError:  # pragma: no cover - defensive
            pass

    # First-token sanity: also incorporate the already-computed
    # steered_last_logits (kept only for diagnostic parity with the
    # ``steered_logits_last`` return value; no effect on decoded text).
    del steered_last_logits
    try:
        return str(decode_fn(ids[0].tolist()))
    except (ValueError, TypeError):
        return f"<steered: feature at layer {target_layer}>"


__all__ = [
    "ActivationSteerer",
    "SteeringResult",
]
