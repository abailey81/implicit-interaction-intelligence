"""Activation patching / causal tracing for I³'s cross-attention SLM.

This module implements a light-weight, random-init-friendly variant of the
ROME / causal-mediation methodology for the :class:`~i3.slm.model.AdaptiveSLM`.
The question it answers is operational rather than theoretical: **which
sub-modules of the SLM are the causal pathway through which an
``AdaptationVector`` influences the next-token distribution?**

Pipeline
--------
The procedure for a single ``component`` is:

1. **Corrupted run.** Run the model with a corrupted (e.g. zeroed)
   conditioning and cache every sub-module's output via forward hooks.
2. **Clean run.** Run the model with the *clean* conditioning, but at
   sub-module ``component`` splice in the corrupted activation that was
   cached in step 1. Capture the resulting next-token logits.
3. **Reference run.** A third, fully-clean forward pass produces the
   "ideal" next-token distribution.
4. **Effect size.** The causal effect is the symmetric KL divergence
   between the patched next-token distribution and the fully-clean
   next-token distribution. Larger values mean that swapping in the
   corrupted activation at ``component`` moved the distribution further
   from the ideal — i.e. the component is on the critical path.

References
----------
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and
  Editing Factual Associations in GPT.* NeurIPS 2022 (ROME).
- Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., &
  Shieber, S. (2020). *Causal Mediation Analysis for Interpreting Neural
  NLP: The Case of Gender Bias.* arXiv:2004.12265.
- Geiger, A., Wu, Z., Lu, H., Rozner, J., Icard, T., Potts, C. (2024).
  *Causal Abstraction: A Theoretical Foundation for Mechanistic
  Interpretability.* arXiv:2301.04709.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component-name canonicalisation.
# ---------------------------------------------------------------------------


def canonical_components(n_layers: int) -> list[str]:
    """Return the canonical ordered list of traced component names.

    Args:
        n_layers: Number of transformer layers in the :class:`AdaptiveSLM`.

    Returns:
        List of component identifiers covering the conditioning
        projector plus every ``{cross,self,ffn}_attn_layer_{i}`` for
        ``i`` in ``range(n_layers)``. The order is stable and matches
        the order used in the report produced by
        ``scripts/run_interpretability_study.py``.
    """
    names: list[str] = ["conditioning_projector"]
    for i in range(n_layers):
        names.append(f"cross_attn_layer_{i}")
    for i in range(n_layers):
        names.append(f"self_attn_layer_{i}")
    for i in range(n_layers):
        names.append(f"ffn_layer_{i}")
    return names


def _resolve_submodule(model: nn.Module, component: str) -> nn.Module:
    """Resolve a canonical ``component`` string to an :class:`nn.Module`.

    Args:
        model: The :class:`AdaptiveSLM` or stub exposing the expected
            attributes.
        component: A canonical component name as produced by
            :func:`canonical_components`.

    Returns:
        The referenced sub-module.

    Raises:
        AttributeError: If the component cannot be located.
        ValueError: If the component string does not match the grammar.
    """
    if component == "conditioning_projector":
        return model.conditioning_projector  # type: ignore[attr-defined]

    parts = component.rsplit("_layer_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"component {component!r} does not match canonical grammar"
        )
    kind, idx_str = parts
    try:
        idx = int(idx_str)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid layer index in {component!r}") from exc

    layers = getattr(model, "layers", None)
    if layers is None or idx >= len(layers):
        raise AttributeError(
            f"model has no layer index {idx} for component {component!r}"
        )
    layer = layers[idx]
    if kind == "cross_attn":
        return layer.cross_attn
    if kind == "self_attn":
        return layer.self_attn
    if kind == "ffn":
        # AdaptiveTransformerBlock names the FFN sub-layer ``ff``.
        return getattr(layer, "ff", getattr(layer, "ffn", layer))
    raise ValueError(f"unknown component kind in {component!r}")


# ---------------------------------------------------------------------------
# Patch state.
# ---------------------------------------------------------------------------


@dataclass
class _PatchState:
    """Private mutable state held by :class:`ActivationPatcher`."""

    cached: dict[str, torch.Tensor] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)


@dataclass
class CausalEffect:
    """Summary of the causal effect of patching a single component.

    Attributes:
        component: Canonical component name (e.g. ``"cross_attn_layer_2"``).
        kl_to_clean: Symmetric KL divergence (nats) between the patched
            and the fully-clean next-token distribution. Larger means
            the component is more strongly on the causal path.
        logit_l2: L2 distance between the patched and the clean
            next-token logits.
        top1_flipped: ``True`` if the argmax next token differs between
            the patched and the clean runs.
    """

    component: str
    kl_to_clean: float
    logit_l2: float
    top1_flipped: bool


# ---------------------------------------------------------------------------
# ActivationPatcher.
# ---------------------------------------------------------------------------


class ActivationPatcher:
    """Context-manager that swaps cached activations into a forward pass.

    The patcher is used in two phases:

    1. :meth:`cache_corrupted` attaches output-capture hooks, runs the
       supplied ``corrupted_run`` callable once, and stores the outputs
       of every traced sub-module.
    2. :meth:`patch` attaches replacement hooks that force the output of
       a single target ``component`` to the cached corrupted tensor, and
       re-runs the model under *clean* conditioning.

    Hooks are always removed on :meth:`__exit__`, so a failure in the
    user's run callable cannot leave dangling hooks on the model.

    Parameters
    ----------
    model : nn.Module
        The :class:`AdaptiveSLM` (or a stub with matching attributes).
        The patcher does not own the model and does not modify its
        parameters.
    components : list[str] | None
        Optional subset of canonical component names to trace. Defaults
        to :func:`canonical_components(n_layers)`.

    Example
    -------
    >>> with ActivationPatcher(model) as patcher:
    ...     patcher.cache_corrupted(lambda: model(ids, zero_adapt, zero_state))
    ...     patched_logits = patcher.patch(
    ...         "cross_attn_layer_2",
    ...         clean_run=lambda: model(ids, clean_adapt, clean_state),
    ...     )
    """

    def __init__(
        self,
        model: nn.Module,
        components: Optional[list[str]] = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module, got {type(model).__name__}"
            )
        self._model = model
        n_layers = len(getattr(model, "layers", []))
        self._components: list[str] = (
            list(components)
            if components is not None
            else canonical_components(n_layers)
        )
        self._state = _PatchState()

    # ------------------------------------------------------------------
    # Context-manager protocol.
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationPatcher":
        """Initialise the patch state. Hooks are attached lazily."""
        self._state = _PatchState()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Remove any outstanding hooks. Idempotent."""
        self._detach_all()

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @property
    def components(self) -> list[str]:
        """Traced component names in canonical order."""
        return list(self._components)

    @property
    def n_attached_hooks(self) -> int:
        """Number of currently-attached hooks (for tests and diagnostics)."""
        return len(self._state.handles)

    def cache_corrupted(self, corrupted_run: Callable[[], object]) -> None:
        """Run ``corrupted_run`` and cache every traced sub-module's output.

        Args:
            corrupted_run: Zero-argument callable that performs a forward
                pass under the corrupted conditioning (e.g. zeroed
                ``AdaptationVector``). Its return value is discarded.

        Raises:
            ValueError: If a component cannot be resolved on the model.
        """
        self._detach_all()
        for comp in self._components:
            submod = _resolve_submodule(self._model, comp)
            handle = submod.register_forward_hook(self._make_capture_hook(comp))
            self._state.handles.append(handle)

        # The forward pass writes into ``self._state.cached`` via the hooks.
        corrupted_run()
        self._detach_all()

    def patch(
        self,
        component: str,
        clean_run: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Re-run the model under clean conditioning with a patched component.

        The replacement forward hook substitutes the cached corrupted
        output for the live output of ``component`` during ``clean_run``.
        All other sub-modules behave normally.

        Args:
            component: Canonical name of the component to patch. Must
                have been included in :attr:`components` and cached by
                :meth:`cache_corrupted`.
            clean_run: Zero-argument callable returning the model's
                logits tensor of shape ``[batch, seq_len, vocab_size]``.

        Returns:
            Logits tensor captured from ``clean_run`` while the
            component was patched.

        Raises:
            KeyError: If ``component`` has not been cached.
            ValueError: If ``clean_run`` does not return a tensor.
        """
        if component not in self._state.cached:
            raise KeyError(
                f"no cached activation for {component!r}; call "
                "cache_corrupted() first"
            )

        submod = _resolve_submodule(self._model, component)
        cached = self._state.cached[component]

        handle = submod.register_forward_hook(
            self._make_replace_hook(cached)
        )
        self._state.handles.append(handle)
        try:
            logits = clean_run()
        finally:
            # Remove only the replacement hook — keep any other hooks
            # the caller may have attached via a nested context manager.
            try:
                handle.remove()
            except RuntimeError:  # pragma: no cover - defensive
                pass
            if handle in self._state.handles:
                self._state.handles.remove(handle)

        if isinstance(logits, tuple):
            # Some AdaptiveSLM variants return (logits, info) — accept both.
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "clean_run must return a torch.Tensor (optionally wrapped "
                "in a tuple whose first element is the tensor)"
            )
        return logits

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _detach_all(self) -> None:
        """Remove all attached hooks. Never raises."""
        for h in self._state.handles:
            try:
                h.remove()
            except RuntimeError:  # pragma: no cover - defensive
                continue
        self._state.handles = []

    def _make_capture_hook(
        self, component: str
    ) -> Callable[[nn.Module, tuple, object], None]:
        """Produce a hook that stores the sub-module's primary output."""
        cache = self._state.cached

        def _hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
            tensor = _extract_primary_tensor(output)
            if tensor is None:
                return
            # Clone-and-detach so later passes cannot aliasing-overwrite us.
            cache[component] = tensor.detach().clone()

        return _hook

    @staticmethod
    def _make_replace_hook(
        cached: torch.Tensor,
    ) -> Callable[[nn.Module, tuple, object], object]:
        """Produce a hook that replaces the sub-module's output tensor."""

        def _hook(_module: nn.Module, _inputs: tuple, output: object) -> object:
            if isinstance(output, tuple):
                if not output:
                    return output
                head = output[0]
                if isinstance(head, torch.Tensor) and head.shape == cached.shape:
                    return (cached.to(dtype=head.dtype, device=head.device), *output[1:])
                return output
            if isinstance(output, torch.Tensor) and output.shape == cached.shape:
                return cached.to(dtype=output.dtype, device=output.device)
            return output

        return _hook


def _extract_primary_tensor(output: object) -> Optional[torch.Tensor]:
    """Return the first tensor of a module output, or ``None``.

    Some sub-modules return ``(hidden, attn_weights)``; others return a
    bare tensor. This helper normalises both.

    Args:
        output: The raw output object from a forward pass.

    Returns:
        The primary tensor, or ``None`` if no tensor is found.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        head = output[0]
        if isinstance(head, torch.Tensor):
            return head
    return None


# ---------------------------------------------------------------------------
# Top-level convenience: trace_causal_effect.
# ---------------------------------------------------------------------------


def _symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Compute symmetric KL between two next-token logit vectors.

    Args:
        p_logits: Logits tensor of shape ``[vocab]`` or ``[1, vocab]``.
        q_logits: Logits tensor of shape ``[vocab]`` or ``[1, vocab]``.

    Returns:
        Symmetric KL divergence in nats, non-negative.
    """
    if p_logits.dim() > 1:
        p_logits = p_logits.reshape(-1)
    if q_logits.dim() > 1:
        q_logits = q_logits.reshape(-1)
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    p = logp.exp()
    q = logq.exp()
    kl_pq = torch.sum(p * (logp - logq))
    kl_qp = torch.sum(q * (logq - logp))
    return float(0.5 * (kl_pq + kl_qp).item())


def _last_position_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the next-token logits for the last sequence position.

    Args:
        logits: Tensor of shape ``[batch, seq_len, vocab_size]`` or
            ``[seq_len, vocab_size]`` or ``[vocab_size]``.

    Returns:
        Tensor of shape ``[vocab_size]``.
    """
    if logits.dim() == 3:
        return logits[0, -1, :]
    if logits.dim() == 2:
        return logits[-1, :]
    return logits


def trace_causal_effect(
    model: nn.Module,
    clean_input: dict[str, torch.Tensor],
    corrupted_input: dict[str, torch.Tensor],
    components: Optional[list[str]] = None,
) -> dict[str, CausalEffect]:
    """Iterate patching over every component and return per-component effects.

    The function runs (n_components + 2) forward passes — one corrupted
    pass to cache activations, one clean pass to establish the reference,
    and one patched pass per component.

    Args:
        model: :class:`AdaptiveSLM` (or stub) to trace. Must expose the
            attributes used by :func:`_resolve_submodule` — namely
            ``conditioning_projector`` and ``layers[*].{self_attn,
            cross_attn,ff}``.
        clean_input: Keyword arguments passed to ``model(...)`` for the
            clean pass. Must include ``input_ids`` at minimum.
        corrupted_input: Keyword arguments passed to ``model(...)`` for
            the corrupted pass (typically with a zeroed
            ``adaptation_vector`` and ``user_state``).
        components: Optional subset of canonical components to probe.
            Defaults to :func:`canonical_components` with ``n_layers``
            taken from the model.

    Returns:
        Ordered dict ``component -> CausalEffect`` with one entry per
        traced component. The caller is responsible for downstream
        aggregation / plotting.
    """
    n_layers = len(getattr(model, "layers", []))
    comps = list(components) if components is not None else canonical_components(n_layers)

    # Reference clean run.
    model.eval()
    with torch.no_grad():
        clean_out = model(**clean_input)
    clean_logits = _last_position_logits(
        clean_out[0] if isinstance(clean_out, tuple) else clean_out
    )

    results: dict[str, CausalEffect] = {}
    with ActivationPatcher(model, components=comps) as patcher:
        with torch.no_grad():
            patcher.cache_corrupted(lambda: model(**corrupted_input))

        for comp in comps:
            with torch.no_grad():
                patched = patcher.patch(
                    comp,
                    clean_run=lambda: model(**clean_input),
                )
            patched_logits = _last_position_logits(patched)
            kl = _symmetric_kl(patched_logits, clean_logits)
            l2 = float(torch.norm(patched_logits - clean_logits).item())
            flipped = bool(
                int(patched_logits.argmax().item())
                != int(clean_logits.argmax().item())
            )
            results[comp] = CausalEffect(
                component=comp,
                kl_to_clean=kl,
                logit_l2=l2,
                top1_flipped=flipped,
            )
    return results


__all__ = [
    "ActivationPatcher",
    "CausalEffect",
    "canonical_components",
    "trace_causal_effect",
]
