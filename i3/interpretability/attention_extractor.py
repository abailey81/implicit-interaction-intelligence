"""Cross-attention weight extractor via forward hooks.

Registers :meth:`torch.nn.Module.register_forward_hook` on every
:class:`~i3.slm.cross_attention.MultiHeadCrossAttention` sub-module of a
loaded :class:`~i3.slm.model.AdaptiveSLM`.  Each hook records the
cross-attention weight tensor for that layer during a forward pass.

Usage::

    with CrossAttentionExtractor(model) as extractor:
        model(input_ids, adaptation_vector, user_state)
    attention_maps = extractor.get_attention_maps()
    # list[Tensor] of length n_layers, each shaped [heads, T, n_cond]

The extractor is a context manager that cleans up its hooks on exit, so
they cannot leak into subsequent forward passes or be accidentally
pickled alongside the model.  The design follows the standard PyTorch
hook-based interpretability pattern documented in
Paszke et al. (2019, "PyTorch: An Imperative Style, High-Performance
Deep Learning Library").

References
----------
- Paszke, A. et al. (2019).  *PyTorch: An Imperative Style, High-
  Performance Deep Learning Library*.  NeurIPS 2019.
- Vaswani, A. et al. (2017).  *Attention Is All You Need*.  NeurIPS 2017
  -- original source of multi-head attention whose weights are
  extracted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

import torch
import torch.nn as nn


@dataclass
class ExtractedAttention:
    """One extracted cross-attention snapshot.

    Attributes:
        layer_index: Zero-based index of the
            :class:`AdaptiveTransformerBlock` that produced the
            attention.
        weights: Cross-attention weights of shape
            ``[heads, seq_len, n_cond]`` (the batch dim has been
            squeezed out if it was 1, otherwise it is kept as
            ``[batch, heads, seq_len, n_cond]``).
    """

    layer_index: int
    weights: torch.Tensor


@dataclass
class _ExtractorState:
    """Private mutable state kept by :class:`CrossAttentionExtractor`.

    Using a dataclass instead of a dict preserves type information and
    makes the extractor reproducible under :mod:`copy.deepcopy`.
    """

    handles: list[torch.utils.hooks.RemovableHandle] = field(
        default_factory=list
    )
    captured: list[torch.Tensor | None] = field(default_factory=list)


class CrossAttentionExtractor:
    """Context-manager hook that records cross-attention weights per layer.

    Parameters
    ----------
    model : nn.Module
        The :class:`~i3.slm.model.AdaptiveSLM` (or any module whose
        descendants include
        :class:`~i3.slm.cross_attention.MultiHeadCrossAttention`
        instances).  The extractor does not own or modify the model --
        it only attaches forward hooks.
    squeeze_batch : bool, default False
        If True, attention tensors with batch dimension 1 are returned
        with the batch dimension removed.

    Attributes
    ----------
    n_layers : int
        Number of cross-attention modules detected at construction time.

    Example
    -------
    >>> with CrossAttentionExtractor(model) as ex:
    ...     model(ids, adapt, state)
    >>> maps = ex.get_attention_maps()
    >>> assert len(maps) == model.config.n_layers
    """

    def __init__(self, model: nn.Module, squeeze_batch: bool = False) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module, got {type(model).__name__}"
            )
        self._model = model
        self._squeeze_batch = bool(squeeze_batch)
        self._state = _ExtractorState()
        # Locate cross-attention modules at construction time so callers
        # can interrogate ``n_layers`` even before entering the context.
        self._cross_modules: list[tuple[int, nn.Module]] = self._find_cross_modules()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of cross-attention modules that will be hooked."""
        return len(self._cross_modules)

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Return the latest captured attention weights, one per layer.

        Returns:
            List of length :attr:`n_layers` containing the most recent
            cross-attention weight tensors.  Each tensor has shape
            ``[heads, seq_len, n_cond]`` when ``squeeze_batch=True``, or
            ``[batch, heads, seq_len, n_cond]`` otherwise.  Layers whose
            forward pass was not invoked (e.g. due to early stopping)
            contain a zero-element tensor as a placeholder.

        Raises:
            RuntimeError: If called before the context manager has been
                entered AND before a forward pass has run.
        """
        return [
            (
                t
                if t is not None
                else torch.empty(0)
            )
            for t in self._state.captured
        ]

    def get_extracted(self) -> list[ExtractedAttention]:
        """Structured variant of :meth:`get_attention_maps`.

        Returns:
            List of :class:`ExtractedAttention` records, one per hooked
            cross-attention module.
        """
        return [
            ExtractedAttention(layer_index=i, weights=t)
            for i, t in enumerate(self.get_attention_maps())
        ]

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> CrossAttentionExtractor:
        """Register one forward hook per cross-attention module."""
        # Initialise the captured slot list to match n_layers so indexing
        # via layer_index is always safe.
        self._state.captured = [None] * len(self._cross_modules)
        for layer_idx, (_, mod) in enumerate(self._cross_modules):
            handle = mod.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._state.handles.append(handle)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Remove all hooks registered during :meth:`__enter__`.

        This is idempotent -- calling ``__exit__`` twice is harmless --
        so catching exceptions in a nested ``with`` block does not leave
        dangling hooks on the model.
        """
        # SEC: best-effort hook cleanup -- a failing ``.remove()`` on one
        # handle must not prevent the others from being removed, or the
        # caller's model will be left with phantom hooks that pin
        # references to the extractor and leak memory.
        for h in self._state.handles:
            try:
                h.remove()
            except Exception:  # pragma: no cover - defensive
                continue
        self._state.handles = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_cross_modules(self) -> list[tuple[int, nn.Module]]:
        """Locate every ``MultiHeadCrossAttention`` descendant in the model.

        Returns:
            Ordered list of (depth-first index, module) tuples.  The
            index is used solely for stable ordering within a single
            extractor instance.

        Notes:
            Uses a late import of
            :class:`~i3.slm.cross_attention.MultiHeadCrossAttention` to
            avoid a package-level circular import on
            ``i3.interpretability`` <-> ``i3.slm``.  Falls back to a
            name-based match if the import fails, so the extractor can
            still be used against a stubbed model in unit tests.
        """
        try:
            from i3.slm.cross_attention import MultiHeadCrossAttention
        except Exception:  # pragma: no cover - defensive
            MultiHeadCrossAttention = None  # type: ignore[assignment]

        found: list[tuple[int, nn.Module]] = []
        for i, mod in enumerate(self._model.modules()):
            if (MultiHeadCrossAttention is not None and isinstance(
                mod, MultiHeadCrossAttention
            )) or (
                MultiHeadCrossAttention is None
                and type(mod).__name__ == "MultiHeadCrossAttention"
            ):
                found.append((i, mod))
        return found

    def _make_hook(
        self, layer_idx: int
    ) -> callable[[nn.Module, tuple, object], None]:
        """Produce a forward hook closure bound to ``layer_idx``.

        The hook extracts the attention-weight tensor from the module's
        output.  :class:`MultiHeadCrossAttention.forward` is documented
        to return ``(output, attn_weights)`` -- we store the second
        element.  Non-conforming modules are tolerated: the hook simply
        stores a zero tensor so downstream code never encounters
        ``None``.
        """
        def _hook(
            _module: nn.Module,
            _inputs: tuple,
            output: object,
        ) -> None:
            weights: torch.Tensor | None = None
            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if isinstance(candidate, torch.Tensor):
                    weights = candidate
            elif isinstance(output, torch.Tensor):
                weights = output
            if weights is None:
                # SEC: defensive fallback so `.captured[i]` is never None
                # for a layer that actually ran.
                self._state.captured[layer_idx] = torch.empty(0)
                return

            # Detach so we do not build a graph through the hook, and
            # move to CPU to avoid pinning GPU memory after the forward
            # pass completes.
            w = weights.detach().cpu()
            if self._squeeze_batch and w.dim() == 4 and w.size(0) == 1:
                w = w.squeeze(0)
            self._state.captured[layer_idx] = w

        return _hook


__all__ = ["CrossAttentionExtractor", "ExtractedAttention"]
