"""Lightweight residual-stream activation cache for SAE training.

The :class:`ActivationCache` wraps a small PyTorch forward-hook
registration helper that captures the output tensors of named
sub-modules during arbitrary forward passes, then stores them on disk
(optionally sharded) for later consumption by
:class:`~i3.interpretability.sparse_autoencoder.SAETrainer`.

Usage::

    cache = ActivationCache()
    for i, layer in enumerate(model.layers):
        cache.register(layer.cross_attn, f"cross_attn_{i}")

    def data_iter() -> Iterator[dict[str, Any]]:
        for prompt_ids, adapt_vec in dataset:
            yield {"input_ids": prompt_ids, "adaptation_vector": adapt_vec}

    cache.collect(model, data_iter(), max_samples=4096)
    cache.save(Path("cache/residuals.pt"))

On disk each layer is stored as a single ``torch.save``-able dict; with
``shard_size`` set, the payload is split across numbered ``.pt`` files
alongside a tiny ``index.json`` manifest so that very large sweeps do
not require a monolithic tensor in RAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Private mutable storage for one hooked sub-module."""

    layer_name: str
    module: nn.Module
    handle: Optional[torch.utils.hooks.RemovableHandle] = None
    captured: list[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ActivationCache.
# ---------------------------------------------------------------------------


class ActivationCache:
    """Captures residual-stream activations from named sub-modules.

    Attributes
    ----------
    max_samples : int
        Default upper bound on the number of ``[seq_len, d_model]``
        activation rows retained per registered layer. Passing
        ``max_samples`` to :meth:`collect` overrides this per-call.
    """

    def __init__(self, max_samples: int = 10_000) -> None:
        if max_samples <= 0:
            raise ValueError(
                f"max_samples must be > 0, got {max_samples}"
            )
        self.max_samples: int = int(max_samples)
        self._entries: dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Registration.
    # ------------------------------------------------------------------

    def register(self, module: nn.Module, layer_name: str) -> None:
        """Register a sub-module whose output should be captured.

        Args:
            module: The :class:`nn.Module` to hook. Its forward output is
                expected either to be a tensor of shape ``[batch, seq,
                d_model]`` or a tuple whose first element is such a
                tensor.
            layer_name: A stable name used as the key under which the
                captured activations are stored.

        Raises:
            ValueError: If ``layer_name`` is already registered.
        """
        if not isinstance(module, nn.Module):
            raise ValueError(
                f"module must be nn.Module, got {type(module).__name__}"
            )
        if layer_name in self._entries:
            raise ValueError(
                f"layer_name {layer_name!r} already registered"
            )
        self._entries[layer_name] = _CacheEntry(
            layer_name=layer_name,
            module=module,
        )

    @property
    def layer_names(self) -> list[str]:
        """Sorted list of registered layer names."""
        return sorted(self._entries.keys())

    def get(self, layer_name: str) -> torch.Tensor:
        """Return the stacked captured activations for ``layer_name``.

        Args:
            layer_name: Name previously passed to :meth:`register`.

        Returns:
            Tensor of shape ``[n_samples, d_model]``.

        Raises:
            KeyError: If the layer has never been registered or nothing
                was captured for it.
        """
        if layer_name not in self._entries:
            raise KeyError(f"layer {layer_name!r} not registered")
        captured = self._entries[layer_name].captured
        if not captured:
            raise KeyError(
                f"no activations captured for layer {layer_name!r}"
            )
        return torch.cat(captured, dim=0)

    # ------------------------------------------------------------------
    # Collection.
    # ------------------------------------------------------------------

    def collect(
        self,
        model: nn.Module,
        data_iterator: Iterable[dict[str, Any]],
        max_samples: Optional[int] = None,
    ) -> dict[str, int]:
        """Run ``model`` on each element of ``data_iterator`` and cache outputs.

        Each element must be a ``dict`` of keyword arguments suitable for
        ``model(**kwargs)``. The function attaches forward hooks to every
        registered sub-module, flattens captured activations from
        ``[batch, seq, d_model]`` (or ``[batch, d_model]``) into
        ``[N, d_model]`` chunks, and stops once any one layer accumulates
        ``max_samples`` rows.

        Args:
            model: The model to run. Must be an :class:`nn.Module`.
            data_iterator: Iterable of ``dict`` kwargs. Exhausted in
                order; entries after the ``max_samples`` threshold is hit
                are skipped.
            max_samples: Per-layer cap for this call. Defaults to
                ``self.max_samples``.

        Returns:
            Mapping ``layer_name -> n_rows_captured``.

        Raises:
            RuntimeError: If no layers have been registered.
        """
        if not self._entries:
            raise RuntimeError("no layers registered; call register() first")

        cap = int(max_samples if max_samples is not None else self.max_samples)
        if cap <= 0:
            raise ValueError(f"max_samples must be > 0, got {cap}")

        # Attach hooks.
        for name, entry in self._entries.items():
            entry.handle = entry.module.register_forward_hook(
                self._make_hook(entry)
            )

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for kwargs in data_iterator:
                    if self._all_layers_full(cap):
                        break
                    model(**kwargs)
        finally:
            for entry in self._entries.values():
                if entry.handle is not None:
                    try:
                        entry.handle.remove()
                    except RuntimeError:  # pragma: no cover - defensive
                        pass
                    entry.handle = None
            if was_training:
                model.train()

        return {
            name: sum(t.size(0) for t in entry.captured)
            for name, entry in self._entries.items()
        }

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
        shard_size: Optional[int] = None,
    ) -> None:
        """Persist the captured activations.

        Args:
            path: Destination file path (``.pt``). When ``shard_size`` is
                set the base path is treated as a directory and
                individual shards are written as ``shard_0000.pt``,
                ``shard_0001.pt``, ... along with an ``index.json``
                manifest.
            shard_size: Optional per-layer shard size. If ``None`` the
                entire cache is written as a single ``torch.save`` blob.

        Raises:
            RuntimeError: If no activations have been captured.
        """
        payload = {
            name: self.get(name) for name in self._entries if self._entries[name].captured
        }
        if not payload:
            raise RuntimeError("nothing to save; collect() was never called")

        dst = Path(path)
        if shard_size is None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, dst)
            return

        if shard_size <= 0:
            raise ValueError(
                f"shard_size must be > 0, got {shard_size}"
            )
        dst.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[str]] = {}
        for name, tensor in payload.items():
            n_shards = (tensor.size(0) + shard_size - 1) // shard_size
            shard_paths: list[str] = []
            for i in range(n_shards):
                start = i * shard_size
                end = min(tensor.size(0), start + shard_size)
                shard_name = f"{name}_shard_{i:04d}.pt"
                torch.save(tensor[start:end].clone(), dst / shard_name)
                shard_paths.append(shard_name)
            manifest[name] = shard_paths
        (dst / "index.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ActivationCache":
        """Load activations previously written by :meth:`save`.

        Args:
            path: Either a ``.pt`` file produced by an un-sharded save or
                a directory containing ``index.json``.

        Returns:
            New :class:`ActivationCache` populated with the loaded
            tensors. Registered sub-module hooks are NOT restored — this
            is a passive container suitable for training.
        """
        src = Path(path)
        cache = cls.__new__(cls)
        cache.max_samples = 10_000
        cache._entries = {}

        if src.is_file():
            payload = torch.load(src, map_location="cpu")
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected dict in {src}, got {type(payload).__name__}"
                )
            for name, tensor in payload.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(
                        f"expected tensor under {name!r}, got {type(tensor).__name__}"
                    )
                placeholder = _CacheEntry(
                    layer_name=name,
                    module=nn.Identity(),
                    captured=[tensor],
                )
                cache._entries[name] = placeholder
            return cache

        if not src.is_dir():
            raise FileNotFoundError(f"activation cache not found at {src}")
        manifest_path = src / "index.json"
        if not manifest_path.exists():
            raise ValueError(
                f"sharded cache missing index.json in {src}"
            )
        manifest = json.loads(manifest_path.read_text())
        for name, shards in manifest.items():
            parts: list[torch.Tensor] = []
            for shard_name in shards:
                parts.append(torch.load(src / shard_name, map_location="cpu"))
            placeholder = _CacheEntry(
                layer_name=name,
                module=nn.Identity(),
                captured=[torch.cat(parts, dim=0)],
            )
            cache._entries[name] = placeholder
        return cache

    # ------------------------------------------------------------------
    # Context-manager: auto-remove hooks if used as ``with`` block.
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationCache":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        for entry in self._entries.values():
            if entry.handle is not None:
                try:
                    entry.handle.remove()
                except RuntimeError:  # pragma: no cover - defensive
                    pass
                entry.handle = None

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _make_hook(self, entry: _CacheEntry) -> Any:
        """Return a forward hook that appends the flattened output tensor."""

        cap_limit = self.max_samples

        def _hook(_m: nn.Module, _inp: tuple, output: object) -> None:
            tensor = _extract_primary_tensor(output)
            if tensor is None:
                return
            flat = tensor.reshape(-1, tensor.shape[-1]).detach().cpu()
            current_total = sum(t.size(0) for t in entry.captured)
            room = cap_limit - current_total
            if room <= 0:
                return
            if flat.size(0) > room:
                flat = flat[:room]
            entry.captured.append(flat)

        return _hook

    def _all_layers_full(self, cap: int) -> bool:
        for entry in self._entries.values():
            total = sum(t.size(0) for t in entry.captured)
            if total < cap:
                return False
        return True


def _extract_primary_tensor(output: object) -> Optional[torch.Tensor]:
    """Return the first tensor in a possibly-tuple forward output.

    Args:
        output: The raw module output. Supported shapes are a bare
            tensor or a tuple whose first element is a tensor.

    Returns:
        The tensor if one was found, else ``None``.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output:
        head = output[0]
        if isinstance(head, torch.Tensor):
            return head
    return None


# ---------------------------------------------------------------------------
# Convenience iterator for SAE training pipelines.
# ---------------------------------------------------------------------------


def iter_prompt_adaptation_pairs(
    prompt_input_ids: list[torch.Tensor],
    adaptation_vectors: list[torch.Tensor],
    user_state_dim: int = 64,
) -> Iterator[dict[str, Any]]:
    """Zip prompts and adaptation vectors into ``model(**kwargs)`` dicts.

    Args:
        prompt_input_ids: List of 1-D tensors of token ids.
        adaptation_vectors: List of 1-D tensors of length ``adaptation_dim``.
        user_state_dim: Dimension of the user-state zero tensor to attach
            to each forward call. Defaults to 64 (I3 default).

    Yields:
        Keyword-argument dicts of the form expected by
        :meth:`AdaptiveSLM.forward`.
    """
    if len(prompt_input_ids) != len(adaptation_vectors):
        raise ValueError(
            "prompt_input_ids and adaptation_vectors must have equal length"
        )
    for ids, adapt in zip(prompt_input_ids, adaptation_vectors):
        yield {
            "input_ids": ids.unsqueeze(0) if ids.dim() == 1 else ids,
            "adaptation_vector": (
                adapt.unsqueeze(0) if adapt.dim() == 1 else adapt
            ),
            "user_state": torch.zeros(1, user_state_dim),
        }


__all__ = [
    "ActivationCache",
    "iter_prompt_adaptation_pairs",
]
