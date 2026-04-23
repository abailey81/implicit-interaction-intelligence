"""Linear probing classifiers for I³'s conditioning representation.

Trains a bank of :class:`LinearProbe` modules — one per
(``AdaptationVector`` dimension, transformer layer) pair — that predict
the adaptation dimension from the hidden-state tensor produced by that
layer. Per-layer per-dimension probe R² scores give the classic
"what-where" picture of linearly decodable information, following
Alain & Bengio (2016) and Hewitt & Liang (2019).

Notes on random-init caveat
---------------------------

Probes on random-init transformer features answer a weaker but still
useful question: **how much of the adaptation signal survives the
random mixing of self-attention and the non-linear MLP layers?** On a
trained model the same code reports the usual Alain-Bengio decoding
curve; on a random-init model it reports the architectural *capacity*
for the information to flow through. The report produced by
``scripts/experiments/interpretability_study.py`` flags this in the "Threats to
Validity" section.

References
----------
- Alain, G., & Bengio, Y. (2016). *Understanding Intermediate Layers
  Using Linear Classifier Probes.* arXiv:1610.01644.
- Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes
  with Control Tasks.* EMNLP 2019.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from i3.interpretability.feature_attribution import ADAPTATION_DIMS

# Soft-import pandas. Any non-dataframe consumer can still work off the
# returned dict.
try:  # pragma: no cover - import path exercised in tests
    import pandas as _pd
    _PD_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - pandas not installed
    _pd = None  # type: ignore[assignment]
    _PD_AVAILABLE = False


# ---------------------------------------------------------------------------
# LinearProbe.
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """A bias-free linear regression probe.

    The probe maps a pooled hidden state ``[d_model]`` to a scalar
    prediction of one :class:`AdaptationVector` dimension. Training is
    plain MSE with L2 regularisation, keeping the probe in the
    linear-decoding regime advocated by Alain & Bengio (2016).

    Parameters
    ----------
    d_model : int
        Dimensionality of the pooled hidden state.
    bias : bool, default True
        Whether to include a bias term. Hewitt & Liang (2019) show that
        bias-inclusion matters for discriminating learned signal from
        probe capacity; we default to ``True`` to keep the baseline
        faithful to common practice but tests can disable it to
        replicate their control-task protocol.
    """

    def __init__(self, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=bias)
        nn.init.zeros_(self.linear.weight)
        if bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the target adaptation dimension.

        Args:
            x: Pooled hidden state, shape ``[batch, d_model]``.

        Returns:
            Scalar predictions, shape ``[batch]``.
        """
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset type.
# ---------------------------------------------------------------------------


@dataclass
class ProbingExample:
    """One labelled example in a probing dataset.

    Attributes:
        input_ids: Tensor of shape ``[seq_len]``.
        adaptation_vector: Tensor of shape ``[adaptation_dim]``.
        user_state: Optional tensor of shape ``[user_state_dim]``.
    """

    input_ids: torch.Tensor
    adaptation_vector: torch.Tensor
    user_state: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# ProbingSuite.
# ---------------------------------------------------------------------------


class ProbingSuite:
    """Train and evaluate linear probes across transformer layers.

    Parameters
    ----------
    n_epochs : int, default 150
        Number of gradient steps per probe. 100–200 is typical for
        linear regression on pooled hidden states.
    lr : float, default 1e-2
        Learning rate for the Adam optimiser.
    weight_decay : float, default 1e-4
        L2 regularisation strength.
    test_fraction : float, default 0.25
        Held-out fraction used to compute R². Drawn deterministically
        from the end of the dataset to avoid pandas/numpy dependence.
    pool : str, default ``"mean"``
        Pooling over the sequence dimension before the probe. One of
        ``"mean"``, ``"last"``, ``"max"``.
    """

    def __init__(
        self,
        n_epochs: int = 150,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        test_fraction: float = 0.25,
        pool: str = "mean",
    ) -> None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(
                f"test_fraction must be in (0, 1), got {test_fraction}"
            )
        if pool not in {"mean", "last", "max"}:
            raise ValueError(
                f"pool must be one of 'mean'|'last'|'max', got {pool!r}"
            )
        self.n_epochs: int = int(n_epochs)
        self.lr: float = float(lr)
        self.weight_decay: float = float(weight_decay)
        self.test_fraction: float = float(test_fraction)
        self.pool: str = pool

    # ------------------------------------------------------------------
    # Main API.
    # ------------------------------------------------------------------

    def train_probes(
        self,
        model: nn.Module,
        adaptation_dataset: Sequence[ProbingExample],
        target_dimension: str,
        layer_indices: Optional[list[int]] = None,
    ) -> dict[int, float]:
        """Train one probe per layer and return held-out R² per layer.

        Args:
            model: :class:`AdaptiveSLM` (or stub). Its forward must
                return ``(logits, layer_info)`` where ``layer_info`` has
                ``"layer_{i}"`` entries. In addition the caller must be
                able to recover per-layer hidden states; this method
                uses forward hooks on ``model.layers[i]`` to capture
                outputs rather than relying on the public return value.
            adaptation_dataset: Sequence of :class:`ProbingExample`.
                Must contain at least one sample whose
                ``adaptation_vector`` varies along ``target_dimension``.
            target_dimension: Name of the target adaptation dimension,
                e.g. ``"cognitive_load"``. Must be in
                :data:`i3.interpretability.feature_attribution.ADAPTATION_DIMS`.
            layer_indices: Optional list of layer indices to probe.
                Defaults to ``range(n_layers)``.

        Returns:
            Mapping ``{layer_idx: r_squared_on_held_out}``. R² is
            unclipped — values may go slightly negative on low-signal
            layers / random-init models. Call sites typically clip at
            zero for plotting.

        Raises:
            ValueError: If ``target_dimension`` is not recognised or
                ``adaptation_dataset`` is empty.
        """
        if not adaptation_dataset:
            raise ValueError("adaptation_dataset is empty")
        if target_dimension not in ADAPTATION_DIMS:
            raise ValueError(
                f"target_dimension {target_dimension!r} not in ADAPTATION_DIMS"
            )
        target_idx = ADAPTATION_DIMS.index(target_dimension)

        layers = getattr(model, "layers", None)
        if layers is None:
            raise AttributeError("model has no 'layers' attribute")

        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        hidden_by_layer, targets = self._collect_activations(
            model=model,
            dataset=list(adaptation_dataset),
            target_idx=target_idx,
            layer_indices=layer_indices,
        )

        results: dict[int, float] = {}
        for li in layer_indices:
            hs = hidden_by_layer[li]
            r2 = self._fit_and_score(hs, targets)
            results[li] = r2
        return results

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _collect_activations(
        self,
        model: nn.Module,
        dataset: list[ProbingExample],
        target_idx: int,
        layer_indices: Iterable[int],
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Forward every example once and capture pooled hidden states.

        Args:
            model: The :class:`AdaptiveSLM` under probe.
            dataset: List of probing examples.
            target_idx: Index of the target dimension in
                :data:`ADAPTATION_DIMS`.
            layer_indices: Layers to collect.

        Returns:
            Tuple ``(hidden_by_layer, targets)`` where ``hidden_by_layer[i]``
            is a tensor of shape ``[N, d_model]`` and ``targets`` is a
            tensor of shape ``[N]``.
        """
        layers = model.layers  # type: ignore[attr-defined]
        layer_indices = list(layer_indices)
        capture: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}

        handles: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(idx: int) -> object:
            def _hook(_m: nn.Module, _inp: tuple, out: object) -> None:
                tensor: Optional[torch.Tensor] = None
                if isinstance(out, tuple) and out:
                    head = out[0]
                    if isinstance(head, torch.Tensor):
                        tensor = head
                elif isinstance(out, torch.Tensor):
                    tensor = out
                if tensor is None:
                    return
                pooled = self._pool(tensor).detach().cpu()
                capture[idx].append(pooled)

            return _hook

        for i in layer_indices:
            handles.append(layers[i].register_forward_hook(_make_hook(i)))

        target_values: list[float] = []
        model.eval()
        try:
            for example in dataset:
                input_ids = example.input_ids.unsqueeze(0)
                adapt = example.adaptation_vector.unsqueeze(0)
                user = (
                    example.user_state.unsqueeze(0)
                    if example.user_state is not None
                    else None
                )
                with torch.no_grad():
                    model(input_ids, adapt, user)
                target_values.append(
                    float(example.adaptation_vector[target_idx].item())
                )
        finally:
            for h in handles:
                try:
                    h.remove()
                except RuntimeError:  # pragma: no cover - defensive
                    continue

        hidden_by_layer: dict[int, torch.Tensor] = {}
        for i in layer_indices:
            stacked = torch.cat(capture[i], dim=0)  # [N, d_model]
            hidden_by_layer[i] = stacked
        targets = torch.tensor(target_values, dtype=torch.float32)
        return hidden_by_layer, targets

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Reduce ``[batch, seq, d]`` to ``[batch, d]``.

        Args:
            hidden: Batched hidden-state tensor.

        Returns:
            Pooled representation.
        """
        if hidden.dim() == 2:
            return hidden
        if self.pool == "mean":
            return hidden.mean(dim=1)
        if self.pool == "last":
            return hidden[:, -1, :]
        return hidden.max(dim=1).values

    def _fit_and_score(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Fit a :class:`LinearProbe` and return R² on the held-out split.

        Args:
            features: Tensor of shape ``[N, d_model]``.
            targets: Tensor of shape ``[N]``.

        Returns:
            Held-out R². May be negative if the probe is worse than
            predicting the mean.
        """
        n = features.size(0)
        n_test = max(1, int(round(n * self.test_fraction)))
        n_train = n - n_test
        if n_train <= 0:
            return float("nan")

        x_train, y_train = features[:n_train], targets[:n_train]
        x_test, y_test = features[n_train:], targets[n_train:]

        probe = LinearProbe(d_model=features.size(1))
        opt = torch.optim.Adam(
            probe.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.n_epochs):
            opt.zero_grad()
            pred = probe(x_train)
            loss = F.mse_loss(pred, y_train)
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            pred_test = probe(x_test)
            ss_res = torch.sum((y_test - pred_test) ** 2)
            ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
        if float(ss_tot.item()) == 0.0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot).item())


# ---------------------------------------------------------------------------
# Selectivity table.
# ---------------------------------------------------------------------------


def compute_probe_selectivity(
    probe_results: dict[str, dict[int, float]],
):
    """Turn a dim->layer->R² dict into a dimension × layer table.

    The returned object is a :class:`pandas.DataFrame` if pandas is
    importable, otherwise a dict-of-dicts with the same structure
    (rows = adaptation dimensions, columns = layer indices).

    Args:
        probe_results: ``{dimension_name: {layer_idx: r_squared}}``.

    Returns:
        Either a :class:`pandas.DataFrame` with dimensions as the index
        and layer indices as the columns, or a plain ``dict`` if pandas
        is unavailable.
    """
    # Collect the full index/columns so missing values become NaN / 0.0.
    dims = list(probe_results.keys())
    layers: set[int] = set()
    for row in probe_results.values():
        layers.update(row.keys())
    ordered_layers = sorted(layers)

    if _PD_AVAILABLE:
        data = {
            li: [probe_results[d].get(li, float("nan")) for d in dims]
            for li in ordered_layers
        }
        frame = _pd.DataFrame(data, index=dims)
        frame.columns.name = "layer"
        frame.index.name = "adaptation_dim"
        return frame

    return {
        d: {li: probe_results[d].get(li, float("nan")) for li in ordered_layers}
        for d in dims
    }


__all__ = [
    "LinearProbe",
    "ProbingExample",
    "ProbingSuite",
    "compute_probe_selectivity",
]
