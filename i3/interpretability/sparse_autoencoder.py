"""Sparse autoencoders for cross-attention-conditioning interpretability.

This module implements a *dictionary-learning* sparse autoencoder (SAE)
of the form popularised by Anthropic's mechanistic-interpretability
programme: a one-hidden-layer autoencoder whose hidden dimension is
larger than its input dimension, trained with an L1 sparsity penalty on
the hidden activations, so that the hidden features form an
*overcomplete, monosemantic* basis for the input distribution.

The I3 use-case is the residual stream of each cross-attention block of
the :class:`~i3.slm.model.AdaptiveSLM`. Feeding the residual-stream
activations into a per-layer SAE produces a dictionary of monosemantic
features that can then be correlated against the eight
``AdaptationVector`` dimensions (``cognitive_load``, ``formality``,
``verbosity``, ``emotionality``, ``directness``, ``emotional_tone``,
``accessibility``, ``reserved``) — which is the job of the companion
module :mod:`i3.interpretability.sae_analysis`.

Architecture
------------
Encoder ``W_e``: ``Linear(d_model -> d_dict) + ReLU``.
Decoder ``W_d``: ``Linear(d_dict -> d_model)`` (optionally tied to
``encoder.weight.T``). Decoder column-norms are projected back to unit
norm after every optimiser step per Bricken et al. (2023) §5.2 — this
removes the trivial sparsity degeneracy where the encoder shrinks its
features and the decoder compensates by scaling its columns up.

Training objective::

    L(x) = ||x - W_d ReLU(W_e (x - b))||^2 + lambda * ||ReLU(W_e (x - b))||_1

Input centring (``normalise_input=True``) subtracts the decoder bias
``b`` from the input before encoding and adds it back after decoding,
which empirically stabilises the L1 penalty (Bricken 2023).

References
----------
- Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly,
  T., et al. (2023). *Towards Monosemanticity: Decomposing Language
  Models With Dictionary Learning.* Anthropic Circuits Thread.
- Templeton, A., Conerly, T., Marcus, J., Bricken, T., et al. (2024).
  *Scaling Monosemanticity: Extracting Interpretable Features from
  Claude 3 Sonnet.* Anthropic Circuits Thread.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023).
  *Sparse Autoencoders Find Highly Interpretable Features in Language
  Models.* arXiv:2309.08600.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SparseAutoencoder.
# ---------------------------------------------------------------------------


class SparseAutoencoder(nn.Module):
    """Dictionary-learning sparse autoencoder.

    The SAE projects a ``d_model``-dimensional activation into a larger
    ``d_dict``-dimensional, ReLU-activated feature space and reconstructs
    the input from that feature space. An L1 penalty on the features
    encourages sparsity so that each feature specialises on a small slice
    of the input distribution (the "monosemanticity" of Bricken 2023).

    Parameters
    ----------
    d_model : int
        Dimensionality of the activations being decomposed. For I3 this
        equals the cross-attention residual-stream width (``d_model`` of
        the :class:`AdaptiveSLM`).
    d_dict : int
        Size of the learned feature dictionary. Must be strictly positive
        and in typical use is 8× ``d_model`` (Bricken 2023).
    sparsity_coef : float, default ``1e-3``
        Coefficient ``lambda`` on the L1 sparsity penalty.
    normalise_input : bool, default ``True``
        If ``True`` the decoder bias is subtracted from the input before
        encoding and added back after decoding (Bricken 2023 §5.2).
    tied_weights : bool, default ``False``
        If ``True`` the decoder weight is kept equal to the encoder
        weight transposed, i.e. ``W_d = W_e.T``. This is rarely used in
        production SAEs but simplifies certain tests and reduces the
        parameter count by roughly a factor of two.

    Attributes
    ----------
    encoder : nn.Linear
        ``d_model -> d_dict`` encoder. ReLU is applied in :meth:`forward`.
    decoder : nn.Linear
        ``d_dict -> d_model`` decoder. Bias is shared with the input
        centring term when ``normalise_input=True``.
    d_model : int
    d_dict : int
    sparsity_coef : float
    normalise_input : bool
    tied_weights : bool
    """

    def __init__(
        self,
        d_model: int,
        d_dict: int,
        sparsity_coef: float = 1e-3,
        normalise_input: bool = True,
        tied_weights: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if d_dict <= 0:
            raise ValueError(f"d_dict must be > 0, got {d_dict}")
        if sparsity_coef < 0.0:
            raise ValueError(
                f"sparsity_coef must be >= 0, got {sparsity_coef}"
            )

        self.d_model: int = int(d_model)
        self.d_dict: int = int(d_dict)
        self.sparsity_coef: float = float(sparsity_coef)
        self.normalise_input: bool = bool(normalise_input)
        self.tied_weights: bool = bool(tied_weights)

        # Encoder has no bias by default; the decoder bias doubles as the
        # input-centring term b. This mirrors Bricken 2023's formulation.
        self.encoder = nn.Linear(d_model, d_dict, bias=True)
        self.decoder = nn.Linear(d_dict, d_model, bias=True)

        # Kaiming init is a reasonable default for ReLU-activated encoders;
        # decoder columns are then explicitly projected to unit norm below.
        nn.init.kaiming_uniform_(self.encoder.weight, a=5 ** 0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5 ** 0.5)
        nn.init.zeros_(self.decoder.bias)

        # Apply the unit-norm constraint once at construction time so the
        # invariant holds even before the first optimiser step.
        with torch.no_grad():
            self._project_decoder_unit_norm()

        if self.tied_weights:
            # When tied, the decoder weight is a view onto encoder.weight.T.
            # We enforce this eagerly here and re-enforce it before each
            # forward pass via ``_maybe_tie``.
            self._maybe_tie()

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, decode, and score the L1 sparsity loss.

        Args:
            x: Input activations of shape ``[batch, d_model]``. Any
                leading shape is flattened for the encoder; callers are
                expected to reshape sequence-dimension tensors before
                invoking this method.

        Returns:
            Tuple ``(reconstruction, features, l1_sparsity_loss)``:

            * ``reconstruction``: Tensor of shape ``[batch, d_model]``.
            * ``features``: Tensor of shape ``[batch, d_dict]`` holding
              the post-ReLU feature activations.
            * ``l1_sparsity_loss``: Scalar tensor equal to
              ``sparsity_coef * mean(sum_j ReLU(W_e x + b_e)_j)``. This
              is returned unreduced so callers can add it to their own
              reconstruction loss.

        Raises:
            ValueError: If ``x`` does not have ``d_model`` as its final
                dimension.
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("x must be a torch.Tensor")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dim must equal d_model={self.d_model}, "
                f"got shape {tuple(x.shape)}"
            )

        self._maybe_tie()

        # Centre the input by the decoder bias if requested. This is the
        # "subtract b then add b back" trick from Bricken 2023.
        if self.normalise_input:
            centred = x - self.decoder.bias
        else:
            centred = x

        pre = F.linear(centred, self.encoder.weight, self.encoder.bias)
        features = F.relu(pre)

        if self.normalise_input:
            reconstruction = (
                F.linear(features, self.decoder.weight) + self.decoder.bias
            )
        else:
            reconstruction = F.linear(
                features, self.decoder.weight, self.decoder.bias
            )

        # Mean L1 per sample so the scale does not depend on batch size.
        l1 = features.abs().sum(dim=-1).mean()
        l1_sparsity_loss = self.sparsity_coef * l1

        return reconstruction, features, l1_sparsity_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the post-ReLU feature activations for ``x``.

        Args:
            x: Input activations of shape ``[batch, d_model]``.

        Returns:
            Feature tensor of shape ``[batch, d_dict]``.
        """
        _, features, _ = self.forward(x)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``d_model``-dim activations from a feature vector.

        Args:
            features: Feature activations of shape ``[batch, d_dict]``.

        Returns:
            Reconstruction of shape ``[batch, d_model]``.
        """
        self._maybe_tie()
        return self.decoder(features)

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-squared-error reconstruction loss.

        Args:
            x: Input activations of shape ``[batch, d_model]``.
            reconstruction: Output of :meth:`forward` of the same shape.

        Returns:
            Scalar tensor with the MSE.
        """
        return F.mse_loss(reconstruction, x)

    @torch.no_grad()
    def project_decoder_unit_norm(self) -> None:
        """Project every decoder column back to unit L2 norm.

        Call this after every optimiser step to enforce the constraint
        recommended by Bricken et al. (2023) §5.2. The method is a thin
        public wrapper around :meth:`_project_decoder_unit_norm`.
        """
        self._project_decoder_unit_norm()

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _project_decoder_unit_norm(self) -> None:
        """In-place unit-norm projection of the decoder columns."""
        w = self.decoder.weight.data  # shape [d_model, d_dict]
        norms = w.norm(dim=0, keepdim=True)
        # Avoid div-by-zero for any column that happens to be exactly 0.
        norms = norms.clamp(min=1e-12)
        w.div_(norms)
        if self.tied_weights:
            # Keep encoder in sync with the new decoder columns.
            self.encoder.weight.data.copy_(w.t())

    def _maybe_tie(self) -> None:
        """Enforce ``W_d = W_e.T`` when ``tied_weights=True``."""
        if not self.tied_weights:
            return
        # Copy encoder.T into decoder.weight in-place so that we preserve
        # the nn.Parameter identity and gradients keep flowing.
        self.decoder.weight.data.copy_(self.encoder.weight.data.t())

    # ------------------------------------------------------------------
    # Representation.
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_dict={self.d_dict}, "
            f"sparsity_coef={self.sparsity_coef}, "
            f"normalise_input={self.normalise_input}, "
            f"tied_weights={self.tied_weights}"
        )


# ---------------------------------------------------------------------------
# SAETrainer.
# ---------------------------------------------------------------------------


@dataclass
class SAETrainingReport:
    """Summary of a single SAE training run.

    Attributes:
        epochs: Number of epochs completed.
        final_loss: Total loss (MSE + L1) at the last step.
        initial_loss: Total loss at the first step (before any updates).
        final_reconstruction_mse: MSE at the last step (unweighted).
        final_mean_sparsity: Fraction of feature entries that are
            exactly zero across the final batch.
        loss_history: Per-epoch mean total loss.
    """

    epochs: int
    final_loss: float
    initial_loss: float
    final_reconstruction_mse: float
    final_mean_sparsity: float
    loss_history: list[float]


class SAETrainer:
    """Mini-batch trainer for a :class:`SparseAutoencoder`.

    The trainer is deliberately minimal — it batches a fixed tensor of
    cached activations, runs Adam for ``epochs`` iterations over shuffled
    mini-batches, enforces the decoder unit-norm constraint after every
    step, and returns both the trained SAE and a summary report.

    Parameters
    ----------
    weight_decay : float, default 0.0
        L2 regularisation on the Adam optimiser. Bricken 2023 recommends
        zero weight decay because the unit-norm projection already acts
        as a norm bound on the decoder.
    device : str | torch.device | None, default ``None``
        Device on which training runs. ``None`` defers to the device of
        the supplied activations tensor.
    seed : int | None, default ``None``
        If set, seeds both the torch global RNG and the mini-batch
        shuffle for deterministic training.
    """

    def __init__(
        self,
        weight_decay: float = 0.0,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        if weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {weight_decay}"
            )
        self.weight_decay: float = float(weight_decay)
        self.device: torch.device | None = (
            torch.device(device) if device is not None else None
        )
        self.seed: int | None = seed

    def fit(
        self,
        activations: torch.Tensor,
        sae: SparseAutoencoder | None = None,
        d_dict: int | None = None,
        sparsity_coef: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        normalise_input: bool = True,
        tied_weights: bool = False,
    ) -> tuple[SparseAutoencoder, SAETrainingReport]:
        """Train an SAE on a cache of activations.

        Args:
            activations: Tensor of shape ``[n_samples, d_model]``. Any
                leading sequence dimension must be flattened in advance.
            sae: Optional pre-constructed :class:`SparseAutoencoder`. If
                ``None`` a new SAE is built using ``d_dict``,
                ``sparsity_coef``, ``normalise_input`` and
                ``tied_weights``.
            d_dict: Overcomplete dictionary size (ignored if ``sae`` is
                supplied). Must be strictly positive.
            sparsity_coef: L1 coefficient for the constructed SAE
                (ignored if ``sae`` is supplied).
            epochs: Number of epochs. Must be >= 1.
            batch_size: Mini-batch size. Capped at the number of samples.
            lr: Adam learning rate. Must be > 0.
            normalise_input: Passed through to the SAE constructor.
            tied_weights: Passed through to the SAE constructor.

        Returns:
            Tuple ``(trained_sae, report)``.

        Raises:
            ValueError: If ``activations`` is not 2-D, ``epochs`` < 1,
                ``batch_size`` < 1, or ``lr`` <= 0.
        """
        if activations.dim() != 2:
            raise ValueError(
                "activations must be 2-D [n_samples, d_model], got "
                f"shape {tuple(activations.shape)}"
            )
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")

        n_samples, d_model = activations.shape
        if sae is None:
            if d_dict is None:
                raise ValueError(
                    "either sae or d_dict must be supplied"
                )
            sae = SparseAutoencoder(
                d_model=d_model,
                d_dict=d_dict,
                sparsity_coef=sparsity_coef,
                normalise_input=normalise_input,
                tied_weights=tied_weights,
            )
        if sae.d_model != d_model:
            raise ValueError(
                f"sae.d_model ({sae.d_model}) != activations "
                f"d_model ({d_model})"
            )

        if self.seed is not None:
            torch.manual_seed(self.seed)
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
        else:
            generator = None

        device = self.device or activations.device
        sae = sae.to(device)
        data = activations.to(device)
        batch_size = min(batch_size, n_samples)

        optimiser = torch.optim.Adam(
            sae.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        loss_history: list[float] = []
        initial_loss: float | None = None
        final_loss = 0.0
        final_recon_mse = 0.0
        final_sparsity = 0.0

        for epoch in range(epochs):
            # Shuffle indices each epoch (deterministic when seeded).
            perm = (
                torch.randperm(n_samples, generator=generator)
                if generator is not None
                else torch.randperm(n_samples)
            )
            epoch_losses: list[float] = []
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch = data[idx]

                reconstruction, features, l1_loss = sae(batch)
                recon_loss = sae.reconstruction_loss(batch, reconstruction)
                loss = recon_loss + l1_loss

                if initial_loss is None:
                    initial_loss = float(loss.detach().item())

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                # Enforce the unit-norm constraint (Bricken 2023 §5.2).
                sae.project_decoder_unit_norm()

                epoch_losses.append(float(loss.detach().item()))
                final_loss = float(loss.detach().item())
                final_recon_mse = float(recon_loss.detach().item())
                final_sparsity = float(
                    (features == 0).float().mean().item()
                )

            loss_history.append(
                sum(epoch_losses) / max(1, len(epoch_losses))
            )

        report = SAETrainingReport(
            epochs=epochs,
            final_loss=final_loss,
            initial_loss=initial_loss if initial_loss is not None else final_loss,
            final_reconstruction_mse=final_recon_mse,
            final_mean_sparsity=final_sparsity,
            loss_history=loss_history,
        )
        return sae, report


# ---------------------------------------------------------------------------
# FeatureDictionary.
# ---------------------------------------------------------------------------


class FeatureDictionary:
    """Post-training container exposing per-feature statistics.

    The dictionary takes a trained :class:`SparseAutoencoder` together
    with the activation cache it was trained on and precomputes the
    feature activations so that subsequent top-k / sparsity / similarity
    queries are O(1) in the number of samples.

    Parameters
    ----------
    sae : SparseAutoencoder
        A trained SAE.
    activations : torch.Tensor
        Tensor of shape ``[n_samples, d_model]`` — the cache of
        residual-stream activations on which queries operate. The same
        activations that were used for training are the typical choice.
    input_ids : torch.Tensor | None, default ``None``
        Optional tensor of shape ``[n_samples, seq_len]`` associating
        each cached activation with the input sequence that produced it.
        Required for :meth:`top_k_activating_inputs` to return something
        more meaningful than integer sample indices.

    Attributes
    ----------
    feature_count : int
        Size of the learned dictionary (``sae.d_dict``).
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        activations: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> None:
        if activations.dim() != 2:
            raise ValueError(
                "activations must be 2-D, got shape "
                f"{tuple(activations.shape)}"
            )
        if activations.shape[-1] != sae.d_model:
            raise ValueError(
                f"activations last dim ({activations.shape[-1]}) must "
                f"equal sae.d_model ({sae.d_model})"
            )

        self.sae: SparseAutoencoder = sae
        self._activations: torch.Tensor = activations.detach()
        self._input_ids: torch.Tensor | None = (
            input_ids.detach() if input_ids is not None else None
        )

        sae.eval()
        with torch.no_grad():
            self._features: torch.Tensor = sae.encode(self._activations).cpu()

    # ------------------------------------------------------------------
    # Properties.
    # ------------------------------------------------------------------

    @property
    def feature_count(self) -> int:
        """Size of the learned feature dictionary."""
        return int(self.sae.d_dict)

    @property
    def features(self) -> torch.Tensor:
        """All precomputed feature activations, shape ``[n_samples, d_dict]``."""
        return self._features

    # ------------------------------------------------------------------
    # Queries.
    # ------------------------------------------------------------------

    def top_k_activating_inputs(
        self,
        feature_idx: int,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return the top-``k`` activating samples for ``feature_idx``.

        Args:
            feature_idx: Zero-based feature index in ``[0, d_dict)``.
            k: Number of top activations to return. Clamped to the
                number of samples.

        Returns:
            List of ``(sample_index, activation)`` pairs sorted in
            descending order of activation. Exactly ``min(k, n_samples)``
            entries.

        Raises:
            ValueError: If ``feature_idx`` is out of range or ``k`` <= 0.
        """
        self._check_feature_idx(feature_idx)
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        k = min(k, self._features.size(0))
        col = self._features[:, feature_idx]
        values, indices = torch.topk(col, k=k, largest=True)
        return [
            (int(i.item()), float(v.item())) for i, v in zip(indices, values)
        ]

    def feature_sparsity(self, feature_idx: int) -> float:
        """Return the fraction of samples on which this feature is zero.

        Args:
            feature_idx: Zero-based feature index.

        Returns:
            Sparsity in ``[0, 1]``: ``1.0`` means the feature never
            fires, ``0.0`` means it always fires.
        """
        self._check_feature_idx(feature_idx)
        col = self._features[:, feature_idx]
        return float((col == 0).float().mean().item())

    def cosine_similarity_between_features(self, i: int, j: int) -> float:
        """Cosine similarity between decoder columns ``i`` and ``j``.

        The decoder column is the natural "direction in residual space"
        associated with a given feature and is the quantity used for
        activation-steering (Turner et al., 2023).

        Args:
            i: First feature index.
            j: Second feature index.

        Returns:
            Cosine similarity in ``[-1, 1]``.
        """
        self._check_feature_idx(i)
        self._check_feature_idx(j)
        w = self.sae.decoder.weight.detach()  # [d_model, d_dict]
        col_i = w[:, i]
        col_j = w[:, j]
        denom = (col_i.norm() * col_j.norm()).clamp(min=1e-12)
        return float((col_i.dot(col_j) / denom).item())

    def decoder_column(self, feature_idx: int) -> torch.Tensor:
        """Return the decoder column for ``feature_idx`` (detached clone)."""
        self._check_feature_idx(feature_idx)
        return (
            self.sae.decoder.weight.detach()[:, feature_idx].clone()
        )

    def mean_activation(self, feature_idx: int) -> float:
        """Mean activation of a feature across all cached samples."""
        self._check_feature_idx(feature_idx)
        return float(self._features[:, feature_idx].mean().item())

    def max_activation(self, feature_idx: int) -> float:
        """Maximum activation of a feature across all cached samples."""
        self._check_feature_idx(feature_idx)
        return float(self._features[:, feature_idx].max().item())

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _check_feature_idx(self, feature_idx: int) -> None:
        if not 0 <= feature_idx < self.feature_count:
            raise ValueError(
                f"feature_idx {feature_idx} out of range "
                f"[0, {self.feature_count})"
            )


# ---------------------------------------------------------------------------
# Convenience.
# ---------------------------------------------------------------------------


def iter_activations_flat(
    activations: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Flatten a sequence of ``[batch, seq, d_model]`` tensors to ``[N, d_model]``.

    Args:
        activations: Iterable of activation tensors of varying leading
            dimensions but identical trailing ``d_model``.

    Returns:
        Single 2-D tensor obtained by reshaping each tensor to
        ``[-1, d_model]`` and concatenating along axis 0.

    Raises:
        ValueError: If the inputs do not share a trailing dimension.
    """
    flat: list[torch.Tensor] = []
    d_model: int | None = None
    for t in activations:
        if not isinstance(t, torch.Tensor):
            raise ValueError("activations must contain torch.Tensor values")
        if d_model is None:
            d_model = t.shape[-1]
        elif t.shape[-1] != d_model:
            raise ValueError(
                "all tensors must share the trailing d_model dimension; "
                f"got {t.shape[-1]} vs {d_model}"
            )
        flat.append(t.reshape(-1, t.shape[-1]))
    if not flat:
        raise ValueError("activations iterable is empty")
    return torch.cat(flat, dim=0)


__all__ = [
    "FeatureDictionary",
    "SAETrainer",
    "SAETrainingReport",
    "SparseAutoencoder",
    "iter_activations_flat",
]
