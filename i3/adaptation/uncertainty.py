"""Monte-Carlo-dropout uncertainty quantification for the adaptation layer.

Pipeline refresher
------------------

The I3 adaptation pipeline is::

    feature_window  --TCN-->  embedding  --controller-->  AdaptationVector

Both the TCN encoder and several sub-components of the
:class:`~i3.adaptation.controller.AdaptationController` contain dropout
layers that are normally inactive at inference time. This module turns
those dropout layers back ON for a bounded number of stochastic forward
passes (MC Dropout, Gal & Ghahramani 2016) and aggregates the resulting
distribution of :class:`AdaptationVector` samples into:

* a mean :class:`AdaptationVector`,
* a per-dimension standard deviation,
* a per-dimension 95 % confidence interval,
* a sample count.

Two policy helpers then translate the uncertainty into *action-level*
semantics:

* :func:`confidence_threshold_policy` — "are we confident enough to
  adapt?".
* :func:`refuse_when_unsure_mask` — "clamp uncertain dimensions back to
  a neutral baseline". This is the architectural instantiation of the
  principle in ``docs/responsible_ai/accessibility_statement.md`` §2:
  the system must not silently override the user's default experience
  when the underlying signal is ambiguous.

References
----------
- Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian
  Approximation: Representing Model Uncertainty in Deep Learning.*
  ICML 2016. https://proceedings.mlr.press/v48/gal16.html
- Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in
  Bayesian Deep Learning for Computer Vision?* NeurIPS 2017.
  https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector, StyleVector

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from i3.adaptation.controller import AdaptationController
    from i3.encoder.tcn import TemporalConvNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Uncertainty schema
# ---------------------------------------------------------------------------

#: Canonical ordering of the 8 :class:`AdaptationVector` dimensions. The
#: :class:`UncertainAdaptationVector` Pydantic schema below uses this
#: ordering for every per-dimension array (std, ci, etc.) so that
#: downstream code can zip them against ``ADAPTATION_DIMS``.
ADAPTATION_DIMS: tuple[str, ...] = (
    "cognitive_load",
    "formality",
    "verbosity",
    "emotionality",
    "directness",
    "emotional_tone",
    "accessibility",
    "reserved",
)


class DimensionInterval(BaseModel):
    """95 % confidence interval on one AdaptationVector dimension.

    Attributes:
        lower: Lower bound of the 95 % CI (inclusive). Clamped to
            ``[0, 1]`` so CI bounds never escape the valid adaptation
            range.
        upper: Upper bound of the 95 % CI (inclusive). Clamped to
            ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: float = Field(..., ge=0.0, le=1.0)
    upper: float = Field(..., ge=0.0, le=1.0)


class UncertainAdaptationVector(BaseModel):
    """Mean :class:`AdaptationVector` decorated with per-dimension uncertainty.

    This is the public return type of :meth:`MCDropoutAdaptationEstimator
    .estimate`. It carries exactly the information needed to decide, per
    dimension, whether the system is confident enough to act on the
    adaptation (see :func:`confidence_threshold_policy` and
    :func:`refuse_when_unsure_mask`).

    Attributes:
        mean: The per-dimension *mean* AdaptationVector serialised via
            :meth:`AdaptationVector.to_dict`.
        std: Per-dimension standard deviation in :data:`ADAPTATION_DIMS`
            order. Length must equal 8.
        ci: Per-dimension 95 % confidence interval in
            :data:`ADAPTATION_DIMS` order. Length must equal 8.
        sample_count: Number of MC Dropout forward passes used to
            estimate ``mean``, ``std`` and ``ci``. Strictly positive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: dict[str, object] = Field(
        ..., description="AdaptationVector.to_dict() of the sample mean."
    )
    std: list[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Per-dimension std in ADAPTATION_DIMS order.",
    )
    ci: list[DimensionInterval] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Per-dimension 95 % CI in ADAPTATION_DIMS order.",
    )
    sample_count: int = Field(..., gt=0)

    # ------------------------------------------------------------------
    # Convenience access.
    # ------------------------------------------------------------------

    def mean_vector(self) -> AdaptationVector:
        """Return the mean as a live :class:`AdaptationVector`."""
        return AdaptationVector.from_dict(dict(self.mean))

    def std_by_name(self) -> dict[str, float]:
        """Return per-dimension std keyed by canonical name."""
        return {name: float(self.std[i]) for i, name in enumerate(ADAPTATION_DIMS)}

    def ci_by_name(self) -> dict[str, DimensionInterval]:
        """Return per-dimension CI keyed by canonical name."""
        return {name: self.ci[i] for i, name in enumerate(ADAPTATION_DIMS)}


# ---------------------------------------------------------------------------
# MC Dropout estimator
# ---------------------------------------------------------------------------


def _enable_dropout_only(module: nn.Module) -> None:
    """Turn on dropout layers while keeping every other module in eval mode.

    This follows the Gal & Ghahramani (2016) convention: batch-norm,
    weight norm, etc. stay in their evaluation configuration; only the
    stochastic *Dropout* layers are re-enabled so the forward pass
    samples from the approximate posterior.

    Args:
        module: Any :class:`torch.nn.Module`. Walked recursively.
    """
    for sub in module.modules():
        if isinstance(
            sub,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
        ):
            sub.train()


def _override_dropout_probability(module: nn.Module, p: float) -> list[tuple[nn.Module, float]]:
    """Temporarily override the dropout probability of every dropout layer.

    Args:
        module: Module tree whose dropout layers will be mutated.
        p: New dropout probability in ``[0, 1)``.

    Returns:
        List of ``(layer, old_p)`` tuples that the caller must pass to
        :func:`_restore_dropout_probability` to undo the override.
    """
    if not (0.0 <= p < 1.0):
        raise ValueError(f"dropout_p must be in [0, 1), got {p!r}")

    saved: list[tuple[nn.Module, float]] = []
    for sub in module.modules():
        if isinstance(
            sub,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
        ):
            saved.append((sub, float(getattr(sub, "p", 0.0))))
            sub.p = float(p)
    return saved


def _restore_dropout_probability(
    saved: Sequence[tuple[nn.Module, float]],
) -> None:
    """Restore the dropout probability overrides made by :func:`_override_dropout_probability`."""
    for layer, old_p in saved:
        layer.p = float(old_p)


@dataclass
class _ModeSnapshot:
    """Captured training/eval mode of an :class:`nn.Module` tree.

    ``_ModeSnapshot`` is a tiny helper used by
    :class:`MCDropoutAdaptationEstimator` to guarantee that even if the
    forward pass raises, the encoder is always restored to the exact
    training-mode flags it had on entry.
    """

    module: nn.Module
    was_training: bool

    def restore(self) -> None:
        """Restore every sub-module's training flag to its captured value."""
        self.module.train(self.was_training)


class MCDropoutAdaptationEstimator:
    """Compute MC-Dropout confidence intervals on the adaptation pathway.

    The estimator wraps the TCN encoder and an
    :class:`AdaptationController` pair. On :meth:`estimate` it:

    1. Snapshots the training/eval mode of the encoder (and of the
       controller's adapters, if any are :class:`nn.Module` subclasses).
    2. Temporarily puts every :class:`nn.Dropout*` layer in ``train``
       mode (Gal & Ghahramani 2016) and overrides its ``p`` to
       ``dropout_p`` so the sampling variance is well-controlled even
       on models whose training dropout was 0.
    3. Runs ``n_samples`` stochastic forward passes through
       ``encoder`` (+ optional ``feature_to_controller_inputs``
       projection, see below) and through ``controller.compute``.
    4. Aggregates the resulting stream of :class:`AdaptationVector`
       into a :class:`UncertainAdaptationVector`.
    5. Restores the encoder (and any dropout overrides) before
       returning — the estimator never leaks training mode into the
       production inference path.

    Args:
        encoder: The :class:`~i3.encoder.tcn.TemporalConvNet`. Must
            accept a tensor of shape ``[batch, seq_len, input_dim]`` or
            ``[seq_len, input_dim]``.
        controller: The :class:`AdaptationController`. Only its
            ``compute`` method is called.
        n_samples: Number of stochastic forward passes. ``>= 2``.
        dropout_p: Dropout probability used during MC estimation. Must
            be in ``[0, 1)``. Default 0.1 matches the TCN's training
            default.
    """

    def __init__(
        self,
        encoder: "TemporalConvNet",
        controller: "AdaptationController",
        n_samples: int = 30,
        dropout_p: float = 0.1,
    ) -> None:
        if not isinstance(encoder, nn.Module):
            raise TypeError(
                f"encoder must be an nn.Module, got {type(encoder).__name__}"
            )
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        if not (0.0 <= float(dropout_p) < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p!r}")

        self._encoder = encoder
        self._controller = controller
        self._n_samples = int(n_samples)
        self._dropout_p = float(dropout_p)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Configured number of stochastic forward passes."""
        return self._n_samples

    @property
    def dropout_p(self) -> float:
        """Configured MC-Dropout probability."""
        return self._dropout_p

    def estimate(self, feature_window: torch.Tensor) -> UncertainAdaptationVector:
        """Run MC Dropout and return a mean + uncertainty summary.

        Args:
            feature_window: Tensor of shape ``[seq_len, input_dim]`` or
                ``[batch, seq_len, input_dim]`` with ``input_dim`` equal
                to the encoder's configured ``input_dim`` (32 by
                default). Batch size must be ``1`` if three-dimensional
                — the estimator is single-user by design.

        Returns:
            An :class:`UncertainAdaptationVector` summarising the
            posterior approximation.

        Raises:
            ValueError: If ``feature_window`` has the wrong rank or a
                batch size other than 1.
        """
        self._check_feature_window(feature_window)

        # Snapshot + override dropout.
        enc_snapshot = _ModeSnapshot(self._encoder, self._encoder.training)
        controller_snapshots: list[_ModeSnapshot] = [enc_snapshot]
        for sub in _controller_modules(self._controller):
            controller_snapshots.append(_ModeSnapshot(sub, sub.training))

        saved_p = _override_dropout_probability(self._encoder, self._dropout_p)
        for sub in _controller_modules(self._controller):
            saved_p.extend(_override_dropout_probability(sub, self._dropout_p))

        try:
            # 1. Eval everything, 2. then re-enable dropout only.
            self._encoder.eval()
            for sub in _controller_modules(self._controller):
                sub.eval()
            _enable_dropout_only(self._encoder)
            for sub in _controller_modules(self._controller):
                _enable_dropout_only(sub)

            samples = self._collect_samples(feature_window)
        finally:
            _restore_dropout_probability(saved_p)
            for snap in controller_snapshots:
                snap.restore()

        return _summarise(samples, sample_count=len(samples))

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _check_feature_window(self, feature_window: torch.Tensor) -> None:
        """Validate the shape/rank of the input window."""
        if not isinstance(feature_window, torch.Tensor):
            raise TypeError(
                "feature_window must be a torch.Tensor, got "
                f"{type(feature_window).__name__}"
            )
        if feature_window.dim() == 2:
            return
        if feature_window.dim() == 3 and feature_window.shape[0] == 1:
            return
        raise ValueError(
            "feature_window must have shape [seq_len, input_dim] or "
            f"[1, seq_len, input_dim], got {tuple(feature_window.shape)}"
        )

    def _collect_samples(
        self, feature_window: torch.Tensor
    ) -> list[AdaptationVector]:
        """Run ``n_samples`` stochastic passes and collect AdaptationVectors.

        The helper accepts either a bare feature window (in which case
        it calls ``controller.compute`` with a *derived* stub
        ``InteractionFeatureVector`` / ``DeviationMetrics``) or a pre-
        computed pair. Because the pipeline's ``AdaptationController``
        is defined over dataclass inputs rather than tensors, we route
        the encoder's stochastic embedding through a *fresh* feature
        vector derivation on every sample (see
        :func:`_derive_controller_inputs`). This is the same convention
        used by :mod:`i3.adaptation.ablation` when the encoder is swapped
        in or out of the adaptation pipeline.
        """
        samples: list[AdaptationVector] = []
        batch = (
            feature_window
            if feature_window.dim() == 3
            else feature_window.unsqueeze(0)
        )
        # Snapshot the controller's mutable style state so MC sampling
        # does not progressively drift the style mirror toward the
        # observed style across the N forward passes. We want to sample
        # the *current* posterior, not an EMA over the posterior.
        saved_style: Optional[StyleVector] = None
        if hasattr(self._controller, "_current_style"):
            saved_style = getattr(self._controller, "_current_style", None)
        try:
            for _ in range(self._n_samples):
                if saved_style is not None:
                    self._controller._current_style = saved_style  # type: ignore[attr-defined]
                with torch.no_grad():
                    embedding = self._encoder(batch)
                features, deviation = _derive_controller_inputs(
                    feature_window=batch[0],
                    embedding=embedding[0],
                )
                vec = self._controller.compute(features, deviation)
                if not isinstance(vec, AdaptationVector):  # pragma: no cover
                    raise TypeError(
                        "controller.compute must return an AdaptationVector, got "
                        f"{type(vec).__name__}"
                    )
                samples.append(vec)
        finally:
            if saved_style is not None:
                self._controller._current_style = saved_style  # type: ignore[attr-defined]
        return samples


# ---------------------------------------------------------------------------
# Policy helpers.
# ---------------------------------------------------------------------------


def confidence_threshold_policy(
    uncertain: UncertainAdaptationVector, threshold: float = 0.15
) -> bool:
    """Return ``True`` iff every dimension's std is below ``threshold``.

    The policy is deliberately *conservative*: if even one dimension
    has a standard deviation above ``threshold`` we refuse to report
    the overall adaptation as confident. This mirrors the "refuse when
    unsure" stance of ``docs/responsible_ai/accessibility_statement.md``
    — it is safer to under-adapt than to over-adapt on a weak signal.

    Args:
        uncertain: A populated :class:`UncertainAdaptationVector`.
        threshold: Per-dimension std ceiling (default ``0.15``). Must
            be strictly positive.

    Returns:
        ``True`` if the system is confident enough to act on every
        dimension; ``False`` otherwise.
    """
    if threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError(f"threshold must be a positive finite float, got {threshold!r}")
    return all(float(s) < threshold for s in uncertain.std)


def refuse_when_unsure_mask(
    uncertain: UncertainAdaptationVector, threshold: float = 0.15
) -> AdaptationVector:
    """Clamp under-confident dimensions back to their neutral baseline.

    For each :class:`AdaptationVector` dimension, if the MC-Dropout
    standard deviation exceeds ``threshold`` we replace the sample
    mean with the neutral baseline (``0.5`` for the four scalar
    dimensions, a neutral :class:`StyleVector.default` for the style
    dimensions, ``0.0`` for accessibility — i.e. "standard mode",
    matching :meth:`AdaptationVector.default`).

    This is the machinery behind the "refuse to act when unsure"
    semantic of the explanation panel. The goal is *not* to hide
    uncertainty but to prevent the UI from rendering a high-confidence
    adaptation arrow when the underlying signal is statistically
    unstable.

    Args:
        uncertain: The posterior approximation from
            :meth:`MCDropoutAdaptationEstimator.estimate`.
        threshold: Per-dimension std ceiling (default ``0.15``).

    Returns:
        A new :class:`AdaptationVector` with every under-confident
        dimension reset to its neutral baseline.
    """
    if threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError(f"threshold must be a positive finite float, got {threshold!r}")

    mean_vec = uncertain.mean_vector()
    neutral = AdaptationVector.default()
    std_by_name = uncertain.std_by_name()

    def _chooser(name: str, mean_val: float, neutral_val: float) -> float:
        return neutral_val if float(std_by_name.get(name, 0.0)) >= threshold else mean_val

    new_style = StyleVector(
        formality=_chooser(
            "formality", mean_vec.style_mirror.formality, neutral.style_mirror.formality
        ),
        verbosity=_chooser(
            "verbosity", mean_vec.style_mirror.verbosity, neutral.style_mirror.verbosity
        ),
        emotionality=_chooser(
            "emotionality",
            mean_vec.style_mirror.emotionality,
            neutral.style_mirror.emotionality,
        ),
        directness=_chooser(
            "directness", mean_vec.style_mirror.directness, neutral.style_mirror.directness
        ),
    )
    return AdaptationVector(
        cognitive_load=_chooser(
            "cognitive_load", mean_vec.cognitive_load, neutral.cognitive_load
        ),
        style_mirror=new_style,
        emotional_tone=_chooser(
            "emotional_tone", mean_vec.emotional_tone, neutral.emotional_tone
        ),
        accessibility=_chooser(
            "accessibility", mean_vec.accessibility, neutral.accessibility
        ),
    )


# ---------------------------------------------------------------------------
# Summary / derivation helpers.
# ---------------------------------------------------------------------------


def _tensor_of(samples: Iterable[AdaptationVector]) -> torch.Tensor:
    """Stack a stream of AdaptationVectors into a ``[n, 8]`` tensor."""
    stacked = torch.stack([v.to_tensor() for v in samples], dim=0)
    if stacked.dim() != 2 or stacked.shape[1] != 8:  # pragma: no cover
        raise ValueError(
            f"expected [n, 8] stacked samples, got {tuple(stacked.shape)}"
        )
    return stacked


def _summarise(
    samples: Sequence[AdaptationVector], sample_count: int
) -> UncertainAdaptationVector:
    """Build an :class:`UncertainAdaptationVector` from a sample stream."""
    stack = _tensor_of(samples)

    mean_t = stack.mean(dim=0)
    # ``unbiased=False`` matches the population-std convention used by
    # numpy's default for a fixed sample size — the MC sample count is
    # exact, not an estimate of a larger parent population.
    std_t = stack.std(dim=0, unbiased=False)

    # 95 % CI via the empirical 2.5 % / 97.5 % quantiles. ``quantile``
    # expects a 1-D quantile tensor and returns a [q, features] result.
    qs = torch.tensor([0.025, 0.975], dtype=stack.dtype, device=stack.device)
    ci_t = torch.quantile(stack, qs, dim=0)
    lower_t = ci_t[0]
    upper_t = ci_t[1]

    ci = [
        DimensionInterval(
            lower=float(_clip01(lower_t[i].item())),
            upper=float(_clip01(upper_t[i].item())),
        )
        for i in range(8)
    ]

    mean_vec = AdaptationVector.from_tensor(mean_t)
    return UncertainAdaptationVector(
        mean=mean_vec.to_dict(),
        std=[float(std_t[i].item()) for i in range(8)],
        ci=ci,
        sample_count=int(sample_count),
    )


def _clip01(x: float) -> float:
    """Clamp ``x`` to ``[0, 1]`` while mapping NaN to 0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(max(0.0, min(1.0, x)))


def _controller_modules(controller: object) -> list[nn.Module]:
    """Return every :class:`nn.Module` owned by the controller.

    Some adaptation dimension adapters may themselves be
    :class:`nn.Module` subclasses (future work — the current batch has
    plain Python classes); this helper is forward-compatible.
    """
    collected: list[nn.Module] = []
    for attr in ("cognitive", "style", "emotional", "accessibility"):
        sub = getattr(controller, attr, None)
        if isinstance(sub, nn.Module):
            collected.append(sub)
    return collected


# ---------------------------------------------------------------------------
# Controller-input derivation.
# ---------------------------------------------------------------------------


def _derive_controller_inputs(
    feature_window: torch.Tensor, embedding: torch.Tensor
) -> tuple[object, object]:
    """Build ``(features, deviation)`` inputs for ``controller.compute``.

    The controller expects an
    :class:`~i3.interaction.types.InteractionFeatureVector` and a
    :class:`~i3.user_model.types.DeviationMetrics`, not tensors. For
    MC-Dropout uncertainty we need a *stable* conversion that threads
    the encoder's stochasticity through the pipeline. The convention
    used here — and mirrored in ``scripts/demos/uncertainty.py`` —
    is to take the last row of ``feature_window`` as the "current
    message" features, and to derive deviation z-scores from the
    magnitude of the encoder embedding (a proxy for the distance of
    the current message from a neutral baseline).

    Args:
        feature_window: ``[seq_len, input_dim]`` tensor of the
            recent feature history.
        embedding: ``[embedding_dim]`` encoder output for this sample.

    Returns:
        ``(features, deviation)`` suitable for
        :meth:`AdaptationController.compute`.
    """
    # Deferred import: avoids a circular dependency at module load.
    from i3.interaction.types import InteractionFeatureVector
    from i3.user_model.types import DeviationMetrics

    last = feature_window[-1]
    if last.numel() != 32:
        # Pad / truncate so an oddly-shaped caller cannot break
        # from_tensor. Downstream adapters are NaN-safe by design.
        padded = torch.zeros(32, dtype=last.dtype, device=last.device)
        n = min(32, int(last.numel()))
        padded[:n] = last.reshape(-1)[:n]
        last = padded
    features = InteractionFeatureVector.from_tensor(last.detach().to("cpu"))

    # Magnitude-based deviation proxy, scaled to roughly the
    # ``[-2, 2]`` z-score range the rest of the pipeline expects.
    mag = float(embedding.detach().norm().item())
    mag_scaled = max(-2.0, min(2.0, mag - 1.0))
    deviation = DeviationMetrics(
        current_vs_baseline=max(0.0, min(1.0, mag / 2.0)),
        current_vs_session=max(0.0, min(1.0, mag / 2.0)),
        engagement_score=0.5,
        magnitude=mag,
        iki_deviation=mag_scaled,
        length_deviation=mag_scaled,
        vocab_deviation=mag_scaled,
        formality_deviation=mag_scaled,
        speed_deviation=mag_scaled,
        engagement_deviation=mag_scaled,
        complexity_deviation=mag_scaled,
        pattern_deviation=mag_scaled,
    )
    return features, deviation


__all__ = [
    "ADAPTATION_DIMS",
    "DimensionInterval",
    "MCDropoutAdaptationEstimator",
    "UncertainAdaptationVector",
    "confidence_threshold_policy",
    "refuse_when_unsure_mask",
]
