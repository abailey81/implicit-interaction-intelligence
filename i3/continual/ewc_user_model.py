"""Composable EWC wrapper around the existing three-timescale UserModel.

This module defines :class:`EWCUserModel`, a thin composition that:

* delegates every existing user-modelling call to the pre-existing
  :class:`~i3.user_model.model.UserModel` (no mutation of that class);
* owns an :class:`~i3.continual.ewc.ElasticWeightConsolidation` or
  :class:`~i3.continual.ewc.OnlineEWC` instance anchored on the shared
  :class:`~i3.encoder.tcn.TemporalConvNet`;
* subscribes to a :class:`~i3.continual.drift_detector.ConceptDriftDetector`
  whose drift alarm triggers an EWC consolidation step.

The *user-model* layer therefore gains long-horizon plasticity-stability
balance for free: user-specific EMA baselines keep adapting (plasticity),
but the shared encoder's representation stops drifting away from
already-learned user-state regions (stability) whenever the drift
detector spots a shift.

Key design property: *EWCUserModel does not modify
:class:`i3.user_model.model.UserModel` in-place*. This is a composition,
not inheritance, so existing tests and call sites continue to work with
vanilla :class:`UserModel` unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import torch
from torch import nn

from i3.continual.drift_detector import (
    ConceptDriftDetector,
    DriftDetectionResult,
)
from i3.continual.ewc import (
    ElasticWeightConsolidation,
    LossClosure,
    OnlineEWC,
)
from i3.continual.replay_buffer import (
    ExperienceReplay,
    ReplaySample,
    ReservoirReplayBuffer,
)

if TYPE_CHECKING:
    from i3.interaction.types import InteractionFeatureVector
    from i3.user_model.model import UserModel
    from i3.user_model.types import DeviationMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EWCUserModel
# ---------------------------------------------------------------------------


class EWCUserModel:
    """Composable EWC + replay + drift wrapper around :class:`UserModel`.

    The wrapper does not subclass :class:`UserModel`; it holds a reference
    and delegates. Every method that mutates user state (``start_session``,
    ``update_state``, ``end_session``) is forwarded unchanged to the inner
    instance. Additionally the wrapper tracks per-update deviation
    magnitude to feed the drift detector.

    Args:
        inner_model: The existing three-timescale user model instance.
            It must expose ``update_state`` and ``deviation_from_baseline``.
        encoder: The shared :class:`~i3.encoder.tcn.TemporalConvNet` (or
            any :class:`torch.nn.Module`) whose parameters EWC protects.
        lambda_ewc: EWC penalty strength passed to the inner regulariser.
        online: When ``True`` uses :class:`OnlineEWC` instead of classic
            :class:`ElasticWeightConsolidation`.
        gamma: Online-EWC decay factor (only consulted when ``online``).
        fisher_estimation_samples: Fisher sample budget per consolidation.
        replay_capacity: Reservoir buffer size; set to ``0`` to disable
            replay integration entirely.
        drift_delta: ADWIN δ parameter controlling detector sensitivity.
        drift_min_sub_window: Minimum sub-window size for ADWIN splits.
        loss_closure: Optional ``(encoder, batch) -> loss`` closure used
            during Fisher estimation.
        auto_consolidate_on_drift: When ``True`` the drift callback
            invokes :meth:`consolidate` with the provided ``drift_
            dataloader`` (if set). When ``False`` the wrapper only
            records the event and leaves consolidation to the caller.
    """

    def __init__(
        self,
        inner_model: "UserModel",
        encoder: nn.Module,
        *,
        lambda_ewc: float = 1000.0,
        online: bool = False,
        gamma: float = 0.95,
        fisher_estimation_samples: int = 200,
        replay_capacity: int = 512,
        drift_delta: float = 0.002,
        drift_min_sub_window: int = 8,
        loss_closure: Optional[LossClosure] = None,
        auto_consolidate_on_drift: bool = True,
    ) -> None:
        self._inner = inner_model
        self._encoder = encoder

        self._ewc: ElasticWeightConsolidation
        if online:
            self._ewc = OnlineEWC(
                encoder,
                lambda_ewc=lambda_ewc,
                gamma=gamma,
                fisher_estimation_samples=fisher_estimation_samples,
                loss_closure=loss_closure,
            )
        else:
            self._ewc = ElasticWeightConsolidation(
                encoder,
                lambda_ewc=lambda_ewc,
                fisher_estimation_samples=fisher_estimation_samples,
                loss_closure=loss_closure,
            )

        # Replay buffer: per-user Fisher cache is augmented with an
        # episodic memory so that consolidations have data to compute
        # the Fisher from even when the caller's training shard is
        # exhausted.
        self._replay: Optional[ExperienceReplay] = None
        if replay_capacity > 0:
            self._replay = ExperienceReplay(
                buffer=ReservoirReplayBuffer(capacity=replay_capacity),
                replay_batch_size=min(16, replay_capacity),
            )

        # Drift detector -- fires the EWC consolidation trigger.
        self._drift_detector: ConceptDriftDetector = ConceptDriftDetector(
            delta=drift_delta,
            min_sub_window=drift_min_sub_window,
            on_drift_detected=self._handle_drift,
        )

        self._auto_consolidate_on_drift: bool = bool(auto_consolidate_on_drift)
        # Optional dataloader stashed by :meth:`set_consolidation_dataloader`
        # for the auto-consolidate path.
        self._consolidation_dataloader: Optional[Iterable[object]] = None

        # Event hook a caller may register for observability.
        self._external_drift_hook: Optional[
            Callable[[DriftDetectionResult], None]
        ] = None

    # ------------------------------------------------------------------
    # Inner-model delegation
    # ------------------------------------------------------------------

    @property
    def inner(self) -> "UserModel":
        """The wrapped :class:`UserModel`."""
        return self._inner

    @property
    def user_id(self) -> str:
        """Delegated user id."""
        return self._inner.user_id

    def start_session(self) -> None:
        """Delegate to the inner model."""
        self._inner.start_session()

    def end_session(self) -> dict:
        """Delegate to the inner model."""
        return self._inner.end_session()

    def update_state(
        self,
        embedding: torch.Tensor,
        features: "InteractionFeatureVector",
    ) -> "DeviationMetrics":
        """Forward to the inner model and feed the drift detector.

        The wrapper consumes the resulting
        :class:`~i3.user_model.types.DeviationMetrics` and drives the
        drift detector with its scalar magnitude.

        Args:
            embedding: 64-dim encoder output.
            features: Interaction feature vector.

        Returns:
            Whatever the inner model returned.
        """
        dev = self._inner.update_state(embedding, features)
        # SEC: feed only finite scalars; the detector already no-ops on
        # NaN/Inf but we avoid the detour through its logger by checking
        # here too.
        try:
            scalar = float(dev.magnitude)
        except (TypeError, ValueError):
            scalar = 0.0
        self._drift_detector.update(scalar)
        return dev

    # ------------------------------------------------------------------
    # EWC-specific API
    # ------------------------------------------------------------------

    @property
    def ewc(self) -> ElasticWeightConsolidation:
        """The underlying EWC regulariser (either vanilla or Online)."""
        return self._ewc

    @property
    def replay(self) -> Optional[ExperienceReplay]:
        """The experience-replay wrapper, or ``None`` if disabled."""
        return self._replay

    @property
    def drift_detector(self) -> ConceptDriftDetector:
        """The drift detector."""
        return self._drift_detector

    def penalty_loss(self) -> torch.Tensor:
        """Proxy to :meth:`ElasticWeightConsolidation.penalty_loss`."""
        return self._ewc.penalty_loss()

    def consolidate(
        self,
        dataloader: Iterable[object],
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Run an EWC consolidation step on the shared encoder."""
        self._ewc.consolidate(dataloader, device=device)

    def set_consolidation_dataloader(
        self, dataloader: Iterable[object]
    ) -> None:
        """Register a default dataloader used by the auto-consolidate path."""
        self._consolidation_dataloader = dataloader

    def observe_training_sample(self, sample: ReplaySample) -> None:
        """Offer a training sample to the reservoir buffer."""
        if self._replay is not None:
            self._replay.observe(sample)

    def register_drift_hook(
        self, hook: Callable[[DriftDetectionResult], None]
    ) -> None:
        """Register an external callback invoked on every drift event.

        The hook runs *after* the optional auto-consolidation step.
        """
        self._external_drift_hook = hook

    # ------------------------------------------------------------------
    # Drift handling
    # ------------------------------------------------------------------

    def _handle_drift(self, result: DriftDetectionResult) -> None:
        """Internal drift callback wired into :class:`ConceptDriftDetector`."""
        logger.info(
            "EWCUserModel drift event for user_id=%s: cut=%s psi=%s",
            self._inner.user_id,
            result.cut_point,
            f"{result.psi:.4f}" if result.psi is not None else "n/a",
        )
        if self._auto_consolidate_on_drift:
            loader = self._consolidation_dataloader
            if loader is None and self._replay is not None:
                # Fall back to replay buffer samples as a dataloader-like
                # iterable; each yielded batch is a single-sample
                # (input_tensor, target) tuple.
                loader = list(self._iter_replay_as_dataloader())
            if loader is not None:
                try:
                    self._ewc.consolidate(loader)
                except (RuntimeError, ValueError) as exc:
                    logger.error(
                        "Auto-consolidation failed: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
            else:
                logger.info(
                    "Drift detected but no consolidation dataloader is "
                    "available; skipping auto-consolidate."
                )

        if self._external_drift_hook is not None:
            try:
                self._external_drift_hook(result)
            except (RuntimeError, ValueError) as exc:
                logger.error(
                    "external drift hook raised %s: %s",
                    type(exc).__name__,
                    exc,
                )

    def _iter_replay_as_dataloader(self) -> Iterable[object]:
        """Yield replay samples as ``(input, target)`` tuples."""
        if self._replay is None:
            return
        for sample in self._replay.buffer:
            inp = sample.input_tensor
            if inp.dim() == 2:
                # TCN expects [batch, seq_len, features]; add batch dim.
                inp = inp.unsqueeze(0)
            target = sample.target
            if target is None:
                # Self-supervised: use encoder output as its own target.
                yield (inp,)
            else:
                yield (inp, target)


__all__ = ["EWCUserModel"]
