"""Continual learning primitives for the I3 user-model + TCN encoder.

This sub-package addresses the *catastrophic-forgetting* risk implicit in
the three-timescale EMA (``i3.user_model``) + gradient-updated TCN
encoder pipeline. A user profile that has observed a fresh, focused user
for months and then encounters a fatigued or motor-impaired pattern will,
without a regulariser, silently overwrite the representations that made
the earlier state distinguishable.

The design composes four orthogonal mechanisms:

1. :class:`~i3.continual.ewc.ElasticWeightConsolidation` -- Kirkpatrick
   et al. 2017 PNAS "Overcoming catastrophic forgetting in neural
   networks". A diagonal Fisher Information matrix estimated per task is
   consolidated with a snapshot of the parameters so the next task's
   training loss is augmented with a quadratic penalty
   ``(λ/2) · Σ F_i (θ_i − θ*_i)²``.
2. :class:`~i3.continual.ewc.OnlineEWC` -- Schwarz et al. 2018 *Progress
   & Compress* running-average Fisher for streaming data where discrete
   task boundaries do not exist.
3. :class:`~i3.continual.replay_buffer.ReservoirReplayBuffer` and
   :class:`~i3.continual.replay_buffer.ExperienceReplay` -- Chaudhry
   et al. 2019 "On Tiny Episodic Memories" combined with Vitter 1985
   reservoir sampling. Unbiased past samples are mixed into new-task
   batches to complement the parameter-space regulariser with a data-
   space one.
4. :class:`~i3.continual.drift_detector.ConceptDriftDetector` -- Bifet &
   Gavaldà 2007 ADWIN-style adaptive windowing on the encoder's
   embedding-error distribution. Detected drift triggers an EWC
   consolidation step, closing the loop on the user-model side.

Finally :class:`~i3.continual.ewc_user_model.EWCUserModel` composes the
three primitives around the *existing*
:class:`~i3.user_model.model.UserModel` without mutating it.

Public surface (stable):
    - :class:`ElasticWeightConsolidation`, :class:`OnlineEWC`
    - :class:`ReservoirReplayBuffer`, :class:`ExperienceReplay`
    - :class:`ConceptDriftDetector`, :func:`population_stability_index`
    - :class:`EWCUserModel`
"""

from __future__ import annotations

from i3.continual.drift_detector import (
    ConceptDriftDetector,
    DriftDetectionResult,
    population_stability_index,
)
from i3.continual.ewc import (
    ElasticWeightConsolidation,
    FisherDict,
    OnlineEWC,
)
from i3.continual.ewc_user_model import EWCUserModel
from i3.continual.replay_buffer import (
    ExperienceReplay,
    ReplaySample,
    ReservoirReplayBuffer,
)

__all__: list[str] = [
    "ConceptDriftDetector",
    "DriftDetectionResult",
    "EWCUserModel",
    "ElasticWeightConsolidation",
    "ExperienceReplay",
    "FisherDict",
    "OnlineEWC",
    "ReplaySample",
    "ReservoirReplayBuffer",
    "population_stability_index",
]
