"""Simulation-based closed-loop evaluation harness.

This package implements the simulation harness described in
``the advancement plan`` (Tier 1, Batch G1). It provides:

* An :class:`HCIPersona` library of eight canonical synthetic users drawn
  from published HCI literature (Epp et al. 2011; Vizer 2009; Zimmermann
  2014).
* A :class:`UserSimulator` that replays plausible keystroke streams and
  linguistic signatures for a given persona under a fixed seed.
* A :class:`ClosedLoopEvaluator` that feeds simulator output through the
  full I3 :class:`~i3.pipeline.engine.Pipeline` and scores persona
  recovery, adaptation-vector error, convergence speed, and router bias
  against ground truth.

Unlike the KL-based ablation studies under
:mod:`i3.eval.ablation_experiment`, this harness can answer the
load-bearing user-modelling question: *does the system actually recover
the user's true state?* All metrics are measured against known
ground-truth adaptation vectors embedded in the persona definitions.

Public surface:
    * :class:`HCIPersona`
    * :class:`UserSimulator`
    * :class:`SimulatedMessage`
    * :class:`ClosedLoopEvaluator`
    * :class:`ClosedLoopResult`
    * :data:`ALL_PERSONAS`
"""

from __future__ import annotations

from i3.eval.simulation.closed_loop import (
    ClosedLoopEvaluator,
    ClosedLoopResult,
)
from i3.eval.simulation.personas import (
    ALL_PERSONAS,
    DYSLEXIC_USER,
    ENERGETIC_USER,
    FATIGUED_DEVELOPER,
    FRESH_USER,
    HIGH_LOAD_USER,
    LOW_VISION_USER,
    MOTOR_IMPAIRED_USER,
    SECOND_LANGUAGE_SPEAKER,
    HCIPersona,
    LinguisticProfile,
    TypingProfile,
)
from i3.eval.simulation.user_simulator import SimulatedMessage, UserSimulator

__all__: list[str] = [
    "ALL_PERSONAS",
    "DYSLEXIC_USER",
    "ENERGETIC_USER",
    "FATIGUED_DEVELOPER",
    "FRESH_USER",
    "HIGH_LOAD_USER",
    "LOW_VISION_USER",
    "MOTOR_IMPAIRED_USER",
    "SECOND_LANGUAGE_SPEAKER",
    "ClosedLoopEvaluator",
    "ClosedLoopResult",
    "HCIPersona",
    "LinguisticProfile",
    "SimulatedMessage",
    "TypingProfile",
    "UserSimulator",
]
