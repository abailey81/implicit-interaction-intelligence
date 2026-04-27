"""Safety planning primitives for I³.

This subpackage hosts the **PDDL-grounded safety planner** that lifts I³'s
privacy-override path from a policy check to a *machine-checkable*
decision. Huawei's public agentic-safety line reports 99.9 % interception
of highest-risk agent operations grounded in Planning Domain Definition
Language (PDDL) formalisms; the classes exported from this package are
the I³-side analogue.

Top-level exports:

* :class:`~i3.safety.pddl_planner.PrivacySafetyPlanner` -- builds and
  executes a plan for the privacy-safety domain.
* :class:`~i3.safety.pddl_planner.SafetyContext` -- the grounded input
  state (predicate valuations) the planner reasons over.
* :class:`~i3.safety.pddl_planner.SafetyPlan` -- an ordered sequence of
  actions produced by the planner.
* :class:`~i3.safety.certificates.SafetyCertificate` -- a Pydantic-v2
  model that serialises a plan's reasoning trace to YAML for audit.

The design follows Ghallab et al. (AAAI 1998) "PDDL — The Planning Domain
Definition Language" and McDermott (1998) PDDL 1.2 with goal conditions.
"""

from __future__ import annotations

from i3.safety.certificates import (
    CertifiedStep,
    SafetyCertificate,
    certificate_from_yaml,
    certificate_to_yaml,
)
from i3.safety.classifier import (
    SafetyClassifier,
    SafetyVerdict,
    get_global_classifier,
)
from i3.safety.pddl_planner import (
    Action,
    PddlDomain,
    PrivacySafetyPlanner,
    SafetyContext,
    SafetyPlan,
    SafetyPlannerError,
)

__all__ = [
    "Action",
    "CertifiedStep",
    "PddlDomain",
    "PrivacySafetyPlanner",
    "SafetyCertificate",
    "SafetyClassifier",
    "SafetyContext",
    "SafetyPlan",
    "SafetyPlannerError",
    "SafetyVerdict",
    "certificate_from_yaml",
    "certificate_to_yaml",
    "get_global_classifier",
]
