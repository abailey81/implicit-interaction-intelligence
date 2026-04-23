"""Public API for the I3 red-team adversarial harness.

This package bundles a curated attack corpus, a parallel runner, four
concrete target surfaces wired to the production subsystems, and a
small library of invariant checkers.  All symbols that higher layers
(``scripts/security/run_redteam.py``, tests, CI) are allowed to import are
re-exported here.
"""

from __future__ import annotations

from i3.redteam.attack_corpus import (
    ATTACK_CATEGORIES,
    ATTACK_CATEGORY_TARGETS,
    ATTACK_CORPUS,
    Attack,
    AttackCategory,
    ExpectedOutcome,
    Severity,
    load_external_corpus,
)
from i3.redteam.attacker import (
    AttackResult,
    FastAPITargetSurface,
    GuardrailsTargetSurface,
    PDDLPlannerTargetSurface,
    RedTeamReport,
    RedTeamRunner,
    SanitizerTargetSurface,
    TargetSurface,
)
from i3.redteam.policy_check import (
    REDTEAM_CANARY,
    verify_pddl_soundness,
    verify_privacy_invariant,
    verify_rate_limit_invariant,
    verify_sensitive_topic_invariant,
)

__all__ = [
    "ATTACK_CATEGORIES",
    "ATTACK_CATEGORY_TARGETS",
    "ATTACK_CORPUS",
    "REDTEAM_CANARY",
    "Attack",
    "AttackCategory",
    "AttackResult",
    "ExpectedOutcome",
    "FastAPITargetSurface",
    "GuardrailsTargetSurface",
    "PDDLPlannerTargetSurface",
    "RedTeamReport",
    "RedTeamRunner",
    "SanitizerTargetSurface",
    "Severity",
    "TargetSurface",
    "load_external_corpus",
    "verify_pddl_soundness",
    "verify_privacy_invariant",
    "verify_rate_limit_invariant",
    "verify_sensitive_topic_invariant",
]
