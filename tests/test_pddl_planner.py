"""Tests for :mod:`i3.safety.pddl_planner` and
:mod:`i3.safety.certificates`.

Coverage goals:
    1. Load-bearing invariant -- sensitive-topic contexts NEVER produce
       a ``route_cloud`` action.
    2. Rate-limited contexts yield a single ``deny_request`` action.
    3. Every emitted plan's preconditions are satisfied by the context.
    4. Certificate round-trips through YAML faithfully.
    5. Ordering invariants (``redact_pii`` always precedes routing;
       routing is always the final step in a successful plan).
    6. Unauthenticated callers are denied.
    7. Domain validation refuses unknown action names during certify.
"""

from __future__ import annotations

import pytest

from i3.safety import (
    PrivacySafetyPlanner,
    SafetyContext,
    SafetyPlannerError,
    certificate_from_yaml,
    certificate_to_yaml,
)
from i3.safety.pddl_planner import PRIVACY_SAFETY_DOMAIN, Action, PddlDomain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def planner() -> PrivacySafetyPlanner:
    """Return a default :class:`PrivacySafetyPlanner`."""
    return PrivacySafetyPlanner()


def _context(**overrides: bool) -> SafetyContext:
    """Construct a :class:`SafetyContext` with sensible defaults.

    Args:
        **overrides: Any predicate attributes to flip.

    Returns:
        A :class:`SafetyContext` suitable for the planner.
    """
    base = {
        "sensitive_topic": False,
        "network_available": True,
        "authenticated_user": True,
        "encryption_key_loaded": True,
        "rate_limited": False,
        "contains_pii": False,
    }
    base.update(overrides)
    return SafetyContext(**base)


# ---------------------------------------------------------------------------
# Invariant: sensitive -> never cloud
# ---------------------------------------------------------------------------


def test_sensitive_topic_never_routes_to_cloud(
    planner: PrivacySafetyPlanner,
) -> None:
    """Plan for a sensitive topic MUST NOT include ``route_cloud``."""
    ctx = _context(sensitive_topic=True)
    plan = planner.plan(ctx)
    assert "route_cloud" not in plan.actions
    assert "route_local" in plan.actions


def test_sensitive_topic_with_pii_redacts_then_routes_local(
    planner: PrivacySafetyPlanner,
) -> None:
    """Sensitive + PII -> redact_pii must precede route_local."""
    ctx = _context(sensitive_topic=True, contains_pii=True)
    plan = planner.plan(ctx)
    assert plan.actions == ("redact_pii", "route_local")


# ---------------------------------------------------------------------------
# Rate limiting & authentication
# ---------------------------------------------------------------------------


def test_rate_limited_context_denies_without_routing(
    planner: PrivacySafetyPlanner,
) -> None:
    """A rate-limited caller should receive a single ``deny_request``."""
    ctx = _context(rate_limited=True, sensitive_topic=True)
    plan = planner.plan(ctx)
    assert plan.actions == ("deny_request",)
    assert "routed_local" not in plan.final_state
    assert "routed_cloud" not in plan.final_state


def test_unauthenticated_caller_is_denied(
    planner: PrivacySafetyPlanner,
) -> None:
    """Without ``authenticated_user`` the planner must deny."""
    ctx = _context(authenticated_user=False)
    plan = planner.plan(ctx)
    assert plan.actions == ("deny_request",)


# ---------------------------------------------------------------------------
# Cloud routing path
# ---------------------------------------------------------------------------


def test_non_sensitive_clean_request_routes_cloud(
    planner: PrivacySafetyPlanner,
) -> None:
    """Non-sensitive + no PII + network => route_cloud."""
    ctx = _context()
    plan = planner.plan(ctx)
    assert plan.actions == ("route_cloud",)


def test_pii_redaction_before_cloud(
    planner: PrivacySafetyPlanner,
) -> None:
    """PII must be redacted before a cloud route fires."""
    ctx = _context(contains_pii=True)
    plan = planner.plan(ctx)
    assert plan.actions == ("redact_pii", "route_cloud")


def test_no_network_falls_back_to_local(
    planner: PrivacySafetyPlanner,
) -> None:
    """Without network, non-sensitive requests still route locally."""
    ctx = _context(network_available=False)
    plan = planner.plan(ctx)
    assert plan.actions == ("route_local",)


# ---------------------------------------------------------------------------
# Precondition correctness for every emitted plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"sensitive_topic": True},
        {"contains_pii": True},
        {"sensitive_topic": True, "contains_pii": True},
        {"network_available": False},
        {"rate_limited": True},
        {"authenticated_user": False},
    ],
)
def test_all_plan_preconditions_satisfied(
    planner: PrivacySafetyPlanner,
    overrides: dict[str, bool],
) -> None:
    """Every plan's steps must have satisfied preconditions.

    The trace records the pre-state at each step; the applied action's
    preconditions must be a subset of that pre-state, and none of the
    action's forbidden predicates may appear in it.
    """
    ctx = _context(**overrides)
    plan = planner.plan(ctx)
    for pre, action_name, _post in plan.trace:
        action = PRIVACY_SAFETY_DOMAIN.actions[action_name]
        assert action.preconditions.issubset(pre), (
            f"Action {action_name} preconditions {action.preconditions} "
            f"not subset of pre-state {sorted(pre)}."
        )
        assert pre.isdisjoint(action.forbidden), (
            f"Action {action_name} forbidden-set {action.forbidden} "
            f"intersects pre-state {sorted(pre)}."
        )


# ---------------------------------------------------------------------------
# Ordering invariants
# ---------------------------------------------------------------------------


def test_redact_always_precedes_routing(
    planner: PrivacySafetyPlanner,
) -> None:
    """If the plan contains redact_pii, it must appear before any route."""
    ctx = _context(sensitive_topic=True, contains_pii=True)
    plan = planner.plan(ctx)
    actions = list(plan.actions)
    assert "redact_pii" in actions
    redact_idx = actions.index("redact_pii")
    for route in ("route_local", "route_cloud"):
        if route in actions:
            assert actions.index(route) > redact_idx


def test_routing_is_final_step_when_request_is_served(
    planner: PrivacySafetyPlanner,
) -> None:
    """For any non-denied plan, the last action is a routing action."""
    ctx = _context()
    plan = planner.plan(ctx)
    assert plan.actions[-1] in {"route_local", "route_cloud"}


# ---------------------------------------------------------------------------
# Certificate round-trips
# ---------------------------------------------------------------------------


def test_certificate_round_trip(
    planner: PrivacySafetyPlanner,
) -> None:
    """YAML serialise + parse reconstructs an equal certificate."""
    ctx = _context(sensitive_topic=True, contains_pii=True)
    plan = planner.plan(ctx)
    cert = planner.certify(plan, context=ctx)

    yaml_text = certificate_to_yaml(cert)
    reloaded = certificate_from_yaml(yaml_text)
    assert reloaded == cert
    assert reloaded.actions == list(plan.actions)
    assert reloaded.domain_name == "privacy_safety_v1"


def test_certificate_from_yaml_rejects_empty() -> None:
    """Empty YAML must not be accepted as a certificate."""
    with pytest.raises(ValueError):
        certificate_from_yaml("")


def test_certificate_from_yaml_rejects_non_mapping() -> None:
    """Top-level non-mapping YAML is rejected."""
    with pytest.raises(ValueError):
        certificate_from_yaml("- a\n- b\n")


# ---------------------------------------------------------------------------
# Certify guards
# ---------------------------------------------------------------------------


def test_certify_rejects_tampered_trace(
    planner: PrivacySafetyPlanner,
) -> None:
    """A trace whose action name is unknown must be rejected."""
    from i3.safety.pddl_planner import SafetyPlan

    # Construct a plan with a fabricated action name (not in the domain).
    fake_plan = SafetyPlan(
        actions=("not_a_real_action",),
        trace=((frozenset(), "not_a_real_action", frozenset({"routed_local"})),),
        final_state=frozenset({"routed_local"}),
    )
    with pytest.raises(SafetyPlannerError):
        planner.certify(fake_plan)


def test_certify_detects_invariant_violation() -> None:
    """A hand-rolled plan that fires route_cloud while sensitive_topic
    holds must be rejected by certify()."""
    from i3.safety.pddl_planner import SafetyPlan

    planner = PrivacySafetyPlanner()
    # Manually build a bogus trace that violates the invariant.
    pre = frozenset(
        {"authenticated_user", "network_available", "sensitive_topic"}
    )
    post = frozenset(pre | {"routed_cloud"})
    bogus = SafetyPlan(
        actions=("route_cloud",),
        trace=((pre, "route_cloud", post),),
        final_state=post,
    )
    with pytest.raises(SafetyPlannerError):
        planner.certify(bogus)


# ---------------------------------------------------------------------------
# Custom domain smoke
# ---------------------------------------------------------------------------


def test_planner_accepts_custom_domain_reference() -> None:
    """A planner can be constructed with a compatible custom domain."""
    # Use the canonical domain but exercise the injection path.
    custom_domain = PddlDomain(
        predicates=PRIVACY_SAFETY_DOMAIN.predicates,
        actions=dict(PRIVACY_SAFETY_DOMAIN.actions),
    )
    planner = PrivacySafetyPlanner(domain=custom_domain)
    ctx = _context()
    plan = planner.plan(ctx)
    assert plan.actions == ("route_cloud",)


def test_action_applicable_respects_forbidden_set() -> None:
    """Action.applicable returns False when a forbidden predicate holds."""
    route_cloud: Action = PRIVACY_SAFETY_DOMAIN.actions["route_cloud"]
    bad_state = frozenset(
        {"authenticated_user", "network_available", "sensitive_topic"}
    )
    assert route_cloud.applicable(bad_state) is False
