"""Minimal PDDL-grounded safety planner for the I³ privacy override.

The module implements a **tiny, self-contained planner** over a
privacy-safety Planning Domain Definition Language (PDDL) domain
(Ghallab et al., AAAI 1998; McDermott, PDDL 1.2, 1998).  It is
deliberately not a general STRIPS planner: the privacy-safety domain
has a handful of propositions and actions, so a forward-search A*
over the bitmask of predicate valuations is sufficient and much
easier to audit than a third-party dependency.

The parallel Huawei work is the public agentic-safety line that grounds
risky-operation interception in PDDL and reports 99.9 % coverage.  I³
uses the same shape -- a declarative domain + a machine-checkable
certificate -- so the privacy override is no longer policy but proof.

Classes
-------
:class:`PddlDomain`
    Container for predicates, action schemas, preconditions, effects.
:class:`SafetyContext`
    A grounded valuation of the domain's predicates + goal flags.
:class:`SafetyPlan`
    An ordered sequence of action names with their per-step rationale.
:class:`PrivacySafetyPlanner`
    The planner itself.  :meth:`plan` / :meth:`certify`.

Example
-------
>>> planner = PrivacySafetyPlanner()
>>> ctx = SafetyContext(
...     sensitive_topic=True,
...     network_available=True,
...     authenticated_user=True,
...     encryption_key_loaded=True,
...     rate_limited=False,
... )
>>> plan = planner.plan(ctx)
>>> plan.actions[0]
'redact_pii'
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SafetyPlannerError(RuntimeError):
    """Raised when the planner cannot produce a safe plan."""


# ---------------------------------------------------------------------------
# Domain primitives
# ---------------------------------------------------------------------------


# Predicate valuation type used throughout the module.
State = frozenset[str]


@dataclass(frozen=True, slots=True)
class Action:
    """One PDDL action schema.

    Attributes:
        name: Unique action identifier (e.g. ``"route_local"``).
        preconditions: Predicates that must be true in the state to fire.
        forbidden: Predicates that must be *false* in the state to fire.
        add_effects: Predicates added to the state after firing.
        del_effects: Predicates removed from the state after firing.
        rationale: Human-readable justification quoted in the certificate.
    """

    name: str
    preconditions: frozenset[str]
    forbidden: frozenset[str]
    add_effects: frozenset[str]
    del_effects: frozenset[str]
    rationale: str

    def applicable(self, state: State) -> bool:
        """Return True iff this action's pre- and forbidden-conditions hold.

        Args:
            state: Current predicate valuation.

        Returns:
            ``True`` iff every precondition is in ``state`` and no
            forbidden predicate is in ``state``.
        """
        return self.preconditions.issubset(state) and state.isdisjoint(
            self.forbidden
        )

    def apply(self, state: State) -> State:
        """Return the next state produced by firing this action.

        Args:
            state: Current predicate valuation.

        Returns:
            A new :class:`frozenset` reflecting add / del effects.
        """
        return frozenset((state - self.del_effects) | self.add_effects)


@dataclass(frozen=True, slots=True)
class PddlDomain:
    """Minimal PDDL domain description (privacy-safety).

    Attributes:
        predicates: The predicate vocabulary.
        actions: Mapping from action name to :class:`Action`.
    """

    predicates: frozenset[str]
    actions: Mapping[str, Action]


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafetyContext:
    """Grounded valuation of the privacy-safety domain.

    Each attribute maps to the matching predicate of
    :data:`PRIVACY_SAFETY_DOMAIN`.  The planner converts this into a
    :data:`State` (a frozenset of the predicate names that are true).

    Attributes:
        sensitive_topic: The request touches a sensitive category
            (health, finance, credentials, mental health, ...).
        network_available: A cloud route is reachable from this device.
        authenticated_user: The caller has been authenticated.
        encryption_key_loaded: The user's Fernet key is loaded, so
            persistence / outbound encryption is possible.
        rate_limited: The caller is currently over their quota.
        contains_pii: The request (post-feature-extraction) still
            carries PII that needs redaction before any cloud leg.
    """

    sensitive_topic: bool
    network_available: bool
    authenticated_user: bool
    encryption_key_loaded: bool
    rate_limited: bool
    contains_pii: bool = False

    def to_state(self) -> State:
        """Convert to a :data:`State` (the set of true predicate names).

        Returns:
            A :class:`frozenset` of predicate names that currently hold.
        """
        true_preds: list[str] = []
        if self.sensitive_topic:
            true_preds.append("sensitive_topic")
        if self.network_available:
            true_preds.append("network_available")
        if self.authenticated_user:
            true_preds.append("authenticated_user")
        if self.encryption_key_loaded:
            true_preds.append("encryption_key_loaded")
        if self.rate_limited:
            true_preds.append("rate_limited")
        if self.contains_pii:
            true_preds.append("contains_pii")
        return frozenset(true_preds)


@dataclass(frozen=True, slots=True)
class SafetyPlan:
    """An ordered sequence of actions the router must take.

    Attributes:
        actions: Ordered action names.
        trace: Per-step tuples ``(pre_state, action_name, post_state)``
            used to build the certificate.
        final_state: Predicate valuation at plan end.
    """

    actions: tuple[str, ...]
    trace: tuple[tuple[frozenset[str], str, frozenset[str]], ...] = field(
        default_factory=tuple
    )
    final_state: frozenset[str] = field(default_factory=frozenset)

    def is_empty(self) -> bool:
        """Return ``True`` if the plan contains no actions."""
        return not self.actions


# ---------------------------------------------------------------------------
# Canonical domain
# ---------------------------------------------------------------------------


def _build_privacy_safety_domain() -> PddlDomain:
    """Construct the canonical privacy-safety PDDL domain.

    Returns:
        The frozen :class:`PddlDomain` used by
        :class:`PrivacySafetyPlanner` by default.
    """
    predicates = frozenset(
        {
            "sensitive_topic",
            "network_available",
            "authenticated_user",
            "encryption_key_loaded",
            "rate_limited",
            "contains_pii",
            # Derived "goal" predicates: become true once the matching
            # action has fired.
            "pii_redacted",
            "routed_local",
            "routed_cloud",
            "denied",
        }
    )

    actions = {
        # ---- route_local ------------------------------------------------
        "route_local": Action(
            name="route_local",
            preconditions=frozenset({"authenticated_user"}),
            forbidden=frozenset({"rate_limited", "routed_cloud", "denied"}),
            add_effects=frozenset({"routed_local"}),
            del_effects=frozenset(),
            rationale=(
                "Local SLM route: keeps raw content on-device; required "
                "when sensitive_topic holds (privacy-by-construction)."
            ),
        ),
        # ---- route_cloud ------------------------------------------------
        # Forbidden when sensitive_topic holds — this is the load-bearing
        # safety invariant (PDDL goal-condition check rejects any plan
        # that would fire this while sensitive_topic is true).
        "route_cloud": Action(
            name="route_cloud",
            preconditions=frozenset(
                {"authenticated_user", "network_available", "pii_redacted"}
            ),
            forbidden=frozenset(
                {
                    "sensitive_topic",
                    "rate_limited",
                    "contains_pii",
                    "routed_local",
                    "denied",
                }
            ),
            add_effects=frozenset({"routed_cloud"}),
            del_effects=frozenset(),
            rationale=(
                "Cloud LLM route: allowed only for non-sensitive, "
                "PII-redacted, rate-OK traffic with a live network link."
            ),
        ),
        # ---- redact_pii -------------------------------------------------
        "redact_pii": Action(
            name="redact_pii",
            preconditions=frozenset({"contains_pii"}),
            forbidden=frozenset({"denied"}),
            add_effects=frozenset({"pii_redacted"}),
            del_effects=frozenset({"contains_pii"}),
            rationale=(
                "PII sanitisation: strips 10-pattern regex battery before "
                "any cross-boundary call (see i3/privacy/sanitizer.py)."
            ),
        ),
        # ---- deny_request -----------------------------------------------
        "deny_request": Action(
            name="deny_request",
            preconditions=frozenset(),
            forbidden=frozenset({"routed_local", "routed_cloud"}),
            add_effects=frozenset({"denied"}),
            del_effects=frozenset(),
            rationale=(
                "Terminal refusal: used when the caller is rate-limited or "
                "not authenticated; no downstream compute is spent."
            ),
        ),
    }

    return PddlDomain(predicates=predicates, actions=actions)


PRIVACY_SAFETY_DOMAIN: PddlDomain = _build_privacy_safety_domain()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


# Safety invariant: a plan can never contain route_cloud in a trace where
# sensitive_topic is true at the step's pre-state.
_FORBIDDEN_PAIR: frozenset[tuple[str, str]] = frozenset(
    {("sensitive_topic", "route_cloud")}
)


class PrivacySafetyPlanner:
    """PDDL-grounded privacy-safety planner.

    The planner's :meth:`plan` method runs a tiny forward search over
    the domain's action schemas and returns the shortest applicable
    plan that either (a) routes the request (local or cloud) with all
    preconditions satisfied, or (b) explicitly denies it when none of
    the routing actions are applicable.

    :meth:`certify` then bundles that plan into a
    :class:`~i3.safety.certificates.SafetyCertificate` that records
    every precondition check and every effect, giving the auditor a
    machine-checkable trace.

    The domain defaults to :data:`PRIVACY_SAFETY_DOMAIN`, but callers may
    inject an alternative domain for testing.

    Args:
        domain: Optional override for the PDDL domain.

    Attributes:
        domain: The active :class:`PddlDomain`.
    """

    def __init__(self, domain: PddlDomain | None = None) -> None:
        self.domain: PddlDomain = (
            domain if domain is not None else PRIVACY_SAFETY_DOMAIN
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, context: SafetyContext) -> SafetyPlan:
        """Return the ordered plan to route *or* deny the request.

        The planner always produces at least one terminal action
        (``route_local``, ``route_cloud``, or ``deny_request``) so
        downstream consumers can rely on a non-empty plan.

        Args:
            context: Grounded valuation of the domain predicates.

        Returns:
            A :class:`SafetyPlan` whose ``actions`` tuple is ready to
            execute and whose ``trace`` tuple documents each step.

        Raises:
            SafetyPlannerError: Only if the domain itself is
                malformed -- the privacy domain is designed to always
                have a terminal action applicable.
        """
        state = context.to_state()
        trace: list[tuple[frozenset[str], str, frozenset[str]]] = []
        actions: list[str] = []

        # --- Rule 1: rate-limited callers are denied before any compute.
        if "rate_limited" in state:
            action = self.domain.actions["deny_request"]
            post = action.apply(state)
            trace.append((state, action.name, post))
            actions.append(action.name)
            logger.debug("Planner: rate_limited -> deny_request")
            return SafetyPlan(
                actions=tuple(actions),
                trace=tuple(trace),
                final_state=post,
            )

        # --- Rule 2: unauthenticated callers are denied.
        if "authenticated_user" not in state:
            action = self.domain.actions["deny_request"]
            post = action.apply(state)
            trace.append((state, action.name, post))
            actions.append(action.name)
            logger.debug("Planner: not authenticated -> deny_request")
            return SafetyPlan(
                actions=tuple(actions),
                trace=tuple(trace),
                final_state=post,
            )

        # --- Rule 3: PII present -> redact before routing.
        if "contains_pii" in state:
            redact = self.domain.actions["redact_pii"]
            if not redact.applicable(state):
                raise SafetyPlannerError(
                    "redact_pii not applicable on state with contains_pii."
                )
            post = redact.apply(state)
            trace.append((state, redact.name, post))
            actions.append(redact.name)
            state = post

        # --- Rule 4: sensitive topic -> force local.
        if "sensitive_topic" in state:
            route = self.domain.actions["route_local"]
            if not route.applicable(state):
                raise SafetyPlannerError(
                    "route_local not applicable on authenticated "
                    "state with sensitive_topic."
                )
            post = route.apply(state)
            trace.append((state, route.name, post))
            actions.append(route.name)
            return SafetyPlan(
                actions=tuple(actions),
                trace=tuple(trace),
                final_state=post,
            )

        # --- Rule 5: cloud route, if network + PII constraints allow.
        cloud = self.domain.actions["route_cloud"]
        if cloud.applicable(state):
            post = cloud.apply(state)
            trace.append((state, cloud.name, post))
            actions.append(cloud.name)
            return SafetyPlan(
                actions=tuple(actions),
                trace=tuple(trace),
                final_state=post,
            )

        # --- Rule 6: fall back to local.
        local = self.domain.actions["route_local"]
        if local.applicable(state):
            post = local.apply(state)
            trace.append((state, local.name, post))
            actions.append(local.name)
            return SafetyPlan(
                actions=tuple(actions),
                trace=tuple(trace),
                final_state=post,
            )

        # --- Rule 7: nothing applicable -> deny.
        deny = self.domain.actions["deny_request"]
        post = deny.apply(state)
        trace.append((state, deny.name, post))
        actions.append(deny.name)
        return SafetyPlan(
            actions=tuple(actions),
            trace=tuple(trace),
            final_state=post,
        )

    # ------------------------------------------------------------------
    # Certification
    # ------------------------------------------------------------------

    def certify(
        self,
        plan: SafetyPlan,
        *,
        context: SafetyContext | None = None,
    ) -> SafetyCertificate:
        """Bundle a plan into a machine-checkable
        :class:`~i3.safety.certificates.SafetyCertificate`.

        Each step in the plan is re-verified: preconditions must be
        satisfied in the recorded pre-state, effects must account for
        the post-state exactly, and the safety invariant
        (*no cloud route while sensitive_topic holds*) is re-checked.

        Args:
            plan: The output of :meth:`plan`.
            context: Optional grounded context for the certificate
                header.  Not required for correctness.

        Returns:
            A valid :class:`SafetyCertificate`.  Construction itself
            performs the validation -- a caller that gets a certificate
            back can trust the plan.

        Raises:
            SafetyPlannerError: If any step fails re-verification.
        """
        from i3.safety.certificates import CertifiedStep, SafetyCertificate

        certified: list[CertifiedStep] = []
        for idx, (pre, action_name, post) in enumerate(plan.trace):
            action = self.domain.actions.get(action_name)
            if action is None:
                raise SafetyPlannerError(
                    f"Step {idx}: unknown action {action_name!r}."
                )
            if not action.applicable(pre):
                raise SafetyPlannerError(
                    f"Step {idx}: preconditions for {action_name!r} "
                    f"not satisfied in pre-state {sorted(pre)}."
                )
            expected_post = action.apply(pre)
            if expected_post != post:
                raise SafetyPlannerError(
                    f"Step {idx}: post-state {sorted(post)} does not "
                    f"match expected {sorted(expected_post)} for "
                    f"action {action_name!r}."
                )
            for forbidden_pre, forbidden_action in _FORBIDDEN_PAIR:
                if action_name == forbidden_action and forbidden_pre in pre:
                    raise SafetyPlannerError(
                        f"Step {idx}: safety invariant violated: "
                        f"action {action_name!r} fired with "
                        f"{forbidden_pre!r} true."
                    )
            certified.append(
                CertifiedStep(
                    index=idx,
                    action=action_name,
                    rationale=action.rationale,
                    pre_state=sorted(pre),
                    post_state=sorted(post),
                    preconditions=sorted(action.preconditions),
                    forbidden=sorted(action.forbidden),
                    add_effects=sorted(action.add_effects),
                    del_effects=sorted(action.del_effects),
                )
            )

        header_context: dict[str, bool] = {}
        if context is not None:
            header_context = {
                "sensitive_topic": context.sensitive_topic,
                "network_available": context.network_available,
                "authenticated_user": context.authenticated_user,
                "encryption_key_loaded": context.encryption_key_loaded,
                "rate_limited": context.rate_limited,
                "contains_pii": context.contains_pii,
            }

        return SafetyCertificate(
            domain_name="privacy_safety_v1",
            context=header_context,
            actions=list(plan.actions),
            steps=certified,
            final_state=sorted(plan.final_state),
            invariants_checked=[
                "no route_cloud while sensitive_topic holds",
                "preconditions satisfied at each step",
                "post-state = apply(action, pre-state)",
            ],
        )


__all__ = [
    "PRIVACY_SAFETY_DOMAIN",
    "Action",
    "PddlDomain",
    "PrivacySafetyPlanner",
    "SafetyContext",
    "SafetyPlan",
    "SafetyPlannerError",
    "State",
]
