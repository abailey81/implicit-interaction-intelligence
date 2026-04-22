"""Machine-checkable safety certificates for the PDDL planner.

A :class:`SafetyCertificate` is the audit artefact produced by
:meth:`i3.safety.pddl_planner.PrivacySafetyPlanner.certify`.  It is
shaped so that every field needed to *re-verify* the plan -- without
re-running the planner -- is present:

* ``steps`` contains the ordered list of :class:`CertifiedStep`
  entries, each carrying its preconditions, forbidden set, add / del
  effects, pre- and post-states.
* ``invariants_checked`` records which safety invariants were
  re-verified during certification.
* ``context`` records the grounded input valuation for the header.

YAML is the preferred serialisation format because the certificate is
designed for human review as well as programmatic ingestion, and YAML's
block layout reads as an indented reasoning trace.
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CertifiedStep(BaseModel):
    """One verified step of a :class:`SafetyCertificate`.

    Attributes:
        index: 0-based step position within the plan.
        action: The PDDL action name that fired at this step.
        rationale: Human-readable justification drawn from the domain.
        pre_state: Sorted list of predicates true immediately before
            the action fires.
        post_state: Sorted list of predicates true immediately after.
        preconditions: The action's declared preconditions.
        forbidden: The action's declared forbidden-predicates.
        add_effects: Predicates the action adds to the state.
        del_effects: Predicates the action removes from the state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    action: str
    rationale: str
    pre_state: list[str]
    post_state: list[str]
    preconditions: list[str]
    forbidden: list[str]
    add_effects: list[str]
    del_effects: list[str]


class SafetyCertificate(BaseModel):
    """Top-level certificate envelope for a planner run.

    The certificate is YAML-round-trippable via
    :func:`certificate_to_yaml` and :func:`certificate_from_yaml`.

    Attributes:
        domain_name: Identifier of the PDDL domain (versioned).
        context: Grounded predicate valuations supplied as the planner's
            :class:`~i3.safety.pddl_planner.SafetyContext`.
        actions: Ordered action names (same as ``plan.actions``).
        steps: Fully-verified :class:`CertifiedStep` entries.
        final_state: The sorted predicate set after the last step.
        invariants_checked: Human-readable list of safety invariants
            that were re-verified during certification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain_name: str
    context: dict[str, bool] = Field(default_factory=dict)
    actions: list[str]
    steps: list[CertifiedStep]
    final_state: list[str]
    invariants_checked: list[str]

    def summary(self) -> str:
        """Return a one-line human summary of the certificate.

        Returns:
            A short string like ``"privacy_safety_v1: redact_pii, route_local
            (2 steps)"`` suitable for log lines.
        """
        joined = ", ".join(self.actions) if self.actions else "<empty>"
        return (
            f"{self.domain_name}: {joined} "
            f"({len(self.steps)} step{'s' if len(self.steps) != 1 else ''})"
        )


# ---------------------------------------------------------------------------
# YAML serialisation
# ---------------------------------------------------------------------------


def certificate_to_yaml(certificate: SafetyCertificate) -> str:
    """Serialise a certificate to a YAML block.

    Args:
        certificate: The certificate to serialise.

    Returns:
        A YAML document string.  Lists are rendered block-style and
        keys are emitted in insertion order for reproducible diffs.
    """
    payload: dict[str, Any] = certificate.model_dump()
    return yaml.safe_dump(
        payload,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def certificate_from_yaml(text: str) -> SafetyCertificate:
    """Parse a certificate YAML document back into a Pydantic object.

    Args:
        text: YAML string produced by :func:`certificate_to_yaml`.

    Returns:
        The reconstructed :class:`SafetyCertificate`.

    Raises:
        ValueError: If the YAML is empty, is not a mapping, or does
            not validate against the certificate schema.
    """
    # SEC: ``yaml.safe_load`` is used to avoid arbitrary-object
    # construction from untrusted certificate blobs.
    loaded = yaml.safe_load(text)
    if loaded is None:
        raise ValueError("Empty YAML document cannot be a certificate.")
    if not isinstance(loaded, dict):
        raise ValueError(
            "Certificate YAML must deserialise to a mapping, "
            f"got {type(loaded).__name__}."
        )
    return SafetyCertificate.model_validate(loaded)


__all__ = [
    "CertifiedStep",
    "SafetyCertificate",
    "certificate_from_yaml",
    "certificate_to_yaml",
]
