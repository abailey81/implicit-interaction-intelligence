# ─── admission.rego ───────────────────────────────────────────────────────
#
# OPA/Gatekeeper admission rules for the I³ service. The package exposes a
# set of `deny[msg]` rules; any non-empty `deny` collection rejects the
# request. This mirrors the Kyverno bundle under ../kyverno/.
#
# Rego v1 syntax.  To lint locally:
#
#   opa check -v1 deploy/policy/opa/
#   opa test deploy/policy/opa/ -v
#
# References:
#   • Rego v1 reference:   https://www.openpolicyagent.org/docs/latest/policy-language/
#   • Gatekeeper library:  https://open-policy-agent.github.io/gatekeeper-library/

package i3.policy.admission

import rego.v1

import data.i3.policy.data as allow

# --------------------------------------------------------------------------
# Rule 1 — reject `:latest` or untagged images
# --------------------------------------------------------------------------

deny contains msg if {
    input.request.kind.kind == "Pod"
    some container in input.request.object.spec.containers
    endswith(container.image, ":latest")
    msg := sprintf(
        "container '%s' uses disallowed `:latest` tag (image=%s)",
        [container.name, container.image],
    )
}

deny contains msg if {
    input.request.kind.kind == "Pod"
    some container in input.request.object.spec.containers
    not contains(container.image, ":")
    msg := sprintf(
        "container '%s' image '%s' is untagged (defaults to :latest)",
        [container.name, container.image],
    )
}

# --------------------------------------------------------------------------
# Rule 2 — require hardened securityContext in the i3 namespace
# --------------------------------------------------------------------------

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    some container in input.request.object.spec.containers
    not container.securityContext.runAsNonRoot == true
    msg := sprintf(
        "container '%s' must set securityContext.runAsNonRoot=true",
        [container.name],
    )
}

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    some container in input.request.object.spec.containers
    not container.securityContext.readOnlyRootFilesystem == true
    msg := sprintf(
        "container '%s' must set securityContext.readOnlyRootFilesystem=true",
        [container.name],
    )
}

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    some container in input.request.object.spec.containers
    container.securityContext.allowPrivilegeEscalation == true
    msg := sprintf(
        "container '%s' must set allowPrivilegeEscalation=false",
        [container.name],
    )
}

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    profile_type := object.get(
        input.request.object.spec, ["securityContext", "seccompProfile", "type"], "",
    )
    profile_type != "RuntimeDefault"
    msg := "pod must use seccompProfile.type=RuntimeDefault"
}

# --------------------------------------------------------------------------
# Rule 3 — only signed images from allow-listed registries / signers
# --------------------------------------------------------------------------

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    some container in input.request.object.spec.containers
    not image_registry_allowed(container.image)
    msg := sprintf(
        "container '%s' image '%s' is not from an allow-listed registry",
        [container.name, container.image],
    )
}

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    some container in input.request.object.spec.containers
    sig := object.get(container, "_signature", {})
    not sig.signer in allow.allowed_signers
    msg := sprintf(
        "container '%s' image signer '%v' not on allow-list",
        [container.name, object.get(sig, "signer", "<missing>")],
    )
}

image_registry_allowed(image) if {
    some prefix in allow.allowed_registries
    contains(image, prefix)
}

# --------------------------------------------------------------------------
# Rule 4 — namespace must contain a NetworkPolicy
# --------------------------------------------------------------------------

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    count(input.context.namespace_networkpolicies) == 0
    msg := sprintf(
        "namespace '%s' has no NetworkPolicy — scheduling is blocked",
        [input.request.namespace],
    )
}

# --------------------------------------------------------------------------
# Rule 5 — block admission on critical/high CVE findings
# --------------------------------------------------------------------------

deny contains msg if {
    input.request.kind.kind == "Pod"
    input.request.namespace == allow.i3_namespace
    some finding in input.context.scan_findings
    finding.severity in allow.blocking_severities
    msg := sprintf(
        "image '%s' has blocking CVE %s (%s)",
        [finding.image, finding.id, finding.severity],
    )
}
