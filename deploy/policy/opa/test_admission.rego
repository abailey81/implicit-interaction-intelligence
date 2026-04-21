# ─── test_admission.rego ─────────────────────────────────────────────────
#
# Rego unit tests.  Run with:
#
#   opa test deploy/policy/opa/ -v
#
# Convention: every test rule starts with `test_` (OPA's default).

package i3.policy.admission_test

import rego.v1

import data.i3.policy.admission

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_hardened_container := {
    "name": "i3",
    "image": "ghcr.io/abailey81/implicit-interaction-intelligence:v1.0.0",
    "securityContext": {
        "runAsNonRoot": true,
        "readOnlyRootFilesystem": true,
        "allowPrivilegeEscalation": false,
        "capabilities": {"drop": ["ALL"]},
    },
    "_signature": {
        "signer": "https://github.com/abailey81/implicit-interaction-intelligence/.github/workflows/docker.yml@refs/heads/main",
    },
}

_good_input := {
    "request": {
        "kind": {"kind": "Pod"},
        "namespace": "i3",
        "object": {
            "spec": {
                "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
                "containers": [_hardened_container],
            },
        },
    },
    "context": {
        "namespace_networkpolicies": [{"name": "default-deny"}],
        "scan_findings": [],
    },
}

# --------------------------------------------------------------------------
# Positive cases
# --------------------------------------------------------------------------

test_hardened_pod_is_admitted if {
    count(admission.deny) == 0 with input as _good_input
}

# --------------------------------------------------------------------------
# :latest tag
# --------------------------------------------------------------------------

test_reject_latest_tag if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/image", "value": "ghcr.io/abailey81/implicit-interaction-intelligence:latest"}],
    )
    count(admission.deny) > 0 with input as bad
}

test_reject_untagged_image if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/image", "value": "ghcr.io/abailey81/implicit-interaction-intelligence"}],
    )
    count(admission.deny) > 0 with input as bad
}

# --------------------------------------------------------------------------
# securityContext
# --------------------------------------------------------------------------

test_reject_run_as_root if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/securityContext/runAsNonRoot", "value": false}],
    )
    count(admission.deny) > 0 with input as bad
}

test_reject_writable_rootfs if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/securityContext/readOnlyRootFilesystem", "value": false}],
    )
    count(admission.deny) > 0 with input as bad
}

test_reject_privilege_escalation if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/securityContext/allowPrivilegeEscalation", "value": true}],
    )
    count(admission.deny) > 0 with input as bad
}

test_reject_missing_seccomp if {
    bad := json.patch(
        _good_input,
        [{"op": "remove", "path": "/request/object/spec/securityContext"}],
    )
    count(admission.deny) > 0 with input as bad
}

# --------------------------------------------------------------------------
# Signer / registry allow-list
# --------------------------------------------------------------------------

test_reject_unknown_registry if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/image", "value": "docker.io/library/nginx:1.27"}],
    )
    count(admission.deny) > 0 with input as bad
}

test_reject_unknown_signer if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/request/object/spec/containers/0/_signature/signer", "value": "https://example.com/evil"}],
    )
    count(admission.deny) > 0 with input as bad
}

# --------------------------------------------------------------------------
# NetworkPolicy requirement
# --------------------------------------------------------------------------

test_reject_no_networkpolicy if {
    bad := json.patch(
        _good_input,
        [{"op": "replace", "path": "/context/namespace_networkpolicies", "value": []}],
    )
    count(admission.deny) > 0 with input as bad
}

# --------------------------------------------------------------------------
# CVE findings
# --------------------------------------------------------------------------

test_reject_on_critical_cve if {
    bad := json.patch(
        _good_input,
        [{"op": "add", "path": "/context/scan_findings/-", "value": {
            "image": "ghcr.io/abailey81/implicit-interaction-intelligence:v1.0.0",
            "id": "CVE-2026-0001",
            "severity": "CRITICAL",
        }}],
    )
    count(admission.deny) > 0 with input as bad
}

test_allow_low_severity_cve if {
    good := json.patch(
        _good_input,
        [{"op": "add", "path": "/context/scan_findings/-", "value": {
            "image": "ghcr.io/abailey81/implicit-interaction-intelligence:v1.0.0",
            "id": "CVE-2026-0002",
            "severity": "LOW",
        }}],
    )
    count(admission.deny) == 0 with input as good
}

# --------------------------------------------------------------------------
# Non-i3 namespaces bypass the namespace-scoped rules
# --------------------------------------------------------------------------

test_other_namespace_not_forced_nonroot if {
    other := json.patch(
        _good_input,
        [
            {"op": "replace", "path": "/request/namespace", "value": "kube-system"},
            {"op": "replace", "path": "/request/object/spec/containers/0/securityContext/runAsNonRoot", "value": false},
        ],
    )
    # The only possible deny here would be :latest, which doesn't apply.
    count(admission.deny) == 0 with input as other
}
