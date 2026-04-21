# OPA Rego Policies for I³

This directory holds [Rego][rego-v1] admission-control policies that are
functionally equivalent to the Kyverno bundle under
[`../kyverno/`](../kyverno/README.md). Use these for clusters running
[OPA Gatekeeper][gk] rather than Kyverno.

## Files

| File                 | Purpose                                                            |
| -------------------- | ------------------------------------------------------------------ |
| `admission.rego`     | `deny` rules for signed images, non-root, network policy, `:latest` |
| `data.rego`          | Allow-lists (signers, registries, acceptable CVE severities)       |
| `test_admission.rego`| Rego unit tests using the `test_` naming convention               |

## Run tests

```bash
# Install the OPA CLI from https://www.openpolicyagent.org/docs/latest/#running-opa
opa test deploy/policy/opa/ -v
```

## Deploy with Gatekeeper

Wrap each rule in a `ConstraintTemplate` and `Constraint` resource when
shipping to Gatekeeper — see:
<https://open-policy-agent.github.io/gatekeeper/website/docs/howto>

## Rego version

All files begin with `import rego.v1` and use the v1 future syntax
(`if`, `contains`, `in`). Older v0 syntax is intentionally avoided.

## References

- OPA / Rego docs: <https://www.openpolicyagent.org/docs/latest/policy-language/>
- Gatekeeper docs: <https://open-policy-agent.github.io/gatekeeper/>

[rego-v1]: https://www.openpolicyagent.org/docs/latest/opa-1/
[gk]: https://open-policy-agent.github.io/gatekeeper/
