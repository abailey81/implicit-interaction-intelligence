# Falco Runtime Security Rules for I³

[Falco](https://falco.org/) is the CNCF-graduated runtime security engine
that watches Linux syscalls (via a kernel module or eBPF probe) and
matches them against declarative rules. For I³ we ship a small custom
rule set tailored to the service's threat model.

## What we alert on

| Rule                                        | Trigger                                                                 |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `Unexpected access to diary.db`             | any process other than the uvicorn worker opens `/app/data/diary.db`    |
| `Unexpected outbound connection`            | outbound connect to anything not on the `i3_allowed_egress` list        |
| `Unexpected container exec`                 | any `exec` into the i3 container                                        |
| `Unexpected write to /app`                  | write to the read-only rootfs (indicates container tampering)          |

## Install

```bash
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm repo update
helm install falco falcosecurity/falco \
  --namespace falco --create-namespace \
  -f deploy/policy/falco/helm-values.yaml
```

The Helm values inject `i3_rules.yaml` via the chart's `customRules` hook
so operators don't need to rebuild the Falco image.

## Tuning

- Expand the `i3_allowed_egress` list when a new upstream (Anthropic,
  OpenTelemetry collector, etc.) is introduced.
- For high-traffic nodes, raise Falco's evaluation throughput with
  `falcoctl` or the eBPF driver (already default in v0.38+).

## References

- Falco rules reference: <https://falco.org/docs/rules/>
- Falco Helm chart: <https://github.com/falcosecurity/charts/tree/master/charts/falco>
