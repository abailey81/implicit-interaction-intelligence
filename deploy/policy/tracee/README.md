# Tracee eBPF Runtime Tracing for I³

[Tracee](https://aquasecurity.github.io/tracee/) is Aqua Security's
eBPF-based runtime-security and forensics tool. Where Falco focuses on
*alerting* from syscalls, Tracee focuses on *tracing* — capturing a rich
stream of security-relevant events (file I/O, network, process exec,
container lifecycle, kernel-object creation) that you can archive for
forensics or feed into a SIEM.

## Install

```bash
helm repo add aqua https://aquasecurity.github.io/helm-charts/
helm repo update
helm install tracee aqua/tracee \
  --namespace tracee-system --create-namespace \
  --set-file policies.i3\.yaml=deploy/policy/tracee/policies/i3_policy.yaml
```

## Policy file

`policies/i3_policy.yaml` is a Tracee `Policy` v1beta1 manifest that:

- scopes events to the `i3` namespace,
- traces file access (open/read/write of sensitive paths),
- traces outbound network connections,
- traces process exec and kernel-module loads.

## Integration with Falco

Falco alerts on the "did something bad happen" side; Tracee is the
"what exactly happened and in what order" side. Pipe Tracee events to
Loki / OpenSearch for replayable forensics.

## References

- Tracee docs: <https://aquasecurity.github.io/tracee/>
- Policy CRD reference: <https://aquasecurity.github.io/tracee/latest/docs/policies/>
