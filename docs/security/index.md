# Security

I³ takes a **privacy-first, architecture-first** posture. This page is a
pointer to the canonical policy and an operational summary for deployers.

!!! warning "Report vulnerabilities privately"
    **tamer.atesyakar@bk.ru** — acknowledged within 2 business days.
    Do **not** open a public issue. See the canonical
    [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md).

## Canonical policy { #policy }

The source of truth is
[`SECURITY.md`](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)
in the repository. It documents:

- Supported versions and patch cadence.
- Reporting procedure and encryption key on request.
- Security architecture — privacy by architecture, defence in depth.
- Threat model, out-of-scope exclusions, and known limitations.

## Guarantees at a glance { #guarantees }

| Guarantee | Enforced by |
|:----------|:-----------|
| Raw user text is never written to disk        | Schema (no text column), privacy auditor |
| Raw user text leaves the device only sanitised| `i3/cloud/client.py` pre-send scan |
| Sensitive topics bypass the learned router    | `i3/router/router.py::IntelligentRouter` privacy override |
| Profiles at rest are Fernet-encrypted         | `i3/privacy/encryption.py` |
| User ids cannot contain path traversal        | Anchored regex in REST and WS |
| WebSocket origins are allow-listed            | `server/websocket.py::_validate_origin` |

## Supported versions { #versions }

| Version | Supported |
|:--------|:---------:|
| 1.0.x   | yes |
| < 1.0   | no  |

## Reporting a vulnerability { #reporting }

1. Email **tamer.atesyakar@bk.ru**.
2. Include reproduction steps, affected version / commit, and any
   proposed mitigation.
3. Expect acknowledgement within 2 business days and a patch target of
   14 days for critical issues.
4. Coordinate disclosure with the maintainer before going public.

## Operational security checklist { #checklist }

Before deploying to production:

- [x] `I3_DEMO_MODE` is **not** set.
- [x] `I3_CORS_ORIGINS` is a concrete list (never `*`).
- [x] `I3_ENCRYPTION_KEY` is stored in a secrets manager and backed up.
- [x] `ANTHROPIC_API_KEY` is stored in a secrets manager; PII
      sanitisation is on.
- [x] TLS with HSTS and a pinned cert chain.
- [x] Rate limits validated end-to-end.
- [x] `/demo/*` routes denied at the ingress.
- [x] OTel collector receiving traces; Prometheus scraping `/metrics`.
- [x] Privacy-auditor alert wired to on-call.
- [x] Backup of `data/profiles.sqlite` scheduled and tested.

## Known limitations { #limitations }

- **No caller authentication** in the shipped REST / WS surface. Any
  client that knows a `user_id` may claim it. Production must layer JWT
  or mTLS. See [REST API § Security-relevant middleware](../api/rest.md#security).
- **Cloud summariser**, when enabled, receives aggregated metadata
  (topics, route counts, adaptation trajectory). Disable in the config
  if your policy forbids metadata egress.
- **Key loss is permanent** by design. Back up the Fernet key.

## Related documents { #related }

- [Privacy architecture](../architecture/privacy.md)
- [ADR-0004 — Privacy by architecture](../adr/0004-privacy-by-architecture.md)
- [ADR-0008 — Fernet](../adr/0008-fernet-over-custom-crypto.md)
- [Runbook](../operations/runbook.md)
- [Model Card § Ethics](../model_card.md#ethics)
