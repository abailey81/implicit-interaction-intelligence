# ADR-0004 — Privacy by architecture

- **Status**: Accepted
- **Date**: 2026-01-15
- **Deciders**: Tamer Atesyakar
- **Technical area**: cross-cutting (privacy)

## Context and problem statement { #context }

I³ builds a rich behavioural model of its user. That model is what makes
the product useful. The non-negotiable constraint is that this must happen
**without persisting the raw content** of anything the user typed.

"Privacy by policy" — a rule written in a design document that writers
can honour or not — is insufficient. Any future refactor can break it.
The guarantee must be architectural: it must be *impossible* for a future
commit to introduce a raw-text leak without the architecture itself
changing.

## Decision drivers { #drivers }

- Raw user text **must never** be written to disk.
- Raw user text may leave the device only after PII sanitisation and only
  on an explicit cloud-route decision.
- Privacy-sensitive topics **must** bypass the router's learned policy.
- The guarantee must be enforced by the type system and the schema, not
  solely by vigilance.
- Defence in depth: encryption at rest is mandatory, even though nothing
  sensitive is written in the first place.

## Considered options { #options }

1. **Policy-based**: write guidelines and rely on review.
2. **Field-level encryption** of raw text on write.
3. **Privacy by architecture**: schema forbids text columns; router
   override forbids cloud routing for sensitive topics; cloud client
   forbids un-sanitised sends. Fernet-at-rest on top.

## Decision outcome { #outcome }

> **Chosen option**: Option 3. Policy-based is insufficient; field-level
> encryption still writes data that a misconfiguration could decrypt
> wholesale. Privacy by architecture makes the leak impossible at the
> data-model layer, with Fernet and sanitisation as two independent
> defence layers.

### Consequences — positive { #pos }

- A future PR that tries to write a text column cannot compile (the
  `DiaryEntry` type has no string field capable of holding it) and
  cannot run (the auditor's periodic scan would catch a hash collision).
- Cloud egress is pinned to sanitised prompts only, enforced by
  `i3/cloud/client.py`.
- Sensitive-topic routing override runs **before** the bandit's posterior
  sample — not as a downstream filter that could be bypassed.
- Aligns with the GDPR data-minimisation principle out of the box.

### Consequences — negative { #neg }

- No "undo" capability from the diary — the raw text is genuinely gone.
  *Mitigation*: this is the explicit goal.
- Session summaries must be generated from metadata only, which limits
  their expressive range. *Mitigation*: templated summaries with TF-IDF
  topics are sufficient for demo; richer analysis can happen on-device,
  ephemerally.
- Debugging a user-reported issue is harder — no transcript to replay.
  *Mitigation*: traces include structured pipeline state (no text),
  which is usually enough.

## Pros and cons of the alternatives { #alternatives }

### Option 1 — Policy-based { #opt-1 }

- Yes Zero implementation effort.
- No Fails silently on the first refactor that forgets the rule.
- No Cannot satisfy a "no raw text on disk" auditor.

### Option 2 — Field-level encryption of raw text { #opt-2 }

- Yes Reversible for legitimate debugging.
- No Still writes data — a key-management failure or a mis-grant is a
  wholesale leak.
- No Complicates backup and migration.
- No Sends a worse signal about the product's posture.

## References { #refs }

- [Privacy architecture](../architecture/privacy.md)
- [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)
- [ADR-0008 — Fernet](0008-fernet-over-custom-crypto.md)
- GDPR Art. 5(1)(c) data minimisation principle.
