# ADR-0008 — Fernet over custom crypto

- **Status**: Accepted
- **Date**: 2026-03-04
- **Deciders**: Tamer Atesyakar
- **Technical area**: privacy, persistence

## Context and problem statement { #context }

`UserProfile` rows and diary `embedding` blobs must be encrypted at rest.
The application needs a symmetric, authenticated encryption primitive
with a small API, a stable library, and good defaults. Key rotation is
important but should not require a downtime.

## Decision drivers { #drivers }

- Authenticated encryption (integrity + confidentiality).
- Vetted primitive; no bespoke construction.
- First-class key-rotation support.
- Stable, well-audited Python library.
- Small API surface so the wrapper at `i3/privacy/encryption.py` is
  easy to review.

## Considered options { #options }

1. **Fernet** (`cryptography.fernet.Fernet` / `MultiFernet`).
2. **AES-GCM** via `cryptography.hazmat.primitives.aead.AESGCM`.
3. **ChaCha20-Poly1305** via the same library.
4. **Custom cipher** (e.g. AES-CTR + HMAC), rolled in our wrapper.

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — Fernet. It gives us AEAD (AES-128-CBC +
> HMAC-SHA256) with a trivially small API, native `MultiFernet` key
> rotation, and no nonce-management responsibility on our side.

### Consequences — positive { #pos }

- Misuse-resistant: nonces, padding, and HMAC are all handled by the
  library.
- `MultiFernet` lets us deploy `new,old` key chains for zero-downtime
  rotation.
- URL-safe base64 keys are trivially storable in secrets systems and
  `.env` files.
- Library is part of `cryptography`, which we already depend on for
  TLS (transitively).

### Consequences — negative { #neg }

- AES-128, not AES-256. *Mitigation*: 128-bit symmetric key is beyond
  brute-force today; the threat model is endpoint compromise, not
  cryptographic attack.
- CBC + HMAC is slightly slower than AES-GCM on AES-NI. *Mitigation*:
  profile budget is negligible at our read/write volume.
- Ciphertext includes a timestamp; uniquely-timed ciphertexts fingerprint
  write order. *Mitigation*: acceptable under our threat model — any
  attacker with the key + file also has row-level access.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — AES-GCM { #opt-2 }

- Yes Faster on modern CPUs with AES-NI.
- Yes 256-bit keys.
- No Caller responsible for nonce uniqueness.
- No Multi-key rotation must be hand-rolled.

### Option 3 — ChaCha20-Poly1305 { #opt-3 }

- Yes Great on devices without AES-NI.
- No Same nonce-management footgun.
- No No library-level multi-key rotation.

### Option 4 — Custom AES-CTR + HMAC { #opt-4 }

- No Cryptographic construction we would have to audit ourselves.
- No "Don't roll your own crypto" — especially in a privacy-first project.
- No Zero upside.

## References { #refs }

- [Privacy architecture](../architecture/privacy.md)
- [ADR-0004 — Privacy by architecture](0004-privacy-by-architecture.md)
- [cryptography.io docs](https://cryptography.io/en/latest/fernet/)
