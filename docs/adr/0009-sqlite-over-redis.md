# ADR-0009 — SQLite over Redis for persistence

- **Status**: Accepted
- **Date**: 2026-03-12
- **Deciders**: Tamer Atesyakar
- **Technical area**: persistence

## Context and problem statement { #context }

Two durable stores are needed:

- `UserProfile` — persistent three-timescale user state (Layer 3).
- `DiaryEntry` — embeddings, topics, metrics per interaction (Layer 7).

Both are low-write, moderate-read workloads with a clear "one row per
user" (profiles) or "append-only" (diary) shape. The data is
Fernet-encrypted at rest. The service targets **on-device / single-node**
deployment first, with optional multi-replica deployment behind a sticky
load balancer.

## Decision drivers { #drivers }

- Zero additional infrastructure — the demo must run with no external
  services.
- Embedded and file-based so a single volume mount covers persistence.
- Native async API.
- ACID guarantees for single-node writes.
- Works with Fernet-wrapped BLOBs.

## Considered options { #options }

1. **SQLite** via `aiosqlite`.
2. **Redis**.
3. **PostgreSQL**.

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — SQLite. It fits the "edge / single node"
> posture, has real transactions, and introduces no infrastructure.

### Consequences — positive { #pos }

- No external service. One SQLite file per node covers both tables.
- Real transactions; writes are ACID. A crash mid-update does not split
  a profile.
- `aiosqlite` integrates with the rest of the async pipeline without
  thread-pool shims.
- Backup is trivial — copy the file (or use `.backup`).

### Consequences — negative { #neg }

- Single-writer concurrency. *Mitigation*: WAL mode + a single server
  replica per node; horizontal scaling uses per-node SQLite plus a
  periodic sync.
- Not multi-master. *Mitigation*: we do not need multi-master — sessions
  are sticky per user.
- Binary blobs make ad-hoc queries awkward. *Mitigation*: `i3/privacy/
  encryption.py` has helpers to unwrap a row for debugging.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — Redis { #opt-2 }

- ✅ In-memory speed.
- ❌ Requires a second process / container for the demo.
- ❌ Persistence semantics (AOF / RDB) are configurable but not ACID.
- ❌ Fernet-at-rest is awkward (Redis' own encryption is a paid feature).
- ❌ Does not fit the "one-node, one-file" edge story.

### Option 3 — PostgreSQL { #opt-3 }

- ✅ Industry-standard, rich query language.
- ✅ Logical replication, high availability.
- ❌ Overkill for on-device / single-node scope.
- ❌ Requires operator to run a DB alongside the app.

## References { #refs }

- [Architecture: Layers (Layer 3, Layer 7)](../architecture/layers.md)
- [ADR-0008 — Fernet](0008-fernet-over-custom-crypto.md)
- `i3/user_model/store.py`, `i3/diary/store.py`
