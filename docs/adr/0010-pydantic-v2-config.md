# ADR-0010 — Pydantic v2 for configuration

- **Status**: Accepted
- **Date**: 2026-03-25
- **Deciders**: Tamer Atesyakar
- **Technical area**: configuration

## Context and problem statement { #context }

I³ has ~17 nested submodels of configuration — encoder, adaptation,
router, SLM, cloud, diary, privacy, profiling, server, observability.
Each has field-level constraints (e.g. `slm.d_model % slm.n_heads == 0`,
`server.cors_origins` must parse as URLs) that we want enforced at load
time, with no chance of runtime drift.

## Decision drivers { #drivers }

- Field-level validation.
- Cross-field (root) validation for non-trivial rules.
- Immutable (frozen) models to prevent runtime mutation.
- YAML + env-var overlay.
- Fast enough not to matter at startup.

## Considered options { #options }

1. **Pydantic v2** with `pydantic-settings`.
2. **Pydantic v1**.
3. **Plain `dataclass` + manual validation**.
4. **Hydra** (`omegaconf`).

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — Pydantic v2. Frozen models, fast
> validation (core written in Rust), richer field validators, better
> serialisation, and direct compatibility with FastAPI.

### Consequences — positive { #pos }

- `model_config = ConfigDict(frozen=True)` makes configs truly
  immutable at runtime.
- `pydantic-settings` gives us env-var overlays out of the box.
- Error messages include the exact failing field path — we surface them
  directly from the server start-up failure.
- Same models are reused by FastAPI for request validation.

### Consequences — negative { #neg }

- The v1 → v2 migration has breaking surface area for contributors who
  remember v1. *Mitigation*: `CONTRIBUTING.md` notes the version and
  links to the migration guide.
- `@validator` is `@field_validator`, `@root_validator` is
  `@model_validator`, and `Config` is `model_config = ConfigDict(...)`.
  Slightly more ceremony, but more explicit. *Mitigation*: consistent
  patterns across the config tree.
- v2 is stricter on implicit coercions. *Mitigation*: we prefer strict
  behaviour in a config layer.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — Pydantic v1 { #opt-2 }

- ✅ Familiar.
- ❌ Deprecated upstream. New FastAPI versions expect v2.
- ❌ Slower validation.
- ❌ `frozen` is awkward pre-v2.

### Option 3 — dataclass + manual validation { #opt-3 }

- ✅ No third-party dependency.
- ❌ Every cross-field rule written by hand.
- ❌ YAML / env overlays require custom parsers.
- ❌ Error messages are bespoke.

### Option 4 — Hydra / omegaconf { #opt-4 }

- ✅ Excellent CLI composition, good for research sweeps.
- ❌ Config is *not* a typed object — `DictConfig` loses IDE help.
- ❌ Field validation is ad-hoc.
- ❌ Overkill for our non-sweep production config.

## References { #refs }

- [Getting Started: Configuration](../getting-started/configuration.md)
- [Pydantic v2 docs](https://docs.pydantic.dev/)
- `i3/config.py`
