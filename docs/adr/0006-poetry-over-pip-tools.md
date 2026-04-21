# ADR-0006 — Poetry over pip-tools

- **Status**: Accepted
- **Date**: 2026-02-02
- **Deciders**: Tamer Atesyakar
- **Technical area**: packaging

## Context and problem statement { #context }

We need a reproducible Python packaging story that:

- Locks the full transitive dependency graph.
- Distinguishes runtime, dev, security, and docs dependency groups.
- Pins floors (we enforce `torch ≥ 2.6`, `cryptography ≥ 43.0`,
  `fastapi ≥ 0.115`) and handles the Python range `3.10–3.12`.
- Publishes the package as a wheel for the demo image and for library
  consumption.
- Is quickly learnable for a new contributor.

## Decision drivers { #drivers }

- Lock file for reproducible installs.
- First-class dependency groups.
- A usable CLI for day-to-day work.
- Widely adopted so contributors recognise it.

## Considered options { #options }

1. **Poetry 1.8+**.
2. **pip-tools** (`pyproject.toml` + `requirements*.txt` generated via
   `pip-compile`).
3. **uv** + **hatch**.

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — Poetry. It is the shortest path to
> groups + lock file + publish-as-wheel while remaining familiar to most
> Python contributors.

### Consequences — positive { #pos }

- `pyproject.toml` is a single source of truth; `poetry install --with
  dev,security,docs` is the common path.
- `poetry.lock` gives reproducible installs across CI, Docker, and
  local.
- `poetry publish` builds a clean wheel for HuggingFace-free edge
  deployment.
- Group semantics (`--with dev`, `--only main`) match our Docker multi-
  stage build cleanly.

### Consequences — negative { #neg }

- Poetry's resolver is slower than `uv`. *Mitigation*: acceptable at our
  dependency count; CI uses `--no-interaction` + cached virtualenv.
- Some corporate proxies have quirky interactions with Poetry's PyPI
  client. *Mitigation*: `POETRY_HTTP_BASIC_*` and `--repositories` work
  around it.
- `poetry export` is in a plugin; we rely on `poetry install` directly
  in the Dockerfile instead of exporting a `requirements.txt`.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — pip-tools { #opt-2 }

- Yes Minimal, composable.
- No No native group concept — requires multiple `requirements*.in`
  files and tooling around them.
- No No built-in publish flow.
- No Discovery is worse for new contributors.

### Option 3 — uv + hatch { #opt-3 }

- Yes Very fast resolver.
- Yes PEP 621 purity with `hatch`.
- No uv is still maturing for group workflows (late 2025 /2026).
- No Ecosystem documentation assumes Poetry or pip-tools.
- No Two tools to learn instead of one.

## References { #refs }

- [Installation](../getting-started/installation.md)
- [pyproject.toml](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/pyproject.toml)
