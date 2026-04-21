# Contributing

Welcome! Contributions to I³ are very welcome — whether that is a bug fix,
a new adapter, an ADR, a tuning experiment, or a documentation improvement.

!!! tip "Canonical guide"
    This page is an overview. The canonical contributor guide is
    [CONTRIBUTING.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/CONTRIBUTING.md)
    in the repository root. Read it before opening a pull request.

## Start here { #start }

1. Read the [Code of Conduct](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/CODE_OF_CONDUCT.md).
2. Read the [Security Policy](../security/index.md) — report
   vulnerabilities privately.
3. Install the dev environment as in [Installation](../getting-started/installation.md#manual).
4. Enable pre-commit: `poetry run pre-commit install`.

## What to work on { #topics }

| I'd like to… | Start with… |
|:-------------|:------------|
| Fix a bug | Open a GitHub Issue or small PR. |
| Add an adapter | `i3/adaptation/dimensions.py`; see [Layers § L4](../architecture/layers.md#l4). |
| Improve the TCN | `i3/encoder/`; see [Research: contrastive loss](../research/contrastive_loss.md). |
| Propose an architectural change | Draft an [ADR](../adr/template.md). |
| Improve docs | Edit pages in `docs/`. This site auto-deploys from `main`. |

## Conventions { #conventions }

### Code style

- Python 3.10+.
- `ruff` for lint and format (line length 100).
- `mypy` with `check_untyped_defs = true` (see `pyproject.toml`).
- Docstrings in Google style; `mkdocstrings` renders them.
- No emojis in source.

### Commit messages

- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `perf:`, `chore:`, `security:`).
- Present tense, imperative mood.
- Reference an issue when one exists.

### Tests

- Every non-trivial change has a test in `tests/`.
- `make test` must pass before review.
- Property tests live under `tests/test_*_property.py`; security tests
  under `tests/test_*_security.py`.

### Security-sensitive changes

If your change touches any of:

- `i3/privacy/**`
- `server/websocket.py`, `server/middleware.py`
- `i3/cloud/**`
- `i3/router/sensitivity.py`

…then label your PR `security` and expect an extra review pass. Never
weaken a constraint without an ADR.

## Pull-request checklist { #pr-checklist }

- [ ] `make test` passes.
- [ ] `make lint` passes.
- [ ] `make security` passes (`bandit`, `pip-audit`).
- [ ] Changelog entry added in the `Unreleased` section.
- [ ] Docs updated if user-facing behaviour changed.
- [ ] If architectural, an ADR has been drafted and linked.
- [ ] No `print()` statements left behind.
- [ ] No Fernet key, Anthropic key, or raw message logged.

## Documentation { #docs }

The site you are reading is built with MkDocs Material. To iterate:

```bash
poetry install --with docs
poetry run mkdocs serve --strict
# Open http://127.0.0.1:8000
```

- **Page structure**: one H1, short opening paragraph, then content.
- **Admonitions** (`!!! note`, `!!! tip`, `!!! warning`) are encouraged.
- **Mermaid diagrams** go in fenced blocks with the `mermaid` language.
- **Cross-references**: prefer relative links to anchors.
- **Strict build**: the CI builds with `--strict`; broken links fail the
  pipeline.

## Review and merge { #review }

- Small PRs merge fast; split large changes into reviewable pieces.
- A maintainer will squash-merge. Keep the PR description rich — it
  becomes the commit message.
- After merge, the docs site redeploys within a couple of minutes via
  GitHub Actions.

## Further reading { #further }

- [Code of Conduct](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/CODE_OF_CONDUCT.md)
- [CONTRIBUTING.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/CONTRIBUTING.md)
- [Security policy](../security/index.md)
- [ADR index](../adr/index.md)
