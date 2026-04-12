# Contributing to Implicit Interaction Intelligence (I³)

Thank you for your interest in contributing to I³! This project is an ambient AI
companion system built around privacy-by-architecture, from-scratch ML components,
and edge-first design. We welcome contributions of all sizes — bug fixes,
features, documentation, tests, and even just thoughtful issue reports.

This document describes the development workflow, coding standards, and review
process. Please read it before submitting your first pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Architectural Principles](#architectural-principles)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running the Project](#running-the-project)
- [Code Style](#code-style)
- [Testing](#testing)
- [Git Workflow](#git-workflow)
- [Conventional Commits](#conventional-commits)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Security Vulnerabilities](#security-vulnerabilities)
- [Documentation](#documentation)
- [Release Process](#release-process)

---

## Code of Conduct

This project and everyone participating in it is governed by the
[I³ Code of Conduct](CODE_OF_CONDUCT.md), adapted from Contributor Covenant v2.1.
By participating, you are expected to uphold this code. Report unacceptable
behavior to **tamer.atesyakar@bk.ru**.

## Architectural Principles

Before contributing, please understand the non-negotiable principles that
shape this codebase. PRs that violate these will be asked to rework.

### 1. Privacy by Architecture

- **No raw user data leaves the device** without explicit user opt-in.
- All on-disk user state is encrypted with Fernet (AES-128-CBC + HMAC).
- PII sanitizer runs on every text input before it reaches the SLM.
- Telemetry is opt-in, aggregated, and never includes free-form text.
- Never introduce unencrypted logging of user interaction content.

### 2. From-Scratch Components

The research story of I³ is that core ML primitives are implemented from
scratch in PyTorch, not pulled from HuggingFace. This applies to:

- The TCN encoder (dilated causal convolutions, layer norm, residuals)
- The SLM (tokenizer, token/position embeddings, transformer blocks,
  cross-attention conditioning, LM head)
- NT-Xent contrastive loss
- Thompson sampling bandit with Laplace posterior

If your PR would replace one of these with a library call, please open a
discussion first. There may be good reasons (speed, correctness), but the
trade-off needs to be explicit.

### 3. Edge-First Performance

I³ targets consumer Huawei devices (Kirin 9000-class SoCs). Every PR should
consider latency, memory, and battery. Use `scripts/profile_edge.py` to
benchmark changes to inference paths.

### 4. Reproducibility

- Seed every random source (numpy, torch, python `random`).
- Pin versions in `pyproject.toml`.
- Never introduce non-determinism into training without a flag to disable it.

---

## Getting Started

### Prerequisites

- **Python 3.10, 3.11, or 3.12** (we test on all three)
- **Poetry 1.8+** for dependency management
- **Git 2.30+**
- (Optional) **CUDA 12.1+** if you want to train on GPU
- (Optional) **Docker** if you prefer containerized development

### Fork and Clone

```bash
# 1. Fork the repo on GitHub (click "Fork" top-right)
# 2. Clone your fork locally
git clone https://github.com/<your-username>/implicit-interaction-intelligence.git
cd implicit-interaction-intelligence

# 3. Add upstream remote so you can pull in latest main
git remote add upstream https://github.com/abailey81/implicit-interaction-intelligence.git
git fetch upstream
```

## Development Setup

### 1. Install Poetry

```bash
# macOS / Linux
curl -sSL https://install.python-poetry.org | python3 -

# Or via pipx (recommended)
pipx install poetry==1.8.0
```

### 2. Install Dependencies

```bash
poetry install --with dev
```

This installs the project in editable mode along with all development
dependencies (pytest, ruff, mypy, bandit, pre-commit, etc.).

### 3. Activate the Virtualenv

```bash
poetry shell        # opens a subshell with the venv active
# OR, for one-off commands:
poetry run <command>
```

### 4. Install Pre-commit Hooks

```bash
poetry run pre-commit install
```

This sets up the hooks defined in `.pre-commit-config.yaml` so that `ruff`,
`bandit`, trailing-whitespace checks, and secret scanning run automatically
on every commit. **Please do not bypass these hooks** unless you have a
very good reason and explicit reviewer approval.

### 5. Generate an Encryption Key

The project requires a Fernet encryption key at runtime. For development:

```bash
poetry run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the output into your `.env` file (see `.env.example` for the full list
of environment variables):

```bash
cp .env.example .env
# edit .env and set I3_ENCRYPTION_KEY=<your-key>
```

## Running the Project

### Start the Server

```bash
make server
# or: poetry run uvicorn server.main:app --reload
```

The FastAPI server will start on `http://localhost:8000`. The WebSocket
endpoint is at `ws://localhost:8000/ws/interact`. API docs (Swagger) are
available at `http://localhost:8000/docs`.

### Run the Demo

```bash
make demo
```

This opens the dark-theme frontend with canvas-based embedding visualization.

### Train the Models

```bash
make train-tcn      # train the TCN encoder with NT-Xent contrastive loss
make train-slm      # train the custom SLM on curated companion dialogue
```

Training config lives in `configs/`. See `training/` for the entrypoints.

### Other Make Targets

Run `make help` to see all 23 targets. Common ones:

```bash
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
make test           # pytest with coverage
make test-unit      # unit tests only (fast)
make test-integration   # integration tests
make profile-edge   # run edge feasibility profiler
make clean          # remove __pycache__, .pyc, build artifacts
```

---

## Code Style

We enforce a consistent code style across the project.

### Formatter: Ruff Format

We use `ruff format` (compatible with Black) with a 100-character line length.

```bash
poetry run ruff format i3/ server/ training/ tests/
```

### Linter: Ruff

We use Ruff with a curated rule set. Configuration lives in `pyproject.toml`
under `[tool.ruff]`.

```bash
poetry run ruff check i3/ server/ training/ tests/
poetry run ruff check --fix i3/ server/ training/ tests/    # auto-fix safe issues
```

Zero lint errors on main is our standard. If you disagree with a rule for a
specific line, use `# noqa: <rule-code>` with a comment explaining why.

### Type Checker: mypy

All new code in `i3/` must be fully type-annotated. mypy runs in CI with
`--ignore-missing-imports` for third-party libs but is strict about project code.

```bash
poetry run mypy i3/
```

Use `from __future__ import annotations` at the top of every module to get
PEP 604 union syntax (`X | Y`) on all supported Python versions.

### Docstrings

We use Google-style docstrings. Every public function, class, and module
should have one. Private helpers (`_leading_underscore`) need docstrings only
if their purpose is non-obvious.

```python
def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
    """Encode a batch of interaction sequences.

    Args:
        x: Input tensor of shape `[batch, time, features]`.
        mask: Optional boolean mask of shape `[batch, time]` where True
            indicates valid timesteps.

    Returns:
        Pooled embedding tensor of shape `[batch, embedding_dim]`.
    """
```

### Naming

- `snake_case` for functions, variables, module names
- `PascalCase` for classes
- `SCREAMING_SNAKE_CASE` for module-level constants
- Use full words over abbreviations (`encoder` not `enc`)
- Prefix private names with `_`

---

## Testing

Tests live in `tests/` and are organized as:

```
tests/
  unit/           # fast, no I/O, no network, no GPU
  integration/    # FastAPI TestClient, SQLite fixtures, small model forward passes
  conftest.py     # shared fixtures
```

### Running Tests

```bash
make test                           # everything, with coverage
poetry run pytest tests/unit -v     # just unit tests
poetry run pytest -k "tcn" -v       # tests matching pattern
poetry run pytest -x --lf           # stop on first failure, run last-failed
```

### Writing Tests

- Use `pytest` style (functions, not `unittest.TestCase`).
- Use `pytest-asyncio` for async code (`@pytest.mark.asyncio`).
- Parametrize aggressively — one `@pytest.mark.parametrize` is worth five tests.
- Every bug fix PR must include a regression test.
- Every new feature PR must include tests for happy path + at least one edge case.
- Aim for ≥80% coverage on new code.

### Fixtures

Common fixtures are defined in `tests/conftest.py`. Please reuse them before
adding new ones.

### Slow Tests

Mark slow tests (>1s) with `@pytest.mark.slow` so they can be skipped in
fast development loops:

```python
@pytest.mark.slow
def test_full_slm_generation():
    ...
```

Run without slow tests: `poetry run pytest -m "not slow"`.

---

## Git Workflow

We follow a **feature-branch workflow** with rebase-on-merge.

### 1. Sync with Upstream

Before starting work, always pull the latest `main`:

```bash
git checkout main
git fetch upstream
git rebase upstream/main
git push origin main
```

### 2. Create a Feature Branch

Name branches descriptively using the pattern `<type>/<short-description>`:

```bash
git checkout -b feat/masked-tcn-pooling
git checkout -b fix/bandit-posterior-nan
git checkout -b docs/update-architecture
```

Branch types: `feat/`, `fix/`, `docs/`, `refactor/`, `test/`, `chore/`, `ci/`,
`perf/`, `security/`.

### 3. Commit Often, Commit Small

- Keep commits atomic — one logical change per commit.
- Write descriptive commit messages (see [Conventional Commits](#conventional-commits)).
- Avoid "WIP" commits in PR branches; squash before opening.

### 4. Keep Your Branch Rebased

Rebase onto upstream `main` regularly to avoid merge conflicts:

```bash
git fetch upstream
git rebase upstream/main
```

If you hit conflicts, resolve them, `git add` the resolved files, then
`git rebase --continue`. Never use `git merge` to reconcile; always rebase.

### 5. Force-push Your Feature Branch (Carefully)

After rebasing, you'll need to force-push to your fork:

```bash
git push --force-with-lease origin feat/masked-tcn-pooling
```

Always use `--force-with-lease` instead of `--force` to avoid clobbering
someone else's work.

---

## Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
for commit messages. This enables automated changelog generation and
semantic versioning.

### Format

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

### Types

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `style` — formatting (no code change)
- `refactor` — code change that neither fixes a bug nor adds a feature
- `perf` — performance improvement
- `test` — adding or updating tests
- `chore` — maintenance (deps, tooling, etc.)
- `ci` — CI configuration
- `security` — security fix

### Examples

```
feat(perception): add masked pooling to TCN encoder

Supports variable-length interaction sequences by ignoring padded
timesteps during average pooling. Addresses #42.

Closes #42
```

```
fix(bandit): guard against NaN in Laplace posterior when Hessian is singular

Falls back to the prior covariance when np.linalg.cholesky raises
LinAlgError. Adds a regression test with a degenerate feature matrix.
```

```
docs(contributing): add section on conventional commits
```

### Breaking Changes

Mark breaking changes with `!` after the type/scope and include a
`BREAKING CHANGE:` footer:

```
feat(api)!: change WebSocket protocol to MsgPack

BREAKING CHANGE: Clients must now send MsgPack-encoded frames. See the
migration guide in docs/migration-v2.md.
```

---

## Pull Request Process

1. **Open a draft PR early** if you want feedback on direction. Mark it as
   "Ready for review" only when you're confident it's complete.
2. **Fill out the PR template** thoroughly — it exists for good reason.
3. **Ensure CI passes**: lint, typecheck, tests across Python 3.10/3.11/3.12
   on Ubuntu and macOS.
4. **Update documentation**: README, ARCHITECTURE, docstrings, CHANGELOG as
   appropriate.
5. **Request review** from a maintainer. If you're not sure who, tag
   `@abailey81`.
6. **Address review comments** by pushing new commits (don't force-push
   during active review — it makes comment threads hard to follow).
7. **Once approved**, a maintainer will squash-and-merge your PR with a
   conventional commit message summarizing the change.

### What Makes a Good PR

- Small and focused (ideally <400 lines changed)
- Clear description of the problem and solution
- Tests included
- No unrelated changes (format-only churn, dependency bumps, etc.)
- Passes all CI checks on first push (or close to it)

### What Gets PRs Rejected

- Introducing unencrypted PII storage
- Replacing from-scratch ML components with library calls without discussion
- Adding new dependencies without justification
- Regressing edge-profile benchmarks without a performance rationale
- Bypassing pre-commit hooks or CI
- Committing secrets (we will revoke them and force-push to remove from history)

---

## Reporting Bugs

Please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).
Before filing:

1. Search existing issues to avoid duplicates.
2. Verify you're on the latest main.
3. Try to produce a minimal reproduction.
4. Include Python version, OS, and I³ version / commit SHA.

## Requesting Features

Please use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).
Explain the problem, not just the solution. The best feature requests are
use-case driven.

## Security Vulnerabilities

**Do not file public issues for security vulnerabilities.** Instead, use
[GitHub Security Advisories](https://github.com/abailey81/implicit-interaction-intelligence/security/advisories/new)
or email **tamer.atesyakar@bk.ru** with the details. We will acknowledge
within 48 hours and aim to have a fix within 14 days for critical issues.

See [SECURITY.md](SECURITY.md) for our full security policy.

---

## Documentation

- **README.md** — overview, quickstart, key features
- **docs/ARCHITECTURE.md** — system design, data flow, component interactions
- **docs/DEMO_SCRIPT.md** — interactive demo walkthrough
- **SECURITY.md** — security model, threat analysis, reporting
- **Docstrings** — every public API

If you change behavior visible to users or developers, update the relevant
docs in the same PR.

## Release Process

Maintainers handle releases. The process is:

1. All desired changes merged to `main` with passing CI.
2. Update `CHANGELOG.md` — move `[Unreleased]` entries under a new
   `[X.Y.Z] - YYYY-MM-DD` header.
3. Bump version in `pyproject.toml` per [SemVer](https://semver.org/):
   - `MAJOR` for breaking changes
   - `MINOR` for backwards-compatible features
   - `PATCH` for backwards-compatible fixes
4. Tag the commit: `git tag -a vX.Y.Z -m "Release vX.Y.Z"` and push.
5. CI builds the artifacts and (optionally) publishes to PyPI.
6. Create a GitHub Release with the changelog excerpt.

---

## Questions?

- **General discussion**: [GitHub Discussions](https://github.com/abailey81/implicit-interaction-intelligence/discussions)
- **Bug reports**: [Issues](https://github.com/abailey81/implicit-interaction-intelligence/issues)
- **Security**: tamer.atesyakar@bk.ru or [Security Advisories](https://github.com/abailey81/implicit-interaction-intelligence/security/advisories/new)

Thank you for helping make I³ better!
