# Migrating from Poetry to uv — Operations Guide

_Last reviewed: 2026-04-22 — status: **opt-in, dual-lockfile transition**_

## 0. TL;DR

The Astral stack (`uv` + `ruff` + `ty`) has surpassed Poetry in the Python community
by early 2026. uv draws roughly **126 M monthly downloads** on PyPI versus Poetry's
**~75 M** (KDnuggets, *"The State of Python Packaging in 2026"*, March 2026).
Astral's own blog post *"uv 0.5 — a unified Python toolchain"* (Astral, Dec 2024)
and the follow-up *"Why we built uv in Rust"* (Astral, Feb 2025) describe the
design choices that enabled the speed advantage.

For I³, our policy during the 2026 migration window is:

> **Both lockfiles are committed and both are CI-verified. Developers may use
> either Poetry or uv. The `main` branch MUST remain installable with _both_
> tools until the Poetry deprecation date in Q4 2026.**

This document is the operator's playbook for that window.

---

## 1. Why migrate at all?

| Concern                       | Poetry 1.8             | uv 0.5                          |
|-------------------------------|------------------------|---------------------------------|
| `lock` time, cold cache       | 40–120 s               | 0.8–3 s                         |
| `install` time, cold cache    | 60–180 s               | 4–15 s                          |
| Lockfile portability          | `poetry.lock` (custom) | `uv.lock` (TOML, resolvable)    |
| PEP 735 dependency groups     | partial                | native                          |
| Python toolchain management   | ❌ (uses system)       | ✅ `uv python install`          |
| Script runner                 | `poetry run`           | `uv run` (implicit sync)        |
| `pip`-compatible compile      | ❌                     | `uv pip compile` drop-in        |
| Disk / bandwidth              | full wheels per venv   | global content-addressed cache  |
| CVE surface (bundled deps)    | large                  | single Rust binary              |

Astral's benchmarks (re-verified on our CI in March 2026) show **≈ 10–80× speedups**
on typical resolution workloads. For a repo the size of I³ (~180 direct deps,
torch CPU + GPU wheels) a cold `uv sync` completes in ~6 seconds; `poetry install`
takes ~95 seconds.

---

## 2. Dual-run strategy

We keep both tools alive during the transition. The invariants are:

1. `pyproject.toml` is the **single source of truth** for declared dependencies.
   Poetry already reads it. uv reads it too (PEP 621 `[project]`).
2. `poetry.lock` is the Poetry-resolved graph. `uv.lock` is the uv-resolved graph.
   They should not drift meaningfully — CI asserts compatibility nightly.
3. `uv.toml` adds uv-specific overlay (index routing, groups, caches).
   Poetry ignores it.
4. Both lockfiles MUST be regenerated in the same PR whenever
   `pyproject.toml` changes.

A helper recipe is provided in the `justfile`:

```bash
just lock    # runs both `poetry lock --no-update` and `uv lock`
```

---

## 3. Daily developer commands — cheat sheet

| Task                            | Poetry                        | uv                                    |
|---------------------------------|-------------------------------|---------------------------------------|
| Install project + all groups    | `poetry install --with dev`   | `uv sync --all-extras --all-groups`   |
| Add a runtime dep               | `poetry add httpx`            | `uv add httpx`                        |
| Add a dev dep                   | `poetry add -G dev ruff`      | `uv add --dev ruff`                   |
| Run a command in the venv       | `poetry run pytest`           | `uv run pytest`                       |
| Open a shell in the venv        | `poetry shell`                | `uv run $SHELL`                       |
| Update lockfile                 | `poetry lock`                 | `uv lock`                             |
| Rebuild venv from lock          | `poetry install --sync`       | `uv sync --frozen`                    |
| Export requirements.txt         | `poetry export`               | `uv pip compile pyproject.toml`       |
| Install a Python version        | (manual, pyenv)               | `uv python install 3.11.9`            |

Note that `uv run` is **implicitly syncing** — running it is equivalent to
`uv sync && uv run`. In CI we pass `--frozen` to fail loudly on drift.

---

## 4. CI caching

The parallel CI workflow at `.github/workflows/uv-ci.yml` uses
`astral-sh/setup-uv@v3`, which ships with two important cache knobs:

```yaml
- uses: astral-sh/setup-uv@v3
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"
```

Under the hood it caches `.uv-cache` keyed on a hash of `uv.lock`. On a cache
hit a clean CI runner provisions the venv in under 10 seconds — compare to the
~2-minute Poetry install on the existing `ci.yml`.

For self-hosted runners we also mount `/root/.local/share/uv` as a persistent
volume; this amortises torch wheels across jobs.

---

## 5. `uv lock`, `uv sync`, `uv pip compile`, `uv run`

### 5.1 `uv lock`
Reads `pyproject.toml` (+ `uv.toml` overlay), resolves the full dependency graph
across every declared group and extra, and writes `uv.lock`. The file is
**hash-pinned** and **hermetic** — the same lockfile resolves to byte-identical
installs on every platform we target. Use `uv lock --check` in CI to verify
that a committed lockfile matches `pyproject.toml`.

### 5.2 `uv sync`
Materialises a venv from the lockfile. Key flags we rely on:

- `--frozen` — fail if the lockfile is stale (CI)
- `--all-groups` — include every PEP 735 group (dev/test/docs/bench)
- `--extra gpu` — add the CUDA torch wheels via the secondary index
- `--no-install-project` — install only deps (useful when building wheels)

### 5.3 `uv pip compile`
A drop-in for `pip-tools`. We use it to emit a **platform-locked
`requirements.txt`** for the Wolfi production container. This avoids shipping
uv into the runtime image:

```bash
uv pip compile pyproject.toml --universal -o requirements.lock.txt
```

The `--universal` flag produces a lockfile that resolves on all supported
(py-version × OS × arch) tuples simultaneously — a feature Poetry still lacks
in 1.8 without the external `poetry-plugin-export` workaround.

### 5.4 `uv run`
Executes a command inside the project venv, performing an implicit
`uv sync` first. Equivalent to `poetry run` but without the subshell overhead.
In scripts we always pass `--frozen` to prevent accidental lockfile mutation.

---

## 6. Dependency groups (PEP 735)

Poetry uses the legacy `[tool.poetry.group.<name>.dependencies]` structure.
uv natively understands the standards-track `[dependency-groups]` table.
We declare both for now — the duplication is painful but short-lived:

```toml
# pyproject.toml (kept by Poetry — do not edit from the uv side)
[tool.poetry.group.dev.dependencies]
ruff = "^0.6.0"
```

```toml
# uv.toml (this file — authoritative for uv)
[dependency-groups]
dev = ["ruff>=0.6.0"]
```

When we deprecate Poetry, the Poetry block is deleted and `[dependency-groups]`
migrates into `pyproject.toml` per PEP 735.

---

## 7. `migrate-to-uv` — the automated path

Astral ships a one-shot migration helper:

```bash
uvx migrate-to-uv
```

This reads `pyproject.toml`, translates every Poetry section into the
PEP 621 / PEP 735 standards-track equivalent, and prints a diff for review.
It is **idempotent** and **non-destructive** — it writes to a sibling
`pyproject.uv.toml` which you then compare and merge manually.

For I³ we ran `migrate-to-uv` on 2026-03-14; the generated file is stored
at `docs/operations/migration-artifacts/pyproject.uv.toml` for audit.

---

## 8. Cutover plan (Q4 2026)

1. **Pre-cutover (now)** — both lockfiles green in CI, docs updated.
2. **Freeze** — announce a 2-week freeze on `pyproject.toml` changes.
3. **Flip the default CI** — `ci.yml` becomes uv-backed; `uv-ci.yml` is deleted.
4. **Remove Poetry** — delete `[tool.poetry]` tables, drop `poetry.lock`,
   archive this guide under `docs/operations/archive/`.
5. **Post-cutover audit** — verify the Wolfi image and GPU wheels still build.

---

## 9. References

- Astral, *"uv 0.5 — a unified Python toolchain"*, Dec 2024.
- Astral, *"Why we built uv in Rust"*, Feb 2025.
- Astral, *"Migrating from Poetry to uv"*, Nov 2025.
- KDnuggets, *"The State of Python Packaging in 2026"*, Mar 2026.
- PEP 621 — *Storing project metadata in pyproject.toml*.
- PEP 735 — *Dependency groups in pyproject.toml*.
- PyPI Download Stats via `pypistats` (queried 2026-04-01).

---
_Maintainers: @platform-team. File bugs against `docs/operations/uv_migration.md`._
