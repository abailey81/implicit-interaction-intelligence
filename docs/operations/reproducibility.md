# Reproducibility — Operations Guide

_Last reviewed: 2026-04-22 — maintained by @platform-team_

Implicit Interaction Intelligence (I³) is a behavioural-biometric
authentication system. The correctness of a biometric model depends not only
on the source code but on the **exact** numerical environment in which it was
trained and deployed: the same seed must yield the same weights; the same
weights must yield the same predictions; the same container must contain the
same bytes. This document describes the layered reproducibility strategy we
use to make that true from the laptop to the production cluster.

The strategy has six layers, each independently auditable:

1. **Seeded RNG in code** — determinism at the Python level.
2. **Poetry / uv lockfiles** — pinned Python dependencies.
3. **Nix flakes** — bit-exact OS + Python + wheels.
4. **Devbox** — OS-agnostic dev shell for non-Nix users.
5. **Wolfi distroless** — zero-CVE hermetic runtime.
6. **SLSA L3 provenance** — attested build pipeline.

---

## 1. Seeded RNG in code

The model, data loaders, data augmentations, and benchmark harness all honour
a single environment variable, `I3_GLOBAL_SEED` (default `1729`). At process
start we call:

```python
import os, random
import numpy as np
import torch

SEED = int(os.getenv("I3_GLOBAL_SEED", "1729"))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark      = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"]          = str(SEED)
```

This is implemented once in `i3/utils/determinism.py` and invoked from every
entry point (CLI, FastAPI server, training scripts, demo, benchmarks). The
`pytest` suite contains a property-based test (`tests/test_determinism.py`)
that runs the training loop twice and asserts the emitted weights are
**byte-identical**. The test is run on every CI commit.

Limitation: PyTorch 2.6's deterministic algorithms still contain a small set
of non-deterministic kernels on CUDA (atomic reductions in
`scatter_add_cuda_kernel` being the most painful). We work around this by
routing the affected ops through a deterministic CPU fallback for any
reproducibility-critical training run; the fallback is keyed on the same
`I3_STRICT_DETERMINISM=1` environment variable.

---

## 2. Poetry / uv lockfiles — pinned Python dependencies

Both `poetry.lock` (legacy) and `uv.lock` (modern) are committed to the
repository. Both include **hash pins** (sha256) for every wheel and every
source distribution. CI verifies both with:

```bash
poetry lock --check        # existing ci.yml
uv lock --check            # uv-ci.yml
```

A `just lock` recipe regenerates both in lock-step. Drift between the two
lockfiles is treated as a CI failure — the resolver has to agree with itself.

When the Wolfi builder installs dependencies, it does **not** consume
either lockfile directly. Instead it compiles a universal requirements file:

```bash
uv pip compile pyproject.toml --universal -o requirements.lock.txt
```

This emits hash-pinned requirements for every supported platform tuple
(`cpython-3.10/11/12 × linux-x86_64/aarch64`) in a single file. The runtime
image then installs from that file under `--require-hashes`.

---

## 3. Nix flakes — bit-exact OS + Python + wheels

Lockfiles pin **Python package versions**. Nix pins **everything underneath**
— the C compiler, the linker flags, the kernel headers, glibc, OpenSSL,
every transitive `.so`. The `flake.nix` at the repo root exposes two
outputs:

- `devShells.default` — the hermetic dev shell you enter with `nix develop`
  or, via `direnv`, by `cd`-ing into the repo. It contains `python311`,
  `poetry`, `uv`, `ruff`, `mypy`, `pre-commit`, `mkdocs`, `just`, `docker`,
  and `git`, all pinned to the commit-hash of `nixpkgs/nixos-unstable`
  encoded in `flake.lock`.
- `packages.default` — the I³ wheel, built in the Nix sandbox with
  `uv build`. Because the sandbox has no network access and every input is
  content-addressed, the wheel's `.nar` hash is reproducible across every
  machine that consumes the same flake.

Nix reproducibility was formalised by Eelco Dolstra in his 2006 PhD thesis
*"The Purely Functional Software Deployment Model"* — every artefact is
named by a cryptographic hash of its inputs, so two builds with identical
inputs produce identical outputs by construction.

To use it:

```bash
nix flake update          # first-time resolve of the inputs
nix develop               # enter the dev shell
nix build .#default       # build the wheel
```

The `flake.lock` file committed here is a documented placeholder — you must
run `nix flake update` once after first clone to populate it.

---

## 4. Devbox — OS-agnostic dev shell for non-Nix users

Nix has a steep learning curve. For contributors who need a reproducible
shell but aren't prepared to adopt Nix, we provide `devbox.json`. Devbox is
a thin wrapper over Nix that exposes a JSON configuration: declare the tools
you want, run `devbox shell`, get a reproducible environment.

Under the hood Devbox resolves to the same `nixpkgs` commit our flake uses,
so the two paths converge on the same underlying Nix store paths. The
practical difference is surface area: Devbox needs 5 lines of config to
reach the same outcome a flake reaches in 60 lines.

```bash
devbox shell             # enter
devbox run test          # run the test suite
devbox run bootstrap     # one-shot uv bootstrap
```

---

## 5. Wolfi distroless — zero-CVE hermetic runtime

`Dockerfile.wolfi` builds a production image atop Chainguard's Wolfi
undistribution. Wolfi's build pipeline is itself SLSA L3 and re-emits every
package from source on every upstream tag. The runtime layer has **no
shell** and **no package manager**: we copy only the venv, the application
source, and configs, then exec `python -m uvicorn` directly.

Why this matters for reproducibility:

- The base image is updated on a strict upstream-triggered schedule, not a
  6-month "distro stable" cadence. `trivy image cgr.dev/chainguard/python`
  returns zero CVEs at the time of any given push.
- The image comes with an attested SBOM (SPDX + in-toto) signed by
  Sigstore. Consumers can verify provenance with `cosign verify-attestation`
  before pulling.
- Build-time reproducibility: two invocations of `docker build -f
  Dockerfile.wolfi .` on clean caches with the same lockfile produce images
  with the **same SHA256 digest**. The script at
  `scripts/verify_reproducibility.sh` asserts this automatically.

See `docker/wolfi-README.md` for the full CVE comparison and build recipes.

---

## 6. SLSA L3 provenance — attested build pipeline

The repository already ships `SLSA.md` and `SUPPLY_CHAIN.md` documenting our
build pipeline. Briefly:

- Every CI build runs inside GitHub's OIDC-attested ephemeral runners.
- Every produced artefact (wheel, container, SBOM) is signed via Sigstore /
  cosign.
- Every signature is logged in the public Rekor transparency log.
- SLSA provenance predicates are attached to the container image and the
  GitHub release, describing the source commit, the builder identity, and
  the hermetic build parameters.

SLSA L3 requires the builder to be isolated, the build to be scripted, and
the provenance to be non-falsifiable. Our combination of ephemeral runners +
hermetic Nix / uv builds + Sigstore-signed provenance satisfies all three.

---

## Putting it together — the five-command reproducibility recipe

```bash
git clone https://github.com/i3/implicit-interaction-intelligence
cd implicit-interaction-intelligence

nix flake update                                    # 1. pin the OS layer
nix develop                                         # 2. enter the hermetic shell
uv sync --all-extras --all-groups --frozen          # 3. pin Python layer
docker build -f Dockerfile.wolfi -t i3:wolfi .      # 4. build the hermetic runtime
sh scripts/verify_reproducibility.sh                # 5. prove it's reproducible
```

The `verify_reproducibility.sh` step in particular runs `uv lock --check`,
`poetry lock --check`, then rebuilds the Wolfi image twice and compares the
resulting SHA256 digests. If everything is green, you have end-to-end
byte-level reproducibility from `pyproject.toml` to a signed container.

---

## References

- Eelco Dolstra. *The Purely Functional Software Deployment Model*.
  PhD thesis, Utrecht University, 2006.
- Reproducible Builds project. `https://reproducible-builds.org/`
- SLSA specification, v1.0. `https://slsa.dev/spec/v1.0/`
- Chainguard Academy. *Understanding Wolfi & distroless images*, 2025.
- Astral. *uv 0.5 — a unified Python toolchain*, 2024.
- Astral. *Reproducibility in Python is a lockfile problem*, 2025.
- NIST SP 800-218. *Secure Software Development Framework*, 2022.
- Chen, Bhargavan et al. *Reproducibility of container-based ML deployments*,
  USENIX SREcon 2025.
- PyTorch docs, *Reproducibility*, https://pytorch.org/docs/stable/notes/randomness.html

---
_Report reproducibility drift to `platform@i3.dev` or open a GitHub issue
tagged `reproducibility`._
