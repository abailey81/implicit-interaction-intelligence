# Dagger Pipeline for I^3

This directory is a placeholder for the local Dagger engine cache. The
module manifest and Python source live in [`../dagger/`](../dagger/).

## Install

1. Install the Dagger CLI (>= v0.11):

   ```sh
   # macOS / Linux
   curl -L https://dl.dagger.io/dagger/install.sh | sh

   # Windows (PowerShell)
   iwr -useb https://dl.dagger.io/dagger/install.ps1 | iex
   ```

2. Install the Python SDK into your venv (for IDE auto-complete):

   ```sh
   pip install dagger-io
   ```

## Usage

All commands are run from the repository root:

```sh
# Static analysis
dagger call lint --source=. stdout

# Pytest matrix
dagger call test --source=. --python-version=3.11 stdout
dagger call test --source=. --python-version=3.12 stdout

# Build the production image and publish it
dagger call build-image --source=. publish --address=ghcr.io/hmi-lab/i3:dev

# Scan an image for CVEs
dagger call scan-image --image=$(dagger call build-image --source=. id)

# Build docs (strict mode)
dagger call docs-build --source=. export --path=./site

# Full release pipeline
dagger call release --source=. --tag=v1.2.3 stdout
```

## Why Dagger?

* The pipeline is **authored in Python** -- same language as the rest of
  I^3, so engineers don't need to learn YAML-with-bash.
* It runs **identically** on a laptop, in GitHub Actions, in GitLab CI,
  or in Buildkite -- see
  <https://docs.dagger.io/integrations> for runners.
* Every step is **content-addressed and cached** via BuildKit, so
  re-running `dagger call test` after changing a single file is
  nearly free.

## References

* Dagger docs: <https://docs.dagger.io>
* Python SDK reference: <https://docs.dagger.io/sdk/python>
* Modules manual: <https://docs.dagger.io/manuals/developer/modules>
