# Running CI Locally with `act`

[`nektos/act`](https://github.com/nektos/act) runs GitHub Actions workflows
inside Docker containers on your laptop. It dramatically shortens the
feedback loop when editing `.github/workflows/*.yml`.

## Install

```sh
# macOS
brew install act

# Linux / WSL
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Windows (Scoop)
scoop install act
```

You also need Docker Desktop (or Colima, or Orbstack) running.

## First-time setup

On first run, `act` asks which container image size to use. For I^3
workflows, pick the **Medium** image (includes Python + tools):

```sh
act --list   # asks for image preference, then lists workflows
```

This writes `~/.config/act/actrc`. Our recommended contents:

```ini
-P ubuntu-latest=catthehacker/ubuntu:act-22.04
-P ubuntu-22.04=catthehacker/ubuntu:act-22.04
-P ubuntu-20.04=catthehacker/ubuntu:act-20.04
--container-architecture linux/amd64
```

## Running specific workflows

```sh
# Run the default push event (runs ci.yml)
act

# Run the pull_request event
act pull_request

# Run a specific workflow
act -W .github/workflows/ci.yml

# Run a specific job
act -j lint

# Pass secrets from a local file
act --secret-file .secrets.local

# Continuous profiling workflow
act -W .github/workflows/pyroscope.yml
```

Create `.secrets.local` (git-ignored!) with:

```ini
ANTHROPIC_API_KEY=sk-ant-...
I3_ENCRYPTION_KEY=...
PYROSCOPE_SERVER_URL=http://host.docker.internal:4040
```

## Caveats

* `act` cannot emulate GitHub-hosted runners 100% -- in particular,
  `${{ secrets.GITHUB_TOKEN }}` is **empty** unless you pass `--token`.
* macOS/ARM users should pass `--container-architecture linux/amd64` to
  avoid "exec format error" on third-party actions.
* For pipelines that need real Kubernetes, prefer `tilt up` or
  `dagger call` -- `act` is best for Actions-specific logic (matrix,
  permissions, environment rules).

## Related

* [Dagger](../../.dagger/README.md) -- pipeline-as-code alternative that
  runs the same steps in `act`, in CI, and locally.
* `nektos/act` docs: <https://nektosact.com>
