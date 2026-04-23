# Tilt: Local Kubernetes Dev Loop

Tilt gives I^3 engineers a tight inner loop against a real Kubernetes
cluster (Kind / k3d / Minikube / Docker Desktop) with **sub-second hot
reload**. No more `docker build && kubectl rollout` cycles.

## Install

```sh
# macOS (Homebrew)
brew install tilt

# Linux / WSL
curl -fsSL https://raw.githubusercontent.com/tilt-dev/tilt/master/scripts/install.sh | bash

# Windows (Scoop)
scoop install tilt
```

You also need a local cluster. We recommend Kind:

```sh
brew install kind
kind create cluster --name i3-dev
kubectl config use-context kind-i3-dev
```

## Usage

From the repo root:

```sh
tilt up                    # interactive TUI + web UI at http://localhost:10350
tilt up --stream           # non-interactive, prints logs
tilt up -- --lite          # minimal mode (skip MkDocs, skip observability stack)
tilt down                  # tear everything down
tilt trigger i3-server     # force-rebuild one resource
```

Once `tilt up` is running, open <http://localhost:10350> for live status of:

* `i3-server` -- the FastAPI server with debugpy on :5678
* `mkdocs-serve` -- live documentation preview on :8001

## How hot reload works

The `Tiltfile` uses `docker_build_with_restart` (from the
`restart_process` extension) to combine BuildKit caching with in-container
file sync. When a file under `i3/` changes:

1. Tilt rsync's the changed files into the running container.
2. The entrypoint process is signalled to restart.
3. Kubernetes sees the pod is still Ready -- no rollout, no image churn.

For dependency changes (`pyproject.toml`), Tilt triggers a
`pip install -e .` inside the container without rebuilding the image.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Error: could not find cluster` | `kubectl config use-context kind-i3-dev` |
| Port 8000 already in use | `lsof -i :8000` and kill it, or edit `Tiltfile` port_forward |
| Live-update fails, full rebuild every time | File is in `ignore` list or outside `sync` source -- check `Tiltfile` |
| Tilt UI shows "pending" forever | `kubectl get pods` -- usually ImagePullBackOff on a private registry |
| `restart_process` extension missing | Tilt auto-downloads extensions on first `tilt up`; behind a proxy, set `TILT_EXT_REGISTRY` |

## References

* Tilt docs: <https://docs.tilt.dev/>
* `restart_process` extension: <https://github.com/tilt-dev/tilt-extensions/tree/master/restart_process>
