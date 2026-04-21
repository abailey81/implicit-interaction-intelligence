# -*- mode: Python -*-
# Tiltfile for Implicit Interaction Intelligence (I^3).
#
# This gives you a hot-reload local Kubernetes dev loop:
#   1. Changes under ./i3/ are rsync'd into the running container.
#   2. The container's process is restarted in-place (no image rebuild).
#   3. The Tilt UI at http://localhost:10350 streams logs and status.
#
# Install:  brew install tilt   (or see https://tilt.dev/install)
# Run:      tilt up
# Stop:     tilt down
#
# Docs: https://docs.tilt.dev/

load('ext://restart_process', 'docker_build_with_restart')

# --- Configuration flags ---------------------------------------------------

config.define_bool('lite', usage='Minimal mode: skip docs + observability')
cfg = config.parse()
lite = cfg.get('lite', False)

# --- Image build with live-update ------------------------------------------

docker_build_with_restart(
    'ghcr.io/hmi-lab/i3',
    context='.',
    dockerfile='Dockerfile.dev',
    entrypoint=['python', '-m', 'i3.server'],
    live_update=[
        sync('./i3', '/app/i3'),
        sync('./configs', '/app/configs'),
        run('pip install -e .', trigger=['./pyproject.toml']),
    ],
    ignore=[
        '.git',
        '**/__pycache__',
        '**/*.pyc',
        'site',
        'dist',
        '.venv',
    ],
)

# --- Kubernetes manifests --------------------------------------------------

k8s_yaml(kustomize('deploy/k8s/overlays/dev'))

k8s_resource(
    'i3-server',
    port_forwards=[
        port_forward(8000, 8000, name='http'),
        port_forward(5678, 5678, name='debugpy'),
    ],
    labels=['backend'],
)

# --- Local docs server (not in k8s) ----------------------------------------

if not lite:
    local_resource(
        'mkdocs-serve',
        'poetry run mkdocs serve --dev-addr 127.0.0.1:8001',
        serve=True,
        resource_deps=[],
        labels=['docs'],
        links=[link('http://127.0.0.1:8001', 'MkDocs')],
    )

# --- Watch additional source paths -----------------------------------------

watch_file('i3/')
watch_file('configs/')

print('Tilt ready. Open http://localhost:10350 for the dashboard.')
