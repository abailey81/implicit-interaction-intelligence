# Developer Experience (2026 Edition)

> *"Great software is the by-product of a great inner loop."* -- this
> guide describes the inner loop we actually run on, and the reasons we
> chose each tool over its older-generation alternative.

This document is a **tour of the 2026 developer-experience stack** that
Implicit Interaction Intelligence (I^3) uses on top of the classic
Makefile/GitHub Actions baseline. If you are onboarding from a team that
still lives in "bash scripts + `kubectl rollout restart`", expect the
feedback loops here to feel **10x faster**. If you are coming from a team
that already runs everything on Dagger + Tilt + Backstage, the content
below will be familiar; skim the **"When *not* to use this"** boxes --
they capture our house style.

The stack, at a glance:

| Concern                      | Old way (pre-2023)                  | I^3 way (2026)                |
| ---------------------------- | ----------------------------------- | ----------------------------- |
| CI/CD pipelines              | YAML + bash in `.github/workflows/` | **Dagger** (Python SDK)       |
| Local k8s dev loop           | `skaffold dev` / raw `kubectl`      | **Tilt** + **Kind**           |
| Continuous profiling         | py-spy snapshots, rarely run        | **Grafana Pyroscope**         |
| Service catalog              | Confluence page, out of date        | **Backstage** + **TechDocs**  |
| Telemetry collection         | Prom Agent + Fluent Bit + OTelCol   | **Grafana Alloy**             |
| Run CI locally               | "It works on the runner"            | **nektos/act** + Dagger       |
| Cloud dev environment        | Local laptop only                   | **GitHub Codespaces**         |

The rest of this page walks through each tool: what problem it solves,
how we use it, what you type, and -- importantly -- when **not** to
reach for it on a small team.

---

## 1. Dagger: pipeline-as-code in Python

### The problem

Every team eventually hits the wall where their GitHub Actions YAML is
just a fragile bash-and-YAML veneer over their real build logic, and
there is no way to run the pipeline on a laptop without pushing commits
and waiting. Worse, reusing steps across workflows means copy-pasting.

### The answer

[Dagger](https://dagger.io) (Python SDK since mid-2023, stable 0.11+ in
2024) models pipelines as *typed Python functions* running inside
containers. Every function takes typed inputs (directories, files,
containers, secrets) and returns typed outputs. The runtime is BuildKit,
so every call is content-addressed and cached.

### How we use it

Our module lives in [`dagger/main.py`](../../dagger/main.py):

```python
@function
async def test(
    self,
    source: dagger.Directory,
    python_version: str = "3.11",
) -> str:
    ...
```

From the repo root:

```sh
# Lint
dagger call lint --source=. stdout

# Matrix test (same function, different python_version)
dagger call test --source=. --python-version=3.11 stdout
dagger call test --source=. --python-version=3.12 stdout

# Build + scan + publish -- identical to what CI runs
dagger call release --source=. --tag=v1.2.3 stdout
```

The CI workflow invokes `dagger call` rather than running its own
`pytest`/`ruff` steps, which means **local and CI literally share the
same code**.

### When *not* to use Dagger

* **Tiny repos with one linear job.** The Dagger install (CLI + daemon)
  is ~150 MB; if your pipeline is five lines of bash, stay in YAML.
* **Environments without Docker.** Dagger needs a container runtime.
  On air-gapped Windows CI boxes, fall back to the classic workflow.
* **Teams unwilling to learn the SDK.** Dagger's value is proportional
  to how much Python you are willing to write.

### References

* Dagger docs: <https://docs.dagger.io>
* Python SDK: <https://docs.dagger.io/sdk/python>
* Dagger modules (2024): <https://docs.dagger.io/manuals/developer/modules>

---

## 2. Tilt: hot-reload local Kubernetes

### The problem

Developing against Kubernetes traditionally means the loop
`edit -> build image -> push -> kubectl apply -> wait for rollout`,
which costs 30-60 s per iteration. Multiply by the number of
inner-loop iterations in a normal day and you lose an hour.

### The answer

[Tilt](https://tilt.dev) watches your filesystem and uses in-container
file-sync (via the `restart_process` extension) to update running
pods **without rebuilding the image**. Typical reload time: 300-800 ms.

### How we use it

Our [`Tiltfile`](../../Tiltfile) is 50 lines of Starlark. The essentials:

```python
load('ext://restart_process', 'docker_build_with_restart')

docker_build_with_restart(
    'ghcr.io/hmi-lab/i3',
    context='.',
    dockerfile='docker/Dockerfile.dev',
    entrypoint=['python', '-m', 'i3.server'],
    live_update=[
        sync('./i3', '/app/i3'),
        run('pip install -e .', trigger=['./pyproject.toml']),
    ],
)

k8s_yaml(kustomize('deploy/k8s/overlays/dev'))
k8s_resource('i3-server', port_forwards=[8000, 5678])
```

### The demo

```sh
kind create cluster --name i3-dev
tilt up
# visit http://localhost:10350 for the Tilt UI
```

Edit `i3/router/routing.py`, hit save -- you will see the pod log update
within a second. Hit `b` in the Tilt TUI to trigger a full image rebuild
if you changed a C extension. Press `q` to quit; `tilt down` tears
everything down.

### The `--lite` flag

Our Tiltfile parses `config.define_bool('lite')` so you can skip the
MkDocs serve and the observability stack when you are working on a
laptop with limited RAM:

```sh
tilt up -- --lite
```

### When *not* to use Tilt

* **Stateless CLI tools.** If your service does not need Kubernetes,
  `uvicorn --reload` or `watchexec` are simpler.
* **Remote clusters you do not own.** Tilt expects write access --
  point it at Kind, k3d, or Docker Desktop.

### References

* Tilt docs: <https://docs.tilt.dev/>
* `restart_process` extension: <https://github.com/tilt-dev/tilt-extensions/tree/master/restart_process>

---

## 3. Pyroscope: continuous profiling

### The problem

You only ever run `py-spy top` when something is already on fire. By
that point, your prod traffic has moved on and the slow flame graph is
ancient history. Worse, allocation profiling ("where do all these
objects come from?") is rarely hooked up at all.

### The answer

[Grafana Pyroscope](https://grafana.com/oss/pyroscope/) (acquired by
Grafana Labs in 2023 and re-released under the Grafana umbrella) is
an always-on, pprof-compatible continuous profiler. The Python SDK
ships a sampling profiler that streams ~1%-overhead profiles to a
Pyroscope server, which stores them in a flame-graph-native format.

### How we use it

[`i3/observability/pyroscope_integration.py`](../../i3/observability/pyroscope_integration.py)
wires the SDK into application startup:

```python
from i3.observability.pyroscope_integration import configure_pyroscope

configure_pyroscope(
    service_name="i3-server",
    server_url=os.environ["PYROSCOPE_SERVER_URL"],
    tags={"region": "eu-west-1", "version": settings.version},
)
```

The import is **soft** -- if `pyroscope-io` is not installed, the call
returns `False` and the app boots normally. This keeps edge deployments
and unit tests lean.

### Reading a flame graph

Open Pyroscope UI, pick the service, choose a time range (say, last
15 minutes of traffic). The flame graph is read **bottom-up**:

* Width of a bar = **fraction of samples** spent in that frame or below.
* Depth = **call-stack depth**.
* Hovering shows self vs total time.

The most useful switch is **"Diff" mode**: pick a baseline (last week)
and a target (now) and Pyroscope colours frames red if they got slower
or green if they got faster. This catches regressions that benchmarks
miss.

### CI integration

Our [`.github/workflows/pyroscope.yml`](../../.github/workflows/pyroscope.yml)
spins up an in-runner Pyroscope server, runs the benchmark suite with
profiling enabled, exports pprof files as artifacts, and comments the
artifact URL on the PR. You can then:

```sh
go tool pprof -http=:0 profiles/cpu.pprof
```

to inspect the profile in your browser -- no Grafana required.

### When *not* to use Pyroscope

* **One-off CPU mysteries.** `py-spy record` is still great.
* **Sub-millisecond hot loops.** Sampling profilers (Pyroscope included)
  need enough samples to be representative; instrument micro-benchmarks
  with `perf_counter` or `pytest-benchmark`.

### References

* Pyroscope docs: <https://grafana.com/docs/pyroscope/latest/>
* Grafana acquisition announcement (2023):
  <https://grafana.com/blog/2023/03/15/pyroscope-grafana-labs-acquisition/>

---

## 4. Backstage: a service catalog that stays in sync

### The problem

Every team eventually asks "how do I find *all* the services, their
owners, their dashboards, their runbooks, and their on-call rotas in
one place?" -- and the usual answer is a Confluence page that diverges
from reality within weeks.

### The answer

[Backstage](https://backstage.io) (Spotify, CNCF Incubating) is a
developer portal whose **source of truth is YAML files in each repo**.
The portal periodically re-reads those files, so the catalog cannot go
stale without a commit.

### How we register I^3

We ship [`backstage/catalog-info.yaml`](../../backstage/catalog-info.yaml)
declaring I^3 as a `Component`, plus entity files for the System and
APIs. A Backstage admin registers the repo once -- from then on, any
PR that edits these files is picked up automatically.

Highlights of our entry:

```yaml
metadata:
  name: i3
  annotations:
    backstage.io/techdocs-ref: dir:.       # auto-build MkDocs
    github.com/project-slug: hmi-lab/i3
    pagerduty.com/integration-key: ...
spec:
  type: service
  lifecycle: experimental
  owner: group:default/hmi-lab
```

The `techdocs-ref` annotation is the *killer feature*: Backstage's
TechDocs plugin clones the repo, runs `mkdocs build`, and serves the
result inside the portal. Your `/docs` tree becomes browsable **without
publishing a separate site**.

### The graph view

Backstage renders a dependency graph (`System` -> `Components` -> `APIs`
-> `Resources`). Our [`backstage/entities/system.yaml`](../../backstage/entities/system.yaml)
groups the FastAPI server, the MCP bridge, and the observability stack
into a single `i3-system`, so newcomers can see the whole territory in
one picture.

### When *not* to use Backstage

* **Small teams (< 10 services).** Backstage is a 1-2 engineer-week
  install. Under ~10 services, a well-maintained README is cheaper.
* **No dedicated platform engineer.** Without someone to maintain the
  Backstage deployment, it will bit-rot faster than your Confluence.

### References

* Backstage docs: <https://backstage.io/docs>
* Descriptor format: <https://backstage.io/docs/features/software-catalog/descriptor-format>
* TechDocs: <https://backstage.io/docs/features/techdocs/>

---

## 5. Grafana Alloy: the one telemetry collector

### The problem

The typical 2020-era observability pipeline required **three** agents:
a Prometheus Agent for metrics, Fluent Bit for logs, and an
OpenTelemetry Collector for traces. Each had its own config language,
its own hot-reload semantics, and its own failure modes.

### The answer

[Grafana Alloy](https://grafana.com/docs/alloy/) (announced April 2024
as the OpenTelemetry Collector Distribution + Prom Agent successor) is
a **single binary** speaking OTLP, Prometheus, Loki, Pyroscope, and a
few dozen cloud-specific protocols. Its config DSL -- River -- is
component-based and statically validated.

### How we use it

[`deploy/observability/grafana-alloy.river`](../../deploy/observability/grafana-alloy.river)
is the complete config. Structure:

1. **Receivers**: `otelcol.receiver.otlp` (from the app),
   `prometheus.scrape` (for I^3's `/metrics`), plus `prometheus.exporter.unix`
   for node-level metrics.
2. **Processors**: `otelcol.processor.attributes` redacts sensitive keys
   that match our structlog redaction list (`authorization`, `api_key`,
   `anthropic_api_key`, `i3_encryption_key`, cookies, ...).
3. **Exporters**: OTLP to Tempo, Prom remote-write to Mimir, Loki push
   to Loki.

Run it stand-alone:

```sh
docker compose -f deploy/observability/docker-compose.alloy.yml up -d
```

### When *not* to use Alloy

* **You already love the OTel Collector.** Alloy and OTel Collector
  overlap; switching costs (for a working pipeline) often outweigh the
  benefits.
* **You only need one protocol.** If you are a pure-Prometheus shop,
  the Prom Agent binary is smaller and simpler.

### References

* Alloy docs: <https://grafana.com/docs/alloy/latest/>
* Alloy launch post (April 2024):
  <https://grafana.com/blog/2024/04/09/grafana-alloy-opentelemetry-collector-with-prometheus-pipelines/>

---

## 6. Codespaces: cloud dev in a browser

[`.github/codespaces/devcontainer.json`](../../.github/codespaces/devcontainer.json)
extends our existing `.devcontainer/devcontainer.json` with
Codespaces-specific secret injection (`ANTHROPIC_API_KEY`,
`I3_ENCRYPTION_KEY`), port labels (Tilt UI auto-opens in the browser),
and the right feature set (Docker-in-Docker for Tilt, kind, act).

Once a teammate lands on the repo, they can click **"Open in
Codespace"** and have a 4-CPU / 8 GB sandbox ready in ~60 s, with:

* `tilt up` ready to run.
* `dagger call lint --source=.` usable immediately.
* Pyroscope reachable at `http://localhost:4040`.

---

## 7. Running CI locally with `act`

See [act-local.md](../../.github/workflows/act-local.md) for the full
walk-through. TL;DR:

```sh
act -W .github/workflows/ci.yml
act -j lint
```

`act` is the **right** choice when the code under test is the workflow
YAML itself (matrix combinations, permissions, reusable workflows). For
the *build logic*, prefer Dagger -- it runs in `act`, in real CI, and
on your laptop identically.

---

## Stack anti-patterns (small teams, take note)

1. **Adopting everything at once.** Pick one tool per quarter. Start
   with Dagger (biggest ROI), then Tilt, then Pyroscope.
2. **Tilt against a shared cluster.** Tilt writes to the cluster it
   sees -- make sure your kubecontext is a *local* Kind cluster, not
   `prod-eu-west-1`.
3. **Backstage without a team.** See the "when not to use" box above.
   A 3-person team will regret the install.
4. **Alloy + OTel Collector side-by-side.** You get two places to look
   when a trace goes missing. Pick one.
5. **Pyroscope on tiny services.** A 50-req/min service generates so
   few samples that the flame graph is useless; skip the agent.

---

## The inner loop, summarised

```sh
# once per machine
brew install tilt kind dagger act
kind create cluster --name i3-dev

# every morning
tilt up                                            # hot-reload k8s
dagger call lint --source=. stdout                 # matches CI
dagger call test --source=. stdout
open http://localhost:10350                        # Tilt UI
open http://localhost:4040                         # Pyroscope
open http://backstage.internal/catalog/default/component/i3   # portal
```

Total time from `git clone` to "first saved edit shows up in the pod":
**under five minutes**. That is the bar to hold.

---

## Extended topics

### 7.1 Migrating legacy Makefile targets into Dagger

Our existing [`Makefile`](../../Makefile) is *not* going away --
Makefiles are still the fastest way to alias a one-liner. The
migration rule of thumb is:

* **Stays in Make**: thin wrappers (`make fmt`, `make shell`), things
  that only a human ever runs interactively.
* **Moves to Dagger**: anything that also runs in CI, anything with
  more than three lines of bash, anything that has to work identically
  on macOS and Linux.

Concretely:

| Make target           | Dagger equivalent                                | Keep in Make? |
| --------------------- | ------------------------------------------------ | ------------- |
| `make lint`           | `dagger call lint --source=. stdout`             | Yes, as alias |
| `make test`           | `dagger call test --source=. stdout`             | Yes, as alias |
| `make docker-build`   | `dagger call build-image --source=.`             | No            |
| `make release-v1.2.3` | `dagger call release --source=. --tag=v1.2.3`    | No            |
| `make shell`          | *(stays Makefile -- interactive)*                | Yes           |

The Makefile keeps its shape for muscle memory; under the hood, the
target shells out to `dagger call`.

### 7.2 Secrets management across the stack

Each tool in this stack has a slightly different idea of where secrets
live. The mapping we standardised on:

| Environment           | Source                                             | Tool reads from            |
| --------------------- | -------------------------------------------------- | -------------------------- |
| Laptop dev            | `.env.local` (git-ignored)                         | `direnv` + `dotenv`        |
| Codespaces            | GitHub Codespaces Secrets (per-repo)               | `remoteEnv` + `secrets`    |
| GitHub Actions        | Repo/org-level Actions Secrets                     | `${{ secrets.X }}`         |
| `act` (local actions) | `.secrets.local` (git-ignored)                     | `act --secret-file ...`    |
| Dagger                | Dagger `Secret` type -- never on stdout or in logs | `dag.set_secret("x", ...)` |
| Tilt                  | Kubernetes `Secret` objects in `deploy/k8s/dev`    | `envFrom: secretRef`       |
| Alloy                 | Env vars injected by compose/k8s                   | `sys.env("X")`             |

Critically, **Dagger's `Secret` type never leaks into logs or cache
keys**. When you pass a secret into a Dagger container, the SDK hashes
it for cache partitioning but never prints the plaintext. This is why
our `release` function takes the registry token as a `Secret`, not a
plain string.

### 7.3 Telemetry correlation: from flame graph to log line

One payoff of the unified stack is **cross-signal correlation** inside
Grafana. Workflow for debugging a latency spike:

1. Alert fires in Alertmanager (fed by Mimir). PagerDuty pages the
   on-call via the Backstage `pagerduty.com/integration-key`
   annotation.
2. On-call opens the Grafana dashboard linked from the Backstage entity
   -- time window is auto-scoped to the alert.
3. Click on a slow span in the Tempo panel -- Grafana's
   "Trace -> Profile" linking takes you straight to the Pyroscope
   flame graph for the same service + time range.
4. The flame graph highlights (say) an unexpected hot path in
   `i3/router/routing.py`. Click "View logs" -- Grafana's
   "Trace -> Logs" linking pivots into Loki, scoped to the same
   `trace_id`.
5. The log line tells you which user session triggered the slow path.
   Fix, deploy, move on.

None of this is possible without a consistent `trace_id` and
`service.name` flowing through every signal. Our structlog configuration
injects both into every log line; Alloy's attribute-redact processor
makes sure no sensitive header survives the journey.

### 7.4 Machine-size recommendations

Rough guide for what you need on a laptop:

* **Lite mode (`tilt up -- --lite`)**: 4 CPU, 8 GB RAM is enough.
* **Full stack (Tilt + Alloy + Pyroscope + MkDocs)**: 8 CPU, 16 GB RAM.
* **Pre-submit full `dagger call release`**: plan for a 2-3 minute run
  and ~20 GB of BuildKit cache after a few days.

Codespaces default (`4 CPU / 8 GB`) works for the inner loop; bump to
`8 CPU / 16 GB` for benchmark work.

---

## References (consolidated)

* Dagger docs, Python SDK, modules: <https://docs.dagger.io>
* Tilt docs and extensions: <https://docs.tilt.dev/>
* Grafana Pyroscope (post-2023 acquisition): <https://grafana.com/docs/pyroscope/latest/>
* Backstage: <https://backstage.io/docs>
* Grafana Alloy launch (2024): <https://grafana.com/blog/2024/04/09/grafana-alloy-opentelemetry-collector-with-prometheus-pipelines/>
* nektos/act: <https://nektosact.com>
* GitHub Codespaces dev containers: <https://docs.github.com/en/codespaces>
