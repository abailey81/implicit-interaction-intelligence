---
title: "I³ — Implicit Interaction Intelligence"
description: "Privacy-first, edge-ready adaptive AI companion."
hide:
  - navigation
  - toc
---

<div class="i3-hero" markdown>

<span class="i3-hero__badge">HMI · Edge AI · Privacy by Architecture</span>

# I³ — Implicit Interaction Intelligence { .i3-hero__title }

<p class="i3-hero__tagline">
  An AI companion that learns from <em>how</em> you interact — keystroke
  dynamics, linguistic complexity, temporal rhythm — and adapts its responses
  across cognitive load, style, tone, and accessibility. No pre-trained
  weights. No HuggingFace. No raw text ever persisted.
</p>

<div class="i3-hero__actions" markdown>
[:material-rocket-launch: Quickstart](getting-started/quickstart.md){ .md-button .md-button--primary }
[:material-book-open-variant: Architecture](architecture/overview.md){ .md-button }
[:material-github: GitHub](https://github.com/abailey81/implicit-interaction-intelligence){ .md-button }
</div>

</div>

## What is I³? { #what-is-i3 }

Conversational AI systems typically respond to **what** you say. I³ responds
to **how** you say it. A ~50K-parameter TCN encoder observes keystroke
dynamics and linguistic features, a ~6.3M-parameter custom transformer
conditions its generation on your current state via cross-attention, and a
contextual Thompson sampling router decides when to consult a cloud LLM at
all.

<div class="i3-grid" markdown>

<div class="i3-card" markdown>
### :material-eye-outline: Implicit, not explicit
The system never asks you to describe yourself. Every adaptation — tone,
verbosity, formality, simplification — is inferred from behavioural signals.
</div>

<div class="i3-card" markdown>
### :material-chip: Custom models
A TCN encoder and a 6.3M-parameter SLM, both built from scratch in PyTorch.
No pre-trained checkpoints, no `transformers` import anywhere.
</div>

<div class="i3-card" markdown>
### :material-shield-lock-outline: Privacy by architecture
Raw user text is never written to disk. The diary stores only embeddings,
topic keywords, and metrics. Fernet-encrypted profiles at rest.
</div>

<div class="i3-card" markdown>
### :material-cellphone-link: Edge-ready
INT8 dynamic quantization shrinks the SLM to ~7 MB. Fits comfortably on a
Kirin 9000 phone; feasible on a Kirin A2 wearable with encoder-only mode.
</div>

<div class="i3-card" markdown>
### :material-dice-multiple-outline: Contextual routing
A Bayesian logistic Thompson sampling bandit decides local vs. cloud per
message, with hard privacy overrides for sensitive topics.
</div>

<div class="i3-card" markdown>
### :material-brain: Architectural adaptation
Cross-attention conditioning weaves user state into every transformer block
at every token position — not just a prompt prefix.
</div>

</div>

## The stack { #stack }

| Layer | Technology |
|:------|:-----------|
| Deep learning | PyTorch 2.6+ (CPU-first; CUDA optional) |
| Web server | FastAPI 0.115+ + Starlette WebSockets |
| Persistence | SQLite via `aiosqlite` (async, Fernet-at-rest) |
| Cloud LLM | Anthropic Claude via direct `httpx` client |
| Encryption | `cryptography` (Fernet, key rotation supported) |
| Configuration | Pydantic v2 (17 nested models, `frozen=True`) |
| Packaging | Poetry (Python 3.10 – 3.12) |
| Observability | OpenTelemetry traces + Prometheus metrics |

## Where to go next { #next }

<div class="i3-grid" markdown>

<div class="i3-card" markdown>
### :material-school-outline: New here?
Start with the [five-minute quickstart](getting-started/quickstart.md), then
skim the [architecture overview](architecture/overview.md).
</div>

<div class="i3-card" markdown>
### :material-flask-outline: Research reader?
The [cross-attention conditioning paper-style note](research/cross_attention.md)
and the [bandit theory write-up](research/bandit_theory.md) are the
architectural centrepieces.
</div>

<div class="i3-card" markdown>
### :material-server-outline: Deploying?
See [Deployment](operations/deployment.md), the [Runbook](operations/runbook.md),
and [Kubernetes](operations/kubernetes.md).
</div>

<div class="i3-card" markdown>
### :material-hammer-wrench: Contributing?
Read [Contributing](contributing/index.md), the [ADR index](adr/index.md),
and the [security policy](security/index.md) first.
</div>

</div>

!!! tip "This site is versioned"
    Every page carries a last-updated timestamp (see the footer) and a
    link to the corresponding source file in the repository. Use the
    edit-pencil in the top right to propose changes directly on GitHub.

!!! note "Licence"
    I³ is released under the [MIT License](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/LICENSE).
    Copyright © 2026 Tamer Atesyakar.
