# Architecture Overview

A bird's-eye view of the I³ system: its seven sequential layers, its two
cross-cutting concerns, and the contracts between them.

!!! tip "Deep dive available"
    The canonical, maths-heavy reference is
    [`docs/ARCHITECTURE.md`](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/docs/ARCHITECTURE.md).
    This page links to it heavily and complements it with Mermaid diagrams
    and the Material navigation.

## The seven layers { #layers }

```mermaid
flowchart TB
    subgraph L1["Layer 1 — Perception · i3/interaction/"]
      direction TB
      P1[KeystrokeMonitor] --> P2[FeatureExtractor]
      P2 --> P3[InteractionFeatureVector · 32-dim]
    end
    subgraph L2["Layer 2 — Encoding · i3/encoder/"]
      direction TB
      E1[TemporalConvNet<br/>dilations 1·2·4·8] --> E2[UserStateEmbedding · 64-dim]
    end
    subgraph L3["Layer 3 — User model · i3/user_model/"]
      direction TB
      U1[Welford online stats] --> U2[Instant / Session / Long-term EMAs]
    end
    subgraph L4["Layer 4 — Adaptation · i3/adaptation/"]
      direction TB
      A1[AdaptationController] --> A2[AdaptationVector · 8-dim]
    end
    subgraph L5["Layer 5 — Router · i3/router/"]
      direction TB
      R1[Thompson sampler<br/>Bayesian logistic + Laplace] --> R2{route?}
    end
    subgraph L6a["Layer 6a — SLM · i3/slm/"]
      direction TB
      S1[AdaptiveSLM ~6.3M<br/>cross-attention conditioning]
    end
    subgraph L6b["Layer 6b — Cloud · i3/cloud/"]
      direction TB
      C1[Claude via httpx<br/>PII-sanitised]
    end
    subgraph L7["Layer 7 — Diary · i3/diary/"]
      direction TB
      D1[Embeddings · topics · metrics only]
    end

    L1 --> L2 --> L3 --> L4 --> L5
    R2 -->|local| S1
    R2 -->|cloud| C1
    S1 --> D1
    C1 --> D1
```

Each layer is a small Python package with a typed, immutable data contract
on both ends. See [Layers](layers.md) for per-layer deep-dives.

## The two cross-cutting concerns { #cross-cutting }

```mermaid
flowchart LR
    subgraph XCC["Cross-cutting"]
        P[Privacy · i3/privacy/<br/>10 PII regex · Fernet · auditor]
        O[Profiling · i3/profiling/<br/>latency · memory · INT8 size]
    end

    L1[Perception] -.-> P
    L5[Router] -.-> P
    L7[Diary] -.-> P
    L2[Encoding] -.-> O
    L6[SLM / Cloud] -.-> O
```

!!! note "Why cross-cutting, not a layer?"
    Privacy and profiling are concerns that every layer depends on, but
    neither produces the primary data output. They are implemented as
    thin libraries consumed by the numbered layers — see
    [Privacy](privacy.md) and
    [ADR 0004 — Privacy by architecture](../adr/0004-privacy-by-architecture.md).

## The 9-step pipeline { #pipeline }

The central `PipelineEngine.process()` awaits a fixed sequence:

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant P as Pipeline
    participant PR as Privacy
    participant F as Features
    participant UM as UserModel
    participant EN as Encoder
    participant AD as Adaptation
    participant RT as Router
    participant LM as SLM / Cloud
    participant DY as Diary

    U->>P: PipelineInput(user_id, text, keystrokes)
    P->>PR: sanitize_pii(text)
    P->>F: extract InteractionFeatureVector
    P->>UM: update Welford stats · compute deviations
    P->>EN: encode → 64-dim UserStateEmbedding
    P->>AD: AdaptationController → 8-dim AdaptationVector
    P->>RT: RoutingDecision (Thompson sample)
    alt route = local
        P->>LM: AdaptiveSLM.generate(conditioning=(a, u))
    else route = cloud
        P->>LM: CloudClient.complete(system=prompt_from(a))
    end
    P->>DY: log embeddings · topics · metrics
    P-->>U: PipelineOutput(response, route, adaptation, latency)
```

All inter-layer IO is `async` and typed with `pydantic.BaseModel` /
`@dataclass(frozen=True)` so data cannot be accidentally mutated in flight.

## Data contracts at a glance { #contracts }

| Layer | Input | Output |
|:------|:------|:-------|
| 1 Perception       | `KeystrokeEvent` stream, raw text      | `InteractionFeatureVector` (32-dim) |
| 2 Encoding         | feature sequence (variable length)     | `UserStateEmbedding` (64-dim)       |
| 3 User Modelling   | `UserStateEmbedding` + feature vector  | `UserProfile`, `DeviationMetrics`   |
| 4 Adaptation       | `UserProfile`, `SessionState`          | `AdaptationVector` (8-dim)          |
| 5 Routing          | `RoutingContext` (12-dim)              | `RoutingDecision { arm, posterior }` |
| 6a Local SLM       | prompt, `AdaptationVector`, embedding  | decoded tokens                      |
| 6b Cloud LLM       | PII-sanitised prompt, system prompt    | decoded tokens                      |
| 7 Diary            | embeddings, topics, metrics            | persisted `DiaryEntry`              |

## Technology footprint { #tech }

| Concern | Technology | Rationale (ADR) |
|:--------|:-----------|:---------------|
| Encoder              | Custom TCN in PyTorch                   | [ADR 0002](../adr/0002-tcn-over-lstm-transformer.md) |
| Language model       | Custom 6.3M-param transformer           | [ADR 0001](../adr/0001-custom-slm-over-huggingface.md) |
| Router               | Contextual Thompson sampling            | [ADR 0003](../adr/0003-thompson-sampling-over-ucb.md) |
| Web server           | FastAPI 0.115+                          | [ADR 0005](../adr/0005-fastapi-over-flask.md) |
| Observability        | OpenTelemetry + Prometheus              | [ADR 0007](../adr/0007-opentelemetry-for-observability.md) |
| At-rest crypto       | `cryptography` Fernet                   | [ADR 0008](../adr/0008-fernet-over-custom-crypto.md) |
| Persistence          | SQLite (`aiosqlite`)                    | [ADR 0009](../adr/0009-sqlite-over-redis.md) |
| Configuration        | Pydantic v2 (`frozen=True`)             | [ADR 0010](../adr/0010-pydantic-v2-config.md) |
| Packaging            | Poetry 1.8                              | [ADR 0006](../adr/0006-poetry-over-pip-tools.md) |

## Where to go next { #next }

<div class="i3-grid" markdown>

<div class="i3-card" markdown>
### :material-layers: Layer-by-layer
[Layers](layers.md) breaks every package down with file links, dimensionality,
and responsibility statements.
</div>

<div class="i3-card" markdown>
### :material-brain: The novel part
[Cross-attention conditioning](cross-attention-conditioning.md) — the
architectural centrepiece.
</div>

<div class="i3-card" markdown>
### :material-dice-multiple-outline: The router
[Router](router.md) — Bayesian logistic regression, Laplace approximation,
privacy overrides.
</div>

<div class="i3-card" markdown>
### :material-shield-lock-outline: Privacy
[Privacy](privacy.md) — ten PII patterns, Fernet, the database auditor,
the "no raw text ever" schema constraint.
</div>

</div>
