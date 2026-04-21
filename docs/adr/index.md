# Architecture Decision Records

An **Architecture Decision Record (ADR)** captures a significant
architectural decision, its context, its consequences, and the alternatives
considered. I³ uses the [MADR 4.0](https://adr.github.io/madr/) format.

!!! tip "Template"
    Start new ADRs from the [template](template.md). Number sequentially —
    do not reuse numbers, even for superseded decisions.

## Status conventions { #status }

| Status | Meaning |
|:-------|:--------|
| **Proposed**   | Drafted, under review, not yet adopted |
| **Accepted**   | Adopted; implementation follows |
| **Deprecated** | Still in effect but superseded soon |
| **Superseded** | No longer in effect; links to the ADR that replaces it |
| **Rejected**   | Considered and explicitly not adopted |

## Index { #index }

| # | Title | Status | Date |
|:-:|:------|:-------|:-----|
| 0001 | [Custom SLM over HuggingFace](0001-custom-slm-over-huggingface.md)              | Accepted | 2025-12-18 |
| 0002 | [TCN encoder over LSTM / Transformer](0002-tcn-over-lstm-transformer.md)        | Accepted | 2025-12-20 |
| 0003 | [Thompson sampling over UCB](0003-thompson-sampling-over-ucb.md)                | Accepted | 2026-01-08 |
| 0004 | [Privacy by architecture](0004-privacy-by-architecture.md)                      | Accepted | 2026-01-15 |
| 0005 | [FastAPI over Flask](0005-fastapi-over-flask.md)                                | Accepted | 2026-01-22 |
| 0006 | [Poetry over pip-tools](0006-poetry-over-pip-tools.md)                          | Accepted | 2026-02-02 |
| 0007 | [OpenTelemetry for observability](0007-opentelemetry-for-observability.md)      | Accepted | 2026-02-18 |
| 0008 | [Fernet over custom crypto](0008-fernet-over-custom-crypto.md)                  | Accepted | 2026-03-04 |
| 0009 | [SQLite over Redis](0009-sqlite-over-redis.md)                                  | Accepted | 2026-03-12 |
| 0010 | [Pydantic v2 for config](0010-pydantic-v2-config.md)                            | Accepted | 2026-03-25 |

## How to write a new ADR { #new }

1. Copy [template.md](template.md) to `NNNN-short-slug.md`.
2. Fill in **Status**, **Context**, **Decision**, **Consequences**,
   **Alternatives**.
3. Add an entry to the index above and to `mkdocs.yml::nav`.
4. Propose via PR. Reviewers look for honesty about trade-offs.

## Reading order { #reading }

If you are new to the project and want to understand *why*, read in this
order:

1. [0004 — Privacy by architecture](0004-privacy-by-architecture.md) —
   the constraint that shaped everything.
2. [0001 — Custom SLM](0001-custom-slm-over-huggingface.md) — the
   architectural centrepiece.
3. [0002 — TCN encoder](0002-tcn-over-lstm-transformer.md).
4. [0003 — Thompson sampling](0003-thompson-sampling-over-ucb.md).
5. The rest can be read in any order.
