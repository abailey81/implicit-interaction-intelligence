# ADR-NNNN — Short noun phrase describing the decision

- **Status**: Proposed | Accepted | Deprecated | Superseded by [ADR-XXXX] | Rejected
- **Date**: `YYYY-MM-DD`
- **Deciders**: List of people
- **Consulted**: List of people
- **Informed**: List of people
- **Technical area**: e.g. encoder, router, persistence

!!! note "Template"
    This is the MADR 4.0 template adapted for I³. When copying, delete
    this admonition and every placeholder before submitting.

## Context and problem statement { #context }

What is the problem? Why now? One-to-three paragraphs of background. State
the forces — technical, organisational, regulatory — that constrain the
decision. Quote numbers (latency targets, parameter budgets, compliance
requirements) wherever possible.

## Decision drivers { #drivers }

Bulleted list of the criteria we will evaluate options against. Examples:

- On-device memory budget ≤ 20 MB.
- P95 generation latency ≤ 200 ms on CPU.
- No external service dependency for the default path.
- Must compose with the existing `AdaptationVector` contract.

## Considered options { #options }

Numbered list of candidate solutions.

1. **Option A** — short name + one-line description.
2. **Option B**.
3. **Option C**.

## Decision outcome { #outcome }

> **Chosen option**: *Option X*, because …

One paragraph with the crisp statement of what we will do and the
single-most-important reason.

### Consequences — positive { #pos }

- What gets easier, cheaper, or possible because of this choice.
- Each item with a concrete reference (file, benchmark, ADR).

### Consequences — negative { #neg }

- What we lose or pay.
- Each item with a mitigation plan or a deliberate acceptance note.

## Pros and cons of the alternatives { #alternatives }

### Option A { #opt-a }

- Yes …
- No …

### Option B { #opt-b }

- Yes …
- No …

### Option C { #opt-c }

- Yes …
- No …

## References { #refs }

- Related ADRs: [ADR-XXXX](XXXX-slug.md).
- External papers / standards.
- Issue tracker / discussion links.
