# ADR-0002 — TCN encoder over LSTM / Transformer

- **Status**: Accepted
- **Date**: 2025-12-20
- **Deciders**: Tamer Atesyakar
- **Technical area**: encoder (Layer 2)

## Context and problem statement { #context }

Layer 2 encodes a sequence of 32-dim interaction feature vectors into a
64-dim user-state embedding. The model must:

- Run in low-single-digit milliseconds on a laptop CPU.
- Have a receptive field of at least ~30 timesteps (a conversational turn).
- Be stable to train with a contrastive NT-Xent objective on synthetic
  data.
- Stay well under 100K parameters so edge memory is dominated by the SLM,
  not the encoder.
- Be implementable entirely from PyTorch primitives (no custom CUDA).

## Decision drivers { #drivers }

- Latency ≤ 5 ms P95 on CPU.
- Receptive field ≥ 30 timesteps.
- Parameters ≤ 100 k.
- Trains stably with NT-Xent (batch sizes of 256 on commodity hardware).

## Considered options { #options }

1. **Temporal Convolutional Network (TCN)** — dilated causal convolutions
   with residual connections.
2. **LSTM / GRU** encoder.
3. **Transformer encoder** (self-attention only).

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — TCN. Dilations `[1, 2, 4, 8]` give a
> ~61-timestep receptive field at four blocks, ~50K parameters, and ~3 ms
> P95 CPU latency. Causal convolutions are the correct inductive bias
> for a sequence with an implicit "now" we are trying to characterise.

### Consequences — positive { #pos }

- Receptive field is deterministic and easy to reason about.
- Parallelism across timesteps during training — faster than an LSTM at
  equal param count.
- Gradient flow through the residual stack is stable; we see no need for
  gradient clipping.
- Pairs naturally with contrastive augmentations (temporal crops do not
  require re-initialising a hidden state).

### Consequences — negative { #neg }

- Fixed receptive field: extending context past 61 steps requires adding
  another dilated block or increasing kernel size. *Mitigation*: Layer 3's
  EMA provides long-horizon memory outside the encoder.
- Cannot learn arbitrary, data-dependent time dependencies as flexibly
  as self-attention. *Mitigation*: attention is overkill for a 32-dim
  vector sequence; the inductive bias of convolutions helps here.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — LSTM / GRU { #opt-2 }

- ✅ Unbounded effective memory.
- ❌ Serial forward pass; 3–4× slower per batch on CPU.
- ❌ Gradient flow fragile on small synthetic datasets.
- ❌ Harder to augment for contrastive learning (stateful).

### Option 3 — Transformer encoder { #opt-3 }

- ✅ Fully parallel, attention is interpretable.
- ❌ \(\mathcal{O}(T^2)\) attention at inference time; with sequences of
  60+ steps this exceeds our 5 ms CPU budget at 64-dim.
- ❌ At 50k params, self-attention is under-provisioned and behaves
  poorly with NT-Xent (collapses to uniform attention).
- ❌ No strong inductive bias for locality, which the keystroke feature
  sequence has.

## References { #refs }

- [Research: Contrastive loss](../research/contrastive_loss.md)
- [Architecture: Encoder (Layer 2)](../architecture/layers.md#l2)
- Bai, S. *et al.* "An empirical evaluation of generic convolutional and
  recurrent networks for sequence modeling." arXiv:1803.01271 (2018).
- Chen, T. *et al.* "A simple framework for contrastive learning of
  visual representations." ICML 2020.
