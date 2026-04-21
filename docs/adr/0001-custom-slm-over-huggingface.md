# ADR-0001 — Custom SLM over HuggingFace

- **Status**: Accepted
- **Date**: 2025-12-18
- **Deciders**: Tamer Atesyakar
- **Technical area**: SLM, cross-attention conditioning

## Context and problem statement { #context }

I³ needs an on-device language model that can be **architecturally**
conditioned on a user-state embedding at every transformer block (see
[Cross-attention conditioning](../architecture/cross-attention-conditioning.md)).
The personalisation signal is a learned vector, not a prompt; the payload
must enter as key-value pairs for a dedicated cross-attention step inside
every block.

Pre-trained HuggingFace decoder models are built without this mechanism.
Adding a cross-attention module between self-attention and feed-forward
is a structural change to every block — the model cannot be retrofitted
without re-training from a random initialisation.

## Decision drivers { #drivers }

- Must support per-block cross-attention to a projected `(a, u)` payload.
- On-device memory budget ≤ 20 MB after INT8 quantization.
- P95 generation latency ≤ 200 ms on a modern laptop CPU.
- No mandatory cloud dependency for the default path.
- Clear, inspectable implementation (interview / teaching artefact).

## Considered options { #options }

1. **Custom 6.3M-parameter transformer built from scratch in PyTorch.**
2. **HuggingFace pre-trained decoder (e.g. TinyLlama-1.1B) + LoRA cross-attention adapter.**
3. **ONNX Runtime + a pre-compiled small decoder.**

## Decision outcome { #outcome }

> **Chosen option**: Option 1 — Custom transformer. Pre-trained models
> cannot accept our conditioning without re-training from scratch, which
> erases the value of using a pre-trained model in the first place.

### Consequences — positive { #pos }

- Cross-attention conditioning can be baked into every block, matching
  the project's architectural thesis.
- No `transformers` / HuggingFace dependency — smaller supply-chain
  surface and simpler edge packaging.
- Implementation is the project's teaching artefact; every component is
  inspectable and unit-tested.
- ~6.3M parameters at `d_model=256, n_heads=4, n_layers=4` fits comfortably
  in the edge budget after INT8.

### Consequences — negative { #neg }

- Dramatically weaker factual and world knowledge than any billion-param
  pre-trained model. *Mitigation*: the router delegates factual queries
  to the cloud arm.
- More training code to own and maintain. *Mitigation*: comprehensive
  tests (`tests/test_slm.py`) and a pinned training recipe.
- We do not benefit from the Hub's checkpoint diversity. *Mitigation*:
  accepted; this is a demonstration system.

## Pros and cons of the alternatives { #alternatives }

### Option 2 — HuggingFace + LoRA adapter { #opt-2 }

- Yes Strong factual recall out of the box.
- Yes Massive community and tooling.
- No LoRA adds parameters into existing linear layers; it cannot synthesise
  new architecture (a dedicated cross-attention between self-attn and FFN).
- No Even with LoRA adapters, pre-trained attention patterns resist
  re-purposing without full fine-tune.
- No Minimum useful size is > 500M params — infeasible for the 20 MB
  INT8 target on a Kirin A2 wearable.

### Option 3 — ONNX Runtime + pre-compiled decoder { #opt-3 }

- Yes Very fast inference on many targets.
- Yes Removes a PyTorch dependency.
- No Cannot modify model graph inside a block at training time.
- No Pre-compiled decoders do not expose cross-attention hooks.
- No Loses the "every layer is custom" pedagogical value.

## References { #refs }

- [Research: Cross-attention conditioning](../research/cross_attention.md)
- [Architecture: SLM (Layer 6a)](../architecture/layers.md#l6a)
- [Model Card](../model_card.md)
- Vaswani, A. *et al.* "Attention is all you need." NeurIPS 2017.
