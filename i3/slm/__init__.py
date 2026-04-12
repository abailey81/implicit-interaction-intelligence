"""Custom Small Language Model built from scratch in PyTorch.

No HuggingFace. No pre-trained weights. Every component implemented from first principles.

Modules:
    tokenizer       -- Word-level tokenizer with vocabulary management
    embeddings      -- Token + sinusoidal positional embeddings
    attention       -- Multi-head self-attention and feed-forward blocks
    cross_attention -- Cross-attention for conditioning on user state
    transformer     -- Pre-LN transformer block with cross-attention
    model           -- Full AdaptiveSLM assembly
    generate        -- Autoregressive generation with sampling
    quantize        -- INT8 dynamic quantization for edge deployment
    train           -- Training loop with cosine warmup scheduling
"""
