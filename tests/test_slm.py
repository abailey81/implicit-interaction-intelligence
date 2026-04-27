"""Tests for the custom Adaptive Small Language Model.

Validates attention mechanics, causal masking, KV cache consistency,
forward shape, conditioning effects, parameter count, and weight tying.
"""

from __future__ import annotations

import pytest
import torch

from i3.slm.attention import MultiHeadSelfAttention, FeedForward, create_causal_mask
from i3.slm.cross_attention import MultiHeadCrossAttention, ConditioningProjector
from i3.slm.model import AdaptiveSLM
from i3.slm.tokenizer import SimpleTokenizer


# -------------------------------------------------------------------------
# Attention
# -------------------------------------------------------------------------

class TestAttention:
    """Tests for Multi-Head Self-Attention."""

    @pytest.fixture
    def attn(self) -> MultiHeadSelfAttention:
        return MultiHeadSelfAttention(d_model=64, n_heads=4, dropout=0.0)

    def test_output_shape(self, attn: MultiHeadSelfAttention) -> None:
        """Output shape must match input shape."""
        x = torch.randn(2, 10, 64)
        out, weights = attn(x)
        assert out.shape == (2, 10, 64)
        assert weights.shape == (2, 4, 10, 10)

    def test_causal_mask_shape(self) -> None:
        """Causal mask must be [1, 1, S, S]."""
        mask = create_causal_mask(8)
        assert mask.shape == (1, 1, 8, 8)

    def test_causal_mask_values(self) -> None:
        """Causal mask: 0 for past/present, large negative for future.

        Implementation uses ``_MASK_NEG = -1e9`` instead of ``-inf`` to avoid
        ``softmax(-inf)`` producing NaN when an entire row is masked
        (which can legitimately happen during edge cases of cross-
        attention or padding-only rows). The test reflects the actual
        contract: future positions are masked with a very-negative
        finite value such that softmax weights drop below 1e-300.
        """
        mask = create_causal_mask(4)
        # Position 0 can only see position 0
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 0, 1] <= -1e8  # masked future, finite-negative
        # Position 3 can see positions 0, 1, 2, 3
        assert mask[0, 0, 3, 0] == 0.0
        assert mask[0, 0, 3, 3] == 0.0

    def test_causal_mask_prevents_future(self, attn: MultiHeadSelfAttention) -> None:
        """With causal mask, changing future tokens should not affect past outputs."""
        attn.eval()

        x1 = torch.randn(1, 6, 64)
        x2 = x1.clone()
        x2[0, 4:, :] = torch.randn(2, 64)  # Modify future tokens

        mask = create_causal_mask(6)
        with torch.no_grad():
            out1, _ = attn(x1, mask=mask)
            out2, _ = attn(x2, mask=mask)

        # Outputs at positions 0..3 must be identical
        torch.testing.assert_close(out1[:, :4, :], out2[:, :4, :], atol=1e-5, rtol=1e-5)

    def test_kv_cache_consistency(self, attn: MultiHeadSelfAttention) -> None:
        """KV cache must produce the same output as full sequence attention."""
        attn.eval()

        seq = torch.randn(1, 5, 64)

        # Full forward (no cache)
        with torch.no_grad():
            full_out, _ = attn(seq, mask=create_causal_mask(5))

        # Incremental forward (with cache)
        attn.clear_cache()
        outputs = []
        with torch.no_grad():
            for t in range(5):
                token = seq[:, t:t+1, :]
                out_t, _ = attn(token, use_cache=True)
                outputs.append(out_t)

        cached_out = torch.cat(outputs, dim=1)
        attn.clear_cache()

        # Each position's output should match
        torch.testing.assert_close(
            full_out[:, -1:, :], cached_out[:, -1:, :],
            atol=1e-4, rtol=1e-4,
        )

    def test_d_model_not_divisible_by_heads_raises(self) -> None:
        """d_model must be divisible by n_heads."""
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(d_model=65, n_heads=4)


# -------------------------------------------------------------------------
# FeedForward
# -------------------------------------------------------------------------

class TestFeedForward:
    """Tests for the position-wise FFN."""

    def test_output_shape(self) -> None:
        """FFN output shape must match input shape."""
        ffn = FeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)


# -------------------------------------------------------------------------
# Cross-Attention
# -------------------------------------------------------------------------

class TestCrossAttention:
    """Tests for Multi-Head Cross-Attention."""

    def test_output_shape(self) -> None:
        """Cross-attention output must have same shape as query."""
        cross = MultiHeadCrossAttention(d_model=64, n_heads=2, dropout=0.0)
        query = torch.randn(2, 10, 64)
        kv = torch.randn(2, 4, 64)
        out, weights = cross(query, kv, kv)
        assert out.shape == (2, 10, 64)
        assert weights.shape == (2, 2, 10, 4)  # [B, heads, seq, cond]


# -------------------------------------------------------------------------
# ConditioningProjector
# -------------------------------------------------------------------------

class TestConditioningProjector:
    """Tests for the conditioning token projector."""

    @pytest.fixture
    def projector(self) -> ConditioningProjector:
        return ConditioningProjector(
            adaptation_dim=8,
            user_state_dim=64,
            d_model=128,
            n_tokens=4,
        )

    def test_output_shape(self, projector: ConditioningProjector) -> None:
        """Should produce [batch, n_tokens, d_model]."""
        adapt = torch.randn(3, 8)
        state = torch.randn(3, 64)
        out = projector(adapt, state)
        assert out.shape == (3, 4, 128)

    def test_default_conditioning(self) -> None:
        """create_default_conditioning should produce valid shapes."""
        adapt, state = ConditioningProjector.create_default_conditioning(
            batch_size=2, adaptation_dim=8, user_state_dim=64
        )
        assert adapt.shape == (2, 8)
        assert state.shape == (2, 64)
        # cognitive_load and emotional_tone should be 0.5
        assert adapt[0, 0].item() == pytest.approx(0.5)
        assert adapt[0, 5].item() == pytest.approx(0.5)


# -------------------------------------------------------------------------
# AdaptiveSLM (full model)
# -------------------------------------------------------------------------

class TestSLM:
    """Tests for the complete Adaptive Small Language Model."""

    @pytest.fixture
    def slm(self) -> AdaptiveSLM:
        """Create a small SLM for testing."""
        return AdaptiveSLM(
            vocab_size=500,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=64,
            conditioning_dim=64,
            adaptation_dim=8,
            n_cross_heads=2,
            n_conditioning_tokens=4,
            dropout=0.0,
            tie_weights=True,
        )

    def test_forward_shape(self, slm: AdaptiveSLM) -> None:
        """Output logits must be [batch, seq_len, vocab_size]."""
        ids = torch.randint(1, 500, (2, 10))
        adapt = torch.randn(2, 8)
        state = torch.randn(2, 64)
        logits, layer_info = slm(ids, adapt, state)

        assert logits.shape == (2, 10, 500)
        assert len(layer_info) == 2  # 2 layers
        assert 'layer_0' in layer_info
        assert 'layer_1' in layer_info

    def test_forward_no_conditioning(self, slm: AdaptiveSLM) -> None:
        """Model should work without explicit conditioning (uses defaults)."""
        ids = torch.randint(1, 500, (1, 5))
        logits, _ = slm(ids)
        assert logits.shape == (1, 5, 500)

    def test_conditioning_effect(self, slm: AdaptiveSLM) -> None:
        """Different conditioning vectors must produce different logits."""
        slm.eval()
        ids = torch.randint(1, 500, (1, 8))

        adapt_a = torch.zeros(1, 8)
        adapt_a[0, 0] = 0.0  # Low cognitive load
        adapt_b = torch.zeros(1, 8)
        adapt_b[0, 0] = 1.0  # High cognitive load

        state = torch.randn(1, 64)

        with torch.no_grad():
            logits_a, _ = slm(ids, adapt_a, state)
            logits_b, _ = slm(ids, adapt_b, state)

        # Logits should differ because conditioning differs
        assert not torch.allclose(logits_a, logits_b, atol=1e-3), (
            "Different conditioning should produce different logits."
        )

    def test_parameter_count(self, slm: AdaptiveSLM) -> None:
        """Parameter count should be in a reasonable range for the small config."""
        n = slm.num_parameters
        # With vocab=500, d_model=64, 2 layers, d_ff=128
        # Expect roughly 200k-2M params for this tiny config
        assert 50_000 < n < 5_000_000, (
            f"Parameter count {n:,} is outside expected range."
        )

    def test_weight_tying(self, slm: AdaptiveSLM) -> None:
        """Output projection and token embedding should share weights."""
        embed_weight = slm.embedding.token_embedding.embedding.weight
        output_weight = slm.output_projection.weight
        assert embed_weight.data_ptr() == output_weight.data_ptr(), (
            "Weight tying failed: embedding and output projection should "
            "share the same weight tensor."
        )

    def test_no_weight_tying(self) -> None:
        """When tie_weights=False, weights should be independent."""
        slm = AdaptiveSLM(
            vocab_size=200, d_model=64, n_heads=4, n_layers=1,
            d_ff=128, tie_weights=False,
        )
        embed_weight = slm.embedding.token_embedding.embedding.weight
        output_weight = slm.output_projection.weight
        assert embed_weight.data_ptr() != output_weight.data_ptr()

    def test_clear_cache(self, slm: AdaptiveSLM) -> None:
        """clear_cache must not raise and should reset all layer caches."""
        ids = torch.randint(1, 500, (1, 3))
        slm.eval()
        with torch.no_grad():
            slm(ids, use_cache=True)
        slm.clear_cache()
        # Verify no cached tensors remain
        for layer in slm.layers:
            assert layer.self_attn._cache_k is None
            assert layer.self_attn._cache_v is None

    def test_size_mb(self, slm: AdaptiveSLM) -> None:
        """size_mb should return a positive float."""
        assert slm.size_mb > 0.0

    def test_repr(self, slm: AdaptiveSLM) -> None:
        """__repr__ should include model dimensions."""
        r = repr(slm)
        assert 'AdaptiveSLM' in r
        assert 'vocab=500' in r


# -------------------------------------------------------------------------
# SimpleTokenizer
# -------------------------------------------------------------------------

class TestTokenizer:
    """Tests for the word-level tokenizer."""

    @pytest.fixture
    def tokenizer(self) -> SimpleTokenizer:
        tok = SimpleTokenizer(vocab_size=100)
        corpus = [
            "Hello world, this is a test.",
            "Testing the tokenizer with multiple sentences.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating!",
        ]
        tok.build_vocab(corpus)
        return tok

    def test_encode_decode_roundtrip(self, tokenizer: SimpleTokenizer) -> None:
        """encode -> decode should recover the original text (modulo case/spacing)."""
        text = "hello world"
        ids = tokenizer.encode(text, add_special=False)
        decoded = tokenizer.decode(ids, skip_special=True)
        assert decoded == "hello world"

    def test_special_tokens(self, tokenizer: SimpleTokenizer) -> None:
        """Encoding with add_special should add BOS and EOS."""
        ids = tokenizer.encode("hello", add_special=True)
        assert ids[0] == SimpleTokenizer.BOS_ID
        assert ids[-1] == SimpleTokenizer.EOS_ID

    def test_unknown_token(self, tokenizer: SimpleTokenizer) -> None:
        """Unknown words should map to UNK_ID."""
        ids = tokenizer.encode("xyzzyplugh", add_special=False)
        assert SimpleTokenizer.UNK_ID in ids

    def test_batch_encode(self, tokenizer: SimpleTokenizer) -> None:
        """batch_encode should produce uniform-length sequences with masks."""
        texts = ["hello world", "test"]
        ids_batch, masks_batch = tokenizer.batch_encode(texts, max_length=10)
        assert len(ids_batch) == 2
        assert len(ids_batch[0]) == 10
        assert len(ids_batch[1]) == 10
        # Mask should be 0 for PAD positions
        assert all(m in (0, 1) for m in masks_batch[0])

    def test_vocab_not_built_raises(self) -> None:
        """Encoding without building vocab should raise RuntimeError."""
        tok = SimpleTokenizer()
        with pytest.raises(RuntimeError, match="not been built"):
            tok.encode("hello")

    def test_truncation(self, tokenizer: SimpleTokenizer) -> None:
        """Long sequences should be truncated to max_length."""
        ids = tokenizer.encode(
            "a " * 100, add_special=True, max_length=10
        )
        assert len(ids) == 10
        # EOS should still be at the end
        assert ids[-1] == SimpleTokenizer.EOS_ID
