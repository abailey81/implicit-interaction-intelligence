"""Unit tests for the interpretability package.

Covers:
    * :class:`FeatureAttributor` -- completeness axiom on a linear
      surrogate (sum of attributions across features equals the model's
      output for that dimension, within numerical tolerance).
    * :class:`CrossAttentionExtractor` -- hook attach / detach counts
      match the number of cross-attention layers; captured attention
      tensors have the expected shape.
    * :class:`TokenHeatmap` -- produces JSON-serialisable output with
      the expected nested shape.
"""

from __future__ import annotations

import json

import pytest
import torch

from i3.interpretability import (
    ADAPTATION_DIMS,
    CrossAttentionExtractor,
    FEATURE_NAMES,
    FeatureAttributor,
    TokenHeatmap,
)
from i3.interpretability.feature_attribution import LinearFeatureAdapter


# -------------------------------------------------------------------------
# FeatureAttributor
# -------------------------------------------------------------------------


class TestFeatureAttributor:
    """Integrated-gradients attributor tests."""

    def test_returns_expected_keys(self) -> None:
        attributor = FeatureAttributor(
            mapping_fn=LinearFeatureAdapter(32, 8),
        )
        x = torch.randn(32)
        result = attributor.attribute(x)
        # Keys match the canonical feature names.
        assert set(result.keys()) == set(FEATURE_NAMES)
        for feat in FEATURE_NAMES:
            assert set(result[feat].keys()) == set(ADAPTATION_DIMS)

    def test_completeness_axiom_linear(self) -> None:
        """For a linear model with zero baseline, sum(attributions) = f(x)."""
        torch.manual_seed(0)
        adapter = LinearFeatureAdapter(32, 8)
        attributor = FeatureAttributor(adapter, n_steps=64)
        x = torch.randn(32)

        with torch.no_grad():
            expected = adapter(x.unsqueeze(0)).squeeze(0)

        result = attributor.attribute(x)
        for j, dim_name in enumerate(ADAPTATION_DIMS):
            total = sum(result[feat][dim_name] for feat in FEATURE_NAMES)
            # Small Riemann-sum discretisation error allowed.
            assert total == pytest.approx(
                float(expected[j].item()), abs=1e-3
            )

    def test_zero_input_gives_zero_attribution(self) -> None:
        """When x == baseline (zeros), IG must produce all-zero attributions."""
        attributor = FeatureAttributor(LinearFeatureAdapter(32, 8))
        x = torch.zeros(32)
        result = attributor.attribute(x)
        for feat in FEATURE_NAMES:
            for dim_name in ADAPTATION_DIMS:
                assert result[feat][dim_name] == pytest.approx(0.0, abs=1e-6)

    def test_wrong_shape_raises(self) -> None:
        attributor = FeatureAttributor(LinearFeatureAdapter(32, 8))
        with pytest.raises(ValueError):
            attributor.attribute(torch.randn(31))
        with pytest.raises(ValueError):
            attributor.attribute(torch.randn(2, 32))

    def test_invalid_n_steps_raises(self) -> None:
        with pytest.raises(ValueError):
            FeatureAttributor(LinearFeatureAdapter(32, 8), n_steps=1)


# -------------------------------------------------------------------------
# CrossAttentionExtractor
# -------------------------------------------------------------------------


class TestCrossAttentionExtractor:
    """Extractor hook lifecycle tests."""

    def _make_model(self) -> "torch.nn.Module":
        """Tiny AdaptiveSLM-compatible model for fast tests."""
        from i3.slm.model import AdaptiveSLM

        return AdaptiveSLM(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_layers=3,
            d_ff=64,
            max_seq_len=16,
            conditioning_dim=64,
            adaptation_dim=8,
            n_cross_heads=2,
            n_conditioning_tokens=4,
            dropout=0.0,
            tie_weights=True,
            padding_idx=0,
        )

    def test_detects_cross_attention_modules(self) -> None:
        model = self._make_model()
        extractor = CrossAttentionExtractor(model)
        assert extractor.n_layers == 3

    def test_hook_attach_detach_count(self) -> None:
        """Entering the context attaches n_layers hooks; exiting removes them."""
        model = self._make_model()
        extractor = CrossAttentionExtractor(model)
        # Enter context -- handles list populated.
        with extractor:
            assert len(extractor._state.handles) == extractor.n_layers
        # Exit -- handles list cleared.
        assert len(extractor._state.handles) == 0

    def test_captures_attention_during_forward(self) -> None:
        """Run a forward pass and ensure each layer recorded an attention map."""
        torch.manual_seed(0)
        model = self._make_model()
        model.eval()
        extractor = CrossAttentionExtractor(model, squeeze_batch=True)

        ids = torch.randint(0, 64, (1, 6))
        adaptation = torch.zeros(1, 8)
        adaptation[0, 0] = 0.5
        user_state = torch.zeros(1, 64)

        with extractor:
            with torch.no_grad():
                model(ids, adaptation, user_state)
            maps = extractor.get_attention_maps()

        assert len(maps) == extractor.n_layers
        for m in maps:
            # [n_cross_heads=2, seq_len=6, n_cond=4]
            assert m.shape == (2, 6, 4)

    def test_hooks_cleaned_up_on_exception(self) -> None:
        """If the forward pass raises, hooks must still be removed on exit."""
        model = self._make_model()
        extractor = CrossAttentionExtractor(model)
        with pytest.raises(RuntimeError):
            with extractor:
                raise RuntimeError("synthetic failure")
        assert len(extractor._state.handles) == 0


# -------------------------------------------------------------------------
# TokenHeatmap
# -------------------------------------------------------------------------


class TestTokenHeatmap:
    """Token heatmap payload tests."""

    def test_builds_json_serialisable_payload(self) -> None:
        heatmap = TokenHeatmap()
        # Two layers, 2 heads, seq_len=3, n_cond=4.
        maps = [torch.rand(2, 3, 4), torch.rand(2, 3, 4)]
        tokens = ["hello", "world", "!"]
        payload = heatmap.build(tokens, maps)
        payload_dict = heatmap.to_dict(payload)
        # Must round-trip through json.
        text = json.dumps(payload_dict)
        assert isinstance(text, str)
        reloaded = json.loads(text)
        assert reloaded["tokens"] == tokens
        # shape is [layers, heads, seq, cond]
        assert reloaded["shape"] == [2, 2, 3, 4]

    def test_handles_mismatched_token_count(self) -> None:
        """Attention seq_len > len(tokens) pads with '<?>' for safety."""
        heatmap = TokenHeatmap()
        maps = [torch.rand(1, 5, 4)]
        payload = heatmap.build(["a", "b"], maps)
        assert len(payload.tokens) == 5
        assert payload.tokens[-1] == "<?>"

    def test_empty_input_is_safe(self) -> None:
        heatmap = TokenHeatmap()
        payload = heatmap.build([], [])
        assert payload.layers == []
        assert payload.shape == (0, 0, 0, 4)
