"""Unit tests for auxiliary conditioning losses.

Covers:
    * Shape / scalar-ness of each loss.
    * Gradient flow -- a non-zero loss produces a non-zero gradient on
      the logits / IDs that fed it.
    * Identity behaviour -- ``c1 == c2`` implies ``ConditioningConsistencyLoss``
      is trivially minimised (zero KL -> loss = 0).
    * Clamping -- consistency loss saturates at the margin and never
      goes below ``-margin``.
"""

from __future__ import annotations

import pytest
import torch

from i3.slm.aux_losses import (
    AdaptationConditioningLoss,
    AdaptationLossOutput,
    ConditioningConsistencyLoss,
    StyleFidelityLoss,
)


# -------------------------------------------------------------------------
# ConditioningConsistencyLoss
# -------------------------------------------------------------------------


class TestConditioningConsistencyLoss:
    """Consistency-loss tests."""

    def test_output_is_scalar(self) -> None:
        loss = ConditioningConsistencyLoss(margin=2.0)
        logits_a = torch.randn(2, 4, 100)
        logits_b = torch.randn(2, 4, 100)
        out = loss(logits_a, logits_b)
        assert out.dim() == 0
        assert torch.is_tensor(out)

    def test_identity_gives_zero(self) -> None:
        """Identical logits => KL = 0 => loss = 0 (trivially minimised)."""
        loss = ConditioningConsistencyLoss(margin=2.0)
        logits = torch.randn(2, 4, 100)
        out = loss(logits, logits.clone())
        # -min(0, margin) = 0.
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_clamping_at_margin(self) -> None:
        """Very divergent logits must saturate at ``-margin``."""
        loss = ConditioningConsistencyLoss(margin=1.5)
        # Build two distributions concentrated on disjoint tokens.
        vocab = 10
        logits_a = torch.full((1, 1, vocab), -1e2)
        logits_a[..., 0] = 1e2  # distribution a: all mass on token 0
        logits_b = torch.full((1, 1, vocab), -1e2)
        logits_b[..., 5] = 1e2  # distribution b: all mass on token 5
        out = loss(logits_a, logits_b)
        # KL is huge; after clamp, loss should equal -margin exactly.
        assert torch.isclose(
            out, torch.tensor(-1.5), atol=1e-5
        )

    def test_gradient_flows(self) -> None:
        """A non-saturated loss produces a gradient on its inputs."""
        loss = ConditioningConsistencyLoss(margin=10.0)
        logits_a = torch.randn(1, 2, 8, requires_grad=True)
        logits_b = torch.randn(1, 2, 8, requires_grad=True)
        out = loss(logits_a, logits_b)
        out.backward()
        assert logits_a.grad is not None
        assert logits_b.grad is not None
        # At least one element has a non-zero gradient.
        assert logits_a.grad.abs().sum().item() > 0.0

    def test_malformed_inputs_are_safe_no_op(self) -> None:
        """Shape mismatch / non-tensor input must return a zero tensor."""
        loss = ConditioningConsistencyLoss(margin=2.0)
        # Non-tensor inputs.
        out = loss(None, None)  # type: ignore[arg-type]
        assert float(out) == 0.0
        # Shape mismatch.
        out2 = loss(torch.randn(1, 2, 5), torch.randn(1, 2, 6))
        assert float(out2) == 0.0
        # Empty tensor.
        out3 = loss(torch.empty(0), torch.empty(0))
        assert float(out3) == 0.0

    def test_invalid_margin_raises(self) -> None:
        with pytest.raises(ValueError):
            ConditioningConsistencyLoss(margin=0.0)
        with pytest.raises(ValueError):
            ConditioningConsistencyLoss(margin=-1.0)


# -------------------------------------------------------------------------
# StyleFidelityLoss
# -------------------------------------------------------------------------


class TestStyleFidelityLoss:
    """Style-fidelity tests."""

    def test_output_is_scalar_and_nonneg(self) -> None:
        loss = StyleFidelityLoss(vocab_size=100, max_seq_len=16)
        ids = torch.randint(0, 100, (2, 16))
        target = {"formality": 0.5, "verbosity": 0.5, "sentiment": 0.0}
        out = loss(ids, target)
        assert out.dim() == 0
        assert float(out) >= 0.0

    def test_accepts_dict_and_tensor(self) -> None:
        loss = StyleFidelityLoss(vocab_size=100, max_seq_len=16)
        ids = torch.randint(0, 100, (2, 16))
        out_dict = loss(
            ids, {"formality": 0.5, "verbosity": 0.5, "sentiment": 0.0}
        )
        out_tensor = loss(ids, torch.tensor([0.5, 0.5, 0.5, 0.5]))
        assert torch.is_tensor(out_dict) and torch.is_tensor(out_tensor)

    def test_adaptation_vector_object(self) -> None:
        from i3.adaptation.types import AdaptationVector

        loss = StyleFidelityLoss(vocab_size=64, max_seq_len=8)
        ids = torch.randint(0, 64, (1, 8))
        out = loss(ids, AdaptationVector.default())
        assert torch.is_tensor(out)
        assert out.dim() == 0

    def test_style_vector_object(self) -> None:
        from i3.adaptation.types import StyleVector

        loss = StyleFidelityLoss(vocab_size=64, max_seq_len=8)
        ids = torch.randint(0, 64, (1, 8))
        out = loss(ids, StyleVector.default())
        assert torch.is_tensor(out)

    def test_malformed_target_is_safe_no_op(self) -> None:
        loss = StyleFidelityLoss(vocab_size=16, max_seq_len=4)
        ids = torch.randint(0, 16, (1, 4))
        out = loss(ids, "not a style")  # type: ignore[arg-type]
        assert float(out) == 0.0


# -------------------------------------------------------------------------
# AdaptationConditioningLoss wrapper
# -------------------------------------------------------------------------


class TestAdaptationConditioningLoss:
    """Wrapper loss tests."""

    def test_default_weights(self) -> None:
        loss = AdaptationConditioningLoss()
        assert loss.alpha_consistency == pytest.approx(0.1)
        assert loss.alpha_fidelity == pytest.approx(0.05)

    def test_default_safe_no_op(self) -> None:
        """With no inputs, every component is zero and total is zero."""
        loss = AdaptationConditioningLoss()
        out = loss()
        assert isinstance(out, AdaptationLossOutput)
        assert float(out.total) == 0.0
        assert float(out.consistency) == 0.0
        assert float(out.fidelity) == 0.0

    def test_both_components_contribute(self) -> None:
        loss = AdaptationConditioningLoss(
            alpha_consistency=0.5,
            alpha_fidelity=0.5,
            margin=10.0,
            vocab_size=64,
            max_seq_len=8,
        )
        logits_c1 = torch.randn(1, 2, 16, requires_grad=True)
        logits_c2 = torch.randn(1, 2, 16, requires_grad=True)
        ids = torch.randint(0, 64, (1, 8))
        out = loss(
            logits_c1=logits_c1,
            logits_c2=logits_c2,
            generated_ids=ids,
            target_style={"formality": 0.5, "verbosity": 0.5, "sentiment": 0.0},
        )
        assert isinstance(out, AdaptationLossOutput)
        # Gradient flows through consistency component.
        out.total.backward()
        assert logits_c1.grad is not None

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            AdaptationConditioningLoss(alpha_consistency=-1.0)
        with pytest.raises(ValueError):
            AdaptationConditioningLoss(alpha_fidelity=-0.01)

    def test_partial_inputs(self) -> None:
        """Supplying only the consistency inputs must still produce a sensible total."""
        loss = AdaptationConditioningLoss(
            alpha_consistency=1.0, alpha_fidelity=1.0, margin=2.0
        )
        logits_c1 = torch.randn(1, 2, 16)
        logits_c2 = torch.randn(1, 2, 16)
        out = loss(logits_c1=logits_c1, logits_c2=logits_c2)
        assert float(out.fidelity) == 0.0
        # total == consistency contribution since fidelity is zero.
        assert torch.isclose(out.total, out.consistency, atol=1e-6)
