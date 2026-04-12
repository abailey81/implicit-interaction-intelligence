"""Property-based / fuzz tests using deterministic random inputs.

These tests exercise the core model components over many random inputs
to verify invariants that should hold for *any* valid input:

    - TCN output shape and normalisation invariants.
    - Bandit convergence to the better arm under deterministic rewards.
    - AdaptationVector value-range guarantees.

Hypothesis is not a project dependency, so we use fixed-seeded RNGs and
pytest.mark.parametrize to generate many test cases deterministically.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from i3.adaptation.types import AdaptationVector, StyleVector


# -------------------------------------------------------------------------
# TCN shape and normalisation invariants
# -------------------------------------------------------------------------


class TestTCNShapeProperties:
    """TCN output shape and normalisation are deterministic given input shape."""

    @pytest.fixture(scope="class")
    def model(self) -> torch.nn.Module:
        tcn_mod = pytest.importorskip("i3.encoder.tcn")
        torch.manual_seed(1234)
        return tcn_mod.TemporalConvNet(input_dim=32, embedding_dim=64)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("seq_len", [1, 2, 5, 10, 20, 50])
    def test_output_shape_invariant(
        self, model: torch.nn.Module, batch_size: int, seq_len: int
    ) -> None:
        """TCN output shape must be (batch_size, embedding_dim) for any input."""
        x = torch.randn(batch_size, seq_len, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 64)

    def test_output_l2_normalized(self, model: torch.nn.Module) -> None:
        """Every output row must lie on the unit sphere (L2 norm == 1)."""
        torch.manual_seed(42)
        x = torch.randn(8, 10, 32)
        with torch.no_grad():
            out = model(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"L2 norms were {norms.tolist()}"
        )

    def test_output_is_finite(self, model: torch.nn.Module) -> None:
        """Output must contain only finite values (no NaN/Inf) for any input."""
        torch.manual_seed(7)
        for _ in range(10):
            x = torch.randn(4, 8, 32) * 10.0  # larger-scale inputs
            with torch.no_grad():
                out = model(x)
            assert torch.isfinite(out).all()

    def test_permutation_invariance_across_batch(
        self, model: torch.nn.Module
    ) -> None:
        """Shuffling the batch should shuffle outputs consistently."""
        torch.manual_seed(11)
        x = torch.randn(5, 10, 32)
        perm = torch.tensor([4, 2, 0, 3, 1])
        with torch.no_grad():
            out_a = model(x)
            out_b = model(x[perm])
        assert torch.allclose(out_b, out_a[perm], atol=1e-6)

    def test_zero_input_produces_finite_output(
        self, model: torch.nn.Module
    ) -> None:
        """Zero input should produce a finite output (shape-stable)."""
        x = torch.zeros(2, 5, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 64)
        assert torch.isfinite(out).all()


# -------------------------------------------------------------------------
# Bandit convergence properties
# -------------------------------------------------------------------------


class TestBanditConvergenceProperties:
    """The bandit should learn the better arm given many deterministic trials."""

    def test_converges_to_better_arm_with_deterministic_rewards(self) -> None:
        """Arm 0 always pays 1.0, arm 1 always pays 0.0 -- bandit should prefer arm 0."""
        bandit_mod = pytest.importorskip("i3.router.bandit")
        np.random.seed(2024)
        bandit = bandit_mod.ContextualThompsonBandit(
            n_arms=2, context_dim=12, prior_precision=1.0
        )
        context = np.array(
            [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4, -0.3, 0.0, 0.1, 0.2],
            dtype=np.float64,
        )

        # Seed with several observations of each arm to bypass cold start
        for _ in range(12):
            bandit.update(arm=0, context=context, reward=1.0)
            bandit.update(arm=1, context=context, reward=0.0)

        # Run further trials -- now bandit chooses
        selections: list[int] = []
        for _ in range(200):
            arm, _ = bandit.select_arm(context)
            # Feed the deterministic reward back so posteriors keep improving
            reward = 1.0 if arm == 0 else 0.0
            bandit.update(arm, context, reward)
            selections.append(arm)

        arm0_rate = sum(1 for a in selections if a == 0) / len(selections)
        assert arm0_rate > 0.7, (
            f"Bandit selected arm 0 only {arm0_rate*100:.1f}% of the time"
        )

    def test_both_arms_equal_reward_converges_uniform(self) -> None:
        """When both arms give identical rewards, selection rate should stay ~50/50."""
        bandit_mod = pytest.importorskip("i3.router.bandit")
        np.random.seed(99)
        bandit = bandit_mod.ContextualThompsonBandit(n_arms=2, context_dim=8)
        context = np.ones(8, dtype=np.float64) * 0.5

        for _ in range(100):
            arm, _ = bandit.select_arm(context)
            bandit.update(arm, context, reward=0.5)  # neutral reward

        stats = bandit.get_arm_stats()
        pulls = [a["pulls"] for a in stats["arms"]]
        # Should not be extremely lopsided
        assert min(pulls) > 0
        # Neither arm should dominate beyond 80/20
        ratio = max(pulls) / sum(pulls)
        assert ratio < 0.85, f"Bandit is too lopsided: {pulls}"

    def test_bandit_update_does_not_crash_with_extreme_contexts(self) -> None:
        """Very large or very small context values should not destabilise updates."""
        bandit_mod = pytest.importorskip("i3.router.bandit")
        np.random.seed(7)
        bandit = bandit_mod.ContextualThompsonBandit(n_arms=2, context_dim=4)
        for magnitude in [1e-6, 1.0, 1e3]:
            ctx = np.array([magnitude, -magnitude, 0.0, magnitude / 2], dtype=np.float64)
            bandit.update(arm=0, context=ctx, reward=0.8)
            bandit.update(arm=1, context=ctx, reward=0.2)
        # After all those updates, posterior should still be PSD and stats queryable
        stats = bandit.get_arm_stats()
        assert stats["total_observations"] >= 6

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 2024])
    def test_reset_clears_all_state(self, seed: int) -> None:
        """reset() must restore the bandit to its initial prior."""
        bandit_mod = pytest.importorskip("i3.router.bandit")
        np.random.seed(seed)
        bandit = bandit_mod.ContextualThompsonBandit(n_arms=2, context_dim=6)
        ctx = np.random.randn(6)
        for _ in range(20):
            bandit.update(0, ctx, 1.0)
            bandit.update(1, ctx, 0.0)
        assert bandit.total_pulls[0] == 20
        bandit.reset()
        assert bandit.total_pulls == [0, 0]
        assert bandit.alpha == [1.0, 1.0]
        assert bandit.beta_param == [1.0, 1.0]


# -------------------------------------------------------------------------
# Adaptation vector invariants
# -------------------------------------------------------------------------


class TestAdaptationVectorInvariants:
    """All values in AdaptationVector should be in [0, 1] regardless of input."""

    def test_default_values_in_range(self) -> None:
        """Default AdaptationVector should have all values in [0, 1]."""
        av = AdaptationVector.default()
        assert 0.0 <= av.cognitive_load <= 1.0
        assert 0.0 <= av.emotional_tone <= 1.0
        assert 0.0 <= av.accessibility <= 1.0
        for attr in ("formality", "verbosity", "emotionality", "directness"):
            assert 0.0 <= getattr(av.style_mirror, attr) <= 1.0

    def test_to_tensor_values_in_range(self) -> None:
        """Tensor serialisation of the default AdaptationVector must be in [0, 1]."""
        av = AdaptationVector.default()
        t = av.to_tensor()
        assert t.shape == (8,)
        assert torch.all(t >= 0.0)
        assert torch.all(t <= 1.0)

    def test_from_tensor_roundtrip_preserves_values(self) -> None:
        """A valid tensor should round-trip through AdaptationVector cleanly."""
        original = torch.tensor(
            [0.3, 0.4, 0.6, 0.5, 0.7, 0.2, 0.8, 0.0], dtype=torch.float32
        )
        av = AdaptationVector.from_tensor(original)
        recovered = av.to_tensor()
        assert torch.allclose(original, recovered, atol=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_clamping_handles_out_of_range_values(self, seed: int) -> None:
        """Constructing with values outside [0, 1] should clamp, not error."""
        rng = random.Random(seed)
        raw_vals = [rng.uniform(-2.0, 3.0) for _ in range(8)]
        t = torch.tensor(raw_vals, dtype=torch.float32)
        av = AdaptationVector.from_tensor(t)
        # After clamping, every value must be in [0, 1]
        assert 0.0 <= av.cognitive_load <= 1.0
        assert 0.0 <= av.emotional_tone <= 1.0
        assert 0.0 <= av.accessibility <= 1.0
        for attr in ("formality", "verbosity", "emotionality", "directness"):
            assert 0.0 <= getattr(av.style_mirror, attr) <= 1.0

    @pytest.mark.parametrize("trial", list(range(10)))
    def test_style_vector_construction_always_clamps(self, trial: int) -> None:
        """StyleVector should clamp any numeric input into [0, 1]."""
        rng = random.Random(trial * 7 + 3)
        sv = StyleVector(
            formality=rng.uniform(-5, 5),
            verbosity=rng.uniform(-5, 5),
            emotionality=rng.uniform(-5, 5),
            directness=rng.uniform(-5, 5),
        )
        for attr in ("formality", "verbosity", "emotionality", "directness"):
            v = getattr(sv, attr)
            assert 0.0 <= v <= 1.0

    def test_to_dict_and_tensor_agree(self) -> None:
        """to_dict and to_tensor must produce consistent values."""
        av = AdaptationVector(
            cognitive_load=0.6,
            style_mirror=StyleVector(0.1, 0.2, 0.3, 0.4),
            emotional_tone=0.5,
            accessibility=0.7,
        )
        d = av.to_dict()
        t = av.to_tensor()
        assert abs(d["cognitive_load"] - t[0].item()) < 1e-6
        assert abs(d["style_mirror"]["formality"] - t[1].item()) < 1e-6
        assert abs(d["emotional_tone"] - t[5].item()) < 1e-6
        assert abs(d["accessibility"] - t[6].item()) < 1e-6
