"""Tests for Batch G3 sparse-autoencoder components.

Covers:
    * :class:`SparseAutoencoder` forward-pass shapes.
    * L1 sparsity loss non-negativity.
    * Tied-weight invariant.
    * Decoder unit-norm constraint.
    * :class:`FeatureDictionary.top_k_activating_inputs` correctness.
    * :func:`identify_monosemantic_features` threshold filtering.
    * :class:`SAETrainer` convergence on a toy Gaussian task.
    * Determinism under a fixed seed.
    * Post-training sparsity threshold.
    * Input-validation ``ValueError`` surfaces.
"""

from __future__ import annotations

import pytest
import torch

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.interpretability.activation_cache import ActivationCache
from i3.interpretability.sae_analysis import (
    compute_per_feature_semantics,
    feature_steering_vector,
    identify_monosemantic_features,
)
from i3.interpretability.sparse_autoencoder import (
    FeatureDictionary,
    SAETrainer,
    SparseAutoencoder,
)


D_MODEL: int = 16
D_DICT: int = 64
N_SAMPLES: int = 512


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def sae() -> SparseAutoencoder:
    """Return a fresh SAE for a single test."""
    torch.manual_seed(0)
    return SparseAutoencoder(d_model=D_MODEL, d_dict=D_DICT, sparsity_coef=1e-3)


@pytest.fixture
def gaussian_activations() -> torch.Tensor:
    """Return a deterministic ``[N, d_model]`` random matrix."""
    torch.manual_seed(123)
    return torch.randn(N_SAMPLES, D_MODEL)


# ---------------------------------------------------------------------------
# SparseAutoencoder.
# ---------------------------------------------------------------------------


class TestSparseAutoencoderShapes:
    """Forward-pass shape invariants."""

    def test_forward_shapes(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(8, D_MODEL)
        recon, features, l1 = sae(x)
        assert recon.shape == (8, D_MODEL)
        assert features.shape == (8, D_DICT)
        assert l1.dim() == 0

    def test_l1_non_negative(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(4, D_MODEL)
        _, _, l1 = sae(x)
        assert float(l1.item()) >= 0.0

    def test_features_non_negative(self, sae: SparseAutoencoder) -> None:
        # ReLU makes features non-negative by construction.
        x = torch.randn(4, D_MODEL)
        _, features, _ = sae(x)
        assert features.min().item() >= 0.0


# ---------------------------------------------------------------------------
# Tied weights.
# ---------------------------------------------------------------------------


class TestTiedWeights:
    """Tied-weight SAE keeps decoder == encoder.T."""

    def test_tied_weights_invariant(self) -> None:
        torch.manual_seed(1)
        tied = SparseAutoencoder(
            d_model=D_MODEL, d_dict=D_DICT, tied_weights=True
        )
        # Force a forward pass to exercise the tie re-application.
        x = torch.randn(2, D_MODEL)
        _ = tied(x)
        assert torch.allclose(
            tied.decoder.weight.data, tied.encoder.weight.data.t(), atol=1e-6
        )


# ---------------------------------------------------------------------------
# Decoder unit-norm projection.
# ---------------------------------------------------------------------------


class TestDecoderUnitNorm:
    """Decoder columns are unit norm after every training step."""

    def test_columns_unit_norm_initially(self, sae: SparseAutoencoder) -> None:
        norms = sae.decoder.weight.data.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_columns_unit_norm_after_training_step(
        self, sae: SparseAutoencoder, gaussian_activations: torch.Tensor
    ) -> None:
        optimiser = torch.optim.Adam(sae.parameters(), lr=1e-3)
        recon, _, l1 = sae(gaussian_activations[:32])
        loss = sae.reconstruction_loss(gaussian_activations[:32], recon) + l1
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        sae.project_decoder_unit_norm()
        norms = sae.decoder.weight.data.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# FeatureDictionary.
# ---------------------------------------------------------------------------


class TestFeatureDictionary:
    """Top-k queries and similarity metrics."""

    def test_top_k_returns_k_entries(
        self,
        sae: SparseAutoencoder,
        gaussian_activations: torch.Tensor,
    ) -> None:
        fd = FeatureDictionary(sae=sae, activations=gaussian_activations)
        result = fd.top_k_activating_inputs(feature_idx=0, k=5)
        assert len(result) == 5
        # Entries are sorted descending.
        values = [v for _, v in result]
        assert values == sorted(values, reverse=True)

    def test_top_k_rejects_invalid(
        self,
        sae: SparseAutoencoder,
        gaussian_activations: torch.Tensor,
    ) -> None:
        fd = FeatureDictionary(sae=sae, activations=gaussian_activations)
        with pytest.raises(ValueError):
            fd.top_k_activating_inputs(feature_idx=-1, k=3)
        with pytest.raises(ValueError):
            fd.top_k_activating_inputs(feature_idx=0, k=0)

    def test_cosine_similarity_range(
        self,
        sae: SparseAutoencoder,
        gaussian_activations: torch.Tensor,
    ) -> None:
        fd = FeatureDictionary(sae=sae, activations=gaussian_activations)
        c = fd.cosine_similarity_between_features(0, 1)
        assert -1.0 <= c <= 1.0
        # Cosine with self is 1 (within numerical error).
        assert fd.cosine_similarity_between_features(0, 0) == pytest.approx(
            1.0, abs=1e-5
        )


# ---------------------------------------------------------------------------
# identify_monosemantic_features.
# ---------------------------------------------------------------------------


class TestMonosemanticFiltering:
    """Threshold filter correctness."""

    def test_threshold_filters(self) -> None:
        from i3.interpretability.sae_analysis import FeatureSemantics

        a = FeatureSemantics(
            feature_idx=0,
            top_dimension_correlations=[("cognitive_load", 0.9)],
            mean_activation=0.1,
            max_activation=1.0,
            sparsity=0.5,
        )
        b = FeatureSemantics(
            feature_idx=1,
            top_dimension_correlations=[("formality", 0.3)],
            mean_activation=0.2,
            max_activation=0.7,
            sparsity=0.4,
        )
        filtered = identify_monosemantic_features([a, b], threshold=0.7)
        assert [s.feature_idx for s in filtered] == [0]

    def test_threshold_range_validation(self) -> None:
        with pytest.raises(ValueError):
            identify_monosemantic_features([], threshold=1.5)


# ---------------------------------------------------------------------------
# SAETrainer convergence.
# ---------------------------------------------------------------------------


class TestSAETrainerConvergence:
    """Toy Gaussian task: loss strictly decreases."""

    def test_converges_on_gaussian(
        self, gaussian_activations: torch.Tensor
    ) -> None:
        trainer = SAETrainer(seed=0)
        _sae, report = trainer.fit(
            gaussian_activations,
            d_dict=D_DICT,
            sparsity_coef=1e-4,
            epochs=20,
            batch_size=64,
            lr=1e-2,
        )
        # Trained loss must be strictly lower than the initial loss.
        assert report.final_loss < report.initial_loss

    def test_post_training_sparsity(
        self, gaussian_activations: torch.Tensor
    ) -> None:
        trainer = SAETrainer(seed=0)
        _sae, report = trainer.fit(
            gaussian_activations,
            d_dict=D_DICT,
            sparsity_coef=1e-2,  # heavier penalty -> more zeros
            epochs=25,
            batch_size=64,
            lr=1e-2,
        )
        assert report.final_mean_sparsity > 0.5


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Identical seed + data -> identical result."""

    def test_seeded_training_is_reproducible(
        self, gaussian_activations: torch.Tensor
    ) -> None:
        trainer_a = SAETrainer(seed=42)
        sae_a, _ = trainer_a.fit(
            gaussian_activations,
            d_dict=D_DICT,
            sparsity_coef=1e-3,
            epochs=5,
            batch_size=64,
            lr=1e-3,
        )
        trainer_b = SAETrainer(seed=42)
        sae_b, _ = trainer_b.fit(
            gaussian_activations,
            d_dict=D_DICT,
            sparsity_coef=1e-3,
            epochs=5,
            batch_size=64,
            lr=1e-3,
        )
        assert torch.allclose(
            sae_a.decoder.weight.data,
            sae_b.decoder.weight.data,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Constructor and forward-path ``ValueError`` coverage."""

    def test_bad_ctor_args(self) -> None:
        with pytest.raises(ValueError):
            SparseAutoencoder(d_model=0, d_dict=10)
        with pytest.raises(ValueError):
            SparseAutoencoder(d_model=10, d_dict=0)
        with pytest.raises(ValueError):
            SparseAutoencoder(d_model=10, d_dict=10, sparsity_coef=-0.1)

    def test_forward_shape_mismatch(self, sae: SparseAutoencoder) -> None:
        bad = torch.randn(3, D_MODEL + 1)
        with pytest.raises(ValueError):
            sae(bad)

    def test_trainer_bad_args(self, gaussian_activations: torch.Tensor) -> None:
        trainer = SAETrainer()
        with pytest.raises(ValueError):
            trainer.fit(
                gaussian_activations.reshape(-1),  # not 2-D
                d_dict=D_DICT,
            )
        with pytest.raises(ValueError):
            trainer.fit(
                gaussian_activations,
                d_dict=D_DICT,
                epochs=0,
            )
        with pytest.raises(ValueError):
            trainer.fit(
                gaussian_activations,
                d_dict=D_DICT,
                lr=0.0,
            )


# ---------------------------------------------------------------------------
# Analysis integration.
# ---------------------------------------------------------------------------


class TestSemanticCorrelation:
    """End-to-end: SAE + cache + per-feature semantics."""

    def test_semantics_shape(
        self,
        sae: SparseAutoencoder,
        gaussian_activations: torch.Tensor,
    ) -> None:
        # Build eight synthetic adaptation vectors, one per row-block.
        n = gaussian_activations.size(0)
        vectors = []
        for i in range(n):
            cl = float((i % 8) / 7.0)
            vectors.append(
                AdaptationVector(
                    cognitive_load=cl,
                    style_mirror=StyleVector.default(),
                    emotional_tone=0.5,
                    accessibility=0.0,
                )
            )
        semantics = compute_per_feature_semantics(
            sae, gaussian_activations, vectors
        )
        assert len(semantics) == D_DICT
        for sem in semantics:
            # Top correlations are a 3-tuple of (name, value) pairs.
            assert len(sem.top_dimension_correlations) == 3
            for name, r in sem.top_dimension_correlations:
                assert -1.0 <= r <= 1.0
                assert isinstance(name, str)


# ---------------------------------------------------------------------------
# Steering vector extraction.
# ---------------------------------------------------------------------------


class TestSteeringVector:
    """feature_steering_vector returns a correctly-shaped decoder column."""

    def test_returns_decoder_column(self, sae: SparseAutoencoder) -> None:
        v = feature_steering_vector(sae, feature_idx=3)
        assert v.shape == (D_MODEL,)
        assert torch.allclose(
            v, sae.decoder.weight.detach()[:, 3], atol=1e-6
        )

    def test_invalid_index(self, sae: SparseAutoencoder) -> None:
        with pytest.raises(ValueError):
            feature_steering_vector(sae, feature_idx=D_DICT + 10)


# ---------------------------------------------------------------------------
# ActivationCache smoke test.
# ---------------------------------------------------------------------------


class TestActivationCache:
    """End-to-end: register, collect, save, load."""

    def test_register_and_get(self) -> None:
        # Drive a tiny Linear via a wrapper so cache.collect(model, iter)
        # reaches the registered inner module's forward hook.
        class Wrapper(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)

        wrapper = Wrapper(torch.nn.Linear(D_MODEL, D_MODEL))

        def data_iter():
            for _ in range(4):
                yield {"x": torch.randn(2, D_MODEL)}

        cache = ActivationCache(max_samples=32)
        cache.register(wrapper.inner, "inner")
        counts = cache.collect(wrapper, data_iter())
        assert counts["inner"] > 0
        tensor = cache.get("inner")
        assert tensor.ndim == 2
        assert tensor.shape[-1] == D_MODEL

    def test_get_missing_raises(self) -> None:
        cache = ActivationCache()
        with pytest.raises(KeyError):
            cache.get("nope")
