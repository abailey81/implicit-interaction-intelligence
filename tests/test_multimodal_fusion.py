"""Tests for :mod:`i3.multimodal.fusion_real` (Batch F-1)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch

from i3.multimodal.fusion_real import MultimodalFusion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def modality_dims() -> dict[str, int]:
    """Canonical four-modality 8-dim layout."""
    return {"keystroke": 8, "voice": 8, "vision": 8, "accelerometer": 8}


@pytest.fixture
def random_vec() -> np.ndarray:
    """Deterministic 8-dim vector for repeatable assertions."""
    rng = np.random.default_rng(seed=7)
    return rng.standard_normal(8).astype(np.float32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Fusion module constructor contracts."""

    def test_default_uses_four_modalities(self) -> None:
        fusion = MultimodalFusion()
        assert len(fusion._modality_order) == 4
        assert fusion.embedding_dim == 64

    def test_rejects_unknown_strategy(self, modality_dims: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="fusion_strategy"):
            MultimodalFusion(
                modality_dim_map=modality_dims,
                fusion_strategy="made_up",  # type: ignore[arg-type]
            )

    def test_rejects_unknown_policy(self, modality_dims: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="missing_modality_policy"):
            MultimodalFusion(
                modality_dim_map=modality_dims,
                missing_modality_policy="panic",  # type: ignore[arg-type]
            )

    def test_rejects_empty_modality_map(self) -> None:
        with pytest.raises(ValueError, match="at least one modality"):
            MultimodalFusion(modality_dim_map={})


# ---------------------------------------------------------------------------
# late_concat
# ---------------------------------------------------------------------------

class TestLateConcat:
    """Contract tests for ``fusion_strategy="late_concat"``."""

    def test_output_shape_64(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims, fusion_strategy="late_concat"
        )
        fusion.eval()
        emb = asyncio.run(
            fusion.fuse(
                keystroke_features=random_vec,
                voice_features=random_vec,
                vision_features=random_vec,
                accelerometer_features=random_vec,
            )
        )
        assert emb.shape == (64,)
        assert emb.dtype == torch.float32

    def test_missing_modality_zero_fill(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims,
            fusion_strategy="late_concat",
            missing_modality_policy="zero_fill",
        )
        fusion.eval()
        # Only keystroke supplied.
        emb = asyncio.run(fusion.fuse(keystroke_features=random_vec))
        assert emb.shape == (64,)

    def test_missing_modality_mask_drop_rescales(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims,
            fusion_strategy="late_concat",
            missing_modality_policy="mask_drop",
        )
        fusion.eval()
        emb = asyncio.run(fusion.fuse(keystroke_features=random_vec))
        # Rescaling should change the final embedding compared to zero_fill.
        fusion_zf = MultimodalFusion(
            modality_dim_map=modality_dims,
            fusion_strategy="late_concat",
            missing_modality_policy="zero_fill",
        )
        fusion_zf.load_state_dict(fusion.state_dict())
        fusion_zf.eval()
        emb_zf = asyncio.run(fusion_zf.fuse(keystroke_features=random_vec))
        assert not torch.allclose(emb, emb_zf)


# ---------------------------------------------------------------------------
# late_gated
# ---------------------------------------------------------------------------

class TestLateGated:
    """Contract tests for ``fusion_strategy="late_gated"``."""

    def test_output_shape_64(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims, fusion_strategy="late_gated"
        )
        fusion.eval()
        emb = asyncio.run(
            fusion.fuse(
                keystroke_features=random_vec,
                voice_features=random_vec,
                vision_features=random_vec,
                accelerometer_features=random_vec,
            )
        )
        assert emb.shape == (64,)

    def test_gate_forcing_single_modality_matches_override(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        """If we force all gates to 1.0 for keystroke only, the gated output
        equals what we'd get from a manual single-modality concat projection.
        """
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims,
            fusion_strategy="late_gated",
            missing_modality_policy="mask_drop",
        )
        fusion.eval()

        # Patch the gate head so gates are exactly [1, 0, 0, 0].
        import torch.nn as nn

        class _FixedGate(nn.Module):
            def forward(self, _x: torch.Tensor) -> torch.Tensor:
                return torch.tensor([1.0, 0.0, 0.0, 0.0])

        fusion.gate_head = _FixedGate()

        emb = asyncio.run(fusion.fuse(keystroke_features=random_vec))
        # Build the expected projection by zero-padding only the keystroke slot.
        with torch.no_grad():
            padded = torch.zeros(32, dtype=torch.float32)
            padded[:8] = torch.from_numpy(random_vec)
            expected_proj = fusion.concat_proj(padded)
            expected = torch.nn.functional.normalize(
                fusion.standalone_head(expected_proj), p=2, dim=-1
            )
        torch.testing.assert_close(emb, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------

class TestAttention:
    """Contract tests for ``fusion_strategy="attention"``."""

    def test_runs_on_random_inputs(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims, fusion_strategy="attention"
        )
        fusion.eval()
        emb = asyncio.run(
            fusion.fuse(
                keystroke_features=random_vec,
                voice_features=random_vec,
                vision_features=random_vec,
                accelerometer_features=random_vec,
            )
        )
        assert emb.shape == (64,)

    def test_attention_tolerates_missing_modalities(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(
            modality_dim_map=modality_dims,
            fusion_strategy="attention",
            missing_modality_policy="mask_drop",
        )
        fusion.eval()
        emb = asyncio.run(
            fusion.fuse(keystroke_features=random_vec, voice_features=random_vec)
        )
        assert emb.shape == (64,)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Fusion must reject ill-shaped inputs with clear errors."""

    def test_requires_at_least_one_modality(
        self, modality_dims: dict[str, int]
    ) -> None:
        fusion = MultimodalFusion(modality_dim_map=modality_dims)
        with pytest.raises(ValueError, match="keystroke_features is required"):
            asyncio.run(fusion.fuse(keystroke_features=None))  # type: ignore[arg-type]

    def test_wrong_shape_raises(self, modality_dims: dict[str, int]) -> None:
        fusion = MultimodalFusion(modality_dim_map=modality_dims)
        with pytest.raises(ValueError, match="shape"):
            asyncio.run(
                fusion.fuse(
                    keystroke_features=np.zeros(9, dtype=np.float32),
                )
            )

    def test_accepts_torch_tensor_input(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(modality_dim_map=modality_dims)
        fusion.eval()
        emb = asyncio.run(
            fusion.fuse(keystroke_features=torch.from_numpy(random_vec))
        )
        assert emb.shape == (64,)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Repeated calls with identical inputs must be deterministic in eval mode."""

    def test_same_input_same_output(
        self, modality_dims: dict[str, int], random_vec: np.ndarray
    ) -> None:
        fusion = MultimodalFusion(modality_dim_map=modality_dims)
        fusion.eval()
        a = asyncio.run(fusion.fuse(keystroke_features=random_vec))
        b = asyncio.run(fusion.fuse(keystroke_features=random_vec))
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
