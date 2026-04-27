"""Tests for :mod:`i3.multimodal.prosody` (voice-prosody flagship #2).

The module is pure-Python + light torch; all tests are CPU-only and
deterministic.
"""

from __future__ import annotations

import math

import pytest
import torch

from i3.multimodal.prosody import (
    PROSODY_FEATURE_KEYS,
    MultimodalFusion,
    ProsodyEncoder,
    ProsodyFeatures,
    prosody_payload_to_tensor,
    validate_prosody_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_payload() -> dict:
    return {
        "speech_rate_wpm_norm": 0.55,
        "pitch_mean_norm": 0.40,
        "pitch_variance_norm": 0.30,
        "energy_mean_norm": 0.65,
        "energy_variance_norm": 0.20,
        "voiced_ratio": 0.78,
        "pause_density": 0.22,
        "spectral_centroid_norm": 0.48,
        "samples_count": 30,
        "captured_seconds": 3.0,
    }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TestValidator:
    def test_good_payload_validates(self) -> None:
        feats = validate_prosody_payload(_good_payload())
        assert isinstance(feats, ProsodyFeatures)
        assert feats.samples_count == 30
        assert feats.captured_seconds == pytest.approx(3.0)

    def test_missing_key_rejected(self) -> None:
        bad = _good_payload()
        del bad["pitch_variance_norm"]
        assert validate_prosody_payload(bad) is None

    @pytest.mark.parametrize("bad_value", [
        float("nan"), float("inf"), float("-inf"), "not a number",
    ])
    def test_invalid_value_rejected(self, bad_value: object) -> None:
        bad = _good_payload()
        bad["pitch_mean_norm"] = bad_value
        assert validate_prosody_payload(bad) is None

    def test_clamp_into_unit_interval(self) -> None:
        bad = _good_payload()
        bad["pitch_mean_norm"] = 5.5     # way out of range
        bad["voiced_ratio"] = -0.4       # negative
        feats = validate_prosody_payload(bad)
        assert feats is not None
        assert feats.pitch_mean_norm == 1.0
        assert feats.voiced_ratio == 0.0

    def test_non_dict_rejected(self) -> None:
        assert validate_prosody_payload("hello") is None
        assert validate_prosody_payload(None) is None
        assert validate_prosody_payload([1, 2, 3]) is None

    def test_optional_fields_default(self) -> None:
        payload = {k: 0.5 for k in PROSODY_FEATURE_KEYS}
        feats = validate_prosody_payload(payload)
        assert feats is not None
        assert feats.samples_count == 0
        assert feats.captured_seconds == 0.0

    def test_excessive_samples_count_clamped(self) -> None:
        bad = _good_payload()
        bad["samples_count"] = 9_999_999  # absurd
        feats = validate_prosody_payload(bad)
        assert feats is not None
        assert feats.samples_count == 100_000


# ---------------------------------------------------------------------------
# Encoder + Fusion
# ---------------------------------------------------------------------------

class TestProsodyEncoder:
    def test_output_shape_unbatched(self) -> None:
        enc = ProsodyEncoder()
        x = torch.zeros(8)
        y = enc(x)
        assert tuple(y.shape) == (32,)

    def test_output_shape_batched(self) -> None:
        enc = ProsodyEncoder()
        x = torch.zeros(4, 8)
        y = enc(x)
        assert tuple(y.shape) == (4, 32)

    def test_finite_outputs(self) -> None:
        enc = ProsodyEncoder()
        x = torch.rand(8)
        y = enc(x)
        assert torch.isfinite(y).all()


class TestMultimodalFusion:
    def test_with_prosody_shape(self) -> None:
        fusion = MultimodalFusion()
        key = torch.randn(64)
        prosody = torch.randn(32)
        out = fusion(key, prosody)
        assert tuple(out.shape) == (96,)

    def test_without_prosody_still_96d(self) -> None:
        fusion = MultimodalFusion()
        key = torch.randn(64)
        out = fusion(key, None)
        assert tuple(out.shape) == (96,)

    def test_batched(self) -> None:
        fusion = MultimodalFusion()
        key = torch.randn(2, 64)
        prosody = torch.randn(2, 32)
        out = fusion(key, prosody)
        assert tuple(out.shape) == (2, 96)

    def test_identity_init_keystroke_dominates_when_prosody_none(self) -> None:
        """At init the residual + identity-init projections mean the head 64
        components of the fused output should track the key embedding far
        more closely than the tail 32 (which got zeroed prosody)."""
        fusion = MultimodalFusion()
        fusion.eval()
        key = torch.randn(64)
        out = fusion(key, None)
        head_norm = float(torch.linalg.norm(out[:64]))
        tail_norm = float(torch.linalg.norm(out[64:]))
        # The tail should be very close to zero (it's the projection of a
        # zero vector through a tiny LayerNorm + Linear).
        assert tail_norm < 1.0
        assert head_norm > 5 * tail_norm

    def test_invalid_out_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            MultimodalFusion(key_dim=64, prosody_dim=32, out_dim=100)

    def test_prosody_dim_mismatch_raises(self) -> None:
        fusion = MultimodalFusion()
        with pytest.raises(ValueError):
            fusion(torch.randn(64), torch.randn(16))


# ---------------------------------------------------------------------------
# Helper API
# ---------------------------------------------------------------------------

class TestPayloadToTensor:
    def test_good_payload_to_tensor(self) -> None:
        t = prosody_payload_to_tensor(_good_payload())
        assert t is not None
        assert tuple(t.shape) == (8,)
        assert torch.isfinite(t).all()

    def test_bad_payload_returns_none(self) -> None:
        assert prosody_payload_to_tensor({"pitch_mean_norm": 0.5}) is None
        assert prosody_payload_to_tensor(None) is None
