"""Tests for :mod:`i3.multimodal.vision` (Batch F-1).

The suite relies on synthetic frames (black, white, gradient) — detecting a
face on them is expected to fail, which exercises the ``None``-return
contract.  The tests run without MediaPipe by confirming the clean-import
fallback; downstream logic tests are skipped in that case.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from i3.multimodal import vision as vision_module
from i3.multimodal.vision import (
    FacialAffectExtractor,
    VisionFeatureVector,
    VisionStreamExtractor,
    _MEDIAPIPE_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def black_frame() -> np.ndarray:
    """A 240 x 320 RGB frame of zeros."""
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    """A 240 x 320 RGB frame of 255s."""
    return np.full((240, 320, 3), 255, dtype=np.uint8)


@pytest.fixture
def gradient_frame() -> np.ndarray:
    """A 240 x 320 RGB gradient frame."""
    h, w = 240, 320
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    band = np.tile(xs, (h, 1))
    return np.stack([band, band, band], axis=-1)


# ---------------------------------------------------------------------------
# Feature vector contract
# ---------------------------------------------------------------------------

class TestVisionFeatureVector:
    """Contract tests for the Pydantic holder."""

    def test_zeros_is_eight_dim(self) -> None:
        arr = VisionFeatureVector.zeros().to_array()
        assert arr.shape == (8,)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, 0.0)

    def test_named_fields_round_trip(self) -> None:
        vec = VisionFeatureVector(
            eye_aspect_ratio=0.3,
            mouth_aspect_ratio=0.4,
            gaze_direction_x=-0.2,
            gaze_direction_y=0.1,
            head_pose_pitch_deg=5.0,
            head_pose_yaw_deg=-3.0,
            brow_furrow_au4=0.25,
            smile_au12=0.15,
        )
        arr = vec.to_array()
        assert arr.shape == (8,)
        assert arr[0] == pytest.approx(0.3)
        assert arr[4] == pytest.approx(5.0)

    def test_pydantic_rejects_out_of_range_gaze(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VisionFeatureVector(gaze_direction_x=10.0)


# ---------------------------------------------------------------------------
# Module import / fallback
# ---------------------------------------------------------------------------

class TestModuleImportFallback:
    """The module must import cleanly when MediaPipe is missing."""

    def test_module_exports_public_names(self) -> None:
        assert "FacialAffectExtractor" in vision_module.__all__
        assert "VisionFeatureVector" in vision_module.__all__
        assert "VisionStreamExtractor" in vision_module.__all__

    def test_extract_raises_runtime_error_without_mediapipe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With MediaPipe masked, ``extract`` raises a clear error.

        Setting ``sys.modules["mediapipe"] = None`` causes Python's import
        machinery to raise ``ImportError`` on subsequent imports; the
        module's soft-import guard catches that and sets the availability
        flag to ``False``.
        """
        monkeypatch.setitem(sys.modules, "mediapipe", None)
        reloaded = importlib.reload(vision_module)
        try:
            assert reloaded._MEDIAPIPE_AVAILABLE is False
            extractor = reloaded.FacialAffectExtractor()
            with pytest.raises(RuntimeError, match="mediapipe"):
                extractor.extract(np.zeros((120, 160, 3), dtype=np.uint8))
        finally:
            sys.modules.pop("mediapipe", None)
            importlib.reload(vision_module)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

pytestmark_requires_mediapipe = pytest.mark.skipif(
    not _MEDIAPIPE_AVAILABLE, reason="mediapipe is not installed"
)


@pytestmark_requires_mediapipe
class TestInputValidation:
    """The extractor must reject malformed frames."""

    def test_non_array_input_raises(self) -> None:
        extractor = FacialAffectExtractor()
        try:
            with pytest.raises(ValueError, match="ndarray"):
                extractor.extract("not a frame")  # type: ignore[arg-type]
        finally:
            extractor.close()

    def test_wrong_shape_raises(self) -> None:
        extractor = FacialAffectExtractor()
        try:
            with pytest.raises(ValueError, match="H, W, 3"):
                extractor.extract(np.zeros((120, 160), dtype=np.uint8))
            with pytest.raises(ValueError, match="H, W, 3"):
                extractor.extract(np.zeros((120, 160, 4), dtype=np.uint8))
        finally:
            extractor.close()


@pytestmark_requires_mediapipe
class TestExtractionOnSyntheticFrames:
    """Synthetic frames do not contain a face -> extractor returns ``None``."""

    def test_black_frame_returns_none(self, black_frame: np.ndarray) -> None:
        extractor = FacialAffectExtractor()
        try:
            result = extractor.extract(black_frame)
            assert result is None
        finally:
            extractor.close()

    def test_white_frame_returns_none(self, white_frame: np.ndarray) -> None:
        extractor = FacialAffectExtractor()
        try:
            result = extractor.extract(white_frame)
            assert result is None
        finally:
            extractor.close()

    def test_gradient_frame_returns_none(self, gradient_frame: np.ndarray) -> None:
        extractor = FacialAffectExtractor()
        try:
            result = extractor.extract(gradient_frame)
            assert result is None
        finally:
            extractor.close()


@pytestmark_requires_mediapipe
class TestStreamingExtractor:
    """Streaming wrapper should rate-limit and return ``None`` without faces."""

    def test_rejects_invalid_fps(self) -> None:
        with pytest.raises(ValueError, match="fps_target"):
            VisionStreamExtractor(fps_target=0.0)

    def test_get_features_is_none_before_any_push(self) -> None:
        streamer = VisionStreamExtractor(fps_target=30.0)
        try:
            assert streamer.get_features() is None
        finally:
            streamer.close()

    def test_push_black_frame_yields_no_features(
        self, black_frame: np.ndarray
    ) -> None:
        streamer = VisionStreamExtractor(fps_target=30.0)
        try:
            # Synthetic frames never detect a face -> buffer stays empty.
            accepted = streamer.push_frame(black_frame)
            assert accepted is False
            assert streamer.get_features() is None
        finally:
            streamer.close()


@pytestmark_requires_mediapipe
class TestContextManager:
    """Extractors support ``with``-style use and close cleanly."""

    def test_extractor_context_manager(self, black_frame: np.ndarray) -> None:
        with FacialAffectExtractor() as extractor:
            result = extractor.extract(black_frame)
            assert result is None
        # After ``__exit__`` the internal mesh should be released.
        assert extractor._face_mesh is None

    def test_streamer_context_manager(self) -> None:
        with VisionStreamExtractor(fps_target=10.0) as streamer:
            assert streamer.get_features() is None
