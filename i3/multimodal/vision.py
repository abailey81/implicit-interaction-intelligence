"""Runnable facial-affect / gaze feature extractor (Batch F-1).

This module complements :mod:`i3.multimodal.voice_real` with the camera-side
analogue of keystroke dynamics.  It wraps MediaPipe Face Mesh (Kartynnik et
al. 2019) to compute eight landmark-derived features from a single RGB frame:

1. eye-aspect ratio (EAR — Soukupova & Cech 2016),
2. mouth-aspect ratio (MAR),
3. horizontal gaze offset,
4. vertical gaze offset,
5. head-pose pitch,
6. head-pose yaw,
7. brow-furrow (AU4 proxy, Ekman & Friesen 1978),
8. smile (AU12 proxy).

MediaPipe, OpenCV and NumPy are **soft-imported**.  Importing this module
never fails; calling :meth:`FacialAffectExtractor.extract` without MediaPipe
raises a clear :class:`RuntimeError`, and the streaming extractor returns
``None`` when no features can be produced.

This is the CV lineage requested by the AI Glasses 12-MP camera + Darwin
Research Centre roadmap.

References
----------
* Kartynnik, Y., Ablavatski, A., Grishchenko, I., Grundmann, M. (2019).
  *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.*
  CVPR Workshop on Computer Vision for AR/VR.
* Soukupova, T., Cech, J. (2016). *Real-Time Eye Blink Detection using
  Facial Landmarks.*  Computer Vision Winter Workshop.
* Ekman, P., Friesen, W. V. (1978). *Facial Action Coding System: A Technique
  for the Measurement of Facial Movement.*  Consulting Psychologists Press.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-import MediaPipe
# ---------------------------------------------------------------------------

try:
    import mediapipe as _mp  # type: ignore[import-not-found]

    _MEDIAPIPE_AVAILABLE = True
except ImportError:  # pragma: no cover - environmental
    _mp = None  # type: ignore[assignment]
    _MEDIAPIPE_AVAILABLE = False


if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    NDArrayF32 = npt.NDArray[np.float32]
else:
    NDArrayF32 = np.ndarray


_INSTALL_HINT = (
    "mediapipe is not installed. Install the optional multimodal group with "
    "`poetry install --with multimodal` to enable facial feature extraction."
)


# ---------------------------------------------------------------------------
# MediaPipe Face-Mesh landmark indices (468-point topology, Kartynnik 2019).
# ---------------------------------------------------------------------------

# Six landmarks per eye (outer, top-outer, top-inner, inner, bottom-inner,
# bottom-outer) as per Soukupova & Cech (2016).
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)

# Mouth corners + top/bottom lip centre.
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14

# Iris centres (Face-Mesh with refine_landmarks=True exposes 478 points; the
# iris centre indices are 468 and 473 for left and right).
_LEFT_IRIS = 468
_RIGHT_IRIS = 473

# Inner-brow landmarks for AU4 (furrow).
_BROW_INNER_LEFT = 55
_BROW_INNER_RIGHT = 285

# Nose tip + chin for head-pose geometry.
_NOSE_TIP = 1
_CHIN = 152
_LEFT_TEMPLE = 234
_RIGHT_TEMPLE = 454


# ---------------------------------------------------------------------------
# Pydantic feature vector
# ---------------------------------------------------------------------------

class VisionFeatureVector(BaseModel):
    """Eight-dimensional facial-affect / gaze feature group.

    Attributes:
        eye_aspect_ratio: Mean EAR across both eyes; low values indicate a
            blink (Soukupova & Cech 2016).
        mouth_aspect_ratio: MAR — vertical/horizontal ratio of mouth
            opening; a speaking-activity proxy.
        gaze_direction_x: Normalised horizontal iris offset from eye centre,
            in ``[-1, 1]``.  Positive = gaze rightwards.
        gaze_direction_y: Normalised vertical iris offset, in ``[-1, 1]``.
        head_pose_pitch_deg: Euler pitch (nose-to-chin tilt) in degrees.
        head_pose_yaw_deg: Euler yaw (left/right rotation) in degrees.
        brow_furrow_au4: AU4 proxy — inter-inner-brow distance normalised
            by inter-pupillary distance.
        smile_au12: AU12 proxy — mouth-corner elevation relative to mouth
            centre.
    """

    eye_aspect_ratio: float = Field(default=0.0, ge=0.0)
    mouth_aspect_ratio: float = Field(default=0.0, ge=0.0)
    gaze_direction_x: float = Field(default=0.0, ge=-1.5, le=1.5)
    gaze_direction_y: float = Field(default=0.0, ge=-1.5, le=1.5)
    head_pose_pitch_deg: float = Field(default=0.0)
    head_pose_yaw_deg: float = Field(default=0.0)
    brow_furrow_au4: float = Field(default=0.0, ge=0.0)
    smile_au12: float = Field(default=0.0)

    def to_array(self) -> NDArrayF32:
        """Return the feature vector as an 8-element ``float32`` numpy array.

        Returns:
            1-D ``np.ndarray`` of shape ``(8,)`` in declaration order.
        """
        return np.asarray(
            [
                self.eye_aspect_ratio,
                self.mouth_aspect_ratio,
                self.gaze_direction_x,
                self.gaze_direction_y,
                self.head_pose_pitch_deg,
                self.head_pose_yaw_deg,
                self.brow_furrow_au4,
                self.smile_au12,
            ],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> VisionFeatureVector:
        """Return a zero-valued vector."""
        return cls()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _landmarks_to_np(landmarks, width: int, height: int) -> np.ndarray:
    """Convert a MediaPipe ``NormalizedLandmarkList`` into an ``(N, 3)`` array.

    Args:
        landmarks: The ``landmark`` iterable from a MediaPipe detection.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Float32 array of ``(x_px, y_px, z_relative)`` rows.
    """
    return np.asarray(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
        dtype=np.float32,
    )


def _eye_aspect_ratio(pts: np.ndarray, indices: tuple[int, ...]) -> float:
    """Compute the Soukupova-Cech eye-aspect ratio for six eye landmarks."""
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in indices]
    v1 = float(np.linalg.norm(p2[:2] - p6[:2]))
    v2 = float(np.linalg.norm(p3[:2] - p5[:2]))
    h = float(np.linalg.norm(p1[:2] - p4[:2])) + 1e-6
    return (v1 + v2) / (2.0 * h)


def _mouth_aspect_ratio(pts: np.ndarray) -> float:
    """Compute the mouth-aspect ratio."""
    vertical = float(np.linalg.norm(pts[_MOUTH_TOP][:2] - pts[_MOUTH_BOTTOM][:2]))
    horizontal = (
        float(np.linalg.norm(pts[_MOUTH_LEFT][:2] - pts[_MOUTH_RIGHT][:2])) + 1e-6
    )
    return vertical / horizontal


def _iris_offset(
    pts: np.ndarray,
    iris_idx: int,
    eye_indices: tuple[int, ...],
) -> tuple[float, float]:
    """Return the normalised (x, y) offset of the iris from the eye centre."""
    iris = pts[iris_idx][:2]
    eye_pts = np.stack([pts[i][:2] for i in eye_indices], axis=0)
    centre = eye_pts.mean(axis=0)
    width = float(np.linalg.norm(eye_pts[0] - eye_pts[3])) + 1e-6
    height = (
        float(np.linalg.norm(eye_pts[1] - eye_pts[5]))
        + float(np.linalg.norm(eye_pts[2] - eye_pts[4]))
    ) / 2.0 + 1e-6
    return (
        float((iris[0] - centre[0]) / (width / 2.0)),
        float((iris[1] - centre[1]) / (height / 2.0)),
    )


def _head_pose(pts: np.ndarray) -> tuple[float, float]:
    """Approximate Euler pitch/yaw in degrees from a handful of landmarks.

    This avoids a full solvePnP by using the simple geometric fact that
    * nose-to-chin vertical tilt ≈ pitch,
    * left-temple-to-right-temple horizontal tilt ≈ yaw.
    """
    nose = pts[_NOSE_TIP][:2]
    chin = pts[_CHIN][:2]
    left = pts[_LEFT_TEMPLE][:2]
    right = pts[_RIGHT_TEMPLE][:2]

    # Pitch: angle between (chin - nose) and the vertical.
    dx = float(chin[0] - nose[0])
    dy = float(chin[1] - nose[1]) + 1e-6
    pitch = math.degrees(math.atan2(dx, dy))

    # Yaw: temple distances are asymmetric under yaw rotation.  We use the
    # ratio of the nose-to-left-temple vs nose-to-right-temple distances.
    d_left = float(np.linalg.norm(nose - left)) + 1e-6
    d_right = float(np.linalg.norm(nose - right)) + 1e-6
    yaw = math.degrees(math.atan2(d_right - d_left, d_right + d_left))
    return pitch, yaw


def _brow_furrow(pts: np.ndarray) -> float:
    """Return the inter-inner-brow distance normalised by inter-pupillary distance."""
    brow = float(
        np.linalg.norm(
            pts[_BROW_INNER_LEFT][:2] - pts[_BROW_INNER_RIGHT][:2]
        )
    )
    ipd = (
        float(np.linalg.norm(pts[_LEFT_IRIS][:2] - pts[_RIGHT_IRIS][:2]))
        + 1e-6
    )
    return brow / ipd


def _smile_au12(pts: np.ndarray) -> float:
    """AU12 (lip-corner puller) proxy: corner elevation vs mouth-centre.

    Returns a signed value — positive means corners are above the mouth
    centre (smile), negative means below (frown).
    """
    left = pts[_MOUTH_LEFT][:2]
    right = pts[_MOUTH_RIGHT][:2]
    centre = (pts[_MOUTH_TOP][:2] + pts[_MOUTH_BOTTOM][:2]) / 2.0
    mouth_width = float(np.linalg.norm(left - right)) + 1e-6
    # In image coordinates, y increases downwards; a smile raises the corners
    # so their y is *smaller* than the centre's.
    elevation = float((centre[1] - (left[1] + right[1]) / 2.0) / mouth_width)
    return elevation


# ---------------------------------------------------------------------------
# Single-frame extractor
# ---------------------------------------------------------------------------

class FacialAffectExtractor:
    """Compute a :class:`VisionFeatureVector` from a single RGB frame.

    The extractor is stateless from the caller's perspective; internally it
    holds a long-lived MediaPipe Face-Mesh instance to avoid per-call
    re-initialisation.  Call :meth:`close` when done (or use it as a context
    manager) to release the native resources.

    Args:
        max_num_faces: Maximum number of faces to detect per frame.
        refine_landmarks: Enable iris-landmark refinement (needed for gaze).
        min_detection_confidence: MediaPipe detection threshold.
        min_tracking_confidence: MediaPipe tracking threshold.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.max_num_faces = int(max_num_faces)
        self.refine_landmarks = bool(refine_landmarks)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self._face_mesh = None  # Lazy-initialised in _ensure_mesh().

    # ------------------------------------------------------------------
    def _ensure_mesh(self) -> None:
        """Instantiate the Face-Mesh model on first use."""
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError(_INSTALL_HINT)
        if self._face_mesh is None:
            self._face_mesh = _mp.solutions.face_mesh.FaceMesh(  # type: ignore[union-attr]
                static_image_mode=False,
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    # ------------------------------------------------------------------
    def extract(self, frame: np.ndarray) -> VisionFeatureVector | None:
        """Extract features from a single RGB frame.

        Args:
            frame: ``H x W x 3`` RGB uint8 numpy array.

        Returns:
            A :class:`VisionFeatureVector` on successful face detection, or
            ``None`` if no face was found in the frame.

        Raises:
            RuntimeError: If ``mediapipe`` is not installed.
            ValueError: If ``frame`` is not a 3-D RGB array.
        """
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError(_INSTALL_HINT)

        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"frame must be numpy.ndarray, got {type(frame).__name__}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have shape (H, W, 3); got {frame.shape}"
            )

        self._ensure_mesh()
        assert self._face_mesh is not None  # for mypy

        height, width = int(frame.shape[0]), int(frame.shape[1])
        results = self._face_mesh.process(frame)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        pts = _landmarks_to_np(landmarks, width, height)

        # Guard against incomplete landmark arrays.
        needed = max(
            _RIGHT_IRIS,
            _CHIN,
            _BROW_INNER_RIGHT,
            _MOUTH_RIGHT,
            _RIGHT_TEMPLE,
        )
        if pts.shape[0] <= needed:
            logger.debug(
                "FacialAffectExtractor: incomplete landmark set "
                "(got %d, need > %d); returning zeros",
                pts.shape[0],
                needed,
            )
            return VisionFeatureVector.zeros()

        ear = 0.5 * (
            _eye_aspect_ratio(pts, _LEFT_EYE) + _eye_aspect_ratio(pts, _RIGHT_EYE)
        )
        mar = _mouth_aspect_ratio(pts)
        gx_l, gy_l = _iris_offset(pts, _LEFT_IRIS, _LEFT_EYE)
        gx_r, gy_r = _iris_offset(pts, _RIGHT_IRIS, _RIGHT_EYE)
        pitch, yaw = _head_pose(pts)
        brow = _brow_furrow(pts)
        smile = _smile_au12(pts)

        return VisionFeatureVector(
            eye_aspect_ratio=float(ear),
            mouth_aspect_ratio=float(mar),
            gaze_direction_x=float(0.5 * (gx_l + gx_r)),
            gaze_direction_y=float(0.5 * (gy_l + gy_r)),
            head_pose_pitch_deg=float(pitch),
            head_pose_yaw_deg=float(yaw),
            brow_furrow_au4=float(brow),
            smile_au12=float(smile),
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the underlying MediaPipe resources."""
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:  # noqa: BLE001 - defensive cleanup
                logger.debug("FacialAffectExtractor: close raised", exc_info=True)
            finally:
                self._face_mesh = None

    def __enter__(self) -> FacialAffectExtractor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Streaming extractor (frame rate-limited)
# ---------------------------------------------------------------------------

class VisionStreamExtractor:
    """Frame-rate-limited streaming wrapper around :class:`FacialAffectExtractor`.

    Maintains a running average of the most recent feature vectors so callers
    can pull a smoothed :class:`VisionFeatureVector` whenever they need one.

    Args:
        fps_target: Approximate features-per-second output rate; frames
            pushed faster are ignored.
        window_size: Number of features to keep in the rolling mean window.
    """

    def __init__(self, fps_target: float = 5.0, window_size: int = 5) -> None:
        if fps_target <= 0:
            raise ValueError(f"fps_target must be positive, got {fps_target}")
        self.fps_target = float(fps_target)
        self.window_size = int(max(1, window_size))
        self._min_interval_s = 1.0 / self.fps_target
        self._extractor = FacialAffectExtractor()
        self._buffer: list[np.ndarray] = []
        self._last_push_ts: float = 0.0

    # ------------------------------------------------------------------
    def push_frame(self, frame: np.ndarray) -> bool:
        """Push a new frame into the stream.

        Args:
            frame: ``H x W x 3`` RGB uint8 numpy array.

        Returns:
            ``True`` if the frame was accepted and processed, ``False`` if it
            was rate-limited away.

        Raises:
            RuntimeError: If ``mediapipe`` is not installed.
        """
        now = time.monotonic()
        if now - self._last_push_ts < self._min_interval_s:
            return False
        self._last_push_ts = now

        vec = self._extractor.extract(frame)
        if vec is None:
            return False

        self._buffer.append(vec.to_array())
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        return True

    # ------------------------------------------------------------------
    def get_features(self) -> VisionFeatureVector | None:
        """Return the rolling-mean :class:`VisionFeatureVector`, if any.

        Returns:
            ``None`` until at least one frame has produced features; otherwise
            a :class:`VisionFeatureVector` holding the windowed mean.
        """
        if not self._buffer:
            return None
        mean = np.mean(np.stack(self._buffer, axis=0), axis=0)
        return VisionFeatureVector(
            eye_aspect_ratio=float(mean[0]),
            mouth_aspect_ratio=float(mean[1]),
            gaze_direction_x=float(mean[2]),
            gaze_direction_y=float(mean[3]),
            head_pose_pitch_deg=float(mean[4]),
            head_pose_yaw_deg=float(mean[5]),
            brow_furrow_au4=float(mean[6]),
            smile_au12=float(mean[7]),
        )

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        self._extractor.close()
        self._buffer.clear()

    def __enter__(self) -> VisionStreamExtractor:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Public module API
# ---------------------------------------------------------------------------

__all__ = [
    "FacialAffectExtractor",
    "VisionFeatureVector",
    "VisionStreamExtractor",
]
