"""Wearable-accelerometer feature extractor.

Produces an 8-dim feature group from a 3-axis accelerometer stream (e.g. from
a Huawei Watch-class wearable or Smart Hanhan's motion sensor).  The shape
matches the keystroke-dynamics group in
:class:`i3.interaction.types.InteractionFeatureVector`.

References
----------
* Bao, L., Intille, S. S. (2004). *Activity recognition from user-annotated
  acceleration data.*  Pervasive Computing.
* Kale, N. et al. (2012). *Studying step count and activity recognition from
  wearables.*  IMWUT.
* Hogan, N., Sternad, D. (2009). *Sensitivity of smoothness measures to
  movement duration, amplitude, and arrests.*  Journal of Motor Behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class AccelerometerSample:
    """A single 3-axis accelerometer reading.

    Attributes:
        timestamp_s: Unix epoch seconds.
        ax: Acceleration on the x-axis in m/s².
        ay: Acceleration on the y-axis in m/s².
        az: Acceleration on the z-axis in m/s² (includes gravity).
    """

    timestamp_s: float
    ax: float
    ay: float
    az: float


@dataclass
class AccelerometerFeatureVector:
    """8-dim wearable-motion feature group.

    Attributes:
        orientation_var: Variance in device orientation (std of the normalised
            gravity-aligned axis); rises with fidgeting.
        step_cadence_hz: Estimated step rate, normalised by 4 Hz.
        jerk_magnitude: RMS of the third-derivative of position
            (m/s³), normalised — captures movement smoothness.
        activity_intensity: L2 norm of linear (gravity-removed) acceleration,
            normalised.
        stillness_ratio: Fraction of samples below a low-activity threshold.
        tremor_energy: Energy in the 4–12 Hz band (physiological tremor),
            normalised.
        orientation_dominant_axis: Encoded dominant gravity axis
            (x=0.0, y=0.5, z=1.0) for coarse posture hints.
        rotation_rate: Proxy estimate of angular speed via gravity-vector
            rotation between consecutive samples.
    """

    orientation_var: float = 0.0
    step_cadence_hz: float = 0.0
    jerk_magnitude: float = 0.0
    activity_intensity: float = 0.0
    stillness_ratio: float = 0.0
    tremor_energy: float = 0.0
    orientation_dominant_axis: float = 0.5
    rotation_rate: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return an 8-element ``float32`` numpy array in declaration order."""
        return np.array(
            [getattr(self, f.name) for f in fields(self)],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> AccelerometerFeatureVector:
        return cls()


def _clip01(x: float) -> float:
    if x != x:
        return 0.0
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_GRAVITY_M_S2 = 9.80665


class AccelerometerFeatureExtractor:
    """Compute :class:`AccelerometerFeatureVector` from an IMU stream.

    Parameters:
        step_band_hz: Band-pass limits for the step detector, in Hz.
        tremor_band_hz: Band-pass limits used for tremor energy.
        stillness_threshold: L2 linear-acceleration ceiling (m/s²) below which
            a sample is classified as "still".
    """

    def __init__(
        self,
        step_band_hz: tuple[float, float] = (1.0, 3.0),
        tremor_band_hz: tuple[float, float] = (4.0, 12.0),
        stillness_threshold: float = 0.2,
    ) -> None:
        self.step_band_hz = step_band_hz
        self.tremor_band_hz = tremor_band_hz
        self.stillness_threshold = stillness_threshold

    def extract(
        self, samples: Sequence[AccelerometerSample]
    ) -> AccelerometerFeatureVector:
        """Compute the 8-dim accelerometer feature vector.

        Args:
            samples: Chronologically ordered IMU samples.

        Returns:
            An :class:`AccelerometerFeatureVector`.  Returns zeros (with a
            warning) on fewer than 8 samples.
        """
        if len(samples) < 8:
            logger.warning("AccelerometerFeatureExtractor: insufficient samples")
            return AccelerometerFeatureVector.zeros()

        ts = np.array([s.timestamp_s for s in samples], dtype=np.float64)
        ax = np.array([s.ax for s in samples], dtype=np.float32)
        ay = np.array([s.ay for s in samples], dtype=np.float32)
        az = np.array([s.az for s in samples], dtype=np.float32)

        duration = max(float(ts[-1] - ts[0]), 1e-3)
        fs = float(len(samples)) / duration

        # -- Magnitudes & linear accel (gravity removed) -------------------
        mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        # Very cheap gravity estimate: mean acceleration vector.
        g = np.array([ax.mean(), ay.mean(), az.mean()], dtype=np.float32)
        g_norm = float(np.linalg.norm(g)) + 1e-9
        g_hat = g / g_norm
        linear = np.stack([ax, ay, az], axis=-1) - g  # broadcast-safe
        linear_mag = np.linalg.norm(linear, axis=-1)

        # -- Activity intensity & stillness --------------------------------
        activity = float(np.mean(linear_mag)) / _GRAVITY_M_S2
        stillness = float((linear_mag < self.stillness_threshold).mean())

        # -- Jerk (discrete derivative) ------------------------------------
        jerk_vec = np.diff(linear, axis=0) / np.maximum(np.diff(ts)[:, None], 1e-3)
        jerk_mag = float(np.sqrt((jerk_vec ** 2).sum(axis=1)).mean())

        # -- Orientation variance: angle between g_hat and each sample ----
        sample_dirs = np.stack([ax, ay, az], axis=-1) / (
            mag[:, None] + 1e-9
        )
        cos_ang = np.clip(sample_dirs @ g_hat, -1.0, 1.0)
        orient_var = float(np.var(np.arccos(cos_ang)))

        # -- Rotation rate: angle change between consecutive orientations -
        dot = np.clip(
            (sample_dirs[:-1] * sample_dirs[1:]).sum(axis=1), -1.0, 1.0
        )
        rot = float(np.mean(np.arccos(dot))) * fs  # rad/s

        # -- Dominant axis -------------------------------------------------
        ax_idx = int(np.argmax(np.abs(g_hat)))
        dom_axis = [0.0, 0.5, 1.0][ax_idx]

        # -- Step cadence via FFT peak in step band -----------------------
        linear_1d = linear_mag - float(np.mean(linear_mag))
        spectrum = np.abs(np.fft.rfft(linear_1d))
        freqs = np.fft.rfftfreq(len(linear_1d), d=1.0 / fs)

        def _band_peak(lo: float, hi: float) -> float:
            m = (freqs >= lo) & (freqs <= hi)
            if not np.any(m):
                return 0.0
            return float(freqs[m][int(np.argmax(spectrum[m]))])

        def _band_energy(lo: float, hi: float) -> float:
            m = (freqs >= lo) & (freqs <= hi)
            if not np.any(m):
                return 0.0
            return float(np.sum(spectrum[m] ** 2)) / (np.sum(spectrum ** 2) + 1e-9)

        step_peak = _band_peak(*self.step_band_hz)
        tremor_energy = _band_energy(*self.tremor_band_hz)

        return AccelerometerFeatureVector(
            orientation_var=_clip01(orient_var / (np.pi ** 2)),
            step_cadence_hz=_clip01(step_peak / 4.0),
            jerk_magnitude=_clip01(jerk_mag / 50.0),
            activity_intensity=_clip01(activity),
            stillness_ratio=_clip01(stillness),
            tremor_energy=_clip01(tremor_energy * 10.0),
            orientation_dominant_axis=dom_axis,
            rotation_rate=_clip01(rot / (2.0 * np.pi)),
        )
