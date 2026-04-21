"""Gaze / eye-tracking feature extractor.

Produces an 8-dim feature group from a sequence of gaze samples (e.g. from a
HarmonyOS AI-Glasses eye tracker or a webcam pipeline like MediaPipe Iris).
The group shape matches
:class:`i3.interaction.types.InteractionFeatureVector`'s keystroke-dynamics
block.

References
----------
* Salvucci, D. D., Goldberg, J. H. (2000). *Identifying fixations and
  saccades in eye-tracking protocols.*  ETRA.
* Duchowski, A. T. (2017). *Eye Tracking Methodology: Theory and Practice.*
  Springer, 3rd ed.
* Kar, A., Corcoran, P. (2017). *A review and analysis of eye-gaze estimation
  systems, algorithms and performance evaluation methods in consumer
  platforms.*  IEEE Access.
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
class GazeSample:
    """A single gaze-tracker sample.

    Attributes:
        timestamp_s: Unix epoch seconds.
        x: Normalised screen / field-of-view coordinate ``[0, 1]``.
        y: Normalised screen / field-of-view coordinate ``[0, 1]``.
        target_id: Optional discrete target / AOI id (0 = none / background).
        valid: True when the tracker reported a high-confidence sample.
    """

    timestamp_s: float
    x: float
    y: float
    target_id: int = 0
    valid: bool = True


@dataclass
class GazeFeatureVector:
    """8-dim gaze feature group.

    Attributes:
        fixation_duration: Mean fixation duration, normalised by 500 ms.
        saccade_rate: Saccades per second, normalised by 5 Hz.
        gaze_target_dwell: Mean dwell on a discrete AOI, normalised by 2 s.
        scanpath_length: Total traversed visual-angle distance in the window,
            normalised.
        fixation_variance: Spatial dispersion of fixations (std over x, y).
        off_screen_ratio: Fraction of samples outside the normalised
            ``[0, 1] × [0, 1]`` viewport.
        blink_rate: Fraction of invalid samples (proxy for blinks/closures),
            normalised.
        smooth_pursuit_ratio: Fraction of samples classified as smooth pursuit
            (low velocity, non-zero velocity).
    """

    fixation_duration: float = 0.0
    saccade_rate: float = 0.0
    gaze_target_dwell: float = 0.0
    scanpath_length: float = 0.0
    fixation_variance: float = 0.0
    off_screen_ratio: float = 0.0
    blink_rate: float = 0.0
    smooth_pursuit_ratio: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return an 8-element ``float32`` numpy array in declaration order."""
        return np.array(
            [getattr(self, f.name) for f in fields(self)],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> GazeFeatureVector:
        return cls()


def _clip01(x: float) -> float:
    if x != x:
        return 0.0
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class GazeFeatureExtractor:
    """Compute :class:`GazeFeatureVector` from a list of :class:`GazeSample`.

    Saccade vs fixation classification uses a simple velocity threshold
    (I-VT algorithm, Salvucci & Goldberg 2000): samples whose instantaneous
    visual-angle velocity exceeds ``velocity_threshold`` are saccades.
    """

    def __init__(
        self,
        velocity_threshold: float = 3.0,  # in normalised-units per second
        smooth_velocity_band: tuple[float, float] = (0.1, 1.0),
        fixation_ceiling_s: float = 0.5,
        dwell_ceiling_s: float = 2.0,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.smooth_velocity_band = smooth_velocity_band
        self.fixation_ceiling_s = fixation_ceiling_s
        self.dwell_ceiling_s = dwell_ceiling_s

    def extract(self, samples: Sequence[GazeSample]) -> GazeFeatureVector:
        """Compute the 8-dim gaze feature vector.

        Args:
            samples: Chronologically ordered gaze samples.

        Returns:
            A :class:`GazeFeatureVector`.  Returns zeros (with a warning) on
            fewer than two valid samples.
        """
        if len(samples) < 2:
            logger.warning("GazeFeatureExtractor: insufficient samples")
            return GazeFeatureVector.zeros()

        valid = [s for s in samples if s.valid]
        blink_rate = 1.0 - (len(valid) / len(samples))

        if len(valid) < 2:
            return GazeFeatureVector(blink_rate=_clip01(blink_rate))

        # Per-sample velocity ----------------------------------------------
        vels = []
        for a, b in zip(valid[:-1], valid[1:]):
            dt = max(b.timestamp_s - a.timestamp_s, 1e-6)
            vels.append(float(np.hypot(b.x - a.x, b.y - a.y)) / dt)
        vels_arr = np.array(vels, dtype=np.float32)

        # Fixations = runs of low-velocity samples -------------------------
        is_fix = vels_arr < self.velocity_threshold
        fixation_runs: list[float] = []
        cur_start: int | None = None
        for i, fix in enumerate(is_fix):
            if fix and cur_start is None:
                cur_start = i
            elif not fix and cur_start is not None:
                dur = valid[i].timestamp_s - valid[cur_start].timestamp_s
                fixation_runs.append(dur)
                cur_start = None
        if cur_start is not None:
            fixation_runs.append(
                valid[-1].timestamp_s - valid[cur_start].timestamp_s
            )

        fix_dur = float(np.mean(fixation_runs)) if fixation_runs else 0.0

        # Saccade rate -----------------------------------------------------
        total_time = max(
            valid[-1].timestamp_s - valid[0].timestamp_s, 1e-6
        )
        saccade_count = int(np.sum(vels_arr >= self.velocity_threshold))
        saccade_rate = saccade_count / total_time

        # Target dwell -----------------------------------------------------
        dwells: list[float] = []
        cur_target = valid[0].target_id
        cur_start_t = valid[0].timestamp_s
        for s in valid[1:]:
            if s.target_id != cur_target:
                if cur_target != 0:
                    dwells.append(s.timestamp_s - cur_start_t)
                cur_target = s.target_id
                cur_start_t = s.timestamp_s
        if cur_target != 0:
            dwells.append(valid[-1].timestamp_s - cur_start_t)
        gaze_dwell = float(np.mean(dwells)) if dwells else 0.0

        # Scanpath length --------------------------------------------------
        scanpath = 0.0
        for a, b in zip(valid[:-1], valid[1:]):
            scanpath += float(np.hypot(b.x - a.x, b.y - a.y))

        # Fixation spatial variance ---------------------------------------
        xs = np.array([s.x for s in valid], dtype=np.float32)
        ys = np.array([s.y for s in valid], dtype=np.float32)
        fix_var = float(np.std(xs) + np.std(ys)) / 2.0

        # Off-screen ratio -------------------------------------------------
        off = [
            1 for s in valid if not (0.0 <= s.x <= 1.0 and 0.0 <= s.y <= 1.0)
        ]
        off_ratio = float(len(off) / len(valid))

        # Smooth pursuit: velocity inside a band ---------------------------
        lo, hi = self.smooth_velocity_band
        smooth_mask = (vels_arr >= lo) & (vels_arr <= hi)
        smooth_ratio = float(smooth_mask.mean())

        return GazeFeatureVector(
            fixation_duration=_clip01(fix_dur / self.fixation_ceiling_s),
            saccade_rate=_clip01(saccade_rate / 5.0),
            gaze_target_dwell=_clip01(gaze_dwell / self.dwell_ceiling_s),
            scanpath_length=_clip01(scanpath),
            fixation_variance=_clip01(fix_var * 2.0),
            off_screen_ratio=_clip01(off_ratio),
            blink_rate=_clip01(blink_rate),
            smooth_pursuit_ratio=_clip01(smooth_ratio),
        )
