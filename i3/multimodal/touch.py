"""Touchscreen-interaction feature extractor.

Produces an 8-dim feature group from a sequence of multi-touch events.  The
shape matches the keystroke-dynamics group in
:class:`i3.interaction.types.InteractionFeatureVector`.

References
----------
* Buschek, D., De Luca, A., Alt, F. (2015). *Improving accuracy, applicability
  and usability of keystroke biometrics on mobile touchscreen devices.*  CHI.
* Miluzzo, E., Varshavsky, A., Balakrishnan, S., Choudhury, R. (2012).
  *TapPrints: Your finger taps have fingerprints.*  MobiSys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event & feature types
# ---------------------------------------------------------------------------

@dataclass
class TouchEvent:
    """A single touch-sample record.

    Attributes:
        timestamp_s: Unix epoch seconds when the sample was recorded.
        x: Normalised screen coordinate ``[0, 1]``.
        y: Normalised screen coordinate ``[0, 1]``.
        pressure: Normalised pressure ``[0, 1]`` (0 for devices that do not
            report pressure).
        pointer_id: Pointer identifier for tracking multi-touch streams.
        phase: One of ``"down"``, ``"move"``, ``"up"``.
    """

    timestamp_s: float
    x: float
    y: float
    pressure: float
    pointer_id: int
    phase: str


@dataclass
class TouchFeatureVector:
    """8-dim touchscreen feature group.

    Attributes:
        pressure_mean: Mean touch pressure across move/down events.
        pressure_var: Variance of touch pressure.
        swipe_velocity: Mean pointer speed during move phases, normalised.
        tap_duration: Mean down→up delta, normalised by a 500 ms ceiling.
        long_press_ratio: Fraction of pointers held longer than 500 ms.
        multi_touch_entropy: Normalised Shannon entropy of concurrent
            pointer count.
        edge_proximity_ratio: Fraction of samples within 10% of any screen
            edge — a marker for grip instability / one-handed use.
        path_curvature: Mean unsigned curvature of swipe paths,
            normalised.
    """

    pressure_mean: float = 0.0
    pressure_var: float = 0.0
    swipe_velocity: float = 0.0
    tap_duration: float = 0.0
    long_press_ratio: float = 0.0
    multi_touch_entropy: float = 0.0
    edge_proximity_ratio: float = 0.0
    path_curvature: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return an 8-element ``float32`` numpy array in declaration order."""
        return np.array(
            [getattr(self, f.name) for f in fields(self)],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> TouchFeatureVector:
        return cls()


def _clip01(x: float) -> float:
    if x != x:
        return 0.0
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class TouchFeatureExtractor:
    """Compute :class:`TouchFeatureVector` from a list of :class:`TouchEvent`.

    Parameters:
        velocity_ceiling: Ceiling for normalisation of swipe velocity
            (in normalised-coord units per second).  Defaults to 5.0,
            corresponding to a full-screen swipe in 200 ms.
        long_press_s: Threshold above which a down→up delta is flagged as a
            long press.
        edge_margin: Fraction-of-screen margin used for edge-proximity.
    """

    def __init__(
        self,
        velocity_ceiling: float = 5.0,
        long_press_s: float = 0.5,
        edge_margin: float = 0.1,
    ) -> None:
        self.velocity_ceiling = velocity_ceiling
        self.long_press_s = long_press_s
        self.edge_margin = edge_margin

    def extract(self, events: Sequence[TouchEvent]) -> TouchFeatureVector:
        """Compute the 8-dim touch feature vector.

        Args:
            events: Chronologically ordered touch samples.

        Returns:
            The computed feature vector.  Returns zeros (with a warning) if
            *events* is empty.
        """
        if len(events) == 0:
            logger.warning("TouchFeatureExtractor: empty event list")
            return TouchFeatureVector.zeros()

        pressures = np.array([e.pressure for e in events], dtype=np.float32)
        pressure_mean = float(np.mean(pressures))
        pressure_var = float(np.var(pressures))

        # -- Velocity per consecutive same-pointer samples ----------------
        by_pointer: dict[int, list[TouchEvent]] = {}
        for e in events:
            by_pointer.setdefault(e.pointer_id, []).append(e)

        velocities: list[float] = []
        tap_durations: list[float] = []
        long_press_flags: list[int] = []
        curvatures: list[float] = []

        for samples in by_pointer.values():
            if len(samples) < 2:
                continue
            deltas = []
            for a, b in zip(samples[:-1], samples[1:]):
                dt = max(b.timestamp_s - a.timestamp_s, 1e-6)
                dist = float(np.hypot(b.x - a.x, b.y - a.y))
                velocities.append(dist / dt)
                deltas.append((b.x - a.x, b.y - a.y))
            # Tap duration = first→last timestamp for this pointer.
            tap_dur = samples[-1].timestamp_s - samples[0].timestamp_s
            tap_durations.append(tap_dur)
            long_press_flags.append(1 if tap_dur >= self.long_press_s else 0)

            # Curvature proxy — mean absolute turn angle between segments.
            if len(deltas) >= 2:
                angles = []
                for (ax, ay), (bx, by) in zip(deltas[:-1], deltas[1:]):
                    na = np.hypot(ax, ay) + 1e-9
                    nb = np.hypot(bx, by) + 1e-9
                    cos = (ax * bx + ay * by) / (na * nb)
                    angles.append(np.arccos(float(np.clip(cos, -1.0, 1.0))))
                curvatures.append(float(np.mean(angles)))

        swipe_velocity = float(np.mean(velocities)) if velocities else 0.0
        tap_duration = float(np.mean(tap_durations)) if tap_durations else 0.0
        long_press_ratio = (
            float(np.mean(long_press_flags)) if long_press_flags else 0.0
        )
        path_curvature = (
            float(np.mean(curvatures)) / float(np.pi) if curvatures else 0.0
        )

        # -- Multi-touch entropy ------------------------------------------
        # Bucket events into 100 ms bins and count concurrent pointers.
        tmin = min(e.timestamp_s for e in events)
        bins: dict[int, set[int]] = {}
        for e in events:
            idx = int((e.timestamp_s - tmin) / 0.1)
            bins.setdefault(idx, set()).add(e.pointer_id)
        counts = np.array([len(s) for s in bins.values()], dtype=np.float32)
        if counts.size > 0:
            p = counts / float(counts.sum())
            p = p[p > 0]
            entropy = float(-(p * np.log2(p)).sum()) if p.size > 0 else 0.0
            # Normalise by the theoretical max over this distribution's support.
            max_entropy = float(np.log2(p.size)) if p.size > 1 else 1.0
            multi_touch_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            multi_touch_entropy = 0.0

        # -- Edge proximity ------------------------------------------------
        edge_flags = [
            1
            for e in events
            if (
                e.x <= self.edge_margin
                or e.x >= 1.0 - self.edge_margin
                or e.y <= self.edge_margin
                or e.y >= 1.0 - self.edge_margin
            )
        ]
        edge_proximity_ratio = float(len(edge_flags) / len(events))

        return TouchFeatureVector(
            pressure_mean=_clip01(pressure_mean),
            pressure_var=_clip01(pressure_var * 4.0),  # amplify small variances
            swipe_velocity=_clip01(swipe_velocity / self.velocity_ceiling),
            tap_duration=_clip01(tap_duration / self.long_press_s),
            long_press_ratio=_clip01(long_press_ratio),
            multi_touch_entropy=_clip01(multi_touch_entropy),
            edge_proximity_ratio=_clip01(edge_proximity_ratio),
            path_curvature=_clip01(path_curvature),
        )
