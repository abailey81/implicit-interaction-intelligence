"""Adaptation controller -- orchestrates all four adaptation dimensions.

The :class:`AdaptationController` is the central coordinator of the
adaptation layer.  It owns one adapter per dimension and combines their
outputs into a single :class:`~src.adaptation.types.AdaptationVector` that
downstream modules (SLM conditioning, prompt templating, response
post-processing) consume.

Lifecycle
~~~~~~~~~
1. **Initialisation**: Created once per user session with the application
   config.  Each dimension adapter receives its own sub-config.
2. **Per-message update**: After every user message, ``compute()`` is called
   with the latest feature vector and deviation metrics.  The returned
   :class:`AdaptationVector` conditions the next AI response.
3. **Reset**: At session boundaries, ``reset()`` returns the style mirror
   to its neutral default so that the next session starts fresh.

Thread safety
~~~~~~~~~~~~~
The controller maintains mutable state (``_current_style``).  In a
multi-user server context, each user **must** have their own controller
instance.  The controller is *not* thread-safe within a single instance.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from i3.adaptation.dimensions import (
    AccessibilityAdapter,
    CognitiveLoadAdapter,
    EmotionalToneAdapter,
    StyleMirrorAdapter,
)
from i3.adaptation.types import AdaptationVector, StyleVector

if TYPE_CHECKING:
    from i3.config import AdaptationConfig
    from i3.interaction.types import InteractionFeatureVector
    from i3.user_model.types import DeviationMetrics


def _sanitize(value: object, default: float) -> float:
    """Coerce *value* to a finite float, falling back to ``default`` on NaN/None.

    SEC: last line of defence -- if a downstream adapter ever returns
    ``NaN``, ``None``, or a non-numeric, this prevents corruption of the
    SLM conditioning tensor.
    """
    if value is None:
        return float(default)
    if not isinstance(value, (int, float, str)):
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(v):
        return float(default)
    if math.isinf(v):
        return float(1.0 if v > 0 else 0.0)
    return v


class AdaptationController:
    """Orchestrate all four adaptation dimension adapters.

    This is the single entry point that the pipeline calls after each user
    message.  It delegates to the four specialised adapters and assembles
    their outputs into a coherent :class:`AdaptationVector`.

    Parameters:
        config: :class:`~src.config.AdaptationConfig` containing sub-configs
            for each dimension adapter.

    Example::

        from i3.config import load_config
        from i3.adaptation import AdaptationController

        cfg = load_config("configs/default.yaml")
        controller = AdaptationController(cfg.adaptation)

        # After each user message:
        vector = controller.compute(features, deviation)
        print(vector.to_tensor())  # 8-dim conditioning vector
    """

    def __init__(self, config: AdaptationConfig) -> None:
        self.config = config
        self.cognitive = CognitiveLoadAdapter(config.cognitive_load)
        self.style = StyleMirrorAdapter(config.style_mirror)
        self.emotional = EmotionalToneAdapter(config.emotional_tone)
        self.accessibility = AccessibilityAdapter(config.accessibility)
        self._current_style: StyleVector = StyleVector.default()

    def compute(
        self,
        features: InteractionFeatureVector,
        deviation: DeviationMetrics,
    ) -> AdaptationVector:
        """Compute a full adaptation vector from current user signals.

        This method is called once per user message.  It:

        1. Computes the target cognitive load from linguistic complexity
           signals and their deviation from the user's baseline.
        2. Smooths the style mirror toward the user's observed
           communication style.
        3. Adjusts emotional tone based on distress signals.
        4. Checks for motor/cognitive difficulty and activates
           accessibility mode if the threshold is exceeded.

        Args:
            features: The 32-dim interaction feature vector extracted from
                the user's latest message and keystroke telemetry.
            deviation: Deviation metrics comparing the user's current
                behaviour to their established personal baseline.

        Returns:
            An :class:`AdaptationVector` with all four dimensions populated
            and clamped to valid ranges.
        """
        # SEC: every adapter call is wrapped so that an unexpected exception
        # or a NaN return value cannot break the entire adaptation pipeline.
        # Each dimension falls back to a neutral default if anything goes
        # wrong.
        try:
            cognitive_load = self.cognitive.compute(features, deviation)
        except Exception:
            cognitive_load = 0.5
        cognitive_load = _sanitize(cognitive_load, default=0.5)

        try:
            new_style = self.style.compute(features, self._current_style)
            if not isinstance(new_style, StyleVector):
                new_style = StyleVector.default()
        except Exception:
            new_style = StyleVector.default()
        # ``StyleVector.__post_init__`` already clamps each axis, so the
        # mutable session state is guaranteed to stay in [0, 1].
        self._current_style = new_style

        try:
            emotional_tone = self.emotional.compute(features, deviation)
        except Exception:
            emotional_tone = 0.5
        emotional_tone = _sanitize(emotional_tone, default=0.5)

        try:
            accessibility = self.accessibility.compute(
                features,
                deviation,
                threshold=self.config.accessibility.detection_threshold,
            )
        except Exception:
            accessibility = 0.0
        accessibility = _sanitize(accessibility, default=0.0)

        return AdaptationVector(
            cognitive_load=cognitive_load,
            style_mirror=self._current_style,
            emotional_tone=emotional_tone,
            accessibility=accessibility,
        )

    @property
    def current_style(self) -> StyleVector:
        """The current style vector (read-only access for diagnostics)."""
        return self._current_style

    def reset(self) -> None:
        """Reset the controller to its neutral default state.

        This should be called at session boundaries so that the style
        mirror starts fresh for the next conversation.  Adapter configs
        are *not* reset -- only the mutable state.
        """
        self._current_style = StyleVector.default()
