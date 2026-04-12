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
        cognitive_load = self.cognitive.compute(features, deviation)

        self._current_style = self.style.compute(features, self._current_style)

        emotional_tone = self.emotional.compute(features, deviation)

        accessibility = self.accessibility.compute(
            features,
            deviation,
            threshold=self.config.accessibility.detection_threshold,
        )

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
