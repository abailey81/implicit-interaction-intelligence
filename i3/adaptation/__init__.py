"""Adaptation layer -- maps user state and deviation metrics to response conditioning.

This package translates implicit interaction signals into a compact
:class:`AdaptationVector` (8-dim) that controls how the AI responds across
four orthogonal dimensions:

1. **Cognitive load** -- response complexity matched to user capacity.
2. **Style mirror** -- communication style mirrored with a smoothing lag.
3. **Emotional tone** -- warmth/support calibrated to detected distress.
4. **Accessibility** -- simplification activated for motor/cognitive difficulty.

Exported classes
----------------
AdaptationVector       8-dim adaptation specification for response generation.
StyleVector            4-dim communication style representation.
AdaptationController   Orchestrator that combines all four dimension adapters.
CognitiveLoadAdapter   Adjusts response complexity to user's cognitive capacity.
StyleMirrorAdapter     Mirrors user's communication style with exponential smoothing.
EmotionalToneAdapter   Calibrates warmth/support based on distress signals.
AccessibilityAdapter   Detects motor/cognitive difficulty and triggers simplification.
"""

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.adaptation.controller import AdaptationController
from i3.adaptation.dimensions import (
    CognitiveLoadAdapter,
    StyleMirrorAdapter,
    EmotionalToneAdapter,
    AccessibilityAdapter,
)

__all__ = [
    "AdaptationVector",
    "StyleVector",
    "AdaptationController",
    "CognitiveLoadAdapter",
    "StyleMirrorAdapter",
    "EmotionalToneAdapter",
    "AccessibilityAdapter",
]
