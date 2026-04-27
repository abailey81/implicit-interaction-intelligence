"""Affect-shift detection for the I3 HMI pipeline.

This sub-package provides three proactive HMI capabilities, all
showcased to Huawei R&D UK as headline implicit-HMI behaviours:

* :mod:`i3.affect.shift_detector` — mid-conversation affect-shift
  check-in (the original pitch piece).
* :mod:`i3.affect.state_classifier` — discrete user-state label for
  the live state badge in the nav.
* :mod:`i3.affect.accessibility_mode` — sticky accessibility-mode
  auto-switch driven by sustained motor-difficulty signals.

All three share the same per-session sliding-window discipline and
1000-session LRU cap.
"""

from i3.affect.accessibility_mode import (
    AccessibilityController,
    AccessibilityModeState,
)
from i3.affect.shift_detector import AffectShift, AffectShiftDetector
from i3.affect.state_classifier import UserStateLabel, classify_user_state

__all__ = [
    "AccessibilityController",
    "AccessibilityModeState",
    "AffectShift",
    "AffectShiftDetector",
    "UserStateLabel",
    "classify_user_state",
]
