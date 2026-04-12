"""Pre-built scenarios for demonstrating I3 capabilities.

Each scenario is a list of messages that simulate a realistic
interaction arc.  The ``speed`` and ``style`` keys are hints for the
demo driver (``demo/run_scenario.py`` or the UI's auto-play mode) to
inject realistic keystroke timings and composition metadata.

Speed guide
-----------
- ``"fast"``      -- ~50 ms inter-key interval, minimal pauses
- ``"normal"``    -- ~120 ms IKI, short pause before send
- ``"slow"``      -- ~250 ms IKI, noticeable pause before send
- ``"very_slow"`` -- ~450 ms IKI, long pauses, high backspace ratio
"""

from __future__ import annotations

from typing import Any

# Approximate inter-key intervals (ms) for each speed tier
SPEED_IKI_MS: dict[str, int] = {
    "fast": 50,
    "normal": 120,
    "slow": 250,
    "very_slow": 450,
}

# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------

DEMO_SCENARIOS: dict[str, list[dict[str, Any]]] = {
    # ---------------------------------------------------------------
    # 1. Energy fade -- user starts excited, gradually tires out
    # ---------------------------------------------------------------
    "energetic_to_tired": [
        {
            "text": "Hey! Just got back from an amazing hike!",
            "speed": "fast",
            "style": "casual",
            "backspace_ratio": 0.05,
        },
        {
            "text": "The views were incredible, you should see the photos!",
            "speed": "fast",
            "style": "excited",
            "backspace_ratio": 0.04,
        },
        {
            "text": "Took like 200 pictures haha",
            "speed": "normal",
            "style": "casual",
            "backspace_ratio": 0.06,
        },
        {
            "text": "yeah it was nice",
            "speed": "slow",
            "style": "simple",
            "backspace_ratio": 0.08,
        },
        {
            "text": "tired now",
            "speed": "very_slow",
            "style": "minimal",
            "backspace_ratio": 0.10,
        },
    ],

    # ---------------------------------------------------------------
    # 2. Accessibility -- motor-difficulty typing pattern
    # ---------------------------------------------------------------
    "accessibility": [
        {
            "text": "hello",
            "speed": "very_slow",
            "style": "minimal",
            "backspace_ratio": 0.30,
        },
        {
            "text": "hw are you",
            "speed": "very_slow",
            "style": "minimal",
            "backspace_ratio": 0.40,
        },
        {
            "text": "cn you help me",
            "speed": "very_slow",
            "style": "minimal",
            "backspace_ratio": 0.35,
        },
    ],

    # ---------------------------------------------------------------
    # 3. Formality shift -- casual chat then switches to work mode
    # ---------------------------------------------------------------
    "casual_to_formal": [
        {
            "text": "yo whats up",
            "speed": "fast",
            "style": "casual",
            "backspace_ratio": 0.03,
        },
        {
            "text": "nm just chillin, u?",
            "speed": "fast",
            "style": "casual",
            "backspace_ratio": 0.05,
        },
        {
            "text": "Actually, I need help drafting an email to my manager.",
            "speed": "normal",
            "style": "semi-formal",
            "backspace_ratio": 0.12,
        },
        {
            "text": (
                "Could you help me write a professional follow-up regarding "
                "the Q2 budget review?"
            ),
            "speed": "slow",
            "style": "formal",
            "backspace_ratio": 0.18,
        },
    ],

    # ---------------------------------------------------------------
    # 4. Frustration build-up -- increasingly terse and fast
    # ---------------------------------------------------------------
    "frustration": [
        {
            "text": "How do I reset my password?",
            "speed": "normal",
            "style": "neutral",
            "backspace_ratio": 0.06,
        },
        {
            "text": "It says invalid link",
            "speed": "fast",
            "style": "terse",
            "backspace_ratio": 0.04,
        },
        {
            "text": "still not working",
            "speed": "fast",
            "style": "terse",
            "backspace_ratio": 0.02,
        },
        {
            "text": "this is broken",
            "speed": "fast",
            "style": "terse",
            "backspace_ratio": 0.01,
        },
    ],

    # ---------------------------------------------------------------
    # 5. Deep thought -- long pauses, heavy editing, careful wording
    # ---------------------------------------------------------------
    "thoughtful_composition": [
        {
            "text": "I've been thinking about something...",
            "speed": "slow",
            "style": "reflective",
            "backspace_ratio": 0.15,
        },
        {
            "text": (
                "Do you think it's possible to measure how much someone "
                "trusts an AI just from how they type?"
            ),
            "speed": "slow",
            "style": "reflective",
            "backspace_ratio": 0.25,
        },
        {
            "text": (
                "Like, not from what they say, but from the rhythm and "
                "hesitation in their keystrokes."
            ),
            "speed": "very_slow",
            "style": "reflective",
            "backspace_ratio": 0.20,
        },
    ],
}


def get_scenario(name: str) -> list[dict[str, Any]]:
    """Return a named scenario or raise ``KeyError``."""
    return DEMO_SCENARIOS[name]


def list_scenarios() -> list[str]:
    """Return the names of all available demo scenarios."""
    return list(DEMO_SCENARIOS.keys())
