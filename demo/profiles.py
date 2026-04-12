"""Pre-built user profiles for demo purposes.

Each profile dictionary mirrors the shape expected by
:meth:`Pipeline.upsert_user_profile` so it can be loaded directly
during demo seeding (``POST /api/demo/seed``).
"""

from __future__ import annotations

from typing import Any

DEMO_PROFILES: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Primary demo user -- has an established baseline so the system
    # can show adaptation from the very first live message.
    # ------------------------------------------------------------------
    "demo_user": {
        "baseline_established": True,
        "total_sessions": 5,
        "total_messages": 26,
        "relationship_strength": 0.35,
        "long_term_style": {
            "formality": 0.40,
            "verbosity": 0.50,
            "emotionality": 0.60,
            "directness": 0.70,
        },
        "baseline_typing_speed_ms": 115.0,       # median IKI in ms
        "baseline_backspace_ratio": 0.08,
        "baseline_pause_before_send_ms": 800.0,
        "preferred_response_length": "medium",
    },

    # ------------------------------------------------------------------
    # Secondary demo user -- brand-new, no baseline yet.
    # Useful for showing the cold-start / calibration phase.
    # ------------------------------------------------------------------
    "demo_new_user": {
        "baseline_established": False,
        "total_sessions": 0,
        "total_messages": 0,
        "relationship_strength": 0.0,
        "long_term_style": {
            "formality": 0.50,
            "verbosity": 0.50,
            "emotionality": 0.50,
            "directness": 0.50,
        },
        "baseline_typing_speed_ms": None,
        "baseline_backspace_ratio": None,
        "baseline_pause_before_send_ms": None,
        "preferred_response_length": None,
    },

    # ------------------------------------------------------------------
    # Accessibility persona -- slower typing, higher edit rate.
    # ------------------------------------------------------------------
    "demo_accessibility": {
        "baseline_established": True,
        "total_sessions": 12,
        "total_messages": 48,
        "relationship_strength": 0.55,
        "long_term_style": {
            "formality": 0.45,
            "verbosity": 0.30,
            "emotionality": 0.50,
            "directness": 0.80,
        },
        "baseline_typing_speed_ms": 420.0,
        "baseline_backspace_ratio": 0.35,
        "baseline_pause_before_send_ms": 2200.0,
        "preferred_response_length": "short",
    },
}
