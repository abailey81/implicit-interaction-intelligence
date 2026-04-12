"""Pre-seed the demo with realistic user profiles and diary entries.

This creates a "yesterday's session" so the diary has content
and the user has a baseline when the live demo starts.  Call via
the ``POST /api/demo/seed`` endpoint or import directly.
"""

from __future__ import annotations

import time
import logging
from typing import Any

from demo.profiles import DEMO_PROFILES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic diary entries that look like prior sessions
# ---------------------------------------------------------------------------

_SEED_DIARY_ENTRIES: list[dict[str, Any]] = [
    {
        "session_id": "seed-session-001",
        "timestamp": time.time() - 86_400,  # ~24 h ago
        "summary": (
            "User greeted the assistant enthusiastically and asked about weekend "
            "hiking trails.  Typing was fast and confident with minimal edits.  "
            "Engagement remained high throughout."
        ),
        "interaction_style": {
            "formality": 0.3,
            "verbosity": 0.6,
            "emotionality": 0.7,
            "directness": 0.8,
        },
        "engagement_score": 0.82,
        "messages_exchanged": 8,
        "notable_shifts": [
            "Energy dipped slightly in the last two messages (shorter replies)."
        ],
    },
    {
        "session_id": "seed-session-002",
        "timestamp": time.time() - 43_200,  # ~12 h ago
        "summary": (
            "Brief check-in session.  User typed slowly and used short sentences.  "
            "Likely tired or distracted.  The assistant shortened its replies and "
            "the user responded positively."
        ),
        "interaction_style": {
            "formality": 0.4,
            "verbosity": 0.3,
            "emotionality": 0.4,
            "directness": 0.7,
        },
        "engagement_score": 0.55,
        "messages_exchanged": 4,
        "notable_shifts": [],
    },
    {
        "session_id": "seed-session-003",
        "timestamp": time.time() - 7_200,  # ~2 h ago
        "summary": (
            "User asked for help drafting an email to a colleague.  Typing "
            "speed was moderate with several long pauses (thinking).  "
            "Formality increased when composing the email body.  The assistant "
            "matched the more formal tone automatically."
        ),
        "interaction_style": {
            "formality": 0.6,
            "verbosity": 0.5,
            "emotionality": 0.3,
            "directness": 0.6,
        },
        "engagement_score": 0.71,
        "messages_exchanged": 6,
        "notable_shifts": [
            "Formality jumped from 0.35 to 0.65 when composing the email.",
            "Pause-before-send increased 3x during the drafting messages.",
        ],
    },
]


async def seed_demo_data(pipeline: Any) -> dict[str, Any]:
    """Seed demo data for the presentation.

    1. Creates a ``demo_user`` with an established behavioural baseline.
    2. Inserts 2--3 synthetic diary entries so the diary panel is not empty.
    3. Sets ``relationship_strength`` to 0.35 (returning-user feel).

    Parameters
    ----------
    pipeline
        The initialised :class:`Pipeline` instance (from ``app.state.pipeline``).

    Returns
    -------
    dict
        Summary of what was seeded (user ids, entry count).
    """
    seeded_users: list[str] = []

    for user_id, profile_data in DEMO_PROFILES.items():
        # Upsert the user profile into the pipeline's store
        await pipeline.upsert_user_profile(user_id, profile_data)

        # Insert synthetic diary entries
        for entry in _SEED_DIARY_ENTRIES:
            await pipeline.add_diary_entry(user_id, entry)

        seeded_users.append(user_id)
        logger.info(
            "Seeded user %s with %d diary entries", user_id, len(_SEED_DIARY_ENTRIES)
        )

    return {
        "users_seeded": seeded_users,
        "diary_entries_per_user": len(_SEED_DIARY_ENTRIES),
    }
