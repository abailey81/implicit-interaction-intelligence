"""Core data types for the interaction monitoring layer.

All types are plain dataclasses so they stay lightweight and free of heavy
framework dependencies.  ``InteractionFeatureVector`` carries the full 32-dim
representation used by downstream adaptation modules.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Keystroke-level event
# ---------------------------------------------------------------------------

@dataclass
class KeystrokeEvent:
    """A single keystroke timing record captured from the client.

    Attributes:
        timestamp: Unix epoch seconds (float) when the key was pressed.
        key_type: One of ``"char"``, ``"backspace"``, ``"enter"``, or other
            modifier labels forwarded by the front-end.
        inter_key_interval_ms: Milliseconds elapsed since the previous
            keystroke in the same composition session.  ``0.0`` for the very
            first key in a session.
    """

    timestamp: float
    key_type: str  # "char", "backspace", "enter"
    inter_key_interval_ms: float


# ---------------------------------------------------------------------------
# Generic interaction event envelope
# ---------------------------------------------------------------------------

@dataclass
class InteractionEvent:
    """Low-level interaction event envelope.

    This is the canonical wire format between the WebSocket transport layer
    and the :class:`InteractionMonitor`.

    Attributes:
        event_type: Discriminator -- ``"keystroke"``, ``"message"``,
            ``"session_start"``, or ``"session_end"``.
        timestamp: Unix epoch seconds.
        data: Arbitrary payload whose schema depends on ``event_type``.
    """

    event_type: str  # "keystroke", "message", "session_start", "session_end"
    timestamp: float
    data: dict[str, Any]


# ---------------------------------------------------------------------------
# 32-dimensional feature vector
# ---------------------------------------------------------------------------

@dataclass
class InteractionFeatureVector:
    """32-dimensional feature vector extracted from user interaction patterns.

    The features are organised into four groups of eight:

    1. **Keystroke dynamics** -- timing and editing behaviour.
    2. **Message content** -- linguistic complexity and style.
    3. **Session dynamics** -- trends and engagement over the conversation.
    4. **Deviation metrics** -- z-score deviations from the user baseline
       (meaningful only after a warm-up period).

    All values are normalised to roughly [0, 1] (or [-1, 1] for
    sentiment/deviation) by :class:`FeatureExtractor`.
    """

    # ---- Keystroke dynamics (8) -----------------------------------------
    mean_iki: float = 0.0              # Mean inter-key interval (normalised)
    std_iki: float = 0.0               # Std of inter-key intervals
    mean_burst_length: float = 0.0     # Mean typing burst length
    mean_pause_duration: float = 0.0   # Mean pause between bursts
    backspace_ratio: float = 0.0       # Backspace frequency
    composition_speed: float = 0.0     # Characters per second
    pause_before_send: float = 0.0     # Pre-send hesitation
    editing_effort: float = 0.0        # Edit distance ratio

    # ---- Message content (8) --------------------------------------------
    message_length: float = 0.0        # Normalised message length
    type_token_ratio: float = 0.0      # Vocabulary richness
    mean_word_length: float = 0.0      # Word sophistication
    flesch_kincaid: float = 0.0        # Readability grade level
    question_ratio: float = 0.0        # Fraction of questions
    formality: float = 0.0             # Language formality score
    emoji_density: float = 0.0         # Emoji usage density
    sentiment_valence: float = 0.0     # Lexicon-based sentiment [-1, 1]

    # ---- Session dynamics (8) -------------------------------------------
    length_trend: float = 0.0          # Message length trend (slope)
    latency_trend: float = 0.0         # Response speed trend
    vocab_trend: float = 0.0           # Vocabulary complexity trend
    engagement_velocity: float = 0.0   # Messages per minute
    topic_coherence: float = 0.0       # Topic consistency (Jaccard)
    session_progress: float = 0.0      # Progress through session [0, 1]
    time_deviation: float = 0.0        # Deviation from typical time
    response_depth: float = 0.0        # Engagement with full response

    # ---- Deviation metrics (8) ------------------------------------------
    iki_deviation: float = 0.0
    length_deviation: float = 0.0
    vocab_deviation: float = 0.0
    formality_deviation: float = 0.0
    speed_deviation: float = 0.0
    engagement_deviation: float = 0.0
    complexity_deviation: float = 0.0
    pattern_deviation: float = 0.0

    # -- Tensor conversion ------------------------------------------------

    def to_tensor(self) -> torch.Tensor:
        """Convert to a 32-dim ``torch.Tensor`` (float32).

        The order of elements matches :data:`FEATURE_NAMES`.
        """
        values = [getattr(self, f.name) for f in fields(self)]
        return torch.tensor(values, dtype=torch.float32)

    @classmethod
    def zeros(cls) -> InteractionFeatureVector:
        """Return a zero-initialised feature vector."""
        return cls()  # All defaults are 0.0

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> InteractionFeatureVector:
        """Reconstruct an ``InteractionFeatureVector`` from a 32-dim tensor.

        Args:
            t: A 1-D tensor with exactly 32 elements.

        Raises:
            ValueError: If the tensor does not have 32 elements.
        """
        if t.numel() != 32:
            raise ValueError(
                f"Expected a 32-element tensor, got {t.numel()} elements."
            )
        values = t.tolist()
        field_names = [f.name for f in fields(cls)]
        return cls(**dict(zip(field_names, values)))


# ---------------------------------------------------------------------------
# Canonical feature-name list (matches dataclass field order)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [f.name for f in dataclasses.fields(InteractionFeatureVector)]
"""Ordered list of all 32 feature names, matching tensor dimension indices."""
