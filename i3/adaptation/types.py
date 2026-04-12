"""Core data types for the adaptation layer.

The adaptation layer translates observed user behaviour (via
:class:`~src.interaction.types.InteractionFeatureVector` and
:class:`~src.user_model.types.DeviationMetrics`) into a compact
:class:`AdaptationVector` that conditions downstream response generation.

Two key types are defined:

- :class:`StyleVector` -- A 4-dimensional representation of communication
  style (formality, verbosity, emotionality, directness).
- :class:`AdaptationVector` -- The complete 8-dimensional adaptation
  specification covering cognitive load, style mirroring, emotional tone,
  and accessibility.

Design rationale
~~~~~~~~~~~~~~~~
The adaptation vector is intentionally low-dimensional (8 floats) so that it
can be injected into the SLM's cross-attention conditioning without adding
significant computational overhead.  All values are clamped to valid ranges
to prevent runaway adaptation and ensure stable behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# StyleVector
# ---------------------------------------------------------------------------

@dataclass
class StyleVector:
    """Multi-dimensional communication style representation.

    Each axis captures an independent aspect of *how* the AI should
    communicate, independent of *what* it says.  The adaptation controller
    evolves this vector over time by mirroring the user's observed style
    with a smoothing lag.

    Attributes:
        formality:    0 = casual / colloquial, 1 = formal / professional.
        verbosity:    0 = concise / terse responses, 1 = elaborate / detailed.
        emotionality: 0 = reserved / neutral affect, 1 = expressive / emotive.
        directness:   0 = indirect / hedged phrasing, 1 = direct / assertive.
    """

    formality: float    # 0=casual, 1=formal
    verbosity: float    # 0=concise, 1=elaborate
    emotionality: float # 0=reserved, 1=expressive
    directness: float   # 0=indirect, 1=direct

    def __post_init__(self) -> None:
        """Clamp all dimensions to [0, 1] on construction."""
        self.formality = _clamp(self.formality)
        self.verbosity = _clamp(self.verbosity)
        self.emotionality = _clamp(self.emotionality)
        self.directness = _clamp(self.directness)

    def to_tensor(self) -> torch.Tensor:
        """Convert to a 4-dim ``torch.Tensor`` (float32).

        Returns:
            1-D tensor with elements ordered as
            ``[formality, verbosity, emotionality, directness]``.
        """
        return torch.tensor(
            [self.formality, self.verbosity, self.emotionality, self.directness],
            dtype=torch.float32,
        )

    @classmethod
    def default(cls) -> StyleVector:
        """Return a neutral mid-range style vector (all dimensions at 0.5).

        This is the starting point before any user data has been observed.
        """
        return cls(formality=0.5, verbosity=0.5, emotionality=0.5, directness=0.5)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> StyleVector:
        """Reconstruct a ``StyleVector`` from a tensor with >= 4 elements.

        Only the first four elements are used; extras are silently ignored.

        Args:
            t: A tensor with at least 4 elements.

        Returns:
            A new ``StyleVector`` with values clamped to [0, 1].
        """
        vals = t.tolist()
        return cls(
            formality=vals[0],
            verbosity=vals[1],
            emotionality=vals[2],
            directness=vals[3],
        )

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dictionary for JSON/logging."""
        return {
            "formality": self.formality,
            "verbosity": self.verbosity,
            "emotionality": self.emotionality,
            "directness": self.directness,
        }


# ---------------------------------------------------------------------------
# AdaptationVector
# ---------------------------------------------------------------------------

@dataclass
class AdaptationVector:
    """Complete adaptation specification for response generation.

    This 8-dimensional vector controls how the AI adapts its responses
    across four orthogonal dimensions:

    1. **Cognitive load** -- Controls response complexity.  When the user
       shows declining linguistic sophistication, we simplify; when they
       engage deeply, we provide richer content.

    2. **Style mirror** -- Matches the user's communication style along
       four sub-dimensions (formality, verbosity, emotionality, directness)
       with a smoothing lag to avoid jarring shifts.

    3. **Emotional tone** -- Ranges from warm/supportive (0.0) to
       neutral/objective (1.0).  The system becomes warmer when it detects
       signs of user distress (engagement drop, slower typing, negative
       sentiment).

    4. **Accessibility** -- Triggers simplification of language and
       interaction patterns when motor or cognitive difficulty is detected
       (e.g., high backspace ratio, slow typing, elevated editing effort).

    Total tensor dimensionality: 8 (1 + 4 + 1 + 1 + 1 reserved).

    Attributes:
        cognitive_load:  0.0 = simplest responses, 1.0 = most complex.
        style_mirror:    Multi-dimensional :class:`StyleVector`.
        emotional_tone:  0.0 = most supportive/warm, 1.0 = most neutral.
        accessibility:   0.0 = standard mode, 1.0 = maximum simplification.
    """

    cognitive_load: float      # 0.0 (simplest) -> 1.0 (most complex)
    style_mirror: StyleVector  # Multi-dimensional style
    emotional_tone: float      # 0.0 (most supportive/warm) -> 1.0 (most neutral)
    accessibility: float       # 0.0 (standard) -> 1.0 (maximum simplification)

    def __post_init__(self) -> None:
        """Clamp scalar dimensions to valid [0, 1] ranges."""
        self.cognitive_load = _clamp(self.cognitive_load)
        self.emotional_tone = _clamp(self.emotional_tone)
        self.accessibility = _clamp(self.accessibility)

    def to_tensor(self) -> torch.Tensor:
        """Convert to an 8-dim ``torch.Tensor`` for SLM conditioning.

        Layout::

            [0] cognitive_load
            [1] style_mirror.formality
            [2] style_mirror.verbosity
            [3] style_mirror.emotionality
            [4] style_mirror.directness
            [5] emotional_tone
            [6] accessibility
            [7] reserved (always 0.0)

        Returns:
            1-D float32 tensor of shape ``(8,)``.
        """
        return torch.tensor(
            [
                self.cognitive_load,
                self.style_mirror.formality,
                self.style_mirror.verbosity,
                self.style_mirror.emotionality,
                self.style_mirror.directness,
                self.emotional_tone,
                self.accessibility,
                0.0,  # Reserved dimension for future use
            ],
            dtype=torch.float32,
        )

    @classmethod
    def default(cls) -> AdaptationVector:
        """Return a neutral default vector suitable for cold-start.

        Cognitive load and emotional tone are mid-range, accessibility is
        off, and style is neutral.
        """
        return cls(
            cognitive_load=0.5,
            style_mirror=StyleVector.default(),
            emotional_tone=0.5,
            accessibility=0.0,
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> AdaptationVector:
        """Reconstruct an ``AdaptationVector`` from a tensor with >= 7 elements.

        The reserved 8th dimension is ignored.  Values are clamped during
        ``__post_init__``.

        Args:
            t: A tensor with at least 7 elements.

        Returns:
            A new ``AdaptationVector``.
        """
        vals = t.tolist()
        return cls(
            cognitive_load=vals[0],
            style_mirror=StyleVector(
                formality=vals[1],
                verbosity=vals[2],
                emotionality=vals[3],
                directness=vals[4],
            ),
            emotional_tone=vals[5],
            accessibility=vals[6],
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to a nested dictionary for JSON/logging.

        Returns:
            Dictionary with keys ``cognitive_load``, ``style_mirror`` (nested),
            ``emotional_tone``, and ``accessibility``.
        """
        return {
            "cognitive_load": self.cognitive_load,
            "style_mirror": self.style_mirror.to_dict(),
            "emotional_tone": self.emotional_tone,
            "accessibility": self.accessibility,
        }
