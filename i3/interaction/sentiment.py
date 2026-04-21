"""Valence-lexicon sentiment scoring — asset-backed module.

A minimal, on-device valence lexicon scorer extracted from
:mod:`i3.interaction.linguistic` for reuse and testing. The lexicon is loaded
from a JSON asset (``i3/interaction/data/sentiment_lexicon.json``); if the
asset is missing the module falls back to a small inline dictionary so the
package keeps working with no behavioural change.

The lexicon is a curated subset inspired by the **NRC Emotion Lexicon**
(Mohammad & Turney, 2013) and **VADER** (Hutto & Gilbert, 2014) valence
entries. No file from those projects is copied verbatim. The project licence
is MIT.

Example:
    >>> from i3.interaction.sentiment import ValenceLexicon
    >>> lex = ValenceLexicon.default()
    >>> lex.score(["i", "am", "very", "happy"])
    0.2
    >>> lex.score(["i", "am", "not", "happy"])   # negation flips valence
    -0.2
    >>> lex.intensity(["awful", "terrible", "bad"])
    0.7833333333333333
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final, Mapping

logger = logging.getLogger(__name__)

__all__ = ["ValenceLexicon", "DEFAULT_LEXICON_PATH"]


# --------------------------------------------------------------------------- #
# Inline fallback (keeps the module behaviourally identical when asset is     #
# missing). Roughly 60 positive / 60 negative entries, mirroring the JSON     #
# asset.                                                                      #
# --------------------------------------------------------------------------- #

_FALLBACK_POSITIVE: Final[dict[str, float]] = {
    "happy": 0.80, "joy": 0.85, "love": 0.90, "excited": 0.80, "great": 0.70,
    "good": 0.60, "wonderful": 0.85, "amazing": 0.90, "fantastic": 0.90,
    "awesome": 0.85, "cool": 0.60, "sweet": 0.65, "nice": 0.55, "kind": 0.68,
    "warm": 0.58, "calm": 0.55, "peaceful": 0.65, "grateful": 0.72,
    "thankful": 0.70, "hope": 0.60, "proud": 0.70, "confident": 0.65,
    "curious": 0.55, "interested": 0.55, "fun": 0.65, "cheerful": 0.72,
    "content": 0.60, "delighted": 0.80, "pleased": 0.65, "satisfied": 0.65,
    "bright": 0.55, "lucky": 0.65, "healthy": 0.60, "safe": 0.55,
    "strong": 0.55, "smart": 0.60, "brave": 0.62, "free": 0.50, "fresh": 0.50,
    "rich": 0.55, "clean": 0.45, "clear": 0.48, "easy": 0.48, "simple": 0.42,
    "quick": 0.48, "successful": 0.72, "perfect": 0.90, "best": 0.75,
    "beautiful": 0.78, "gorgeous": 0.80, "lovely": 0.75, "pretty": 0.55,
    "charming": 0.65, "elegant": 0.62, "graceful": 0.60, "brilliant": 0.80,
    "inspiring": 0.72, "uplifting": 0.72, "encouraging": 0.68,
    "rewarding": 0.68,
}

_FALLBACK_NEGATIVE: Final[dict[str, float]] = {
    "sad": -0.70, "angry": -0.72, "upset": -0.65, "bad": -0.60,
    "terrible": -0.90, "horrible": -0.88, "awful": -0.85, "worse": -0.65,
    "worst": -0.80, "hate": -0.90, "disgust": -0.85, "fear": -0.68,
    "afraid": -0.58, "worried": -0.52, "stressed": -0.60, "tired": -0.42,
    "exhausted": -0.55, "sick": -0.55, "ill": -0.48, "pain": -0.60,
    "hurt": -0.60, "broken": -0.60, "lost": -0.45, "confused": -0.50,
    "annoyed": -0.60, "frustrated": -0.68, "depressed": -0.78,
    "anxious": -0.60, "nervous": -0.50, "scared": -0.60, "lonely": -0.60,
    "bored": -0.48, "boring": -0.48, "hard": -0.35, "difficult": -0.45,
    "impossible": -0.60, "wrong": -0.55, "fail": -0.62, "failed": -0.65,
    "failure": -0.68, "miss": -0.35, "missed": -0.40, "missing": -0.42,
    "problem": -0.50, "trouble": -0.48, "issue": -0.42, "bug": -0.50,
    "error": -0.55, "mistake": -0.48, "poor": -0.55, "weak": -0.48,
    "slow": -0.38, "ugly": -0.65, "dirty": -0.45, "dark": -0.30,
    "cold": -0.30, "hungry": -0.35, "thirsty": -0.30, "empty": -0.40,
    "broke": -0.55, "rude": -0.65, "mean": -0.55, "cruel": -0.78,
}

# Negation window: words after these tokens flip valence for the next N tokens.
_NEGATION_TOKENS: Final[frozenset[str]] = frozenset(
    {"not", "n't", "no", "never"}
)
_NEGATION_WINDOW: Final[int] = 3

DEFAULT_LEXICON_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "data" / "sentiment_lexicon.json"
)


class ValenceLexicon:
    """Lexicon-based valence scorer with negation handling.

    The scorer stores one ``dict[str, float]`` mapping lowercased tokens to
    valences in ``[-1, 1]``. Negation flips the valence of the next
    :data:`_NEGATION_WINDOW` tokens (default 3) when a negation marker is
    encountered. Out-of-vocabulary tokens contribute ``0.0``.

    Attributes:
        valence: The immutable underlying token -> valence mapping.
        negation_tokens: Tokens that trigger polarity flipping.
        negation_window: How many following tokens a negation affects.
    """

    valence: Mapping[str, float]
    negation_tokens: frozenset[str]
    negation_window: int

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        valence: Mapping[str, float],
        *,
        negation_tokens: frozenset[str] = _NEGATION_TOKENS,
        negation_window: int = _NEGATION_WINDOW,
    ) -> None:
        """Initialise the lexicon with an explicit mapping.

        Args:
            valence: Mapping from lowercased tokens to valences in
                ``[-1, 1]``.
            negation_tokens: Tokens that flip polarity for following tokens.
            negation_window: Number of tokens after a negation marker whose
                valence should be flipped.

        Raises:
            ValueError: If ``negation_window`` is negative.
        """
        if negation_window < 0:
            raise ValueError(
                f"negation_window must be >= 0; got {negation_window!r}."
            )
        self.valence = dict(valence)
        self.negation_tokens = negation_tokens
        self.negation_window = negation_window

    @classmethod
    def default(cls) -> "ValenceLexicon":
        """Load the bundled lexicon from the JSON asset.

        Falls back to the inline dictionaries defined at module scope if the
        asset file is missing or fails to parse. A warning is logged in the
        fallback case so deployment drift is visible.

        Returns:
            A :class:`ValenceLexicon` instance.
        """
        return cls.from_json(DEFAULT_LEXICON_PATH)

    @classmethod
    def from_json(cls, path: str | Path) -> "ValenceLexicon":
        """Load a lexicon from a JSON file.

        The JSON file is expected to contain ``"positive"`` and/or
        ``"negative"`` top-level objects mapping token -> float. Any other
        keys (e.g. ``"__meta__"``) are ignored. If the file cannot be read
        or parsed the inline fallback is used.

        Args:
            path: Filesystem path to the JSON asset.

        Returns:
            A :class:`ValenceLexicon` populated from the file, or from the
            inline fallback if loading failed.
        """
        p = Path(path)
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "ValenceLexicon: failed to load %s (%s); using inline fallback.",
                p,
                exc,
            )
            return cls._from_fallback()

        merged: dict[str, float] = {}
        for section in ("positive", "negative"):
            entries = data.get(section, {})
            if not isinstance(entries, dict):
                continue
            for token, score in entries.items():
                if isinstance(token, str) and isinstance(score, (int, float)):
                    merged[token.lower()] = float(score)

        if not merged:
            logger.warning(
                "ValenceLexicon: asset %s parsed but empty; using inline fallback.",
                p,
            )
            return cls._from_fallback()

        return cls(merged)

    @classmethod
    def _from_fallback(cls) -> "ValenceLexicon":
        """Build a lexicon from the inline fallback dictionaries."""
        merged: dict[str, float] = {}
        merged.update(_FALLBACK_POSITIVE)
        merged.update(_FALLBACK_NEGATIVE)
        return cls(merged)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def score(self, tokens: list[str]) -> float:
        """Compute the mean valence of ``tokens`` with negation handling.

        Tokens are lowercased before lookup. After a negation marker
        (``not``, ``n't``, ``no``, ``never``), the next
        :attr:`negation_window` tokens have their valence flipped. The mean
        is taken over the **total token count**, clamped to ``[-1, 1]``.

        Args:
            tokens: Input tokens. May be empty.

        Returns:
            Mean valence in ``[-1, 1]``. Returns ``0.0`` for empty input.
        """
        if not tokens:
            return 0.0

        lowered = [t.lower() for t in tokens]
        total = 0.0
        negation_ttl = 0
        for token in lowered:
            if token in self.negation_tokens:
                negation_ttl = self.negation_window
                continue
            v = self.valence.get(token, 0.0)
            if v != 0.0 and negation_ttl > 0:
                v = -v
            total += v
            if negation_ttl > 0:
                negation_ttl -= 1

        mean = total / len(lowered)
        if mean > 1.0:
            return 1.0
        if mean < -1.0:
            return -1.0
        return mean

    def intensity(self, tokens: list[str]) -> float:
        """Mean absolute valence of *scored* tokens.

        Only tokens present in the lexicon contribute to the denominator;
        negation is irrelevant for intensity since we take ``abs(...)``.

        Args:
            tokens: Input tokens. May be empty.

        Returns:
            Mean absolute valence in ``[0, 1]``; ``0.0`` when no tokens are
            scored.
        """
        if not tokens:
            return 0.0
        scored = [abs(self.valence[t.lower()]) for t in tokens if t.lower() in self.valence]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the number of entries in the lexicon."""
        return len(self.valence)

    def __contains__(self, token: object) -> bool:
        """Case-insensitive membership test."""
        if not isinstance(token, str):
            return False
        return token.lower() in self.valence

    def __repr__(self) -> str:
        """Compact debug representation."""
        return (
            f"ValenceLexicon(size={len(self.valence)}, "
            f"negation_window={self.negation_window})"
        )
