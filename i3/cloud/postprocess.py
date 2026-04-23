"""Post-processing of cloud LLM responses to enforce adaptation constraints.

Even though the system prompt guides the LLM toward the desired style,
the model may overshoot or undershoot on length, vocabulary complexity,
or other measurable axes.  :class:`ResponsePostProcessor` applies
deterministic corrections after the response is received to guarantee
compliance with the :class:`~src.adaptation.types.AdaptationVector`.

Design rationale
~~~~~~~~~~~~~~~~
Post-processing is intentionally *conservative* -- it truncates and
simplifies but never *generates* new content.  This ensures that the
semantic meaning of the response is preserved while its surface form
is brought into alignment with the adaptation parameters.
"""

from __future__ import annotations

import logging
import re

from i3.adaptation.types import AdaptationVector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary simplification map
# ---------------------------------------------------------------------------

_SIMPLIFICATION_MAP: dict[str, str] = {
    "approximately": "about",
    "sufficient": "enough",
    "commence": "start",
    "terminate": "end",
    "utilize": "use",
    "facilitate": "help",
    "demonstrate": "show",
    "subsequently": "then",
    "nevertheless": "still",
    "consequently": "so",
    "furthermore": "also",
    "regarding": "about",
    "previously": "before",
    "currently": "now",
    "additional": "more",
    "significant": "big",
    "implement": "do",
    "indicate": "show",
    "establish": "set up",
    "maintain": "keep",
    "obtain": "get",
    "require": "need",
    "assist": "help",
    "inform": "tell",
    "participate": "join",
    "accomplish": "do",
    "investigate": "look into",
    "comprehend": "understand",
}

# SEC: Sentence-boundary pattern.  We split on `.`, `!`, or `?` followed
# by whitespace.  CPython's stdlib :mod:`re` module requires
# *fixed-width* lookbehind assertions, so the previous variable-width
# pattern raised :class:`re.error` at import time on every supported
# Python version (3.10 / 3.11 / 3.12).  Instead we split unconditionally
# and use :data:`_ABBREVIATIONS` in :meth:`_split_sentences` to merge
# sentences that were split right after a known abbreviation.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

# Common abbreviations whose trailing dot must NOT be treated as a
# sentence boundary.  Stored without the trailing dot for cheap
# `endswith()` checking.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Jr.", "Sr.", "vs.", "etc.", "e.g.", "i.e."}
)


class ResponsePostProcessor:
    """Post-processes cloud LLM responses to enforce adaptation constraints.

    Even though the system prompt guides the LLM, we verify and adjust
    the response to match the adaptation vector's requirements on:

    1. **Length** -- Responses are truncated to a maximum sentence count
       derived from ``cognitive_load``.
    2. **Vocabulary** -- When ``accessibility`` is high, complex words are
       replaced with simpler alternatives.
    3. **Non-emptiness** -- An empty response is replaced with a safe
       fallback string.

    Usage::

        processor = ResponsePostProcessor()
        final_text = processor.process(raw_response, adaptation_vector)
    """

    # Fallback text when the response is empty after processing.
    EMPTY_FALLBACK: str = "I'm here if you'd like to chat."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        response: str,
        adaptation_vector: AdaptationVector,
    ) -> str:
        """Apply all post-processing steps to a cloud LLM response.

        Args:
            response: The raw text returned by the cloud LLM.
            adaptation_vector: The current adaptation specification used
                to determine length limits and vocabulary simplification.

        Returns:
            The post-processed response string, guaranteed to be
            non-empty and within the adaptation constraints.
        """
        original_len = len(response)

        # 1. Length enforcement (cognitive load -> max sentences)
        response = self._enforce_length(
            response, adaptation_vector.cognitive_load
        )

        # 2. Vocabulary simplification for accessibility
        if adaptation_vector.accessibility > 0.5:
            response = self._simplify_vocabulary(response)

        # 3. Non-emptiness guarantee
        if not response.strip():
            response = self.EMPTY_FALLBACK

        result = response.strip()

        if len(result) != original_len:
            logger.debug(
                "Post-processed response: %d -> %d chars  "
                "(cognitive_load=%.2f, accessibility=%.2f)",
                original_len,
                len(result),
                adaptation_vector.cognitive_load,
                adaptation_vector.accessibility,
            )

        return result

    # ------------------------------------------------------------------
    # Length enforcement
    # ------------------------------------------------------------------

    def _enforce_length(self, text: str, cognitive_load: float) -> str:
        """Truncate response to an appropriate sentence count for the cognitive load.

        The mapping is::

            cognitive_load  ->  max_sentences
            0.0 - 0.2       ->  1
            0.2 - 0.4       ->  2
            0.4 - 0.6       ->  3
            0.6 - 0.8       ->  4
            0.8 - 1.0       ->  5

        Args:
            text: The input text to potentially truncate.
            cognitive_load: Value in [0, 1] indicating acceptable complexity.

        Returns:
            Text truncated to at most ``max_sentences`` sentences.
        """
        max_sentences = max(1, int(cognitive_load * 5) + 1)
        # Clamp to a minimum of 1 and maximum of 5
        max_sentences = min(max_sentences, 5)

        sentences = self._split_sentences(text)
        if len(sentences) <= max_sentences:
            return text

        truncated = " ".join(sentences[:max_sentences])
        logger.debug(
            "Truncated from %d to %d sentences (cognitive_load=%.2f)",
            len(sentences),
            max_sentences,
            cognitive_load,
        )
        return truncated

    # ------------------------------------------------------------------
    # Vocabulary simplification
    # ------------------------------------------------------------------

    @staticmethod
    def _simplify_vocabulary(text: str) -> str:
        """Replace complex words with simpler alternatives.

        Both lowercase and capitalised forms are handled.  The
        replacement is performed via whole-word boundary matching to
        avoid corrupting substrings (e.g., "maintained" should not
        become "keeptained").

        Args:
            text: The input text.

        Returns:
            Text with complex words replaced by simpler equivalents.
        """
        for complex_word, simple_word in _SIMPLIFICATION_MAP.items():
            # Case-sensitive replacement with word boundaries
            pattern = re.compile(
                r"\b" + re.escape(complex_word) + r"\b", re.IGNORECASE
            )
            def _replacement(match: re.Match[str], simple: str = simple_word) -> str:
                original = match.group(0)
                # Preserve capitalisation of the first character
                if original[0].isupper():
                    return simple[0].upper() + simple[1:]
                return simple
            text = pattern.sub(_replacement, text)

        return text

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex boundary detection.

        Handles common abbreviations (``Mr.``, ``Dr.``, ``e.g.`` etc.)
        by post-merging fragments that ended with an abbreviation rather
        than a real sentence terminator.  This sidesteps CPython's
        fixed-width-lookbehind restriction in :mod:`re` while preserving
        the original semantics.

        Args:
            text: The input text.

        Returns:
            A list of sentence strings.  Empty strings are excluded.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        raw = [
            s.strip()
            for s in _SENTENCE_BOUNDARY_RE.split(text.strip())
            if s.strip()
        ]

        # Merge fragments that end with a known abbreviation back onto
        # the following fragment, e.g. ["Hi Dr.", "Smith.", "How are you?"]
        # -> ["Hi Dr. Smith.", "How are you?"].
        merged: list[str] = []
        for fragment in raw:
            if merged and any(
                merged[-1].endswith(abbr) for abbr in _ABBREVIATIONS
            ):
                merged[-1] = merged[-1] + " " + fragment
            else:
                merged.append(fragment)
        return merged
