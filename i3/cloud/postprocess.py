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
    # Plain-English replacements for technical or formal vocabulary
    # commonly seen in the demo corpus (science, history, devices).
    "convert": "turn",
    "process": "way",
    "captures": "catches",
    "produces": "makes",
    "consume": "use",
    "transmit": "send",
    "receive": "get",
    "compute": "work out",
    "calculate": "work out",
    "identify": "spot",
    "determine": "find",
    "approximately": "about",
    "approximate": "rough",
    "components": "parts",
    "architecture": "design",
    "organisation": "team",
    "organization": "team",
    "internationally": "around the world",
    "primarily": "mainly",
    "essentially": "really",
    "particularly": "especially",
    "generally": "usually",
    "typical": "normal",
    "occurs": "happens",
    "formed": "made",
    "comprised": "made of",
    "consists": "is made",
    "operates": "works",
    "function": "job",
    "purpose": "point",
    "ability": "way to",
    "via": "through",
    "such as": "like",
    "in order to": "to",
    "in addition": "also",
    "with regard to": "about",
}

# Formality-up synonym map — used when adaptation requests high
# formality.  We deliberately keep this SHORT and only swap obviously
# casual fillers (gonna, yeah, kinda) for their formal equivalents.
# The previous version mapped ordinary verbs like "tell→inform" and
# "help→assist", which made the response sound robotic and stilted
# ("Inform me the task", "What would you like assist with").  Removed
# all such core-verb substitutions; only surface-register fillers
# remain.
_FORMALITY_UP_MAP: dict[str, str] = {
    "kinda": "somewhat",
    "kind of": "somewhat",
    "sort of": "somewhat",
    "yeah": "yes",
    "yep": "yes",
    "nope": "no",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "must",
    "real quick": "promptly",
    "tons of": "many",
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

# Contraction expansion table — used when the adaptation vector demands
# high formality so casual contractions are rewritten to their formal
# long-form equivalents.  Order matters: longer keys must come first so
# we don't expand "I'm" before catching "I'm not".
_CONTRACTION_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("won't", "will not"),
    ("can't", "cannot"),
    ("shan't", "shall not"),
    ("couldn't", "could not"),
    ("shouldn't", "should not"),
    ("wouldn't", "would not"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("wasn't", "was not"),
    ("weren't", "were not"),
    ("hasn't", "has not"),
    ("haven't", "have not"),
    ("hadn't", "had not"),
    ("doesn't", "does not"),
    ("don't", "do not"),
    ("didn't", "did not"),
    ("I'm", "I am"),
    ("I've", "I have"),
    ("I'll", "I will"),
    ("I'd", "I would"),
    ("you're", "you are"),
    ("you've", "you have"),
    ("you'll", "you will"),
    ("you'd", "you would"),
    ("we're", "we are"),
    ("we've", "we have"),
    ("we'll", "we will"),
    ("we'd", "we would"),
    ("they're", "they are"),
    ("they've", "they have"),
    ("they'll", "they will"),
    ("they'd", "they would"),
    ("it's", "it is"),
    ("that's", "that is"),
    ("there's", "there is"),
    ("here's", "here is"),
    ("what's", "what is"),
    ("who's", "who is"),
    ("where's", "where is"),
    ("how's", "how is"),
    ("let's", "let us"),
)

# Reverse table — used when adaptation requests low formality.  Picks
# the most natural casual contractions only; we don't aggressively
# rewrite every "is not" as that often sounds wrong (e.g. "it is not
# a problem" -> "it isn't a problem" is fine, but the rule fires on
# "is not yet" too which can feel off).  Restricting the source side
# to a subset keeps the transformation safe.
_CONTRACTION_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    ("I am", "I'm"),
    ("you are", "you're"),
    ("we are", "we're"),
    ("they are", "they're"),
    ("it is", "it's"),
    ("that is", "that's"),
    ("there is", "there's"),
    ("you will", "you'll"),
    ("I will", "I'll"),
    ("we will", "we'll"),
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("cannot", "can't"),
    ("will not", "won't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
)

# Verbosity-low qualifier strip list — soft hedges and filler that get
# trimmed when verbosity is low so the response sounds tight and direct.
_HEDGE_RE = re.compile(
    r"\b(?:I think|I believe|in my opinion|sort of|kind of|maybe|"
    r"perhaps|it seems(?: like)?|as far as I can tell|to be honest|"
    r"I'd say|I would say|just|really|actually|basically|literally)\b,?\s*",
    re.IGNORECASE,
)

# Low-verbosity follow-up suffixes the response can be trimmed of.
_TRAILING_FOLLOWUP_RE = re.compile(
    r"\s*(?:Anything else\?|Tell me more.?|"
    r"What (?:else|about you)\??|"
    r"Want me to (?:expand|elaborate|continue)\??)\s*\.?\s*$",
    re.IGNORECASE,
)

# Verbosity-high follow-up suffixes appended when the user wants more
# elaboration.  Picked at random elsewhere; here we just expose the
# pool.  Each entry must be one short clause — we do not generate new
# semantic content, only invite a continuation.
_VERBOSE_FOLLOWUP_OPTIONS: tuple[str, ...] = (
    "Want me to dig deeper on any of that?",
    "Happy to expand on the parts that matter most to you.",
    "Tell me which angle to drill into and I'll go further.",
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

    def adapt_with_log(
        self,
        response: str,
        adaptation_vector: AdaptationVector,
    ) -> tuple[str, list[dict[str, str]]]:
        """Apply every adaptation transformation and return a change log.

        Walks each axis of ``adaptation_vector`` and applies the
        corresponding rewriting rule when the value crosses the
        relevant threshold.  Returns the rewritten response plus a list
        of ``{"axis": ..., "value": ..., "change": ...}`` entries that
        the websocket layer can ship to the UI so the user can see
        exactly *how* the visible reply was shaped by their typing.

        Args:
            response: The raw response (from retrieval, SLM, or cloud).
            adaptation_vector: Live :class:`AdaptationVector`.

        Returns:
            A two-tuple ``(rewritten_text, change_log)``.  ``change_log``
            is empty when no axis crossed its threshold.
        """
        if not response:
            return self.EMPTY_FALLBACK, []

        log: list[dict[str, str]] = []
        text = response.strip()

        # 1. Cognitive load -> max sentence count.  Aggressive trimming
        #    when load is high, looser cap when low.
        before_sentences = self._split_sentences(text)
        text = self._enforce_length(text, adaptation_vector.cognitive_load)
        after_sentences = self._split_sentences(text)
        if len(after_sentences) < len(before_sentences):
            log.append({
                "axis": "cognitive_load",
                "value": f"{adaptation_vector.cognitive_load:.2f}",
                "change": (
                    f"trimmed {len(before_sentences)}→"
                    f"{len(after_sentences)} sentences"
                ),
            })

        # 2. Formality -> contraction expansion / contraction + synonym
        #    swap for casual vocabulary.  Both transforms are tracked
        #    as one log entry so the chip in the UI stays compact.
        formality = adaptation_vector.style_mirror.formality
        if formality > 0.65:
            transformed = self._expand_contractions(text)
            transformed = self._apply_synonym_map(transformed, _FORMALITY_UP_MAP)
            if transformed != text:
                log.append({
                    "axis": "formality",
                    "value": f"{formality:.2f}",
                    "change": "raised register",
                })
                text = transformed
        elif formality < 0.35:
            transformed = self._apply_contractions(text)
            if transformed != text:
                log.append({
                    "axis": "formality",
                    "value": f"{formality:.2f}",
                    "change": "kept contractions casual",
                })
                text = transformed

        # 3. Verbosity -> hedge stripping or follow-up appending
        verbosity = adaptation_vector.style_mirror.verbosity
        if verbosity < 0.35:
            stripped = _HEDGE_RE.sub("", text)
            stripped = _TRAILING_FOLLOWUP_RE.sub("", stripped).strip()
            stripped = re.sub(r"\s{2,}", " ", stripped)
            if stripped and stripped != text:
                log.append({
                    "axis": "verbosity",
                    "value": f"{verbosity:.2f}",
                    "change": "trimmed hedges and follow-ups",
                })
                text = stripped
        elif verbosity > 0.7 and not _TRAILING_FOLLOWUP_RE.search(text):
            # Deterministic pick — avoid randomness so the same prompt
            # under the same adaptation always produces the same trace.
            idx = int(verbosity * 100) % len(_VERBOSE_FOLLOWUP_OPTIONS)
            text = text.rstrip() + " " + _VERBOSE_FOLLOWUP_OPTIONS[idx]
            log.append({
                "axis": "verbosity",
                "value": f"{verbosity:.2f}",
                "change": "appended follow-up invitation",
            })

        # 4. Accessibility -> simpler vocabulary
        accessibility = adaptation_vector.accessibility
        if accessibility > 0.5:
            new_text = self._simplify_vocabulary(text)
            if new_text != text:
                log.append({
                    "axis": "accessibility",
                    "value": f"{accessibility:.2f}",
                    "change": "swapped complex words for simpler ones",
                })
                text = new_text

        # 5. Non-emptiness guarantee
        text = text.strip()
        if not text:
            text = self.EMPTY_FALLBACK

        return text, log

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
        """Truncate response so a stressed user gets a short, calm reply.

        High cognitive_load → user is mentally taxed → short reply.
        Low cognitive_load → user has spare bandwidth → longer reply OK.

        Mapping (inverted vs. the prior version)::

            cognitive_load  ->  max_sentences
            0.0 - 0.2       ->  6   (relaxed user — generous)
            0.2 - 0.4       ->  5
            0.4 - 0.6       ->  4
            0.6 - 0.8       ->  2
            0.8 - 1.0       ->  1   (stressed — single concise sentence)

        Also strips parenthetical asides under high load so what is
        kept is even tighter.
        """
        cl = max(0.0, min(1.0, float(cognitive_load)))
        if cl >= 0.8:
            max_sentences = 1
        elif cl >= 0.6:
            max_sentences = 2
        elif cl >= 0.4:
            max_sentences = 4
        elif cl >= 0.2:
            max_sentences = 5
        else:
            max_sentences = 6

        # Drop parenthetical asides on stressed users — they don't need
        # the colour commentary.
        if cl >= 0.6:
            text = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", text).strip()
            text = re.sub(r"\s+—\s+[^.!?]+(?=[.!?])", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip()

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
    def _apply_synonym_map(text: str, mapping: dict[str, str]) -> str:
        """Apply a longest-key-first whole-word synonym substitution.

        Used by the formality-up rewriter to swap casual vocabulary for
        more formal equivalents (e.g. ``"yeah"`` → ``"yes"``,
        ``"a lot of"`` → ``"a great deal of"``).  Whole-word matching
        (with multi-word phrase support) prevents corrupting
        substrings.
        """
        for short, long in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            pattern = re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)

            def _sub(m: re.Match[str], _l: str = long) -> str:
                original = m.group(0)
                if original[:1].isupper():
                    return _l[0].upper() + _l[1:]
                return _l

            text = pattern.sub(_sub, text)
        return text

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Rewrite casual contractions to formal long forms.

        Applied when ``formality`` exceeds 0.65 — the response sounds
        less colloquial without changing its meaning.  Word-boundary
        regex preserves capitalisation of the first letter so
        ``"I'm here"`` becomes ``"I am here"`` (not ``"i am here"``).
        """
        for short, long in _CONTRACTION_EXPANSIONS:
            pattern = re.compile(r"\b" + re.escape(short) + r"\b")

            def _sub(m: re.Match[str], _l: str = long) -> str:
                original = m.group(0)
                if original[:1].isupper():
                    return _l[0].upper() + _l[1:]
                return _l

            text = pattern.sub(_sub, text)
        return text

    @staticmethod
    def _apply_contractions(text: str) -> str:
        """Rewrite stiff long forms as casual contractions.

        Mirror of :meth:`_expand_contractions` for low-formality
        responses.  Only the safest substitutions are applied so we
        don't introduce ambiguity (``"is not"`` → ``"isn't"`` is fine,
        but ``"will not yet"`` → ``"won't yet"`` is awkward — see the
        reverse table for the curated subset).
        """
        for long, short in _CONTRACTION_CONTRACTIONS:
            pattern = re.compile(r"\b" + re.escape(long) + r"\b")

            def _sub(m: re.Match[str], _s: str = short) -> str:
                original = m.group(0)
                if original[:1].isupper():
                    return _s[0].upper() + _s[1:]
                return _s

            text = pattern.sub(_sub, text)
        return text

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
