"""Text normalisation and cleaning for ingested records.

The :class:`Cleaner` applies a deterministic, documented sequence of
transforms to every record:

1. HTML entity decoding (``&amp;`` → ``&``).
2. Unicode NFKC normalisation so visually identical glyphs are
   represented identically.
3. Zero-width character removal (ZWSP, ZWNJ, ZWJ, BOM, …) — common
   adversarial-unicode and copy-paste-from-Word artefacts.
4. Newline canonicalisation (``\\r\\n`` / ``\\r`` → ``\\n``).
5. Whitespace collapse (multiple spaces → single, trim ends).
6. Control-character stripping (keep ``\\n`` and ``\\t``).
7. Optional case-fold of stopwords only, preserving content-word
   casing (rare; off by default).

Every stage is a pure function that takes and returns a string so the
pipeline is easy to reason about and easy to test.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

#: Unicode categories to strip entirely (Cc = control, Cf = format).
#: Preserves ``\n`` and ``\t`` because those are meaningful in
#: dialogue transcripts.
_CONTROL_CATEGORIES: frozenset[str] = frozenset({"Cc", "Cf"})
_PRESERVED_CONTROLS: frozenset[str] = frozenset({"\n", "\t"})

#: Common zero-width and bidi-override characters.  These are either
#: adversarial (bidi override used to cloak PII) or copy-paste
#: artefacts from Word / rich-text sources.
_ZERO_WIDTH_CHARS: tuple[str, ...] = (
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "⁠",  # WORD JOINER
    "﻿",  # BYTE ORDER MARK / ZWNBSP
    "‪",  # LEFT-TO-RIGHT EMBEDDING
    "‫",  # RIGHT-TO-LEFT EMBEDDING
    "‬",  # POP DIRECTIONAL FORMATTING
    "‭",  # LEFT-TO-RIGHT OVERRIDE
    "‮",  # RIGHT-TO-LEFT OVERRIDE
    "⁦",  # LEFT-TO-RIGHT ISOLATE
    "⁧",  # RIGHT-TO-LEFT ISOLATE
    "⁨",  # FIRST STRONG ISOLATE
    "⁩",  # POP DIRECTIONAL ISOLATE
    "­",  # SOFT HYPHEN
)

_ZERO_WIDTH_RE: re.Pattern[str] = re.compile(
    "[" + "".join(re.escape(c) for c in _ZERO_WIDTH_CHARS) + "]"
)

_MULTI_SPACE_RE: re.Pattern[str] = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE: re.Pattern[str] = re.compile(r"\n{3,}")


def normalise_unicode(text: str) -> str:
    """Apply NFKC normalisation.

    NFKC combines decomposition (canonical and compatibility) with
    canonical composition, so ``"é"`` as ``"e"`` + combining acute
    collapses to the precomposed code point, and ``"①"`` becomes
    ``"1"``.  This is exactly what we want for on-disk comparison.
    """
    return unicodedata.normalize("NFKC", text)


def strip_zero_width(text: str) -> str:
    """Remove zero-width and bidi-override characters."""
    return _ZERO_WIDTH_RE.sub("", text)


def _strip_controls(text: str) -> str:
    """Remove every control character except ``\\n`` and ``\\t``."""
    out: list[str] = []
    for ch in text:
        if ch in _PRESERVED_CONTROLS:
            out.append(ch)
        elif unicodedata.category(ch) in _CONTROL_CATEGORIES:
            continue
        else:
            out.append(ch)
    return "".join(out)


def _canonical_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_whitespace(text: str) -> str:
    # Canonicalise runs of horizontal whitespace first.
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # Trim each line's leading + trailing whitespace.
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Configuration for :class:`Cleaner`.

    All flags default to ``True`` — the defaults are the canonical
    pipeline.  Set flags to ``False`` to debug individual stages or
    to preserve raw form for a specific consumer.
    """

    decode_html_entities: bool = True
    normalise_unicode: bool = True
    strip_zero_width: bool = True
    strip_controls: bool = True
    canonical_newlines: bool = True
    collapse_whitespace: bool = True


class Cleaner:
    """Applies a configured sequence of text transforms.

    The cleaner is stateless — it can be shared across threads and
    reused across pipeline runs.  Stage order is fixed and documented.
    """

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """Apply every enabled stage in order, returning the cleaned text."""
        if self.config.decode_html_entities:
            text = html.unescape(text)
        if self.config.normalise_unicode:
            text = normalise_unicode(text)
        if self.config.strip_zero_width:
            text = strip_zero_width(text)
        if self.config.canonical_newlines:
            text = _canonical_newlines(text)
        if self.config.strip_controls:
            text = _strip_controls(text)
        if self.config.collapse_whitespace:
            text = _collapse_whitespace(text)
        return text

    def __call__(self, text: str) -> str:  # convenience
        return self.clean(text)


__all__ = [
    "Cleaner",
    "CleaningConfig",
    "normalise_unicode",
    "strip_zero_width",
]
