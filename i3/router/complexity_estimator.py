"""Per-prompt complexity estimator for the cloud-vs-edge router.

This module is the deterministic, CPU-only signal-extractor that the
LinUCB contextual bandit consumes when deciding whether to route a turn
to the on-device SLM or to the optional cloud LLM fallback.

Design rationale
~~~~~~~~~~~~~~~~
The existing :class:`i3.router.complexity.QueryComplexityEstimator`
collapses six heuristics into a single scalar that lives in
``RoutingContext.query_complexity``.  That estimator is *general* — it
serves a number of downstream consumers (the privacy override gate,
the complexity floor, the bandit, the analyst surfaces).

The cloud-vs-edge decision benefits from a more *narrowly-scoped*
score that asks the question literally on the tin:

    "How hard is this prompt for the local SLM?"

The five sub-scores combined here intentionally weight the things the
4 M-param on-device transformer struggles with:

  1. **length_factor** — long prompts blow past the SLM's coherent-
     generation window.  ``len / 100`` capped to 1.0.
  2. **rare_token_factor** — the on-device tokenizer's vocab is small;
     queries with a high fraction of out-of-vocab words are likely to
     produce poorer SLM outputs and benefit from a cloud route.
  3. **open_ended_factor** — phrases like "explain", "describe",
     "compare", "write a short essay" are precisely the prompts for
     which the SLM emits the most word-salad.  Detected via a curated
     trigger list.
  4. **multi_clause_factor** — counts comma / "and" / "but" / "or"
     splits as a proxy for sub-clause depth.
  5. **retrieval_miss_factor** — the strongest single signal we have:
     when the retriever's top cosine is BELOW threshold the SLM is the
     only option on the local side and is statistically the worst-
     quality response surface, so cloud should be preferred.

The composite is the arithmetic mean, clipped to [0, 1].  No ML
involved — every step is a fast surface-feature scan that runs in
microseconds on CPU.

Privacy
~~~~~~~
This module only consumes the prompt text.  It never persists the text
and never makes a network call.  Used as a feature input to the
LinUCB router, the result is a single float — the underlying prompt
never leaves the host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Open-ended trigger lists
# ---------------------------------------------------------------------------

# Single-word triggers — the prompt opens with one of these and is
# almost certainly an open-ended request the small SLM will flounder
# on.  We match in ``_open_ended_factor`` against the lower-cased
# leading token.
_OPEN_ENDED_LEAD: frozenset[str] = frozenset(
    {
        "explain", "describe", "compare", "contrast", "elaborate",
        "summarise", "summarize", "discuss", "analyse", "analyze",
        "evaluate", "review", "outline", "narrate", "justify",
        "argue", "critique", "debate", "design", "draft",
        "imagine",
    }
)

# Multi-word phrase triggers — searched anywhere in the lower-cased
# prompt (whole-word).
_OPEN_ENDED_PHRASE: tuple[str, ...] = (
    "write a", "write an", "write me",
    "tell me about", "tell me everything",
    "go through", "walk me through", "step by step",
    "in detail", "in depth", "deep dive",
    "compare and contrast",
    "pros and cons",
    "what are all", "give me all", "list all",
)

# Multi-clause splitters — bare regex, whole-word for connectives.
_CLAUSE_SPLITTERS: tuple[str, ...] = (",", ";", ":", " — ", " - ", " – ")
_CLAUSE_CONNECTIVES_RE = re.compile(
    r"\b(?:and|but|or|because|although|though|while|whereas|"
    r"however|moreover|furthermore|therefore|thus|hence)\b",
    re.IGNORECASE,
)

# SEC: Hard cap on prompt size for the estimator.  Mirrors the cap on
# QueryComplexityEstimator so adversarial multi-megabyte prompts cannot
# slow this down.
_MAX_TEXT_LEN = 32_768


@dataclass
class ComplexityEstimate:
    """Structured result of a single complexity estimation pass.

    Attributes:
        score: Composite complexity in [0, 1].  Higher = harder for the
            on-device SLM, so the bandit should prefer the cloud arm.
        factors: Per-sub-score breakdown for explainability.  Keys:
            ``length_factor``, ``rare_token_factor``, ``open_ended_factor``,
            ``multi_clause_factor``, ``retrieval_miss_factor``.
        notes: One-line plain-English description of what dominated the
            score, suitable for the routing-decision tooltip.
    """

    score: float
    factors: dict[str, float] = field(default_factory=dict)
    notes: str = ""


class PromptComplexityEstimator:
    """Estimates per-prompt complexity from cheap surface features.

    Used by the LinUCB router to decide between the edge SLM and the
    cloud LLM arm.

    Five sub-scores combine into a composite:

      1. length_factor       — longer prompts are harder
                              (``len(words) / 100`` capped at 1.0)
      2. rare_token_factor   — fraction of tokens that the on-device
                              tokenizer would treat as rare / OOV
      3. open_ended_factor   — does the prompt match an open-ended
                              pattern ("explain", "describe", "write a",
                              "compare and contrast"…)?
      4. multi_clause_factor — how many sub-clauses?
                              (split on commas / "and" / "but" / etc.)
      5. retrieval_miss_factor — passed-in cosine to the top retrieval
                              candidate.  ``1 - cosine`` clipped to
                              ``[0, 1]`` when below the threshold.

    Composite = arithmetic mean of the five, clipped to [0, 1].

    Parameters:
        tokenizer: Optional on-device tokenizer used to estimate the
            rare-token factor.  Anything with a ``vocab`` (mapping or
            iterable) attribute or an ``encode(text) -> list[int]``
            method works.  When ``None``, the estimator falls back to
            an English-frequency heuristic (rare = word longer than 9
            chars or contains digits / hyphens).
    """

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer
        # Cache the vocab as a frozenset of strings if we can extract
        # one — gives O(1) membership lookup at estimate-time.
        self._vocab_strings: frozenset[str] | None = None
        if tokenizer is not None:
            vocab = getattr(tokenizer, "vocab", None)
            if isinstance(vocab, dict):
                try:
                    self._vocab_strings = frozenset(
                        str(k).lower() for k in vocab.keys()
                    )
                except Exception:  # pragma: no cover - defensive
                    self._vocab_strings = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        prompt: str,
        *,
        retrieval_top_score: float | None = None,
    ) -> ComplexityEstimate:
        """Compute a :class:`ComplexityEstimate` for *prompt*.

        Args:
            prompt: Raw user prompt.  Truncated to
                :data:`_MAX_TEXT_LEN` if longer.
            retrieval_top_score: Optional top-1 cosine score from the
                local retriever.  When provided and below ``0.65`` the
                ``retrieval_miss_factor`` rises towards 1.  When
                ``None``, that factor stays at 0 and only the four
                surface-feature factors contribute to the composite.

        Returns:
            A populated :class:`ComplexityEstimate` whose ``score`` is in
            ``[0, 1]`` and whose ``factors`` and ``notes`` describe the
            breakdown.
        """
        text = prompt or ""
        if not isinstance(text, str):
            text = str(text)
        if len(text) > _MAX_TEXT_LEN:
            text = text[:_MAX_TEXT_LEN]
        text = text.strip()

        if not text:
            return ComplexityEstimate(
                score=0.0,
                factors={
                    "length_factor": 0.0,
                    "rare_token_factor": 0.0,
                    "open_ended_factor": 0.0,
                    "multi_clause_factor": 0.0,
                    "retrieval_miss_factor": 0.0,
                },
                notes="empty prompt",
            )

        length_factor = self._length_factor(text)
        rare_factor = self._rare_token_factor(text)
        open_factor = self._open_ended_factor(text)
        clause_factor = self._multi_clause_factor(text)
        miss_factor = self._retrieval_miss_factor(retrieval_top_score)

        factors = {
            "length_factor": round(length_factor, 4),
            "rare_token_factor": round(rare_factor, 4),
            "open_ended_factor": round(open_factor, 4),
            "multi_clause_factor": round(clause_factor, 4),
            "retrieval_miss_factor": round(miss_factor, 4),
        }
        # Composite: arithmetic mean of five, clipped.
        composite = (
            length_factor
            + rare_factor
            + open_factor
            + clause_factor
            + miss_factor
        ) / 5.0
        composite = max(0.0, min(1.0, composite))

        notes = self._dominant_factor_note(factors, composite)
        return ComplexityEstimate(
            score=composite,
            factors=factors,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _length_factor(text: str) -> float:
        """Word-count / 100, clipped to ``[0, 1]``.

        ``100 words`` is the saturation point.  At that point the SLM is
        almost certainly going to lose track of the prompt; below it
        the contribution scales roughly linearly.
        """
        word_count = len(text.split())
        return float(min(1.0, word_count / 100.0))

    def _rare_token_factor(self, text: str) -> float:
        """Fraction of tokens the SLM tokenizer would treat as rare/OOV.

        Two paths:

        * If a tokenizer with a ``vocab`` attribute was supplied at
          construction time, fraction = ``# OOV / # tokens``.  We treat
          a token as OOV when it is not present in the vocab (case-
          insensitive).
        * Otherwise, fall back to an English-frequency heuristic:
          rare = word length > 9, or word contains digits, or word
          contains a hyphen / underscore.  This isn't a real OOV check
          but it correlates well enough to give the bandit signal.
        """
        words = [
            w.strip(".,;:!?()[]{}\"'").lower()
            for w in text.split()
        ]
        words = [w for w in words if w]
        if not words:
            return 0.0
        if self._vocab_strings is not None:
            rare = sum(1 for w in words if w not in self._vocab_strings)
            return float(rare / len(words))
        # Heuristic fallback.
        rare = 0
        for w in words:
            if len(w) > 9:
                rare += 1
            elif any(ch.isdigit() for ch in w):
                rare += 1
            elif "-" in w or "_" in w:
                rare += 1
        return float(rare / len(words))

    @staticmethod
    def _open_ended_factor(text: str) -> float:
        """Score the prompt's "open-endedness" against trigger lists.

        Returns ``1.0`` when the prompt opens with one of
        :data:`_OPEN_ENDED_LEAD` *or* contains any of
        :data:`_OPEN_ENDED_PHRASE` (which are the strongest signals).
        Otherwise scales by the number of trigger hits found anywhere
        in the lower-cased text (max ``3`` hits → ``1.0``).
        """
        text_lower = text.lower().strip()
        if not text_lower:
            return 0.0
        leading = text_lower.split()[0] if text_lower.split() else ""
        leading = leading.strip(".,;:!?()[]{}\"'")
        if leading in _OPEN_ENDED_LEAD:
            return 1.0
        # Multi-word phrase scan.
        phrase_hits = sum(1 for p in _OPEN_ENDED_PHRASE if p in text_lower)
        if phrase_hits > 0:
            return float(min(1.0, 0.6 + 0.2 * phrase_hits))
        # Fallback: anywhere-occurrence of single-word triggers.
        word_set = set(
            w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()
        )
        hits = sum(1 for w in word_set if w in _OPEN_ENDED_LEAD)
        if hits == 0:
            return 0.0
        return float(min(1.0, 0.3 * hits))

    @staticmethod
    def _multi_clause_factor(text: str) -> float:
        """Estimate sub-clause count via punctuation + connective hits.

        Each comma / semicolon / colon / em-dash counts once; each
        whole-word connective (``and``, ``but``, ``or``, ``because``,
        ``although``…) counts once.  Scales so 3 splits → 1.0.
        """
        count = 0
        for sep in _CLAUSE_SPLITTERS:
            count += text.count(sep)
        count += len(_CLAUSE_CONNECTIVES_RE.findall(text))
        return float(min(1.0, count / 3.0))

    @staticmethod
    def _retrieval_miss_factor(retrieval_top_score: float | None) -> float:
        """Convert retrieval cosine into a "miss" factor.

        When ``retrieval_top_score`` is ``None`` we don't penalise — the
        caller didn't have one available and the four surface-feature
        factors carry the load.  When a score is provided:

        * ``score >= 0.85`` → ``0.0``  (very confident retrieval, SLM
          path is fine)
        * ``score >= 0.65`` → linear ramp from ``0.0`` at ``0.85`` to
          ``0.4`` at ``0.65`` (borderline — slight cloud preference)
        * ``score < 0.65``  → linear ramp from ``0.4`` at ``0.65`` to
          ``1.0`` at ``0.0`` (true miss — strongly cloud)
        """
        if retrieval_top_score is None:
            return 0.0
        try:
            s = float(retrieval_top_score)
        except (TypeError, ValueError):
            return 0.0
        if s >= 0.85:
            return 0.0
        if s >= 0.65:
            # 0.85 -> 0.0, 0.65 -> 0.4
            return float(0.4 * (0.85 - s) / 0.20)
        # 0.65 -> 0.4, 0.0 -> 1.0
        s = max(0.0, s)
        return float(0.4 + 0.6 * (0.65 - s) / 0.65)

    @staticmethod
    def _dominant_factor_note(
        factors: dict[str, float], composite: float
    ) -> str:
        """One-line plain-English explanation of which factor dominated.

        Used by the routing-decision tooltip in the UI so a reviewer
        can see at a glance *why* the prompt scored what it did.
        """
        if not factors:
            return "no factors"
        # Find the single largest sub-score.
        dominant_name, dominant_value = max(
            factors.items(), key=lambda kv: kv[1]
        )
        readable = dominant_name.replace("_", " ").replace("factor", "").strip()
        return (
            f"composite {composite:.2f} "
            f"(dominant: {readable} {dominant_value:.2f})"
        )
