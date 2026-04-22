"""Metric implementations for ImplicitAdaptBench.

Every metric returns a value in ``[0, 1]`` (higher is better) so that they
can be combined into a single aggregate via
:func:`aggregate_score`. Metrics are deliberately **rule-based** — the
benchmark is intended to run on a laptop in seconds and must not depend on
a trained judge model.

The mapping from each metric to a formal definition is given in
``docs/research/implicit_adapt_bench.md`` §3. Short definitions are repeated
in each function's docstring.

All functions are pure (no state mutation) and type-annotated.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence

from benchmarks.implicit_adapt_bench.data_schema import (
    BenchmarkRecord,
    BenchmarkSubmission,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Soft-imported analysers from the I³ codebase
# ---------------------------------------------------------------------------

# We soft-import so the metric module remains usable in environments where
# the full I³ tree is not installed (e.g. an external scorer that ``pip
# install``s the benchmark package on its own).

try:
    from i3.interaction.linguistic import LinguisticAnalyzer as _LinguisticAnalyzer
    from i3.interaction.sentiment import ValenceLexicon as _ValenceLexicon
except ImportError as exc:  # pragma: no cover - only triggered when I³ is absent
    logger.info(
        "i3.interaction.linguistic / sentiment not importable (%s); "
        "metrics will use the minimal built-in fallbacks.",
        exc,
    )
    _LinguisticAnalyzer = None  # type: ignore[assignment]
    _ValenceLexicon = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Reference-data helpers
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE: re.Pattern[str] = re.compile(r"[A-Za-z']+")

# A small, curated English idiom set. The benchmark does *not* aim for
# comprehensive idiom detection — it only needs a signal that
# accessibility-mode responses avoid metaphorical phrasings that are known
# to be hard for non-native speakers and readers with cognitive accessibility
# needs.
_IDIOM_PHRASES: tuple[str, ...] = (
    "piece of cake",
    "break a leg",
    "under the weather",
    "hit the sack",
    "bite the bullet",
    "spill the beans",
    "let the cat out of the bag",
    "cost an arm and a leg",
    "on the ball",
    "the ball is in your court",
    "in hot water",
    "a dime a dozen",
    "beat around the bush",
    "once in a blue moon",
    "read between the lines",
    "see eye to eye",
    "the last straw",
    "jump the gun",
    "burn the midnight oil",
    "call it a day",
)

# Common English contraction / slang markers — a tiny fallback list used only
# when the I³ ``LinguisticAnalyzer`` is not importable.
_FALLBACK_INFORMAL: frozenset[str] = frozenset(
    {
        "gonna",
        "wanna",
        "gotta",
        "yeah",
        "nah",
        "dunno",
        "ain't",
        "y'all",
        "kinda",
        "sorta",
        "lemme",
        "'cause",
    }
)


# ---------------------------------------------------------------------------
# Style label parsing
# ---------------------------------------------------------------------------


def _parse_style_label(label: str) -> dict[str, str]:
    """Parse a structured style label of the form ``"<formality>_<length>_<tone>"``.

    The benchmark data generator emits labels such as
    ``"warm_casual_short"``. This helper splits them into the three
    underlying buckets, tolerating alternative orderings by bucketing each
    token against a known vocabulary.

    Args:
        label: The reference style label (e.g., ``"warm_casual_short"``).

    Returns:
        A dict with keys ``formality``, ``tone``, and ``length`` — each may
        be ``""`` if the label was silent on that axis.
    """
    formality_vocab: frozenset[str] = frozenset({"casual", "neutral", "formal"})
    tone_vocab: frozenset[str] = frozenset({"warm", "reserved", "objective"})
    length_vocab: frozenset[str] = frozenset({"short", "medium", "long"})

    out: dict[str, str] = {"formality": "", "tone": "", "length": ""}
    for token in label.lower().split("_"):
        if token in formality_vocab:
            out["formality"] = token
        elif token in tone_vocab:
            out["tone"] = token
        elif token in length_vocab:
            out["length"] = token
    return out


# ---------------------------------------------------------------------------
# Low-level feature helpers (with built-in fallbacks)
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    """Split ``text`` into sentences using ``.!?`` and newline boundaries.

    Args:
        text: Free text.

    Returns:
        Non-empty stripped sentence strings.
    """
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    return parts


def _words(text: str) -> list[str]:
    """Extract lowercase word tokens using :data:`_WORD_RE`."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _count_syllables(word: str) -> int:
    """Estimate syllables via the vowel-group heuristic.

    Args:
        word: A single word.

    Returns:
        A syllable count, always >= 1 for non-empty words.
    """
    word = word.lower()
    if not word:
        return 1
    vowels = set("aeiou")
    count = 0
    prev_vowel = False
    for ch in word:
        if ch in vowels:
            if not prev_vowel:
                count += 1
            prev_vowel = True
        else:
            prev_vowel = False
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_kincaid(text: str) -> float:
    """Flesch-Kincaid grade level of ``text``.

    Uses :class:`i3.interaction.linguistic.LinguisticAnalyzer` when
    available, falling back to an identical in-module implementation so that
    the metric module does not hard-depend on the I³ tree.

    Args:
        text: Free text.

    Returns:
        A non-negative grade-level estimate; ``0.0`` for empty text.
    """
    if _LinguisticAnalyzer is not None:
        try:
            return float(_LinguisticAnalyzer().flesch_kincaid_grade(text))
        except ValueError:
            # LinguisticAnalyzer should not raise on arbitrary text, but if
            # a future change does, fall through to the local computation.
            logger.debug("LinguisticAnalyzer FK failed; using fallback.")
    sentences = _sentences(text)
    words = _words(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    grade = (
        0.39 * (len(words) / len(sentences))
        + 11.8 * (syllables / len(words))
        - 15.59
    )
    return max(0.0, float(grade))


def _formality_score(text: str) -> float:
    """Formality score in ``[0, 1]``.

    Uses the I³ :class:`LinguisticAnalyzer` if available; otherwise falls
    back to a small informal-word stoplist.

    Args:
        text: Free text.

    Returns:
        ``1.0`` for fully formal text, lower for informal text; ``1.0`` for
        empty text.
    """
    if _LinguisticAnalyzer is not None:
        try:
            return float(_LinguisticAnalyzer().formality_score(text))
        except ValueError:
            logger.debug("LinguisticAnalyzer formality failed; using fallback.")
    words = _words(text)
    if not words:
        return 1.0
    informal = sum(1 for w in words if w in _FALLBACK_INFORMAL)
    return max(0.0, min(1.0, 1.0 - informal / len(words)))


def _sentiment_valence(text: str) -> float:
    """Signed valence in ``[-1, 1]`` via the I³ ValenceLexicon.

    Falls back to ``0.0`` if the lexicon is unavailable.

    Args:
        text: Free text.

    Returns:
        Mean valence of the tokens, or ``0.0`` on an empty/unscorable text.
    """
    if _ValenceLexicon is None:
        return 0.0
    try:
        lex = _ValenceLexicon.default()
        return float(lex.score(_words(text)))
    except (OSError, ValueError) as exc:
        logger.debug("ValenceLexicon unavailable (%s); returning 0.0", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def style_match_score(generated: str, reference_style_label: str) -> float:
    """How well ``generated`` matches a structured reference style label.

    The score is a weighted average of three sub-scores — formality,
    verbosity (length bucket), and emotionality (tone bucket). Each
    sub-score is ``1 - |observed - target|`` clamped to ``[0, 1]``. Axes
    that the label is silent about contribute a neutral ``0.5``.

    Args:
        generated: The system's free-text response.
        reference_style_label: A label such as ``"warm_casual_short"``.

    Returns:
        A float in ``[0, 1]``; ``1.0`` is a perfect match.
    """
    parsed = _parse_style_label(reference_style_label)

    # -- Formality ---------------------------------------------------------
    observed_formality = _formality_score(generated)
    formality_target_by_bucket: dict[str, float] = {
        "casual": 0.2,
        "neutral": 0.5,
        "formal": 0.9,
        "": 0.5,
    }
    target_formality = formality_target_by_bucket[parsed["formality"]]
    formality_sub = 1.0 - abs(observed_formality - target_formality)

    # -- Length (verbosity proxy) -----------------------------------------
    tokens = _words(generated)
    n_tokens = len(tokens)
    if parsed["length"] == "short":
        length_sub = 1.0 if n_tokens <= 40 else max(0.0, 1.0 - (n_tokens - 40) / 80.0)
    elif parsed["length"] == "medium":
        if 40 <= n_tokens <= 120:
            length_sub = 1.0
        elif n_tokens < 40:
            length_sub = max(0.0, n_tokens / 40.0)
        else:  # n_tokens > 120
            length_sub = max(0.0, 1.0 - (n_tokens - 120) / 120.0)
    elif parsed["length"] == "long":
        length_sub = min(1.0, n_tokens / 120.0)
    else:
        length_sub = 0.5

    # -- Tone / emotionality ----------------------------------------------
    valence = _sentiment_valence(generated)
    tone_target: float
    if parsed["tone"] == "warm":
        tone_target = 0.5  # target positive valence
    elif parsed["tone"] == "objective":
        tone_target = 0.0  # target neutral valence
    elif parsed["tone"] == "reserved":
        tone_target = -0.1
    else:
        tone_target = 0.0
    # Valence is in [-1, 1]; normalise distance by 2 so sub-score in [0, 1].
    tone_sub = max(0.0, 1.0 - abs(valence - tone_target) / 2.0)

    # Equal-weight average.
    return float(max(0.0, min(1.0, (formality_sub + length_sub + tone_sub) / 3.0)))


def cognitive_load_fidelity(generated: str, target_load: float) -> float:
    """Whether the Flesch-Kincaid grade matches the target cognitive load.

    The benchmark maps ``target_load`` in ``[0, 1]`` to a Flesch-Kincaid
    grade range:

    * ``target_load <= 0.33``  → grades 3–6  (simple prose).
    * ``0.33 < target_load <= 0.66`` → grades 6–10 (everyday prose).
    * ``target_load > 0.66``   → grades 10–14 (technical prose).

    The sub-score is ``1`` if ``generated``'s FK grade falls in the target
    range, and decays linearly (by one-grade increments) outside it.

    Args:
        generated: The system's free-text response.
        target_load: A number in ``[0, 1]`` where ``0`` is simplest and ``1``
            is most complex.

    Returns:
        A float in ``[0, 1]``.

    Raises:
        ValueError: If ``target_load`` is not a finite number in ``[0, 1]``
            (with a small slack).
    """
    if target_load < -0.1 or target_load > 1.1:
        raise ValueError(
            f"target_load must be in [0, 1], got {target_load}."
        )
    target_load = max(0.0, min(1.0, float(target_load)))

    grade = _flesch_kincaid(generated)
    if target_load <= 0.33:
        lo, hi = 3.0, 6.0
    elif target_load <= 0.66:
        lo, hi = 6.0, 10.0
    else:
        lo, hi = 10.0, 14.0

    if lo <= grade <= hi:
        return 1.0
    distance = (lo - grade) if grade < lo else (grade - hi)
    # One-grade decay; distance of 3 grades -> 0.25, 4 grades -> 0.0.
    return float(max(0.0, 1.0 - distance / 4.0))


def accessibility_mode_appropriateness(
    generated: str, target_accessibility: float
) -> float:
    """Three-axis accessibility-mode check.

    When ``target_accessibility > 0.7`` (the mode is explicitly *on*), the
    response should:

    1. use predominantly short sentences (<= 15 words each);
    2. avoid idioms from a small curated list;
    3. include at least one yes/no (binary-choice) question to confirm.

    When the mode is *off*, the score is fixed at ``1.0`` — the benchmark
    does not penalise a system for producing rich language when
    accessibility was not requested.

    Args:
        generated: The system's free-text response.
        target_accessibility: Target accessibility value in ``[0, 1]``.

    Returns:
        A float in ``[0, 1]``.

    Raises:
        ValueError: If ``target_accessibility`` is outside ``[0, 1]``.
    """
    if target_accessibility < -0.1 or target_accessibility > 1.1:
        raise ValueError(
            f"target_accessibility must be in [0, 1], got {target_accessibility}."
        )
    target_accessibility = max(0.0, min(1.0, float(target_accessibility)))

    if target_accessibility <= 0.7:
        return 1.0

    sentences = _sentences(generated)
    if not sentences:
        # An empty response cannot honour any accessibility axis.
        return 0.0

    # -- 1. Short-sentence ratio ------------------------------------------
    short_count = sum(1 for s in sentences if len(_words(s)) <= 15)
    short_ratio = short_count / len(sentences)
    short_sub = min(1.0, short_ratio)

    # -- 2. Absence of idioms ---------------------------------------------
    lowered = generated.lower()
    idiom_hits = sum(1 for phrase in _IDIOM_PHRASES if phrase in lowered)
    idiom_sub = 1.0 if idiom_hits == 0 else max(0.0, 1.0 - idiom_hits / 3.0)

    # -- 3. Yes/no question marker ----------------------------------------
    yn_sub = 1.0 if _has_yes_no_question(generated) else 0.0

    return float(max(0.0, min(1.0, (short_sub + idiom_sub + yn_sub) / 3.0)))


def _has_yes_no_question(text: str) -> bool:
    """Heuristic — does ``text`` contain a yes/no-style question?

    Yes/no question: a sentence ending in ``?`` that starts (after the first
    capital/lowercase token) with a copula or auxiliary verb (``is``,
    ``are``, ``do``, ``does``, ``can``, ``should``, ``will``, ``would``,
    ``have``, ``has``, ``did``, ``may``, ``shall``).

    Args:
        text: Free text.

    Returns:
        ``True`` if at least one candidate yes/no question is detected.
    """
    yn_starters: frozenset[str] = frozenset(
        {
            "is",
            "are",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "will",
            "would",
            "have",
            "has",
            "had",
            "may",
            "might",
            "shall",
            "am",
            "were",
            "was",
        }
    )
    for s in _sentences(text):
        if not s.endswith("?"):
            continue
        words = _words(s)
        if words and words[0] in yn_starters:
            return True
    return False


def preference_rate(
    submissions: Sequence[BenchmarkSubmission],
    reference: Sequence[BenchmarkRecord],
) -> float:
    """Mean held-out human-preference score for matching submissions.

    Only records on the ``held_out_human`` split have a
    ``human_preference_score``. This metric is the average of those scores,
    taken over every submission whose ``record_id`` matches a record with a
    non-``None`` score.

    Args:
        submissions: Candidate submissions.
        reference: The gold record set (any split may be supplied; records
            without a human-preference score are silently skipped).

    Returns:
        A float in ``[0, 1]``; ``0.0`` when no overlap exists.
    """
    by_id: dict[str, BenchmarkRecord] = {r.record_id: r for r in reference}
    values: list[float] = []
    for sub in submissions:
        rec = by_id.get(sub.record_id)
        if rec is None or rec.human_preference_score is None:
            continue
        values.append(float(rec.human_preference_score))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def runtime_budget_compliance(
    submissions: Sequence[BenchmarkSubmission],
    budget_p95_ms: float = 200.0,
) -> float:
    """Fraction of submissions whose ``runtime_ms_p95`` is under the budget.

    Args:
        submissions: Candidate submissions.
        budget_p95_ms: Per-record p95 latency budget in milliseconds
            (default ``200.0`` — the Batch A H3 latency target).

    Returns:
        A float in ``[0, 1]``; ``1.0`` means every submission met the
        budget. Returns ``1.0`` on an empty input (vacuously true).

    Raises:
        ValueError: If ``budget_p95_ms`` is not strictly positive.
    """
    if budget_p95_ms <= 0.0:
        raise ValueError(
            f"budget_p95_ms must be > 0, got {budget_p95_ms}."
        )
    if not submissions:
        return 1.0
    compliant = sum(1 for s in submissions if s.runtime_ms_p95 <= budget_p95_ms)
    return float(compliant / len(submissions))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


DEFAULT_METRIC_WEIGHTS: dict[str, float] = {
    "style_match": 0.35,
    "cognitive_load_fidelity": 0.25,
    "accessibility_appropriateness": 0.15,
    "preference_rate": 0.15,
    "runtime_budget_compliance": 0.10,
}
"""Canonical default weights used for leaderboard ranking.

These weights are chosen to reflect the benchmark's emphasis: *style match*
is the core responsiveness signal; *preference* is held-out human validation;
*runtime* acts as a small penalty on systems that fail the latency target.
"""


def aggregate_score(
    metrics: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted average of per-metric scalars, clamped to ``[0, 1]``.

    Args:
        metrics: Mapping of metric name -> score in ``[0, 1]``. Metrics not
            covered by ``weights`` are ignored; missing metrics named by
            ``weights`` contribute ``0.0`` (they are penalised, not dropped).
        weights: Mapping of metric name -> weight. Weights are renormalised
            so they sum to ``1`` before aggregation. Defaults to
            :data:`DEFAULT_METRIC_WEIGHTS`.

    Returns:
        A float in ``[0, 1]``.

    Raises:
        ValueError: If ``weights`` is empty or all zero.
    """
    w = dict(weights) if weights is not None else dict(DEFAULT_METRIC_WEIGHTS)
    total_w = sum(max(0.0, float(v)) for v in w.values())
    if total_w <= 0.0:
        raise ValueError("aggregate_score: weights must contain a positive value.")
    acc = 0.0
    for name, weight in w.items():
        value = float(metrics.get(name, 0.0))
        # Clamp each metric to [0, 1] defensively.
        value = max(0.0, min(1.0, value))
        acc += max(0.0, float(weight)) * value
    return float(max(0.0, min(1.0, acc / total_w)))


def compute_all_metrics(
    submissions: Sequence[BenchmarkSubmission],
    records: Sequence[BenchmarkRecord],
    *,
    budget_p95_ms: float = 200.0,
) -> dict[str, float]:
    """Compute the full metric bundle for a submission set.

    Args:
        submissions: The candidate submissions.
        records: The gold record set.
        budget_p95_ms: Latency budget forwarded to
            :func:`runtime_budget_compliance`.

    Returns:
        A dict keyed by the metric names used in :data:`DEFAULT_METRIC_WEIGHTS`.
    """
    by_id: dict[str, BenchmarkRecord] = {r.record_id: r for r in records}
    style_scores: list[float] = []
    load_scores: list[float] = []
    access_scores: list[float] = []
    for sub in submissions:
        rec = by_id.get(sub.record_id)
        if rec is None:
            continue
        style_scores.append(
            style_match_score(sub.generated_text, rec.reference_style_label)
        )
        load_scores.append(
            cognitive_load_fidelity(
                sub.generated_text, rec.target_adaptation_vector[0]
            )
        )
        access_scores.append(
            accessibility_mode_appropriateness(
                sub.generated_text, rec.target_adaptation_vector[6]
            )
        )

    def _mean(xs: Iterable[float]) -> float:
        xs_list = list(xs)
        return float(sum(xs_list) / len(xs_list)) if xs_list else 0.0

    return {
        "style_match": _mean(style_scores),
        "cognitive_load_fidelity": _mean(load_scores),
        "accessibility_appropriateness": _mean(access_scores),
        "preference_rate": preference_rate(submissions, records),
        "runtime_budget_compliance": runtime_budget_compliance(
            submissions, budget_p95_ms=budget_p95_ms
        ),
    }
