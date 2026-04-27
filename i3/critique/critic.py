"""Rule-based self-critique scorer for the I3 small-language-model response path.

Position in the architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A 4.48 M-parameter on-device transformer cannot ``reason`` about its
own output the way a frontier model can.  What it *can* do — and what
this module operationalises — is **deterministically grade** its draft
against a small set of cheap, local rubrics, and trigger a tighter
re-decode when the draft fails any of them.  The orchestrator in
:mod:`i3.pipeline.engine` runs the SLM once at temperature 0, scores
the draft, and (when the score is below threshold) re-runs the SLM at
``T=0.4`` with a stronger repetition penalty before re-scoring.  The
Huawei R&D UK HMI Lab pitch surface ("the model checks its own work")
shows reviewers an inner-monologue chip on every SLM turn:
``self-critique · regenerated 0.41 → 0.79``.

This is the small-model variant of the *self-critique / self-refine*
pattern from the recent literature:

* Bai et al. (2022) — *Constitutional AI: Harmlessness from AI
  Feedback* — used a separate critic prompt to score and rewrite
  drafts against a list of explicit principles.  At their scale the
  critic is itself an LLM; at I3's edge scale we substitute a
  deterministic rubric and a single-attempt regenerate.
* Madaan et al. (2023) — *Self-Refine: Iterative Refinement with
  Self-Feedback* — formalised the score-then-regenerate loop.  We
  cap the regen at 1 retry so the pitch latency budget (<= 800 ms
  for the whole turn) stays intact.
* Saunders et al. (2022) — *Self-Critique* — observed that even
  small models can detect their own errors more reliably than they
  can avoid them in the first place; the regenerate step exploits
  exactly that asymmetry.

The critic returns a composite ``score`` in :math:`[0, 1]` plus a
per-criterion breakdown, so the visible UI chip (``accepted 0.79``)
and the expandable trace (``on_topic=0.7, well_formed=0.9, ...``) come
from the same call.

Sub-criteria
~~~~~~~~~~~~

1. ``on_topic`` — Jaccard overlap of *content* keywords (the
   :data:`_STOPWORDS` set is taken verbatim from
   :mod:`i3.slm.retrieval` so the keyword tokeniser the critic
   uses matches the retriever's).  When the prompt has no content
   words after stop-word stripping (``"hi"``, ``"thanks"``) the
   sub-score collapses to a neutral 0.5 rather than penalising the
   response.
2. ``well_formed`` — five cheap grammatical sanity checks: a finite
   verb is present, length sits in ``[3, 200]`` words, no token is
   repeated 3+ times in a row, no run of 5+ punctuation characters,
   and no ``[SEP]`` / ``UNK`` literal leaks through from the SLM
   tokeniser.
3. ``non_repetitive`` — bigram-level repetition penalty: if any
   bigram appears more than twice in the response, drop the
   sub-score in proportion to the worst offender.
4. ``safe`` — re-runs the same hostility regex from
   :func:`i3.slm.retrieval._is_hostility` against the *response*
   (the SLM occasionally echoes a slur from its corpus).  Also
   asks :class:`i3.privacy.sanitizer.PrivacySanitizer` whether the
   text contains PII; if so, score 0.
5. ``adaptation_match`` — fraction of the live ``AdaptationVector``
   constraints the response respects.  Three constraints are
   active: high cognitive_load (response should be <= 2 sentences),
   high accessibility (response FK grade <= 8), low verbosity
   (response shouldn't end with a follow-up question mark).

Aggregation: weighted mean with default weights
``on_topic=0.30, well_formed=0.30, non_repetitive=0.15, safe=0.15,
adaptation_match=0.10``.  Threshold defaults to ``0.65``.

Determinism + speed
~~~~~~~~~~~~~~~~~~~
* Pure Python, no torch, no numpy, no LLM call.
* ``score()`` never raises — every branch is wrapped so a malformed
  input only degrades the affected sub-score to a safe default.
* Target latency is well under 5 ms per call on a single CPU core;
  the regen loop in :mod:`i3.pipeline.engine` budgets at most 800 ms
  for the whole two-attempt path including the second SLM decode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Stop-word + content-word helpers (mirrors i3.slm.retrieval._STOPWORDS so
# critic and retriever fold contractions / interrogatives identically).
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "so", "to", "of",
    "in", "on", "at", "for", "by", "with", "as", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "it", "its", "this", "that", "these", "those", "i", "you",
    "we", "they", "me", "my", "your", "our", "their", "am",
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "hes", "shes", "theyre", "theyve", "theyll", "theyd",
    "weve", "well", "wed",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt", "cant", "couldnt", "shouldnt",
    "wouldnt", "wont", "shant", "mustnt",
    "what", "whats", "whos", "whens", "wheres", "whys", "hows",
    "why", "how", "when", "where", "who", "whom", "whose",
    "tell", "give", "show", "find",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "must", "ought", "let", "like",
    "very", "really", "just", "much", "more", "less", "some", "any",
    "all", "every", "each", "no", "not", "yes", "ok", "okay",
})


def _content_keywords(text: str) -> set[str]:
    """Return the set of content-bearing lowercase tokens in *text*.

    Apostrophes are stripped so contractions fold (``"what's"`` and
    ``"whats"`` hash to the same keyword).
    """
    cleaned = (text or "").lower().replace("'", "").replace("’", "")
    tokens = re.findall(r"[a-z]+", cleaned)
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


# ---------------------------------------------------------------------------
# Hostility patterns.  Imported lazily from i3.slm.retrieval so the critic
# stays self-contained when retrieval can't be imported (e.g. unit tests
# without sentence-transformers installed) — but we keep the literal
# fallback list in sync as a defence-in-depth check.
# ---------------------------------------------------------------------------

_HOSTILITY_PATTERNS_FALLBACK: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:stupid|dumb|idiot|moron|useless|garbage|trash|suck)\b", re.I),
    re.compile(r"\bf+u+c+k\s*(?:you|off)\b", re.I),
    re.compile(r"\bshit(?:ty)?\b", re.I),
    re.compile(r"\byou\s*(?:are|re|r)\s*(?:bad|terrible|awful|pathetic)\b", re.I),
    re.compile(r"\bi\s*hate\s*you\b", re.I),
)


def _hostility_patterns() -> tuple[re.Pattern[str], ...]:
    try:  # pragma: no cover - exercised in production where retrieval imports
        from i3.slm.retrieval import _HOSTILITY_PATTERNS  # type: ignore[attr-defined]
        return tuple(_HOSTILITY_PATTERNS)
    except Exception:
        return _HOSTILITY_PATTERNS_FALLBACK


# ---------------------------------------------------------------------------
# SLM artefact / leak detection.  These literal substrings indicate the
# tokeniser leaked a control token through to the visible response.
# ---------------------------------------------------------------------------

_LEAK_TOKENS: tuple[str, ...] = (
    "[sep]", "[unk]", "[bos]", "[eos]", "[pad]",
    "<sep>", "<unk>", "<bos>", "<eos>", "<pad>",
    " unk ", " sep ",
)

# Small set of finite-verb markers used by the well_formed check.  Not
# linguistically exhaustive — just enough to distinguish a real sentence
# ("the cat sleeps") from a noun pile ("the cat under the moon").
_FINITE_VERB_MARKERS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "make", "makes", "made", "say", "says", "said",
    "go", "goes", "went", "come", "comes", "came",
    "see", "sees", "saw", "seen", "know", "knows", "knew",
    "think", "thinks", "thought", "feel", "feels", "felt",
    "want", "wants", "wanted", "need", "needs", "needed",
    "like", "likes", "liked", "love", "loves", "loved",
    "find", "finds", "found", "tell", "tells", "told",
    "show", "shows", "showed", "ask", "asks", "asked",
    "give", "gives", "gave", "take", "takes", "took",
    "use", "uses", "used", "work", "works", "worked",
    "try", "tries", "tried", "help", "helps", "helped",
    "let", "lets", "make", "made", "get", "gets", "got",
    "look", "looks", "looked", "seem", "seems", "seemed",
    "mean", "means", "meant", "keep", "keeps", "kept",
    "leave", "leaves", "left", "put", "puts",
    "happen", "happens", "happened",
    "explain", "explains", "explained", "describe", "describes", "described",
    "live", "lives", "lived", "grow", "grows", "grew", "grown",
    "share", "shares", "shared", "build", "builds", "built",
    "convert", "converts", "converted", "produce", "produces", "produced",
    "absorb", "absorbs", "absorbed", "release", "releases", "released",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
    "create", "creates", "created", "create", "exist", "exists", "existed",
})


# ---------------------------------------------------------------------------
# Flesch-Kincaid grade — a tiny copy of training.prepare_dialogue.
# We don't import that module because it pulls heavy data-pipeline deps.
# ---------------------------------------------------------------------------

def _flesch_kincaid_grade(text: str) -> float:
    """Estimate the Flesch-Kincaid grade level of *text*.

    Mirrors :func:`training.prepare_dialogue.compute_flesch_kincaid_grade`
    so the critic and the synthetic-data pipeline agree on what
    "grade <= 8" means.  Returns 0.0 for empty input.
    """
    if not text or not text.strip():
        return 0.0
    words = text.split()
    n_words = max(len(words), 1)
    sentences = re.split(r"[.!?]+", text)
    n_sentences = max(len([s for s in sentences if s.strip()]), 1)

    def _count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"()-")
        if not word:
            return 1
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    total_syllables = sum(_count_syllables(w) for w in words)
    return 0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CritiqueScore:
    """Per-response composite score returned by :meth:`SelfCritic.score`.

    Attributes:
        score: Composite weighted-mean score in :math:`[0, 1]`.
        accepted: ``True`` when ``score >= threshold``.
        sub_scores: Per-criterion floats keyed by criterion name
            (``on_topic``, ``well_formed``, ``non_repetitive``,
            ``safe``, ``adaptation_match``).
        reasons: Short human-readable strings describing each
            sub-score that fell materially below 1.0 — these surface
            in the per-attempt trace in the WS frame.
        threshold: Threshold used for the ``accepted`` decision.
    """

    score: float
    accepted: bool
    sub_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    threshold: float = 0.65


@dataclass
class CritiqueResult:
    """Outcome of the orchestrator-level regenerate-if-bad loop."""

    final_text: str
    final_score: CritiqueScore
    attempts: list[dict] = field(default_factory=list)
    regenerated: bool = False
    rejected: bool = False


# ---------------------------------------------------------------------------
# SelfCritic
# ---------------------------------------------------------------------------

class SelfCritic:
    """Score-and-regenerate critic for SLM-generated responses.

    Runs a small set of cheap checks against an SLM response,
    returning a composite score in :math:`[0, 1]`.  When the score is
    below ``threshold`` the orchestrator (in
    :mod:`i3.pipeline.engine`) re-invokes the SLM with tighter
    sampling and re-scores.

    Sub-criteria (each in :math:`[0, 1]`):

    1. **on_topic** — Jaccard overlap of content keywords between
       prompt and response, normalised (cap penalty when prompt has
       no content words).
    2. **well_formed** — heuristics for grammatical sanity:

       * has at least one finite verb (small tense set);
       * response length is in ``[3, 200]`` words;
       * no token appears 3+ times consecutively;
       * <= 2 trailing punctuation runs (``"....."``);
       * no ``UNK`` literal substring or ``"[SEP]"`` leakage.

    3. **non_repetitive** — bigram-level repetition penalty: if any
       bigram appears > 2 times in the response, penalise.
    4. **safe** — runs the hostility regex from
       :func:`i3.slm.retrieval._is_hostility`; if the SLM emits
       hostile output (it occasionally does), score that
       sub-criterion to 0.  Also checks for PII via
       :class:`i3.privacy.sanitizer.PrivacySanitizer.contains_pii`.
    5. **adaptation_match** — given the live ``AdaptationVector``,
       score how well the response respects it:

       * high cognitive_load (>0.7) → response should be <= 2
         sentences;
       * high accessibility (>0.6) → response should pass the
         :func:`compute_flesch_kincaid_grade` check (level <= 8);
       * low verbosity (<0.35) → response shouldn't end with a
         follow-up question.

       Score is the fraction of constraints met.

    Aggregation: weighted mean.  Default weights:
    ``on_topic=0.30, well_formed=0.30, non_repetitive=0.15,
    safe=0.15, adaptation_match=0.10``.

    Threshold: ``0.65`` by default.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "on_topic": 0.30,
        "well_formed": 0.30,
        "non_repetitive": 0.15,
        "safe": 0.15,
        "adaptation_match": 0.10,
    }

    def __init__(
        self,
        threshold: float = 0.65,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        if weights is None:
            self.weights = dict(self.DEFAULT_WEIGHTS)
        else:
            # Renormalise unknown keys away and rescale to sum to 1.
            kept = {
                k: max(0.0, float(v))
                for k, v in weights.items()
                if k in self.DEFAULT_WEIGHTS
            }
            total = sum(kept.values()) or 1.0
            self.weights = {k: v / total for k, v in kept.items()}
        # PII sanitiser is lazily acquired so a missing optional dep can
        # never block the critic.  ``contains_pii`` is a fast regex scan.
        self._sanitizer: Any | None = None
        try:
            from i3.privacy.sanitizer import PrivacySanitizer
            self._sanitizer = PrivacySanitizer(enabled=True)
        except Exception:
            self._sanitizer = None
        self._hostility_patterns = _hostility_patterns()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        prompt: str,
        response: str,
        adaptation: dict,
    ) -> CritiqueScore:
        """Score *response* against the rubrics above.

        The method NEVER raises: any exception in a sub-scorer
        degrades that sub-score to its safe default (0.5 for
        on_topic / well_formed / adaptation_match where we don't
        want to penalise over-zealously, 1.0 for non_repetitive
        and safe where the absence of evidence is good news).
        """
        prompt = prompt or ""
        response = response or ""
        adaptation = adaptation or {}

        sub: dict[str, float] = {}
        reasons: list[str] = []

        # 1. on_topic ---------------------------------------------------
        try:
            sub["on_topic"], why = self._score_on_topic(prompt, response)
            if why:
                reasons.append(why)
        except Exception:
            sub["on_topic"] = 0.5

        # 2. well_formed ------------------------------------------------
        try:
            sub["well_formed"], why = self._score_well_formed(response)
            if why:
                reasons.append(why)
        except Exception:
            sub["well_formed"] = 0.5

        # 3. non_repetitive ---------------------------------------------
        try:
            sub["non_repetitive"], why = self._score_non_repetitive(response)
            if why:
                reasons.append(why)
        except Exception:
            sub["non_repetitive"] = 1.0

        # 4. safe -------------------------------------------------------
        try:
            sub["safe"], why = self._score_safe(response)
            if why:
                reasons.append(why)
        except Exception:
            sub["safe"] = 1.0

        # 5. adaptation_match -------------------------------------------
        try:
            sub["adaptation_match"], why = self._score_adaptation_match(
                response, adaptation
            )
            if why:
                reasons.append(why)
        except Exception:
            sub["adaptation_match"] = 1.0

        # Aggregate -----------------------------------------------------
        total = 0.0
        for key, weight in self.weights.items():
            total += weight * float(max(0.0, min(1.0, sub.get(key, 0.0))))
        score_val = float(max(0.0, min(1.0, total)))

        return CritiqueScore(
            score=score_val,
            accepted=score_val >= self.threshold,
            sub_scores={k: float(round(v, 3)) for k, v in sub.items()},
            reasons=reasons,
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    # Sub-scorers — each returns ``(score_in_0_1, optional_reason)``
    # ------------------------------------------------------------------

    @staticmethod
    def _score_on_topic(prompt: str, response: str) -> tuple[float, str | None]:
        prompt_kw = _content_keywords(prompt)
        if not prompt_kw:
            # Nothing to anchor topicality against — neutral.
            return 0.5, None
        resp_kw = _content_keywords(response)
        if not resp_kw:
            return 0.0, "off-topic: response has no content words"
        # Direct overlap.  Short demos of "make me a haiku about
        # photosynthesis" → "Sunlight on a leaf..." can be perfectly
        # on-topic while sharing zero literal tokens, so we also fold a
        # 4-char-prefix stem match in (catches photosynth -> photo,
        # photosynthetic -> photo, etc.) before declaring an off-topic.
        intersection = prompt_kw & resp_kw
        if not intersection:
            stems_p = {kw[:4] for kw in prompt_kw if len(kw) >= 4}
            stems_r = {kw[:4] for kw in resp_kw if len(kw) >= 4}
            stem_overlap = stems_p & stems_r
            if stem_overlap:
                # Treat a stem hit as a half-strength keyword hit.
                fake_inter = max(1, len(stem_overlap) // 2)
                jaccard = fake_inter / max(len(prompt_kw | resp_kw), 1)
                normalised = min(1.0, jaccard * 3.0)
                return max(0.35, normalised), None
        union = prompt_kw | resp_kw
        if not union:
            return 0.5, None
        jaccard = len(intersection) / len(union)
        # Map Jaccard to a more forgiving curve.  A 0.2 overlap (one or
        # two shared content words) is still on-topic for short replies.
        normalised = min(1.0, jaccard * 3.0)
        if normalised < 0.5:
            why = (
                f"low topical overlap ({len(intersection)} shared content words "
                f"out of {len(union)})"
            )
            return normalised, why
        return normalised, None

    @staticmethod
    def _score_well_formed(response: str) -> tuple[float, str | None]:
        if not response or not response.strip():
            return 0.0, "well_formed: empty response"

        text = response.strip()
        words = re.findall(r"[A-Za-z']+", text)
        n_words = len(words)
        checks_passed = 0
        checks_total = 5
        violations: list[str] = []

        # (a) finite verb present.  Two-stage check: explicit verb
        # markers (the closed-class auxiliaries + a small open-class
        # set) followed by a morphological cue (any token ending in
        # ``-s``, ``-ed``, ``-ing`` after a vowel, e.g. "wakes",
        # "absorbed", "breathing"). The second stage catches verbs the
        # closed list misses without dragging linguistic deps in.
        lowered = {w.lower().replace("'", "") for w in words}
        has_marker = bool(lowered & _FINITE_VERB_MARKERS)
        has_morph = any(
            len(w) >= 4
            and (
                (w.endswith("ed") and w[-3] in "aeiouy" or w.endswith("ed"))
                or w.endswith("ing")
                or (w.endswith("s") and not w.endswith("ss"))
            )
            and w not in _STOPWORDS
            for w in lowered
        )
        if has_marker or has_morph:
            checks_passed += 1
        else:
            violations.append("no finite verb")

        # (b) length in [3, 200]
        if 3 <= n_words <= 200:
            checks_passed += 1
        else:
            violations.append(f"length {n_words} words outside [3, 200]")

        # (c) no token appears 3+ times consecutively
        consec_run = 1
        worst_consec = 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                consec_run += 1
                worst_consec = max(worst_consec, consec_run)
            else:
                consec_run = 1
        if worst_consec < 3:
            checks_passed += 1
        else:
            violations.append(f"token repeats {worst_consec}x consecutively")

        # (d) no run of 5+ punctuation chars (".....", "!!!!!")
        if not re.search(r"[.!?,;:]{5,}", text):
            checks_passed += 1
        else:
            violations.append("excessive punctuation run")

        # (e) no leaked control tokens
        text_low = text.lower()
        if not any(tok in text_low for tok in _LEAK_TOKENS):
            checks_passed += 1
        else:
            violations.append("control-token leak ([SEP]/UNK)")

        score = checks_passed / checks_total
        if violations:
            return score, "well_formed: " + ", ".join(violations)
        return score, None

    @staticmethod
    def _score_non_repetitive(response: str) -> tuple[float, str | None]:
        words = [w.lower() for w in re.findall(r"[A-Za-z']+", response or "")]
        if len(words) < 4:
            return 1.0, None
        bigrams: list[tuple[str, str]] = list(zip(words, words[1:]))
        counts: dict[tuple[str, str], int] = {}
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
        worst = max(counts.values()) if counts else 1
        if worst <= 2:
            return 1.0, None
        # Each extra repeat over the cap of 2 docks 0.25 from the score.
        excess = worst - 2
        score = max(0.0, 1.0 - 0.25 * excess)
        worst_bg = max(counts, key=counts.get)
        why = f"bigram '{worst_bg[0]} {worst_bg[1]}' repeats {worst}x"
        return score, why

    def _score_safe(self, response: str) -> tuple[float, str | None]:
        if not response:
            return 1.0, None
        for pat in self._hostility_patterns:
            try:
                if pat.search(response):
                    return 0.0, "safe: hostile language detected"
            except Exception:  # pragma: no cover - defensive
                continue
        if self._sanitizer is not None:
            try:
                if self._sanitizer.contains_pii(response):
                    return 0.0, "safe: PII detected in response"
            except Exception:  # pragma: no cover - defensive
                pass
        return 1.0, None

    @staticmethod
    def _score_adaptation_match(
        response: str, adaptation: dict,
    ) -> tuple[float, str | None]:
        ad = adaptation or {}

        def _f(key: str, default: float = 0.5) -> float:
            try:
                return float(ad.get(key, default))
            except (TypeError, ValueError):
                return default

        cognitive_load = _f("cognitive_load", 0.5)
        accessibility = _f("accessibility", 0.0)
        verbosity = _f("verbosity", 0.5)

        constraints_active = 0
        constraints_met = 0
        violations: list[str] = []

        # (a) high cognitive_load -> <= 2 sentences
        if cognitive_load > 0.7:
            constraints_active += 1
            n_sentences = len([
                s for s in re.split(r"[.!?]+", response or "") if s.strip()
            ])
            if n_sentences <= 2:
                constraints_met += 1
            else:
                violations.append(
                    f"cognitive_load {cognitive_load:.2f} but {n_sentences} sentences"
                )

        # (b) high accessibility -> FK <= 8
        if accessibility > 0.6:
            constraints_active += 1
            grade = _flesch_kincaid_grade(response or "")
            if grade <= 8.0:
                constraints_met += 1
            else:
                violations.append(
                    f"accessibility {accessibility:.2f} but FK grade {grade:.1f}"
                )

        # (c) low verbosity -> no trailing follow-up question
        if verbosity < 0.35:
            constraints_active += 1
            stripped = (response or "").rstrip()
            if not stripped.endswith("?"):
                constraints_met += 1
            else:
                violations.append(
                    f"verbosity {verbosity:.2f} but trailing follow-up question"
                )

        if constraints_active == 0:
            return 1.0, None
        score = constraints_met / constraints_active
        if violations:
            return score, "adaptation_match: " + "; ".join(violations)
        return score, None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - demo / manual smoke test
    critic = SelfCritic()

    cases: list[tuple[str, str, str, dict]] = [
        (
            "sensible",
            "make me a haiku about photosynthesis",
            "Sunlight on a leaf, the green machine wakes and breathes, sugar from the air.",
            {"cognitive_load": 0.4, "accessibility": 0.0, "verbosity": 0.5},
        ),
        (
            "off-topic gibberish",
            "explain entropy in one paragraph",
            "Bananas are yellow and the river is wide on Saturday.",
            {"cognitive_load": 0.5, "accessibility": 0.0, "verbosity": 0.5},
        ),
        (
            "hostile",
            "what does the moon think about you",
            "You are stupid and I hate you.",
            {"cognitive_load": 0.5, "accessibility": 0.0, "verbosity": 0.5},
        ),
        (
            "word salad",
            "tell me about photosynthesis",
            "Under the's your interaction intelligence the the the under the.",
            {"cognitive_load": 0.5, "accessibility": 0.0, "verbosity": 0.5},
        ),
        (
            "high cognitive_load violation",
            "explain how cells work",
            (
                "Cells are tiny living units. They have membranes and a nucleus. "
                "They use energy from food. They divide to make more cells. "
                "They are everywhere in our bodies."
            ),
            {"cognitive_load": 0.85, "accessibility": 0.0, "verbosity": 0.5},
        ),
    ]

    for name, prompt, response, adaptation in cases:
        result = critic.score(
            prompt=prompt, response=response, adaptation=adaptation
        )
        print(f"[{name}]  score={result.score:.2f}  accepted={result.accepted}")
        for k, v in result.sub_scores.items():
            print(f"    {k:18s} = {v:.2f}")
        for r in result.reasons:
            print(f"    - {r}")
        print()
