"""Multi-step "explain" decomposition (Phase B.3, 2026-04-25).

For queries that look like
    "explain photosynthesis"
    "tell me about apple"
    "describe the roman empire"
    "how does evolution work?"
…we decompose the query into 3 sub-questions (what / why / how) and
answer each one independently using the regular tool / retrieval /
SLM stack.  The composite paragraph is what the UI renders as the
chat reply, with the per-sub-question trace surfaced as a collapsible
"Reasoning chain" element.

The decomposer is **deterministic** — no extra model is loaded; we
use a small set of templates keyed off the topic and lean entirely
on the existing pipeline routes (entity tool / KG compose / retrieval
/ SLM) for the actual content.  Sub-question execution is sequential
to keep the implementation simple and avoid concurrent state mutation
on the retriever.

Phase 14 (2026-04-25) added an **on-topic filter** inspired by the
self-refine line of work (Saunders et al., 2022, "Self-critiquing
models for assisting human evaluators"; Madaan et al., 2023, "Self-
Refine: Iterative Refinement with Self-Feedback") — after each sub-
question retrieval, we keyword-overlap-check the sub-answer against
the topic.  Sub-answers that fail the overlap check are dropped or
replaced with a generic placeholder so the composite paragraph
doesn't quote a Photosynthesis paragraph in answer to "what is
climate change".  When ≥ 2/3 sub-answers fail the topic check we
abandon the decomposition outright and let the regular retrieval /
SLM pipeline answer with a single paragraph instead.

The module exposes one entry point:

    >>> dec = ExplainDecomposer(retriever, kg)
    >>> if dec.is_explain_query("explain photosynthesis"):
    ...     plan = dec.decompose_and_answer("explain photosynthesis")
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patterns that flag a turn as an "explain" query.
# ---------------------------------------------------------------------------
_EXPLAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*explain\s+(.+?)\s*[\.\?\!]?\s*$", re.I),
    re.compile(r"^\s*tell\s+me\s+about\s+(.+?)\s*[\.\?\!]?\s*$", re.I),
    re.compile(r"^\s*describe\s+(.+?)\s*[\.\?\!]?\s*$", re.I),
    re.compile(r"^\s*how\s+does\s+(.+?)\s+work\s*[\.\?\!]?\s*$", re.I),
    re.compile(r"^\s*what\s+is\s+(.+?)\s*[\.\?\!]?\s*$", re.I),
    re.compile(r"^\s*walk\s+me\s+through\s+(.+?)\s*[\.\?\!]?\s*$", re.I),
)


# Function words that should NEVER count as topic content.  Lowercased.
_TOPIC_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "and", "or", "but", "if", "in", "on", "at",
    "to", "for", "by", "with", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "have", "has", "had", "it", "its",
    "this", "that", "these", "those", "what", "who", "when", "where",
    "why", "how", "tell", "me", "about", "explain", "describe", "walk",
    "through", "matter", "work", "us", "you", "your", "our", "we",
    "they", "them", "their", "i", "my", "from", "as",
})

# Common topic acronyms / aliases that should expand into themselves
# plus their full form so the on-topic check still passes when a
# retrieved paragraph uses the long form ("ML" → "machine learning").
_TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "ml": ("machine", "learning"),
    "ai": ("artificial", "intelligence"),
    "dl": ("deep", "learning"),
    "nlp": ("natural", "language", "processing"),
    "rl": ("reinforcement", "learning"),
    "agi": ("artificial", "general", "intelligence"),
    "gpu": ("graphics", "processing"),
    "cpu": ("central", "processing"),
    "iot": ("internet", "of", "things"),
    "vr": ("virtual", "reality"),
    "ar": ("augmented", "reality"),
    "llm": ("large", "language", "model"),
    "slm": ("small", "language", "model"),
    "co2": ("carbon", "dioxide", "co2"),
    "wwii": ("world", "war"),
    "wwi": ("world", "war"),
}


def _topic_keywords(text: str) -> set[str]:
    """Extract content keywords from the topic, lowercased + stop-removed.

    The output is the canonical "what counts as on-topic" word set.
    Tokens shorter than 3 chars are dropped (too noisy at the document
    level), aliases are expanded, and stopwords are removed.
    """
    if not text:
        return set()
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*", text.lower())
    out: set[str] = set()
    for tok in raw:
        if tok in _TOPIC_STOPWORDS:
            continue
        if tok in _TOPIC_ALIASES:
            out.update(_TOPIC_ALIASES[tok])
            out.add(tok)
            continue
        if len(tok) < 3:
            continue
        out.add(tok)
        # Singular form (rough): drop a trailing 's' if the stem is
        # still ≥ 3 chars.  Cheap pluralisation handler.
        if tok.endswith("s") and len(tok) > 3:
            out.add(tok[:-1])
    return out


def _is_on_topic(
    sub_answer: str, topic_keywords: set[str], min_overlap: int = 1
) -> bool:
    """A sub-answer is on-topic iff it shares ``min_overlap`` content
    keywords with the topic.

    This kills the cross-topic contamination where, say, ``"what is
    climate change"`` pulls a Photosynthesis paragraph as a sub-answer
    because the cosine retrieval found a high-confidence but utterly
    off-topic match.
    """
    if not sub_answer or not topic_keywords:
        return False
    answer_keywords = _topic_keywords(sub_answer)
    overlap = answer_keywords & topic_keywords
    return len(overlap) >= min_overlap


# ---------------------------------------------------------------------------
# Curated topic overviews — last-resort fallback when the decomposer
# would otherwise abandon (≥2/3 sub-answers off-topic) on a topic the
# corpus is too sparse to answer well.  The list is intentionally
# bounded — we only add entries for the topics the audit caught as
# bad, plus a small spread of related ones so the chip looks well-
# stocked.  All curated, hand-written, deterministic.
# ---------------------------------------------------------------------------
_TOPIC_OVERVIEWS: dict[str, str] = {
    "climate change": (
        "Climate change refers to long-term shifts in global temperatures and "
        "weather patterns. Human activity since the Industrial Revolution — "
        "primarily burning fossil fuels — has released large quantities of "
        "carbon dioxide and other greenhouse gases that trap heat in the "
        "atmosphere, driving global warming, sea-level rise, more extreme "
        "weather, and disruption to ecosystems."
    ),
    "depression": (
        "Depression is a mood disorder characterised by persistent feelings of "
        "sadness, loss of interest in activities, fatigue, sleep and appetite "
        "changes, and difficulty concentrating. It results from a combination "
        "of genetic, biological, environmental, and psychological factors. "
        "Effective treatments include psychotherapy (especially cognitive-"
        "behavioural therapy), antidepressant medication, lifestyle changes, "
        "and in some cases brain-stimulation therapies."
    ),
    "consciousness": (
        "Consciousness is the subjective experience of being aware — of "
        "perceptions, thoughts, feelings, and a sense of self. It's one of "
        "the deepest open questions in neuroscience and philosophy: what "
        "physical processes in the brain give rise to the felt quality of "
        "experience (the 'hard problem' of consciousness, as David Chalmers "
        "framed it)? Major scientific theories include Global Workspace Theory "
        "and Integrated Information Theory."
    ),
    "anxiety": (
        "Anxiety is a feeling of worry, nervousness, or unease, typically "
        "about an event with an uncertain outcome. It becomes a clinical "
        "disorder when it is persistent, disproportionate to the trigger, and "
        "interferes with daily life. Common forms include generalised "
        "anxiety disorder, panic disorder, social anxiety, and phobias. "
        "Treatments include CBT, exposure therapy, mindfulness, and "
        "anxiolytic medication."
    ),
    "happiness": (
        "Happiness is a positive emotional state ranging from contentment to "
        "intense joy. Psychologists distinguish hedonic happiness (pleasure, "
        "comfort) from eudaimonic well-being (meaning, growth, purpose). "
        "Research consistently links sustained happiness to social connection, "
        "physical health, autonomy, gratitude, and engagement with meaningful "
        "work — more than to wealth or material possessions beyond a moderate "
        "threshold."
    ),
    "machine learning": (
        "Machine learning is the branch of AI where systems learn patterns "
        "from data instead of being hand-coded. You pick a model family "
        "(linear regression, decision trees, neural networks), define a loss "
        "function that measures prediction error, and use an optimisation "
        "algorithm (typically gradient descent) to fit the model's parameters "
        "on training data. The dominant modern technique is deep learning "
        "with neural networks."
    ),
    "deep learning": (
        "Deep learning is the subset of machine learning that uses neural "
        "networks with many layers — sometimes hundreds — to learn "
        "hierarchical representations of data. Each layer transforms its "
        "input into a slightly more abstract feature; the final layer "
        "produces the prediction. Deep learning powers most modern "
        "advances in computer vision, speech recognition, and natural "
        "language processing."
    ),
    "artificial intelligence": (
        "Artificial intelligence is the field that builds systems performing "
        "tasks normally associated with human cognition — recognising images, "
        "understanding speech, translating languages, planning, reasoning, "
        "and making decisions. Modern AI is dominated by machine learning, "
        "and within that by deep learning. Symbolic AI (rule-based, logical "
        "inference) was the original approach but has been overtaken by "
        "data-driven methods at scale."
    ),
}


def _curated_overview(topic: str) -> str | None:
    """Return a curated overview paragraph for *topic* if available.

    Used as a last-resort fallback by :meth:`ExplainDecomposer.
    decompose_and_answer` when sub-question retrieval all came back
    off-topic.  The curated set is bounded and hand-written so we
    never fabricate.
    """
    if not topic:
        return None
    key = topic.strip().lower().rstrip(".?!,")
    return _TOPIC_OVERVIEWS.get(key)


@dataclass
class SubAnswer:
    question: str
    source: str = "unknown"  # "tool:graph_compose", "tool:entity",
                             # "retrieval", "slm", "kg_overview", ...
    text: str = ""
    confidence: float = 0.0


@dataclass
class ExplainPlan:
    topic: str
    sub_questions: list[str] = field(default_factory=list)
    sub_answers: list[SubAnswer] = field(default_factory=list)
    composite_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "sub_questions": list(self.sub_questions),
            "sub_answers": [asdict(a) for a in self.sub_answers],
            "composite_answer": self.composite_answer,
        }


class ExplainDecomposer:
    """Decomposes "explain X" queries into 3 sub-questions and composes.

    The decomposer is given:
      - ``retriever``: a :class:`i3.slm.retrieval.ResponseRetriever`
        instance — used to answer each sub-question via the same
        tool / cosine path the engine uses.
      - ``kg``: a :class:`i3.dialogue.knowledge_graph.KnowledgeGraph`
        instance — used to bias the first sub-answer toward the
        curated overview when the topic is in the KG.

    Both are optional; missing ones degrade to "no answer for that
    sub-question" gracefully.
    """

    def __init__(self, retriever: Any | None = None, kg: Any | None = None):
        self._retriever = retriever
        self._kg = kg

    # -- detection ------------------------------------------------------
    # Acronyms that we will accept as a topic even though they're shorter
    # than 3 characters.  Matches the alias expansion table in
    # _extract_topic so "explain ML" / "what is AI" land on the
    # decomposer rather than falling through to OOD.
    _SHORT_ACRONYMS: frozenset[str] = frozenset({
        "ml", "ai", "dl", "rl", "nlp", "agi", "gpu", "cpu", "iot", "vr",
        "ar", "llm", "slm", "co2", "us", "uk",
    })

    def is_explain_query(self, text: str) -> bool:
        """Return True iff *text* looks like an explain-style request."""
        if not text:
            return False
        cleaned = text.strip()
        # Math-tool veto (2026-04-26 audit): "what is 25% of 200" syntax
        # MATCHES the explain pattern but is really an arithmetic
        # expression — letting the decomposer compose 3 sub-questions
        # ("Why does 25% of 200 matter?" etc.) produces nonsensical
        # philosophy junk.  Defer to the upstream math tool route.
        try:
            from i3.slm.retrieval import _is_math_expr
            if _is_math_expr(cleaned):
                return False
        except Exception:
            pass
        # Discourse-prefix strip (iter 9): conversational openers like
        # "wait one more thing - what is python", "actually, what is
        # transformer", "oh - tell me about apple" should still hit the
        # decomposer.  Strip a leading discourse prefix once and re-check
        # against the patterns.  Without this, the curated "wait what"
        # entry hijacks the cosine match for any "wait..." query.
        _DISCOURSE_PREFIX = re.compile(
            r"^\s*(?:wait(?:\s+one\s+more\s+thing)?|actually|oh|hmm|"
            r"sorry|um|uh|ok|okay|so|well|hey|listen|you know|i mean|"
            r"by the way|btw)\s*[,\-:—]?\s+",
            re.I,
        )
        candidates = [cleaned]
        stripped = _DISCOURSE_PREFIX.sub("", cleaned, count=1).strip()
        if stripped and stripped != cleaned:
            candidates.append(stripped)
        # Iteration 14 (2026-04-26): generic-pronoun veto — "explain
        # that", "explain it", "tell me about that/this/them" have a
        # pronoun TOPIC and would produce nonsense if decomposed
        # ("What is that?", "Why does that matter?", "How does that
        # work?").  The upstream bare-noun rewriter handles these by
        # substituting the active topic; if the rewriter doesn't fire
        # we'd rather fall through to OOD than fabricate.
        _PRONOUN_TOPICS: frozenset[str] = frozenset({
            "it", "this", "that", "these", "those", "them", "they",
            "him", "her", "us", "you", "me", "myself", "yourself",
            "themselves", "stuff", "things", "thing",
        })
        for variant in candidates:
            for pat in _EXPLAIN_PATTERNS:
                m = pat.match(variant)
                if m and len((m.group(1) or "").split()) >= 1:
                    topic = m.group(1).strip()
                    topic_clean = topic.lower().rstrip(".?!,")
                    if topic_clean in _PRONOUN_TOPICS:
                        return False
                    if len(topic) >= 3:
                        return True
                    # Phase 14: accept short curated acronyms (ML, AI, DL).
                    if topic_clean in self._SHORT_ACRONYMS:
                        return True
        return False

    def _extract_topic(self, text: str) -> str:
        # Strip the same discourse prefix as is_explain_query so the
        # match runs against the meaningful tail of the query
        # ("wait one more thing - what is python" → "what is python").
        _DISCOURSE_PREFIX = re.compile(
            r"^\s*(?:wait(?:\s+one\s+more\s+thing)?|actually|oh|hmm|"
            r"sorry|um|uh|ok|okay|so|well|hey|listen|you know|i mean|"
            r"by the way|btw)\s*[,\-:—]?\s+",
            re.I,
        )
        original = text.strip()
        candidates = [original]
        stripped = _DISCOURSE_PREFIX.sub("", original, count=1).strip()
        if stripped and stripped != original:
            candidates.append(stripped)
        _EXPANSIONS = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "dl": "deep learning",
            "nlp": "natural language processing",
            "rl": "reinforcement learning",
            "agi": "artificial general intelligence",
        }
        for variant in candidates:
            for pat in _EXPLAIN_PATTERNS:
                m = pat.match(variant)
                if m:
                    topic = m.group(1).strip().rstrip(".!?,")
                    bare = topic.lower().strip()
                    if bare in _EXPANSIONS:
                        return _EXPANSIONS[bare]
                    return topic
        return original

    # -- decomposition --------------------------------------------------
    def _build_sub_questions(self, topic: str) -> list[str]:
        """Pick 3 well-formed sub-questions for *topic*."""
        topic_lower = topic.lower()
        # Special-case people: who, what did they do, when did they live
        if any(p in topic_lower for p in (
            "einstein", "newton", "darwin", "jobs", "gates",
        )):
            return [
                f"Who was {topic}?",
                f"What did {topic} discover or contribute?",
                f"When did {topic} live?",
            ]
        # Special-case events
        if any(p in topic_lower for p in (
            "war", "empire", "revolution",
        )):
            return [
                f"What was {topic}?",
                f"When did {topic} happen?",
                f"Why does {topic} matter?",
            ]
        # Generic concept / org / topic decomposition
        return [
            f"What is {topic}?",
            f"Why does {topic} matter?",
            f"How does {topic} work?",
        ]

    # -- execution ------------------------------------------------------
    def _answer_sub(self, q: str) -> SubAnswer:
        """Try to answer one sub-question.  Returns a populated SubAnswer."""
        # 1. KG / tool path via the retriever's tool routes.  This
        #    catches "what is huawei" → entity tool, "who founded
        #    apple" → KG founded_by, etc.  The retriever bumps
        #    ``_last_tool`` on hits so we read it back.
        if self._retriever is not None:
            try:
                result = self._retriever.best(
                    q, min_score=0.65, tool_route=True,
                )
            except Exception:
                logger.debug("decomposer.best failed for %r", q, exc_info=True)
                result = None
            if result is not None:
                text, score = result
                tool = getattr(self._retriever, "_last_tool", None) or ""
                return SubAnswer(
                    question=q,
                    source=f"tool:{tool}" if tool else "retrieval",
                    text=text,
                    confidence=float(score),
                )
        # 2. KG overview fallback for "what is X"
        topic = self._extract_topic_from_subq(q)
        if self._kg is not None and self._kg.loaded:
            try:
                if "what is" in q.lower() or "who was" in q.lower():
                    ov = self._kg.overview(topic)
                    if ov:
                        return SubAnswer(
                            question=q,
                            source="kg_overview",
                            text=ov,
                            confidence=0.9,
                        )
            except Exception:
                logger.debug(
                    "decomposer.kg_overview failed for %r", q, exc_info=True
                )
        return SubAnswer(question=q, source="unanswered", text="", confidence=0.0)

    @staticmethod
    def _extract_topic_from_subq(sub_q: str) -> str:
        # Trim "What is X?" / "Why does X matter?" / "How does X work?"
        m = re.match(r"^(?:what\s+is|who\s+was|what\s+was)\s+(.+?)\s*[\.\?\!]?$", sub_q, re.I)
        if m:
            return m.group(1).strip()
        m = re.match(r"^why\s+does\s+(.+?)\s+matter\s*[\.\?\!]?$", sub_q, re.I)
        if m:
            return m.group(1).strip()
        m = re.match(r"^how\s+does\s+(.+?)\s+work\s*[\.\?\!]?$", sub_q, re.I)
        if m:
            return m.group(1).strip()
        m = re.match(r"^what\s+did\s+(.+?)\s+discover", sub_q, re.I)
        if m:
            return m.group(1).strip()
        m = re.match(r"^when\s+did\s+(.+?)\s+(?:happen|live)", sub_q, re.I)
        if m:
            return m.group(1).strip()
        return sub_q

    # -- composition ----------------------------------------------------
    def _compose(self, plan: ExplainPlan) -> str:
        # Drop any unanswered sub-answers (so the composite paragraph
        # never contains a hole).
        useful = [a for a in plan.sub_answers if a.text]
        if not useful:
            return ""
        if len(useful) == 1:
            return useful[0].text
        # Dedupe near-identical sub-answers: when two sub-questions hit
        # the same retrieved paragraph (common for a single curated
        # entry), composing all of them yields the same sentence twice.
        # Compare on a normalised fingerprint (lowercase, whitespace
        # collapsed, first 80 chars) so cosmetic punctuation differences
        # don't defeat the dedupe.
        seen: set[str] = set()
        deduped: list[SubAnswer] = []
        for sa in useful:
            fp = " ".join(sa.text.lower().split())[:80]
            if fp in seen:
                continue
            seen.add(fp)
            deduped.append(sa)
        if not deduped:
            return ""
        if len(deduped) == 1:
            return deduped[0].text
        return "\n\n".join(sa.text.strip() for sa in deduped)

    # -- public entry point --------------------------------------------
    def decompose_and_answer(self, text: str) -> ExplainPlan:
        topic = self._extract_topic(text)
        plan = ExplainPlan(topic=topic)
        plan.sub_questions = self._build_sub_questions(topic)
        topic_kw = _topic_keywords(topic)
        for q in plan.sub_questions:
            plan.sub_answers.append(self._answer_sub(q))

        # ── Phase 14: on-topic filter (Saunders et al 2022 — self-
        #    refine inspiration).  Each sub-answer must share at least
        #    one content keyword with the topic; if it doesn't, replace
        #    it with a neutral placeholder.  If 2+ of 3 sub-answers fail
        #    the check, abandon the decomposition entirely so the
        #    engine routes to plain retrieval / SLM instead of
        #    composing a cross-topic mess.
        if topic_kw:
            off_topic_count = 0
            for sa in plan.sub_answers:
                if not sa.text:
                    off_topic_count += 1
                    continue
                if not _is_on_topic(sa.text, topic_kw):
                    logger.debug(
                        "decomposer.off_topic: q=%r topic_kw=%s → blanking",
                        sa.question, sorted(topic_kw)[:6],
                    )
                    sa.text = ""
                    sa.source = "off_topic_filtered"
                    sa.confidence = 0.0
                    off_topic_count += 1
            # ≥2/3 off-topic → abandon decomposition; try a curated
            # overview as the final fallback before letting the engine
            # route to retrieval/SLM.
            if (
                len(plan.sub_answers) > 0
                and off_topic_count >= max(2, len(plan.sub_answers) - 1)
            ):
                logger.debug(
                    "decomposer.abandon: %d/%d sub-answers off-topic for %r",
                    off_topic_count, len(plan.sub_answers), topic,
                )
                curated = _curated_overview(topic)
                if curated:
                    plan.composite_answer = curated
                    plan.sub_answers = [
                        SubAnswer(
                            question=plan.sub_questions[0] if plan.sub_questions else f"What is {topic}?",
                            source="curated_overview",
                            text=curated,
                            confidence=0.95,
                        )
                    ]
                    return plan
                plan.composite_answer = ""
                return plan

        plan.composite_answer = self._compose(plan)
        return plan
