"""Per-exchange diary logger with TF-IDF topic extraction.

The :class:`DiaryLogger` sits between the interaction pipeline and the
:class:`~src.diary.store.DiaryStore`, recording one entry per user-AI
exchange.  It performs two key functions:

1. **Topic extraction** -- Extracts the top-N keywords from each message
   using a lightweight TF-IDF implementation built from scratch (no
   sklearn dependency).  The raw message text is used *only* for this
   extraction and is **never persisted**.

2. **Session-level accumulation** -- Tracks per-session running totals of
   engagement, cognitive load, accessibility, and topic keywords so that
   a session summary can be generated at close time without replaying
   individual exchanges.

PRIVACY GUARANTEE
~~~~~~~~~~~~~~~~~
The ``message_text`` parameter in :meth:`log_exchange` is consumed solely
for keyword extraction.  It is not written to the database, not sent to
the cloud, and not retained in memory beyond the scope of the method
call.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from i3.diary.store import DiaryStore

if TYPE_CHECKING:
    from i3.adaptation.types import AdaptationVector
    from i3.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# English stopwords (~175 common terms)
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset({
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "whose",
    # Prepositions
    "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "out", "up", "down", "off", "against",
    "along", "around", "among", "upon", "within", "without",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    # Auxiliary / modal verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    # Common adverbs
    "not", "no", "very", "just", "also", "too", "more", "most",
    "already", "still", "even", "now", "then", "here", "there",
    "when", "where", "why", "how", "all", "each", "every",
    "always", "never", "often", "sometimes", "usually", "really",
    "quite", "rather", "perhaps", "almost", "enough",
    # Common verbs / contractions
    "get", "got", "go", "going", "gone", "come", "came",
    "make", "made", "take", "took", "taken",
    "know", "known", "think", "thought",
    "say", "said", "tell", "told",
    "give", "gave", "given", "see", "saw", "seen",
    "want", "need", "like", "use", "used",
    "try", "keep", "let", "put", "set", "seem", "help",
    "show", "ask", "work", "call", "find", "found",
    # Misc function words
    "if", "because", "as", "until", "while", "of", "than",
    "such", "some", "any", "many", "much", "few", "other",
    "only", "own", "same", "well", "back", "way",
    "thing", "things", "one", "two", "first",
    "new", "old", "good", "great", "right", "long",
    "little", "big", "small",
    # Discourse markers
    "oh", "ok", "okay", "yeah", "yes", "no", "hi", "hello",
    "hey", "thanks", "thank", "please", "sorry",
})


# ---------------------------------------------------------------------------
# Background IDF corpus -- log(N/df) for common English terms
# ---------------------------------------------------------------------------

# Pre-computed IDF scores from a small background reference corpus.
# Terms not in this map receive a default "rare term" IDF boost.
_DEFAULT_IDF: dict[str, float] = {
    # Communication / meta
    "help": 1.2, "question": 2.0, "answer": 2.1, "explain": 2.6,
    "understand": 2.3, "mean": 2.2, "example": 2.4, "thing": 1.4,
    "thought": 2.1, "feel": 2.0, "say": 1.7, "talk": 2.0,
    "speak": 2.3, "hear": 2.3, "listen": 2.5,
    # Time
    "time": 1.5, "day": 1.6, "week": 2.0, "month": 2.2, "year": 1.8,
    "hour": 2.1, "minute": 2.3, "today": 1.8, "tomorrow": 2.3,
    "yesterday": 2.4, "morning": 2.4, "evening": 2.6, "night": 2.2,
    "weekend": 2.7, "date": 2.2, "deadline": 3.1, "late": 2.0,
    "early": 2.1, "now": 1.5, "soon": 2.2,
    # Technology — software engineering
    "code": 3.5, "python": 4.0, "javascript": 4.2, "typescript": 4.5,
    "function": 3.2, "method": 3.0, "class": 3.0, "object": 2.8,
    "variable": 3.2, "library": 3.0, "package": 3.1, "module": 3.2,
    "api": 3.3, "endpoint": 3.8, "request": 2.8, "response": 2.8,
    "database": 3.4, "server": 3.0, "client": 2.8, "deploy": 3.7,
    "deployment": 3.8, "docker": 4.0, "kubernetes": 4.5, "container": 3.8,
    "pipeline": 3.2, "repository": 3.5, "commit": 3.0, "branch": 3.1,
    "merge": 3.3, "pull": 2.8, "push": 2.8, "issue": 2.5,
    "ticket": 3.2, "bug": 3.0, "debug": 3.8, "fix": 2.5,
    "refactor": 4.0, "optimise": 3.8, "optimize": 3.8, "benchmark": 3.9,
    "test": 2.7, "testing": 2.9, "unit": 2.9, "integration": 3.3,
    "regression": 3.6, "build": 2.8, "compile": 3.5, "lint": 3.8,
    "format": 3.0, "syntax": 3.4, "runtime": 3.3, "memory": 2.9,
    "cpu": 3.5, "gpu": 3.6, "latency": 3.7, "throughput": 4.0,
    "scalability": 4.2, "concurrency": 4.1, "async": 3.7, "await": 3.8,
    "thread": 3.3, "process": 2.5, "queue": 3.2, "cache": 3.2,
    # Technology — data + ML
    "data": 2.8, "dataset": 3.4, "model": 2.5, "training": 2.9,
    "inference": 3.6, "prediction": 3.3, "classifier": 3.9, "regression": 3.7,
    "neural": 3.8, "network": 2.8, "transformer": 4.0, "embedding": 4.2,
    "attention": 3.5, "layer": 3.0, "weight": 3.0, "gradient": 4.1,
    "loss": 2.8, "accuracy": 3.0, "precision": 3.0, "recall": 3.2,
    "tokenizer": 4.4, "vocabulary": 3.8, "corpus": 4.0, "annotation": 4.1,
    # File / data operations
    "file": 3.0, "folder": 3.0, "directory": 3.3, "path": 2.8,
    "upload": 3.3, "download": 3.0, "export": 3.0, "import": 3.0,
    "backup": 3.3, "restore": 3.3, "archive": 3.4, "compress": 3.7,
    "search": 2.6, "find": 2.0, "filter": 3.0, "sort": 2.6,
    "save": 2.2, "copy": 2.4, "paste": 2.9, "delete": 2.6,
    "rename": 3.3, "share": 2.5, "link": 2.4, "url": 3.3,
    # Productivity
    "email": 3.2, "inbox": 3.8, "message": 2.3, "chat": 2.9,
    "schedule": 3.5, "calendar": 3.5, "meeting": 3.3, "agenda": 3.9,
    "note": 2.4, "notes": 2.4, "document": 2.9, "spreadsheet": 4.2,
    "presentation": 3.9, "slide": 3.4, "outline": 3.5, "draft": 3.2,
    "review": 2.6, "approve": 3.3, "feedback": 3.1, "comment": 2.7,
    "project": 2.7, "task": 2.9, "todo": 3.2, "milestone": 3.9,
    "deliverable": 4.0, "kickoff": 4.2, "stakeholder": 4.1,
    "blocker": 3.8, "priority": 3.2,
    # Thinking / creation
    "idea": 2.8, "plan": 2.9, "goal": 2.7, "strategy": 3.3,
    "approach": 2.8, "method": 3.0, "solution": 2.8, "problem": 2.5,
    "decision": 3.0, "option": 2.8, "choice": 2.8, "reason": 2.3,
    "cause": 2.5, "effect": 2.6, "analysis": 3.3, "research": 3.0,
    "insight": 3.5, "hypothesis": 3.9, "conclusion": 3.3,
    "write": 2.3, "read": 2.2, "learn": 2.4, "study": 2.7,
    "create": 2.5, "design": 3.0, "draft": 3.2, "publish": 3.3,
    "edit": 2.7, "revise": 3.4, "rewrite": 3.3,
    # Daily life
    "weather": 3.1, "temperature": 3.4, "rain": 3.3, "sunny": 3.6,
    "hot": 2.8, "cold": 2.6,
    "music": 3.4, "song": 3.1, "album": 3.5, "playlist": 3.8,
    "movie": 3.3, "film": 3.3, "show": 2.2, "series": 2.9,
    "book": 3.0, "novel": 3.5, "chapter": 3.3, "page": 2.6,
    "game": 3.1, "play": 2.3, "win": 2.5, "lose": 2.4,
    "news": 2.8, "article": 3.1, "story": 3.0, "report": 2.8,
    "blog": 3.5, "podcast": 3.8,
    "image": 3.2, "photo": 3.4, "video": 3.1, "camera": 3.3,
    "sport": 3.2, "exercise": 3.5, "workout": 3.8, "run": 2.1,
    "walk": 2.4, "gym": 3.5, "yoga": 3.9, "swim": 3.6,
    "recipe": 4.0, "food": 3.0, "cook": 3.1, "dinner": 3.2,
    "lunch": 3.0, "breakfast": 3.3, "coffee": 3.1, "tea": 3.2,
    "travel": 3.6, "flight": 3.7, "hotel": 3.5, "trip": 3.2,
    "holiday": 3.5, "vacation": 3.6,
    "money": 3.0, "budget": 3.5, "shopping": 3.6, "price": 2.8,
    "cost": 2.8, "expense": 3.3, "invoice": 3.8,
    "home": 2.5, "family": 2.9, "friend": 3.0, "school": 2.8,
    "work": 1.7, "job": 2.3, "career": 3.2, "office": 2.8,
    "language": 3.2, "translate": 4.0, "math": 3.5, "science": 3.3,
    "history": 3.1, "art": 3.4, "education": 3.4, "course": 2.9,
    # Health / wellbeing
    "health": 3.2, "doctor": 3.3, "medicine": 3.7, "appointment": 3.6,
    "sleep": 3.0, "rest": 2.8, "tired": 2.4, "energy": 2.9,
    "pain": 3.0, "headache": 3.9, "stress": 2.8, "anxious": 3.4,
    "happy": 2.4, "sad": 2.6, "focus": 2.8, "mood": 3.0,
    # Affective / evaluative
    "great": 1.8, "good": 1.4, "nice": 1.8, "amazing": 2.5,
    "awesome": 2.5, "wonderful": 2.9, "beautiful": 2.7,
    "bad": 1.7, "wrong": 2.0, "terrible": 2.9, "awful": 3.0,
    "difficult": 2.1, "easy": 2.0, "important": 2.0, "useful": 2.4,
    "interesting": 2.3, "boring": 2.9, "exciting": 2.8,
    # Conversation / request patterns
    "please": 1.5, "thanks": 1.7, "thank": 1.7, "hello": 2.2,
    "hi": 2.0, "welcome": 2.6, "sorry": 2.1, "excuse": 3.0,
    "sure": 1.9, "maybe": 2.3, "probably": 2.4, "perhaps": 2.7,
    # Actions
    "go": 1.4, "come": 1.7, "stay": 2.1, "leave": 2.3,
    "open": 2.0, "close": 2.2, "start": 2.0, "stop": 2.2,
    "begin": 2.5, "end": 2.1, "finish": 2.4, "continue": 2.5,
    "send": 2.3, "receive": 2.7, "reply": 2.8, "forward": 3.0,
}

# IDF assigned to terms absent from the background corpus (rare = high info).
_RARE_TERM_IDF: float = 5.0


# ---------------------------------------------------------------------------
# Tokeniser helper
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{1,29}")
"""Match lowercase alpha-start tokens between 2-30 characters."""


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens, remove stopwords.

    Returns a list of cleaned tokens suitable for TF-IDF scoring.  Tokens
    shorter than 2 characters or present in :data:`STOPWORDS` are dropped.
    """
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# DiaryLogger
# ---------------------------------------------------------------------------

class DiaryLogger:
    """Logs each user-AI exchange to the diary without storing raw text.

    After each exchange the logger:

    1. Extracts topic keywords from the current message via TF-IDF (the
       raw text is **not** retained or persisted).
    2. Records timestamp, user-state embedding, adaptation vector,
       route decision, response latency, engagement signal, and extracted
       topics to the :class:`DiaryStore`.
    3. Accumulates per-session running statistics (engagement, cognitive
       load, accessibility, topics) for later session summarisation.

    Parameters
    ----------
    store:
        The :class:`DiaryStore` instance to persist exchanges.
    config:
        The I3 :class:`~src.config.Config` object (used for diary settings).
    """

    def __init__(self, store: DiaryStore, config: Config) -> None:
        self.store = store
        self.config = config

        # Per-session accumulators (keyed by session_id)
        self._session_topics: dict[str, list[str]] = {}
        self._session_engagement: dict[str, list[float]] = {}
        self._session_cognitive: dict[str, list[float]] = {}
        self._session_accessibility: dict[str, list[float]] = {}
        self._session_emotions: dict[str, list[str]] = {}

        # Pre-computed IDF scores (background corpus)
        self._idf_scores: dict[str, float] = dict(_DEFAULT_IDF)

    # ------------------------------------------------------------------
    # Exchange logging
    # ------------------------------------------------------------------

    async def log_exchange(
        self,
        session_id: str,
        user_state_embedding: torch.Tensor,
        adaptation_vector: AdaptationVector,
        route_chosen: str,
        response_latency_ms: int,
        engagement_signal: float,
        message_text: str,
        *,
        cognitive_load: float | None = None,
        accessibility: float | None = None,
        emotion_label: str | None = None,
    ) -> None:
        """Log a single exchange.  ``message_text`` is used ONLY for topic
        extraction and is **never persisted**.

        Parameters
        ----------
        session_id:
            The active session identifier.
        user_state_embedding:
            The encoder's current user-state embedding (Tensor).
        adaptation_vector:
            The adaptation vector applied for this exchange.
        route_chosen:
            Router decision (e.g. ``"local_slm"`` or ``"cloud_llm"``).
        response_latency_ms:
            End-to-end response time in milliseconds.
        engagement_signal:
            Estimated engagement level for this exchange (0-1).
        message_text:
            The raw user message -- used **only** for TF-IDF topic
            extraction.  This string is never stored, logged, or
            transmitted.
        cognitive_load:
            Optional cognitive-load estimate (0-1) from the adaptation
            vector.  If ``None``, extracted from ``adaptation_vector``.
        accessibility:
            Optional accessibility level (0-1).  If ``None``, extracted
            from ``adaptation_vector``.
        emotion_label:
            Optional detected emotion label (e.g. ``"happy"``,
            ``"frustrated"``).  Accumulated for dominant-emotion
            computation at session close.
        """
        # -- Topic extraction (text is discarded after this) -------------
        topics = self.extract_topics(message_text)

        # -- Derive scalar metrics from the adaptation vector ------------
        adapt_dict = adaptation_vector.to_dict()
        cog = cognitive_load if cognitive_load is not None else adapt_dict.get("cognitive_load", 0.5)
        acc = accessibility if accessibility is not None else adapt_dict.get("accessibility", 0.0)

        # -- Accumulate session-level statistics -------------------------
        self._session_topics.setdefault(session_id, []).extend(topics)
        self._session_engagement.setdefault(session_id, []).append(engagement_signal)
        self._session_cognitive.setdefault(session_id, []).append(cog)
        self._session_accessibility.setdefault(session_id, []).append(acc)
        if emotion_label:
            self._session_emotions.setdefault(session_id, []).append(emotion_label)

        # -- Persist to store --------------------------------------------
        embedding_bytes = user_state_embedding.detach().cpu().numpy().tobytes()
        await self.store.log_exchange(
            session_id=session_id,
            user_state_embedding=embedding_bytes,
            adaptation_vector=adapt_dict,
            route_chosen=route_chosen,
            response_latency_ms=response_latency_ms,
            engagement_signal=engagement_signal,
            topics=topics,
        )
        logger.debug(
            "Exchange logged for session %s: topics=%s, engagement=%.2f",
            session_id,
            topics,
            engagement_signal,
        )

    # ------------------------------------------------------------------
    # TF-IDF topic extraction
    # ------------------------------------------------------------------

    def extract_topics(self, text: str, n_topics: int = 3) -> list[str]:
        """Extract the top-N topic keywords from ``text`` using TF-IDF.

        Implementation:

        1. Tokenise: lowercase, remove punctuation, drop stopwords.
        2. Compute **term frequency** (TF) = count(term) / total_tokens.
        3. Look up **inverse document frequency** (IDF) from the
           pre-computed background map.  Unknown terms receive a high IDF
           (rare terms are more informative).
        4. Score each unique term as TF * IDF.
        5. Return the top-N terms by score.

        Parameters
        ----------
        text:
            Raw user message (consumed here, never stored).
        n_topics:
            Number of top keywords to return.

        Returns
        -------
        list[str]
            Top-N topic keywords sorted by TF-IDF score descending.
            May return fewer than ``n_topics`` if the message is very
            short or consists entirely of stopwords.
        """
        # SEC: defensive guard -- if upstream passes None or a non-string
        # (e.g. a redacted placeholder object), return [] instead of
        # crashing.  This keeps the privacy invariant: even malformed
        # input cannot leak via an exception traceback.
        if not text or not isinstance(text, str):
            return []
        tokens = _tokenize(text)
        if not tokens:
            return []

        total = len(tokens)
        term_freq = Counter(tokens)

        scored: list[tuple[str, float]] = []
        for term, count in term_freq.items():
            tf = count / total
            idf = self._idf_scores.get(term, _RARE_TERM_IDF)
            scored.append((term, tf * idf))

        # Sort by score descending, break ties alphabetically
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [term for term, _score in scored[:n_topics]]

    def update_idf(self, corpus_term_counts: dict[str, int], corpus_size: int) -> None:
        """Update IDF scores from observed corpus statistics.

        Computes IDF = log(corpus_size / document_frequency) for each
        term and merges into the existing IDF map.

        Parameters
        ----------
        corpus_term_counts:
            Mapping of term -> number of documents containing that term.
        corpus_size:
            Total number of documents in the corpus.
        """
        if corpus_size <= 0:
            return
        for term, df in corpus_term_counts.items():
            if df > 0:
                self._idf_scores[term] = math.log(corpus_size / df)
        logger.debug(
            "IDF scores updated: %d terms from corpus of %d documents",
            len(corpus_term_counts),
            corpus_size,
        )

    # ------------------------------------------------------------------
    # Session summary data
    # ------------------------------------------------------------------

    async def get_session_summary_data(self, session_id: str) -> dict[str, Any]:
        """Prepare aggregated data for session summary generation.

        Returns a dictionary of **metadata only** -- no raw text is
        included.  This data is suitable for passing to
        :class:`~src.diary.summarizer.SessionSummarizer`.

        Parameters
        ----------
        session_id:
            The session to summarise.

        Returns
        -------
        dict
            Keys include ``message_count``, ``topics``,
            ``mean_engagement``, ``mean_cognitive_load``,
            ``mean_accessibility``, ``dominant_emotion``.
        """
        engagements = self._session_engagement.get(session_id, [0.5])
        cognitive = self._session_cognitive.get(session_id, [0.5])
        accessibility = self._session_accessibility.get(session_id, [0.0])
        topics = list(set(self._session_topics.get(session_id, [])))

        return {
            "message_count": len(engagements),
            "topics": topics,
            "mean_engagement": float(np.mean(engagements)),
            "mean_cognitive_load": float(np.mean(cognitive)),
            "mean_accessibility": float(np.mean(accessibility)),
            "dominant_emotion": self._compute_dominant_emotion(session_id),
            "mean_energy": float(np.mean(engagements)),  # proxy for energy
        }

    def _compute_dominant_emotion(self, session_id: str) -> str:
        """Return the most frequently observed emotion label for a session.

        Falls back to ``"neutral"`` when no emotion labels have been
        recorded.
        """
        emotions = self._session_emotions.get(session_id, [])
        if not emotions:
            return "neutral"
        counter = Counter(emotions)
        return counter.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_session_accumulators(self, session_id: str) -> None:
        """Remove in-memory accumulators for a closed session.

        Should be called after the session summary has been generated and
        persisted to free memory.

        Parameters
        ----------
        session_id:
            The session whose accumulators should be cleared.
        """
        self._session_topics.pop(session_id, None)
        self._session_engagement.pop(session_id, None)
        self._session_cognitive.pop(session_id, None)
        self._session_accessibility.pop(session_id, None)
        self._session_emotions.pop(session_id, None)
        logger.debug("Session accumulators cleared: %s", session_id)
