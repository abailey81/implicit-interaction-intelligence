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
    "help": 1.2, "weather": 3.1, "music": 3.4, "code": 3.5,
    "python": 4.0, "question": 2.0, "answer": 2.1, "time": 1.5,
    "problem": 2.5, "error": 3.0, "data": 2.8, "file": 3.0,
    "search": 2.6, "email": 3.2, "schedule": 3.5, "meeting": 3.3,
    "project": 2.7, "task": 2.9, "idea": 2.8, "plan": 2.9,
    "write": 2.3, "read": 2.2, "learn": 2.4, "explain": 2.6,
    "create": 2.5, "build": 2.8, "design": 3.0, "test": 2.7,
    "debug": 3.8, "fix": 2.5, "update": 2.4, "change": 2.1,
    "recipe": 4.0, "travel": 3.6, "health": 3.2, "exercise": 3.5,
    "book": 3.0, "movie": 3.3, "game": 3.1, "news": 2.8,
    "story": 3.0, "image": 3.2, "photo": 3.4, "video": 3.1,
    "money": 3.0, "budget": 3.5, "shopping": 3.6, "price": 2.8,
    "home": 2.5, "family": 2.9, "friend": 3.0, "school": 2.8,
    "language": 3.2, "translate": 4.0, "math": 3.5, "science": 3.3,
    "history": 3.1, "art": 3.4, "sport": 3.2, "food": 3.0,
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
