"""Hierarchical session memory (Phase B.5, 2026-04-25).

The legacy 4-turn rolling history is fine for short interactions but
loses signal across longer sessions.  This module layers three
additional memory tiers on top of the existing per-session pair list
that the engine already maintains:

1. **Topic stack** — the canonical entities mentioned across the last
   ~10 turns, with recency-decayed weights.
2. **User-stated facts** — explicit factual claims the user made about
   themselves (``"I'm a developer"``, ``"I prefer bullet points"``,
   ``"I work in finance"``, ``"I live in London"``).  Stored as
   ``(predicate, object, confidence, decay)`` triples and capped per
   session.
3. **Session topic thread** — a single-paragraph summary of what the
   conversation has been about, regenerated every 5 turns from the
   topic stack.

All three layers are **bounded** (max 50 topics, max 100 user facts,
max 200 chars in the thread summary) so a long-running server cannot
balloon memory through any single client.

The module is **pure stdlib** (no LLM call, no model weights) — it's
deliberately a rule-based synopsis so it stays fast and deterministic.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User-stated fact extraction patterns
# ---------------------------------------------------------------------------
# Each entry is (predicate, regex).  Group 1 captures the object.
# Patterns terminate at: end-of-string, comma, period, "and", "but", "or"
# so multi-clause messages don't produce greedy captures.
_OBJ_TAIL = r"([a-z][a-z\s\-]{1,40}?)(?=\s+(?:and|but|or|because|so|when|while)\b|[,.!?]|$)"


_USER_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("role", re.compile(r"\bi(?:'m|\s+am)\s+(?:a|an)\s+" + _OBJ_TAIL, re.I)),
    ("preference", re.compile(r"\bi\s+prefer\s+" + _OBJ_TAIL, re.I)),
    ("preference", re.compile(r"\bi\s+like\s+" + _OBJ_TAIL, re.I)),
    ("dislike", re.compile(r"\bi\s+(?:hate|dislike|don't\s+like)\s+" + _OBJ_TAIL, re.I)),
    ("works_in", re.compile(r"\bi\s+work(?:\s+in|\s+at|\s+for)\s+" + _OBJ_TAIL, re.I)),
    ("lives_in", re.compile(r"\bi\s+live(?:\s+in)?\s+" + _OBJ_TAIL, re.I)),
    ("studies", re.compile(r"\bi\s+study(?:\s+in|\s+at)?\s+" + _OBJ_TAIL, re.I)),
    ("name", re.compile(r"\bmy\s+name\s+is\s+([a-z][a-z\s\-]{1,40}?)(?=[,.!?]|$)", re.I)),
    ("name", re.compile(r"\bi'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")),
)

# Stop-words that produce noisy "fact" extraction
_STOPWORDS_FACT = {
    "really", "very", "totally", "kind", "sort", "bit", "little",
    "going", "trying", "thinking", "feeling", "sure", "ready", "tired",
    "sorry", "happy", "sad", "okay", "alright", "still", "now", "today",
    "yesterday", "tomorrow", "here", "there", "good", "bad", "fine",
}


@dataclass
class UserFact:
    predicate: str
    object: str
    confidence: float = 1.0
    decay: float = 1.0  # 1.0 fresh, scaled toward 0 as turns pass

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate": self.predicate,
            "object": self.object,
            "confidence": round(self.confidence, 3),
            "decay": round(self.decay, 3),
        }


@dataclass
class TopicEntry:
    canonical: str
    weight: float = 1.0  # exponentially decayed each turn

    def to_dict(self) -> dict[str, object]:
        return {"canonical": self.canonical, "weight": round(self.weight, 3)}


@dataclass
class SessionMemory:
    """All hierarchical state for a single session."""

    user_facts: list[UserFact] = field(default_factory=list)
    topic_stack: deque[TopicEntry] = field(default_factory=lambda: deque(maxlen=50))
    turn_count: int = 0
    thread_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "user_facts": [f.to_dict() for f in self.user_facts[-30:]],
            "topic_stack": [t.to_dict() for t in list(self.topic_stack)[:10]],
            "turn_count": self.turn_count,
            "thread_summary": self.thread_summary,
        }


# ---------------------------------------------------------------------------
# HierarchicalMemory
# ---------------------------------------------------------------------------
class HierarchicalMemory:
    """Per-session hierarchical memory with bounds.

    Bounds:
        - 100 user facts max per session (oldest dropped).
        - 50 topic entries max in the topic stack (deque ``maxlen``).
        - Thread summary regenerated every ``summary_every`` turns (default 5).
    """

    MAX_FACTS_PER_SESSION = 100
    DECAY_PER_TURN = 0.85

    def __init__(self, summary_every: int = 5) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        self._summary_every = max(1, int(summary_every))

    # -- per-session lifecycle ------------------------------------------
    def _key(self, user_id: str, session_id: str) -> str:
        return f"{user_id or '_'}::{session_id or '_'}"

    def _get(self, user_id: str, session_id: str) -> SessionMemory:
        k = self._key(user_id, session_id)
        if k not in self._sessions:
            self._sessions[k] = SessionMemory()
        return self._sessions[k]

    def end_session(self, user_id: str, session_id: str) -> None:
        self._sessions.pop(self._key(user_id, session_id), None)

    # -- write path -----------------------------------------------------
    def observe(
        self,
        user_id: str,
        session_id: str,
        *,
        user_message: str,
        recent_entities: Iterable[str] | None = None,
    ) -> None:
        """Update the memory from a new turn.

        *recent_entities* are the canonicals from the EntityTracker for
        this turn (org/topic/person frames).  ``user_message`` is the
        raw text we'll scan for user-stated facts.
        """
        sm = self._get(user_id, session_id)
        sm.turn_count += 1

        # 1) Decay topic-stack weights and add fresh entities at weight 1.0
        for entry in sm.topic_stack:
            entry.weight = max(0.0, entry.weight * self.DECAY_PER_TURN)
        for ent in recent_entities or []:
            ent_l = (ent or "").strip().lower()
            if not ent_l:
                continue
            existing = next(
                (e for e in sm.topic_stack if e.canonical == ent_l), None
            )
            if existing is not None:
                # bump weight back to fresh
                existing.weight = 1.0
            else:
                sm.topic_stack.appendleft(TopicEntry(ent_l, 1.0))
        # Drop near-zero entries to keep the stack tight.
        # (deque doesn't allow filtered pop, so rebuild)
        if sm.topic_stack:
            kept = [e for e in sm.topic_stack if e.weight > 0.05]
            sm.topic_stack = deque(kept, maxlen=50)

        # 2) Extract user-stated facts
        for predicate, pattern in _USER_FACT_PATTERNS:
            for m in pattern.finditer(user_message or ""):
                obj = m.group(1).strip().lower()
                # Trim trailing function/stop tokens.
                obj = re.sub(r"\b(too|now|today|please|thanks)\b\s*$", "", obj).strip()
                # Reject pure stopword captures
                head = obj.split()[0] if obj else ""
                if not obj or head in _STOPWORDS_FACT or len(obj) < 2:
                    continue
                # Dedupe: replace existing same-predicate-same-object,
                # and cap total facts per session.
                fact = UserFact(predicate=predicate, object=obj, confidence=0.9)
                # Decay older facts a bit on each new turn so stale
                # claims fade out.
                for f in sm.user_facts:
                    f.decay *= self.DECAY_PER_TURN
                sm.user_facts = [
                    f for f in sm.user_facts
                    if not (f.predicate == predicate and f.object == obj)
                ]
                sm.user_facts.append(fact)
        # Bound facts list
        if len(sm.user_facts) > self.MAX_FACTS_PER_SESSION:
            sm.user_facts = sm.user_facts[-self.MAX_FACTS_PER_SESSION:]

        # 3) Regenerate thread summary every N turns.
        if sm.turn_count % self._summary_every == 0:
            sm.thread_summary = self._build_thread_summary(sm)

    # -- read path ------------------------------------------------------
    def get(
        self, user_id: str, session_id: str
    ) -> SessionMemory:
        return self._get(user_id, session_id)

    def to_dict(self, user_id: str, session_id: str) -> dict[str, object]:
        return self._get(user_id, session_id).to_dict()

    # -- summary builder ------------------------------------------------
    @staticmethod
    def _build_thread_summary(sm: SessionMemory) -> str:
        topics = sorted(
            sm.topic_stack, key=lambda e: e.weight, reverse=True
        )[:5]
        topic_names = [t.canonical for t in topics]
        if not topic_names:
            return ""
        if len(topic_names) == 1:
            t_str = topic_names[0]
        elif len(topic_names) == 2:
            t_str = f"{topic_names[0]} and {topic_names[1]}"
        else:
            t_str = ", ".join(topic_names[:-1]) + f", and {topic_names[-1]}"
        summary = f"Discussion about {t_str}."
        # Append the strongest 1-2 user facts.
        if sm.user_facts:
            top_facts = sorted(
                sm.user_facts,
                key=lambda f: f.confidence * f.decay,
                reverse=True,
            )[:2]
            facts_str = "; ".join(
                f"user is a {f.object}" if f.predicate == "role"
                else f"user prefers {f.object}" if f.predicate == "preference"
                else f"user {f.predicate.replace('_', ' ')} {f.object}"
                for f in top_facts
            )
            summary = f"{summary} ({facts_str})"
        return summary[:200]
