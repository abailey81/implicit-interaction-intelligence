"""Interaction Diary system for Implicit Interaction Intelligence (I3).

Provides privacy-safe logging and summarisation of user-AI interaction
sessions.  **No raw user text is ever persisted** -- the diary records only
embeddings, scalar metrics, topic keywords (extracted via TF-IDF), and
adaptation parameters.

Key components:

- :class:`DiaryStore` -- Async SQLite persistence layer with tables for
  ``sessions`` (session-level summaries) and ``exchanges`` (per-message
  records).
- :class:`DiaryLogger` -- Per-exchange recorder that extracts topic keywords,
  accumulates session-level statistics, and delegates to the store.
- :class:`SessionSummarizer` -- Generates warm, diary-style session summaries
  from aggregated metadata using either a cloud LLM or a local template
  fallback.
"""

from i3.diary.logger import DiaryLogger
from i3.diary.store import DiaryStore
from i3.diary.summarizer import SessionSummarizer

__all__ = [
    "DiaryLogger",
    "DiaryStore",
    "SessionSummarizer",
]
