"""Per-session dialogue-state machinery for the I3 pipeline.

Currently exposes :class:`EntityTracker` and friends from
:mod:`i3.dialogue.coref`, the lightweight pronoun / co-reference
resolver that lets the engine rewrite follow-up questions like
``"where are they located?"`` to ``"where is huawei located?"``
*before* retrieval / SLM see them.

The module is intentionally pure-Python (no spaCy / NLTK / NER model)
so it stays inside the from-scratch I3 stack constraints.
"""

from i3.dialogue.coref import (
    EntityFrame,
    EntityTracker,
    ResolutionResult,
    resolution_to_dict,
)
from i3.dialogue.knowledge_graph import (
    KGRelation,
    KnowledgeGraph,
    get_global_kg,
)

__all__ = [
    "EntityFrame",
    "EntityTracker",
    "ResolutionResult",
    "resolution_to_dict",
    "KGRelation",
    "KnowledgeGraph",
    "get_global_kg",
]
