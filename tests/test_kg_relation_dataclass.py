"""Iter 120 — KGRelation dataclass invariants."""
from __future__ import annotations

import pytest

from i3.dialogue.knowledge_graph import KGRelation


def test_minimal_construction():
    r = KGRelation(subject="python", predicate="founded_by",
                   object="guido van rossum")
    assert r.subject == "python"
    assert r.predicate == "founded_by"
    assert r.object == "guido van rossum"
    assert r.confidence == 1.0
    assert r.year is None


def test_with_year():
    r = KGRelation(subject="python", predicate="founded_in",
                   object="1991", year=1991)
    assert r.year == 1991


def test_to_dict_shape():
    r = KGRelation("x", "p", "o", confidence=0.85, year=2020)
    d = r.to_dict()
    assert d["subject"] == "x"
    assert d["predicate"] == "p"
    assert d["object"] == "o"
    assert d["confidence"] == 0.85
    assert d.get("year") == 2020


def test_frozen_dataclass_immutable():
    """KGRelation is frozen — assignment after init must raise."""
    r = KGRelation("x", "p", "o")
    with pytest.raises((AttributeError, Exception)):
        r.subject = "y"  # type: ignore[misc]


def test_equality_is_value_based():
    """Two KGRelations with the same fields should be equal."""
    a = KGRelation("x", "p", "o", confidence=0.5, year=2020)
    b = KGRelation("x", "p", "o", confidence=0.5, year=2020)
    assert a == b


def test_hashable():
    """Frozen dataclass should be hashable for set / dict-key use."""
    r = KGRelation("x", "p", "o")
    s = {r, r}
    assert len(s) == 1
