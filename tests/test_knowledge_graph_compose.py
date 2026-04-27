"""Iter 94 — KnowledgeGraph.compose_answer per-predicate + alias tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from i3.dialogue.knowledge_graph import KnowledgeGraph


def _kg(rels):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump({"relations": rels}, f)
        path = Path(f.name)
    return KnowledgeGraph(catalogue_path=path)


def test_founded_by_with_year_renders_year():
    kg = _kg([{"s": "python", "p": "founded_by",
               "o": "guido van rossum", "y": 1991}])
    out = kg.compose_answer("python", "founded_by")
    assert "guido" in out.lower()
    assert "1991" in out


def test_founded_by_without_year_omits_year():
    kg = _kg([{"s": "company_x", "p": "founded_by",
               "o": "alice"}])
    out = kg.compose_answer("company_x", "founded_by")
    assert "alice" in out.lower()
    # No year — must not interpolate undefined
    assert "None" not in out
    assert "1991" not in out


def test_predicate_alias_chain_picks_alternate():
    """`discovered_by` falls back to `founded_by` when the catalogue
    only stores the latter."""
    kg = _kg([{"s": "evolution", "p": "founded_by",
               "o": "charles darwin"}])
    out = kg.compose_answer("evolution", "discovered_by")
    assert "darwin" in out.lower()


def test_unknown_predicate_returns_empty():
    kg = _kg([{"s": "python", "p": "famous_for", "o": "data science"}])
    assert kg.compose_answer("python", "totally_unknown_predicate") == ""


def test_unknown_subject_returns_empty():
    kg = _kg([{"s": "python", "p": "famous_for", "o": "data science"}])
    assert kg.compose_answer("totally_unknown_xyz", "famous_for") == ""


def test_compose_handles_multiple_objects():
    kg = _kg([
        {"s": "company_y", "p": "founded_by", "o": "alice"},
        {"s": "company_y", "p": "founded_by", "o": "bob"},
    ])
    out = kg.compose_answer("company_y", "founded_by")
    # Either lists both or picks one — either way must include at
    # least one founder name.
    assert "alice" in out.lower() or "bob" in out.lower()


def test_compose_returns_string_not_none(kg=None):
    kg = _kg([{"s": "x", "p": "p", "o": "o"}])
    out = kg.compose_answer("x", "p")
    assert isinstance(out, str)
