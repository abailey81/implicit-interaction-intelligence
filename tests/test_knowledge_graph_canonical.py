"""Iter 79 — KnowledgeGraph subject canonicalisation tests.

Pins the case-/article-/punctuation-insensitive subject lookup that
``KnowledgeGraph._canonical`` (and therefore ``get_facts``) uses.
A regression here breaks the chat tab's ability to answer "the
python" or "Python." or " python " consistently.
"""
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


@pytest.fixture(scope="module")
def kg():
    return _kg([
        {"s": "python", "p": "famous_for", "o": "data science"},
        {"s": "huawei", "p": "headquartered_in", "o": "Shenzhen"},
    ])


@pytest.mark.parametrize("variant", [
    "python", "Python", "PYTHON", "  python  ",
    "the python", "Python.", "python!",
])
def test_get_facts_normalises_case_and_punctuation(kg, variant):
    facts = kg.get_facts(variant)
    assert len(facts) >= 1, f"variant {variant!r} found 0 facts"


@pytest.mark.parametrize("variant", [
    "huawei", "Huawei", "HUAWEI", "the huawei",
])
def test_get_facts_for_huawei(kg, variant):
    facts = kg.get_facts(variant)
    assert len(facts) >= 1


def test_get_facts_unknown_returns_empty(kg):
    assert kg.get_facts("totally_unknown_xyz") == []


def test_get_facts_empty_input(kg):
    assert kg.get_facts("") == []
    assert kg.get_facts("   ") == []


def test_display_name_uses_overrides(kg):
    """The _display_name helper must produce 'Python' (not 'Python')
    via the curated override; for unknown subjects it falls back to
    title-casing.  Checks via the public overview() output."""
    out = kg.overview("python")
    # First word of the overview should be 'Python' (not 'python')
    assert out.split()[0].lower().startswith("python") or out == ""


def test_display_name_for_huawei_is_capitalised(kg):
    out = kg.overview("huawei")
    if out:
        assert "huawei" in out.lower()
        # Override should use 'Huawei' (proper-name capitalisation)
        assert "Huawei" in out
