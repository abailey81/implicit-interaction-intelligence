"""Iter 70 — KnowledgeGraph.overview() dedupe regression test.

Iter 51 introduced a pairwise-overlap (Jaccard ≥ 0.6) dedupe pass in
``KnowledgeGraph.overview()`` so a `founded_by` sentence that already
mentions the year ("Python was founded by Guido van Rossum in 1991.")
doesn't get followed by a redundant `founded_in` sentence.  These
tests pin that behaviour: a future refactor that removes the pass
silently regresses the chat tab's overview output.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from i3.dialogue.knowledge_graph import KnowledgeGraph


def _kg_with(relations: list[dict]) -> KnowledgeGraph:
    """Construct a KnowledgeGraph backed by an in-memory JSON catalogue."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump({"relations": relations}, f)
        path = Path(f.name)
    return KnowledgeGraph(catalogue_path=path)


def test_overview_returns_string_for_known_subject():
    kg = _kg_with([
        {"s": "python", "p": "founded_by",
         "o": "guido van rossum", "y": 1991},
        {"s": "python", "p": "famous_for",
         "o": "data science"},
    ])
    out = kg.overview("python")
    assert isinstance(out, str)
    assert len(out) > 0


def test_overview_returns_empty_for_unknown_subject():
    kg = _kg_with([
        {"s": "python", "p": "famous_for",
         "o": "data science"},
    ])
    out = kg.overview("totally_unknown_subject_xyz")
    assert out == ""


def test_overview_caps_at_three_sentences():
    """Overview is bounded — never floods the chat with 10 sentences
    even when the catalogue has many slots filled."""
    kg = _kg_with([
        {"s": "python", "p": "founded_by",
         "o": "guido van rossum"},
        {"s": "python", "p": "founded_in",
         "o": "1991"},
        {"s": "python", "p": "famous_for",
         "o": "data science, ML, web backends, scripting"},
        {"s": "python", "p": "headquartered_in",
         "o": "global"},
        {"s": "python", "p": "competitor_of",
         "o": "rust"},
        {"s": "python", "p": "ceo",
         "o": "guido van rossum"},
    ])
    out = kg.overview("python")
    # Sentence count = number of '.' (rough but reliable for this format)
    n_sentences = out.count(".")
    assert n_sentences <= 3, f"overview returned {n_sentences} sentences: {out!r}"


def test_overview_dedupes_year_overlap():
    """Iter 51 contract: when one slot's sentence already names the
    year, the year-only slot must not produce a redundant sentence."""
    kg = _kg_with([
        {"s": "python", "p": "founded_by",
         "o": "guido van rossum", "y": 1991},
        {"s": "python", "p": "founded_in",
         "o": "1991"},
        {"s": "python", "p": "famous_for",
         "o": "data science"},
    ])
    out = kg.overview("python").lower()
    # 1991 should appear at most once
    assert out.count("1991") <= 1, f"year duplicated in overview: {out!r}"


def test_overview_handles_empty_catalogue():
    kg = _kg_with([])
    assert kg.overview("python") == ""
