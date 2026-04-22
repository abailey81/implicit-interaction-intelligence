"""Tests that pin the adversarial corpus shape.

These tests are read-only over :data:`ATTACK_CORPUS` and over the
:func:`load_external_corpus` loader.  They exist to fail loudly if
anyone silently drops attacks, flips an `expected_outcome`, or
reintroduces schema drift into the frozen dataclass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest

from i3.redteam.attack_corpus import (
    ATTACK_CATEGORIES,
    ATTACK_CATEGORY_TARGETS,
    ATTACK_CORPUS,
    Attack,
    ExpectedOutcome,
    Severity,
    load_external_corpus,
)


def test_corpus_has_at_least_fifty_attacks() -> None:
    """The corpus must contain at least 50 attacks (batch requirement)."""
    assert len(ATTACK_CORPUS) >= 50, (
        f"Corpus has only {len(ATTACK_CORPUS)} attacks, need >= 50"
    )


def test_every_attack_has_required_string_fields() -> None:
    """id, description, payload, source_citation must be non-empty."""
    for a in ATTACK_CORPUS:
        assert a.id, f"Attack has empty id: {a!r}"
        assert a.description, f"Attack {a.id} has empty description"
        # payload may be a dict or a non-empty str
        if isinstance(a.payload, str):
            assert a.payload, f"Attack {a.id} has empty string payload"
        else:
            assert isinstance(a.payload, dict)
            assert a.payload, f"Attack {a.id} has empty dict payload"
        assert a.source_citation, f"Attack {a.id} has empty source_citation"


def test_attack_ids_are_unique() -> None:
    """Attack identifiers must be unique across the corpus."""
    ids = [a.id for a in ATTACK_CORPUS]
    assert len(ids) == len(set(ids)), (
        f"Duplicate attack IDs: {sorted(set(i for i in ids if ids.count(i) > 1))}"
    )


def test_every_expected_outcome_is_allowed() -> None:
    """`expected_outcome` must be one of the declared Literal values."""
    allowed = set(get_args(ExpectedOutcome))
    for a in ATTACK_CORPUS:
        assert a.expected_outcome in allowed, (
            f"Attack {a.id} has disallowed expected_outcome "
            f"{a.expected_outcome!r}"
        )


def test_every_severity_is_allowed() -> None:
    """`severity` must be low / medium / high / critical."""
    allowed = set(get_args(Severity))
    assert allowed == {"low", "medium", "high", "critical"}
    for a in ATTACK_CORPUS:
        assert a.severity in allowed, (
            f"Attack {a.id} has disallowed severity {a.severity!r}"
        )


def test_every_category_is_declared() -> None:
    """Every attack's category must be a member of ATTACK_CATEGORIES."""
    allowed = set(ATTACK_CATEGORIES)
    for a in ATTACK_CORPUS:
        assert a.category in allowed, (
            f"Attack {a.id} has unknown category {a.category!r}"
        )


def test_category_counts_meet_targets() -> None:
    """Each category must meet its declared minimum count."""
    counts: dict[str, int] = {}
    for a in ATTACK_CORPUS:
        counts[a.category] = counts.get(a.category, 0) + 1
    for cat, target in ATTACK_CATEGORY_TARGETS.items():
        assert counts.get(cat, 0) >= target, (
            f"Category {cat!r} has {counts.get(cat, 0)} attacks, "
            f"target >= {target}"
        )


def test_prompt_injection_bucket_has_at_least_ten() -> None:
    """Redundant guard: prompt_injection must have >= 10 attacks."""
    n = sum(1 for a in ATTACK_CORPUS if a.category == "prompt_injection")
    assert n >= 10, f"prompt_injection has only {n} attacks"


def test_jailbreak_bucket_has_at_least_eight() -> None:
    n = sum(1 for a in ATTACK_CORPUS if a.category == "jailbreak")
    assert n >= 8, f"jailbreak has only {n} attacks"


def test_privacy_override_bypass_has_at_least_six() -> None:
    n = sum(
        1 for a in ATTACK_CORPUS if a.category == "privacy_override_bypass"
    )
    assert n >= 6, f"privacy_override_bypass has only {n} attacks"


def test_rate_limit_abuse_has_at_least_five() -> None:
    n = sum(1 for a in ATTACK_CORPUS if a.category == "rate_limit_abuse")
    assert n >= 5, f"rate_limit_abuse has only {n} attacks"


def test_load_external_corpus_round_trip(tmp_path: Path) -> None:
    """`load_external_corpus` must round-trip the default corpus."""
    path = tmp_path / "corpus.json"
    serialised = [a.model_dump(mode="json") for a in ATTACK_CORPUS]
    path.write_text(json.dumps(serialised), encoding="utf-8")
    loaded = load_external_corpus(path)
    assert len(loaded) == len(ATTACK_CORPUS)
    for orig, round_tripped in zip(ATTACK_CORPUS, loaded):
        assert orig.id == round_tripped.id
        assert orig.category == round_tripped.category
        assert orig.expected_outcome == round_tripped.expected_outcome
        assert orig.severity == round_tripped.severity


def test_load_external_corpus_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_external_corpus(tmp_path / "does-not-exist.json")


def test_load_external_corpus_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "not-a-list.json"
    path.write_text(json.dumps({"oops": True}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_external_corpus(path)


def test_attack_is_frozen() -> None:
    """:class:`Attack` has ``frozen=True``; assignment must fail."""
    a = ATTACK_CORPUS[0]
    with pytest.raises(Exception):
        a.id = "mutated"  # type: ignore[misc]
