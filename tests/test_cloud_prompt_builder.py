"""Iter 106 — PromptBuilder system-prompt assembly tests."""
from __future__ import annotations

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.cloud.prompt_builder import PromptBuilder


@pytest.fixture
def builder():
    return PromptBuilder()


def _vec(load=0.5, access=0.0, tone=0.5):
    return AdaptationVector(
        cognitive_load=load,
        style_mirror=StyleVector.default(),
        emotional_tone=tone,
        accessibility=access,
    )


def test_build_returns_non_empty_string(builder):
    out = builder.build_system_prompt(_vec())
    assert isinstance(out, str)
    assert len(out) > 0


def test_build_includes_base_role_text(builder):
    out = builder.build_system_prompt(_vec())
    assert "AI companion" in out or "warm" in out.lower()


def test_build_does_not_leak_format_placeholders(builder):
    """The system prompt template must be fully interpolated.  Bare
    ``{adaptation_instructions}`` would mean a missing substitution."""
    out = builder.build_system_prompt(_vec())
    for placeholder in ("{adaptation_instructions}", "{user_context}"):
        assert placeholder not in out, \
            f"unsubstituted {placeholder} in system prompt"


def test_build_with_user_summary(builder):
    out = builder.build_system_prompt(
        _vec(),
        user_summary={"sessions_total": 5, "favourite_topics": ["python"]},
    )
    assert isinstance(out, str)


def test_high_accessibility_changes_prompt(builder):
    a = builder.build_system_prompt(_vec(access=0.0))
    b = builder.build_system_prompt(_vec(access=0.95))
    # The two prompts should differ (high accessibility adds simplification
    # instructions or similar).
    assert a != b, "accessibility doesn't influence the prompt"


def test_high_cognitive_load_changes_prompt(builder):
    a = builder.build_system_prompt(_vec(load=0.1))
    b = builder.build_system_prompt(_vec(load=0.9))
    assert a != b, "cognitive_load doesn't influence the prompt"


def test_includes_no_pii_from_summary(builder):
    """Even if user_summary contains PII-like values, the builder
    should treat them as opaque tags (no email regexes leaked back)."""
    out = builder.build_system_prompt(
        _vec(),
        user_summary={"alias": "tester"},
    )
    assert "@" not in out  # no accidental email injection
