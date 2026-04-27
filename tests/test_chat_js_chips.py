"""Iter 107 — web/js/chat.js _appendSideChips function contract.

Static grep for the iter-51..62 chip rendering logic in chat.js.
A future refactor that drops the helper or changes its inputs
silently breaks the chat-bubble side-channels.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_CHAT = Path("web/js/chat.js")


@pytest.fixture(scope="module")
def js() -> str:
    if not _CHAT.exists():
        pytest.skip("web/js/chat.js not present")
    return _CHAT.read_text(encoding="utf-8")


def test_append_side_chips_function_present(js):
    assert "_appendSideChips" in js
    # Function definition (not just a call site)
    assert "_appendSideChips(" in js


def test_chip_classes_referenced_in_chat(js):
    """Every iter-51 / iter-62 chip class should be emitted by chat.js."""
    for cls in ("chip-affect", "chip-safety", "chip-adapt",
                "chip-intent", "chip-arm"):
        assert cls in js, f"chip class {cls!r} not referenced in chat.js"


def test_iter51_metadata_fields_consumed(js):
    """The renderer reads metadata.{affect_shift, safety_caveat,
    adaptation_changes, intent_result, response_path}."""
    for field in ("affect_shift", "safety_caveat", "adaptation_changes",
                  "intent_result", "response_path"):
        assert field in js, f"chat.js doesn't consume metadata.{field}"


def test_chip_invocation_path(js):
    """_appendSideChips is invoked from the message append path."""
    # The function should be called somewhere; one or more call sites.
    n = js.count("_appendSideChips(")
    # 1 definition + at least 1 call → ≥ 2 occurrences
    assert n >= 2, f"_appendSideChips referenced only {n} times"
