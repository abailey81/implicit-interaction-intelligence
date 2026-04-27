"""Iter 96 — CSS class coverage for the iter-62 cascade-arm chips.

The chat.js renderer emits chips with classes chip-arm-{a,b,c,r,t}.
Each must have a CSS rule in huawei_tabs.css; missing rules silently
render unstyled chips.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_CSS = Path("web/css/huawei_tabs.css")


@pytest.fixture(scope="module")
def css() -> str:
    if not _CSS.exists():
        pytest.skip("web/css/huawei_tabs.css not present")
    return _CSS.read_text(encoding="utf-8")


@pytest.mark.parametrize("cls", [
    "chip-arm",       # base
    "chip-arm-a",     # SLM
    "chip-arm-b",     # Qwen LoRA
    "chip-arm-c",     # Cloud
    "chip-arm-r",     # Retrieval
    "chip-arm-t",     # Tool
])
def test_iter62_chip_arm_classes_present(css, cls):
    assert f".message-chip.{cls}" in css or f".{cls}" in css, \
        f"missing CSS rule for {cls}"


@pytest.mark.parametrize("cls", [
    "chip-affect",    # iter 51
    "chip-safety",    # iter 51
    "chip-adapt",     # iter 51
    "chip-intent",    # iter 51
])
def test_iter51_chip_classes_present(css, cls):
    assert f".message-chip.{cls}" in css or f".{cls}" in css, \
        f"missing CSS rule for {cls}"


def test_cascade_arms_grid_class_present(css):
    """Iter 53 + iter 55 use .cascade-arms-grid + .cascade-arm-card."""
    assert ".cascade-arms-grid" in css
    assert ".cascade-arm-card" in css


def test_huawei_status_pill_classes_present(css):
    """Iter 51 stack-tab subsystem cards use .huawei-status-pill."""
    assert ".huawei-status-pill" in css
    assert ".huawei-status-pill.ok" in css


def test_intent_json_block_class_present(css):
    """Iter 51 Intent dashboard tab uses .intent-json-block for syntax
    highlighting."""
    assert ".intent-json-block" in css
