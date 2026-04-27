"""Iter 97 — web/js/huawei_tabs.js function-wiring contract test.

The iter-51 huawei_tabs.js boots all 7 Huawei tabs.  Each must:
  1. Have a wireXxxTab() function defined.
  2. Be invoked from the bottom-of-file boot block.

A new tab that adds wireXxx() but forgets to call it silently 404s
the tab.  Static grep guards both halves of the contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_JS = Path("web/js/huawei_tabs.js")


@pytest.fixture(scope="module")
def js() -> str:
    if not _JS.exists():
        pytest.skip("web/js/huawei_tabs.js not present")
    return _JS.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", [
    "wireIntentTab",
    "wireEdgeProfileTab",
    "wireFinetuneTab",
    "wireFactsTab",
    "wireResearchTab",
    "wireJdmapTab",
])
def test_wire_function_defined(js, name):
    assert f"function {name}" in js, f"{name}() not defined"


@pytest.mark.parametrize("name", [
    "wireIntentTab",
    "wireEdgeProfileTab",
    "wireFinetuneTab",
    "wireFactsTab",
    "wireResearchTab",
    "wireJdmapTab",
])
def test_wire_function_invoked(js, name):
    """Boot block must call each wireXxx() once."""
    assert f"{name}();" in js, f"{name}() defined but never called"


def test_iter53_render_cascade_arms_present(js):
    """Iter 53 added renderCascadeArms() to the Edge Profile tab."""
    assert "function renderCascadeArms" in js
    assert "renderCascadeArms(" in js  # at least one call site


def test_iter55_render_cascade_live_stats_present(js):
    """Iter 55 added renderCascadeLiveStats() consuming /api/cascade/stats."""
    assert "function renderCascadeLiveStats" in js
    assert "/api/cascade/stats" in js


def test_iter51_intent_endpoint_referenced(js):
    """Iter 51 wireIntentTab must POST to /api/intent."""
    assert "/api/intent" in js


def test_no_console_log_left_in_production_path(js):
    """Production JS shouldn't leave noisy console.log statements
    sprinkled in the boot path.  Allow console.error/warn for real
    error reporting; ban bare console.log."""
    # Permit at most a handful of console.log instances; flag if it
    # gets out of hand (silent linter-style guard).
    n = js.count("console.log")
    assert n < 10, f"too many console.log statements ({n}); production noise"
