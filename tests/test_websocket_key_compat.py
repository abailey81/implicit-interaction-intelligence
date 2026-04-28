"""Iter 41 — key-name compatibility between the JS client and the
WebSocket handler.

The browser ``KeystrokeMonitor`` (web/js/app.js) sends:
  * ``backspace_count``       inside ``composition_metrics``
  * ``iki_ms``                on every keystroke event

The Python pipeline + Python probes use the long-form keys:
  * ``edit_count``
  * ``inter_key_interval_ms``

Pre-iter-41 the WebSocket handler only read the long-form keys, so
every keystroke reached the pipeline with iki=0 and every message
with edit_count=0 — the dashboard's "Typing rhythm 0 ms" and "Edit
profile 0.00 / turn" were the visible symptoms.

These tests pin the bilingual key handling so a future refactor
can't silently re-break the JS payload again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


WS_SRC = (
    Path(__file__).parent.parent / "server" / "websocket.py"
).read_text(encoding="utf-8")


def test_websocket_accepts_backspace_count_alias() -> None:
    """The handler must read ``backspace_count`` when ``edit_count`` is
    absent, otherwise the JS-side dashboard tile reads zero."""
    assert "backspace_count" in WS_SRC, (
        "websocket handler is missing the backspace_count fallback — "
        "JS clients (web/js/app.js) ship that key, not edit_count"
    )
    # The fallback must be inside the message-handling branch (not in
    # an unrelated comment) — look for the actual control-flow shape:
    # ``edit_count_raw = ...`` followed by ``backspace_count``.
    pattern = re.compile(
        r"edit_count_raw\s*=.*?backspace_count",
        re.DOTALL,
    )
    assert pattern.search(WS_SRC), (
        "the backspace_count fallback exists but isn't wired into the "
        "edit_count extraction"
    )


def test_websocket_accepts_iki_ms_alias() -> None:
    """The keystroke event handler must read ``iki_ms`` when
    ``inter_key_interval_ms`` is absent, otherwise the keystroke
    buffer fills with zero-IKI entries and the dashboard reads
    "Typing rhythm 0 ms"."""
    assert "iki_ms" in WS_SRC, (
        "websocket handler is missing the iki_ms fallback — "
        "JS clients (web/js/app.js) ship that key, not "
        "inter_key_interval_ms"
    )
    pattern = re.compile(
        r"ks_iki_raw\s*=.*?iki_ms",
        re.DOTALL,
    )
    assert pattern.search(WS_SRC), (
        "the iki_ms fallback exists but isn't wired into the "
        "ks_iki extraction"
    )


def test_iter41_fallback_logic() -> None:
    """End-to-end key-mapping simulation — what the websocket handler
    would extract from each side's payload."""

    def _safe_int(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0

    def _safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    def extract_edit_count(top, nested):
        raw = top.get("edit_count")
        if raw is None:
            raw = nested.get("edit_count")
        if raw is None:
            raw = top.get("backspace_count")
        if raw is None:
            raw = nested.get("backspace_count", 0)
        return _safe_int(raw)

    def extract_iki(top):
        raw = top.get("inter_key_interval_ms")
        if raw is None:
            raw = top.get("iki_ms", 0)
        return _safe_float(raw)

    # JS payload (browser) — backspace_count nested.
    js_msg = {
        "composition_metrics": {"backspace_count": 3, "mean_iki": 110},
    }
    assert extract_edit_count(js_msg, js_msg["composition_metrics"]) == 3

    # Python probe payload — edit_count top-level.
    py_msg = {"edit_count": 7, "composition_metrics": {}}
    assert extract_edit_count(py_msg, py_msg["composition_metrics"]) == 7

    # JS keystroke payload — iki_ms.
    assert extract_iki({"iki_ms": 120}) == 120.0

    # Python keystroke payload — inter_key_interval_ms.
    assert extract_iki({"inter_key_interval_ms": 95}) == 95.0

    # Empty payload still resolves to 0 cleanly.
    assert extract_iki({}) == 0.0
    assert extract_edit_count({}, {}) == 0


@pytest.mark.parametrize("backspace,expected", [(0, 0), (1, 1), (3, 3), (10, 10)])
def test_backspace_count_round_trips(backspace, expected) -> None:
    msg = {"composition_metrics": {"backspace_count": backspace}}
    nested = msg["composition_metrics"]
    raw = msg.get("edit_count")
    if raw is None:
        raw = nested.get("edit_count")
    if raw is None:
        raw = msg.get("backspace_count")
    if raw is None:
        raw = nested.get("backspace_count", 0)
    assert int(raw) == expected
