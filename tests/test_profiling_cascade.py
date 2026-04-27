"""Iter 53/54 — profiling endpoint shape + cascade_arms contract.

These tests exercise ``Pipeline.get_profiling_report`` directly (no
HTTP / no FastAPI), so they're fast and have no network dependency.
They cover the contract the dashboard's Edge Profile tab depends on.
"""
from __future__ import annotations

import asyncio

import pytest

from i3.pipeline.engine import Pipeline


def _bare_pipeline() -> Pipeline:
    return Pipeline.__new__(Pipeline)


def _report() -> dict:
    p = _bare_pipeline()
    return asyncio.get_event_loop().run_until_complete(p.get_profiling_report())


def test_top_level_shape():
    rep = _report()
    assert isinstance(rep, dict)
    for key in ("components", "total_latency_ms", "memory_mb",
                "fits_budget", "budget_ms", "device_class",
                "cascade_arms"):
        assert key in rep, f"missing {key!r} in profiling report"


def test_components_shape():
    rep = _report()
    comps = rep["components"]
    assert isinstance(comps, list) and len(comps) >= 9
    for c in comps:
        for f in ("name", "params_m", "fp32_mb", "int8_mb", "p50_ms"):
            assert f in c, f"component missing {f!r}: {c}"
        assert isinstance(c["name"], str)
        assert isinstance(c["params_m"], (int, float))


def test_includes_cascade_arm_components():
    """Iter 52 added Qwen + Gemini rows; verify they're surfaced."""
    rep = _report()
    names = [c["name"] for c in rep["components"]]
    assert any("Qwen" in n for n in names), names
    assert any("Gemini" in n for n in names), names


def test_cascade_arms_block():
    rep = _report()
    arms = rep["cascade_arms"]
    assert isinstance(arms, dict)
    assert set(arms.keys()) == {"A_slm", "B_qwen_intent", "C_gemini_cloud"}
    for k, v in arms.items():
        assert isinstance(v, dict)
        assert "latency_ms" in v
        assert "memory_mb" in v
        assert "fires" in v
        assert isinstance(v["latency_ms"], (int, float))


def test_budget_invariants():
    rep = _report()
    assert rep["budget_ms"] > 0
    assert rep["total_latency_ms"] > 0
    # Arm A (SLM, every-turn) must fit the budget
    assert rep["cascade_arms"]["A_slm"]["latency_ms"] <= rep["budget_ms"]
    assert rep["fits_budget"] is True


def test_arm_b_only_when_command():
    """The Qwen arm is only paid on command-shaped turns; verify
    the report annotates that (so the dashboard can render the
    'fires' caption)."""
    rep = _report()
    assert "command" in rep["cascade_arms"]["B_qwen_intent"]["fires"].lower()


def test_arm_c_is_opt_in():
    rep = _report()
    fires = rep["cascade_arms"]["C_gemini_cloud"]["fires"].lower()
    assert "opt-in" in fires or "consent" in fires or "explicit" in fires
