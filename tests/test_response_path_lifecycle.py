"""Iter 121 — Pipeline._last_response_path setter / cascade-classifier
end-to-end test that exercises the per-turn lifecycle without booting
the full Pipeline.
"""
from __future__ import annotations

import pytest

from i3.pipeline.engine import Pipeline


def test_classifier_handles_all_documented_paths():
    classify = Pipeline._classify_cascade_arm
    cases = [
        ("slm",                  "local_slm", "slm"),
        ("retrieval",            "local_slm", "retrieval"),
        ("retrieval_borderline", "local_slm", "retrieval"),
        ("tool:intent",          "local_slm", "qwen_intent"),
        ("tool:fact",            "local_slm", "tool"),
        ("tool:recap",           "local_slm", "tool"),
        ("tool:safety",          "local_slm", "tool"),
        ("tool:name",            "local_slm", "tool"),
        ("tool:clarify",         "local_slm", "tool"),
        ("explain_decomposed",   "local_slm", "slm"),
        ("cloud_llm",            "cloud_llm", "gemini_cloud"),
        ("ood",                  "local_slm", "slm"),
        ("none",                 "local_slm", "slm"),
        ("unknown",              "?",         "other"),
    ]
    for path, route, expected in cases:
        out = classify(path, route)
        assert out == expected, f"({path!r}, {route!r}) -> {out!r} (want {expected!r})"


def test_classifier_case_insensitive():
    """Mixed case should still classify correctly."""
    assert Pipeline._classify_cascade_arm("SLM", "LOCAL_SLM") == "slm"
    assert Pipeline._classify_cascade_arm("Tool:Intent", "?") == "qwen_intent"


def test_classifier_handles_none_or_empty():
    """Should not crash on empty / None inputs."""
    assert Pipeline._classify_cascade_arm("", "") == "other"
    assert Pipeline._classify_cascade_arm(None, None) == "other"  # type: ignore[arg-type]


def test_classifier_returns_str():
    out = Pipeline._classify_cascade_arm("slm", "local_slm")
    assert isinstance(out, str)
    assert out  # non-empty


def test_classifier_arm_name_in_known_set():
    """The classifier must always return one of the known arm names."""
    known = {"slm", "qwen_intent", "gemini_cloud",
             "retrieval", "tool", "other"}
    for path in ("slm", "tool:intent", "cloud_llm", "retrieval",
                 "tool:fact", "anything_else"):
        out = Pipeline._classify_cascade_arm(path, "?")
        assert out in known, f"unknown arm {out!r}"
