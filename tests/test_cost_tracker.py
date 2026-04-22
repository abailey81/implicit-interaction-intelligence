"""Tests for :mod:`i3.cloud.cost_tracker`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from i3.cloud.cost_tracker import CostReport, CostTracker
from i3.cloud.providers.base import TokenUsage


def _write_pricing(tmp_path: Path) -> Path:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "as_of": "2026-04-22",
                "models": {
                    "test_provider": {
                        "model-a": {
                            "input": 3.0,
                            "output": 15.0,
                            "cached_input": 0.3,
                        },
                        "model-b": {"input": 1.0, "output": 4.0},
                        "*": {"input": 0.5, "output": 2.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_record_computes_cost() -> None:
    tmp = Path.cwd() / "_tmp_pricing"
    tmp.mkdir(exist_ok=True)
    try:
        path = _write_pricing(tmp)
        tracker = CostTracker(pricing_path=path)
        usage = TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
        )
        cost = tracker.record("test_provider", "model-a", usage, 123)
        # 1 MTok in + 1 MTok out at (3, 15) = 18 USD.
        assert cost == pytest.approx(18.0)
    finally:
        for child in tmp.iterdir():
            child.unlink()
        tmp.rmdir()


def test_aggregate_across_multiple_calls(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    tracker.record(
        "test_provider",
        "model-a",
        TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        10,
    )
    tracker.record(
        "test_provider",
        "model-a",
        TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
        20,
    )
    report = tracker.report()
    assert report.total_calls == 2
    assert report.total_prompt_tokens == 300
    assert report.total_completion_tokens == 150


def test_pricing_lookup_via_wildcard(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    # Unknown model within known provider -> "*" fallback.
    cost = tracker.record(
        "test_provider",
        "unknown-model",
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000),
        0,
    )
    assert cost == pytest.approx(0.5)  # input rate 0.5 * 1 MTok.


def test_unknown_provider_model_is_tracked_but_free(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    cost = tracker.record(
        "nonexistent_provider",
        "foo",
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        1,
    )
    assert cost == 0.0
    report = tracker.report()
    assert ("nonexistent_provider", "foo") in report.unknown_models


def test_cached_tokens_cheaper(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    # 1 MTok prompt split 50/50 cached vs fresh; 0 completion.
    usage = TokenUsage(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        total_tokens=1_000_000,
        cached_tokens=500_000,
    )
    cost = tracker.record("test_provider", "model-a", usage, 0)
    # 500k fresh * $3 + 500k cached * $0.3 = $1.5 + $0.15 = $1.65.
    assert cost == pytest.approx(1.65)


def test_report_by_provider_buckets(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    tracker.record(
        "test_provider",
        "model-a",
        TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        5,
    )
    tracker.record(
        "test_provider",
        "model-b",
        TokenUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
        50,
    )
    report = tracker.report()
    assert "test_provider" in report.by_provider
    assert report.by_provider["test_provider"]["calls"] == 2
    assert len(report.by_model) == 2


def test_report_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    tracker.record(
        "test_provider",
        "model-a",
        TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        1,
    )
    d = tracker.report().to_dict()
    json.dumps(d)  # must not raise


def test_reset_clears_ledger(tmp_path: Path) -> None:
    path = _write_pricing(tmp_path)
    tracker = CostTracker(pricing_path=path)
    tracker.record(
        "test_provider",
        "model-a",
        TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        1,
    )
    tracker.reset()
    report = tracker.report()
    assert report.total_calls == 0
    assert report.total_prompt_tokens == 0


def test_real_pricing_file_is_valid_json() -> None:
    """Sanity-check the shipped pricing_2026.json."""
    from i3.cloud.cost_tracker import _PRICING_PATH  # type: ignore[attr-defined]

    data = json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert "models" in data
    # All 11 providers must have at least one model priced.
    required = {
        "anthropic",
        "openai",
        "google",
        "azure",
        "bedrock",
        "mistral",
        "cohere",
        "ollama",
        "openrouter",
        "litellm",
        "huawei_pangu",
    }
    missing = required - set(data["models"])
    assert not missing, f"pricing missing for: {missing}"
