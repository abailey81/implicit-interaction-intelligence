"""Iter 92 — pricing_2026.json integrity tests.

The CostTracker reads i3/cloud/pricing_2026.json on every report().
A malformed table silently zeros out cost calculations; these tests
pin the schema + a few sanity invariants.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


_PRICING = Path("i3/cloud/pricing_2026.json")


@pytest.fixture(scope="module")
def table():
    if not _PRICING.exists():
        pytest.skip("pricing_2026.json not present")
    return json.loads(_PRICING.read_text(encoding="utf-8"))


def test_top_level_metadata_present(table):
    for k in ("schema_version", "as_of", "currency", "unit", "models"):
        assert k in table, f"top-level key {k!r} missing"


def test_currency_is_usd(table):
    assert table["currency"] == "USD"


def test_unit_is_per_million_tokens(table):
    assert "M" in str(table["unit"]).lower() or \
           "million" in str(table["unit"]).lower() or \
           "1e6" in str(table["unit"])


def test_models_is_dict(table):
    assert isinstance(table["models"], dict)
    assert len(table["models"]) >= 5


def _iter_models(table):
    """Yield (provider, model_id, rate_dict) tuples across the
    nested {provider: {model_id: {rate_keys}}} schema."""
    for provider, models in table["models"].items():
        if not isinstance(models, dict):
            continue
        for model_id, info in models.items():
            if isinstance(info, dict):
                yield provider, model_id, info


def test_each_model_has_input_and_output_rate(table):
    """Every entry must have at least input + output rates so the
    CostTracker can price a call."""
    for provider, model_id, info in _iter_models(table):
        keys = set(info.keys())
        has_input = bool(keys & {"input", "input_per_mtok",
                                  "prompt", "input_tokens"})
        has_output = bool(keys & {"output", "output_per_mtok",
                                   "completion", "output_tokens"})
        assert has_input, f"{provider}/{model_id}: no input-rate in {keys}"
        assert has_output, f"{provider}/{model_id}: no output-rate in {keys}"


def test_no_negative_rates(table):
    for provider, model_id, info in _iter_models(table):
        for k, v in info.items():
            if isinstance(v, (int, float)):
                assert v >= 0, f"{provider}/{model_id}.{k} negative: {v}"


def test_includes_at_least_anthropic_and_openai_and_gemini(table):
    """Minimum vendor coverage so the iter-52 cascade arm C (Gemini)
    is priced."""
    providers = {p.lower() for p in table["models"].keys()}
    assert "anthropic" in providers
    assert "openai" in providers
    assert any(p in providers for p in ("google", "gemini"))


def test_round_trips_through_json_dumps(table):
    s = json.dumps(table)
    parsed = json.loads(s)
    assert parsed["schema_version"] == table["schema_version"]
