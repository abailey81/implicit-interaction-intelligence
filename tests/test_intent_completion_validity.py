"""Iter 132 — Intent dataset completion-string validity (deeper sweep)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


_DATA_DIR = Path("data/processed/intent")


@pytest.fixture(scope="module")
def all_rows():
    if not _DATA_DIR.exists():
        pytest.skip("intent dataset not present")
    rows = []
    for split in ("train", "val", "test"):
        p = _DATA_DIR / f"{split}.jsonl"
        if not p.exists():
            pytest.skip(f"{p} missing")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def test_every_completion_parses_as_json(all_rows):
    n_bad = 0
    for r in all_rows:
        try:
            json.loads(r["completion"])
        except Exception:
            n_bad += 1
    assert n_bad == 0, f"{n_bad} unparseable completions"


def test_completion_action_matches_output_action(all_rows):
    n_mismatch = 0
    for r in all_rows:
        completion = json.loads(r["completion"])
        if completion.get("action") != r["output"]["action"]:
            n_mismatch += 1
    assert n_mismatch == 0, f"{n_mismatch} action mismatches"


def test_completion_params_match_output_params(all_rows):
    n_mismatch = 0
    for r in all_rows:
        completion = json.loads(r["completion"])
        if completion.get("params", {}) != r["output"].get("params", {}):
            n_mismatch += 1
    assert n_mismatch == 0, f"{n_mismatch} param mismatches"


def test_no_empty_inputs(all_rows):
    for r in all_rows:
        assert isinstance(r["input"], str) and r["input"].strip(), \
            f"empty input row: {r}"


def test_action_distribution_balanced(all_rows):
    """Per-action count should be reasonable for a stratified dataset."""
    from collections import Counter
    counts = Counter(r["output"]["action"] for r in all_rows)
    # Expect at least 5 actions with > 100 examples each.
    well_represented = [a for a, c in counts.items() if c >= 100]
    assert len(well_represented) >= 5, \
        f"only {len(well_represented)} actions ≥ 100 examples"
