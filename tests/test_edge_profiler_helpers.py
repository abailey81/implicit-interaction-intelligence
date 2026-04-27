"""Iter 104 — i3.edge.profiler helper-function invariants."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from i3.edge.profiler import (
    EdgeReport,
    _count_unique_params,
    _percentile,
    _serialised_size_mb,
    _to_dtype_copy,
)


# ---------------------------------------------------------------------------
# Param counting
# ---------------------------------------------------------------------------

def test_count_unique_params_handles_tied_weights():
    """A tied-weights model should not double-count the shared tensor."""
    embed = nn.Embedding(100, 16)
    proj = nn.Linear(16, 100)
    proj.weight = embed.weight  # tie
    model = nn.Sequential(embed, proj)
    n = _count_unique_params(model)
    # Embedding has 100*16=1600 params; proj.bias has 100 → 1700 unique total
    # (proj.weight is shared with embed.weight so not double-counted).
    assert n <= 1700 + 16  # tolerant upper bound
    assert n >= 1600


def test_count_unique_params_distinct_modules():
    a = nn.Linear(8, 16)
    b = nn.Linear(16, 4)
    model = nn.Sequential(a, b)
    n = _count_unique_params(model)
    expected = (8 * 16 + 16) + (16 * 4 + 4)
    assert n == expected


# ---------------------------------------------------------------------------
# Serialised size
# ---------------------------------------------------------------------------

def test_serialised_size_positive():
    model = nn.Linear(64, 32)
    size = _serialised_size_mb(model)
    assert size > 0


def test_serialised_size_scales_with_model():
    small = _serialised_size_mb(nn.Linear(8, 8))
    large = _serialised_size_mb(nn.Linear(512, 512))
    assert large > small


# ---------------------------------------------------------------------------
# Dtype copy
# ---------------------------------------------------------------------------

def test_to_dtype_copy_changes_dtype():
    model = nn.Linear(16, 8)
    bf16 = _to_dtype_copy(model, torch.bfloat16)
    for p in bf16.parameters():
        assert p.dtype == torch.bfloat16


def test_to_dtype_copy_preserves_shapes():
    model = nn.Linear(16, 8)
    fp16 = _to_dtype_copy(model, torch.float16)
    for orig, new in zip(model.parameters(), fp16.parameters()):
        assert orig.shape == new.shape


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def test_percentile_median_of_five():
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)


def test_percentile_p95():
    p95 = _percentile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 95)
    assert 9.0 <= p95 <= 10.0


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single_element():
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 95) == 42.0


# ---------------------------------------------------------------------------
# EdgeReport
# ---------------------------------------------------------------------------

def test_edge_report_to_dict_round_trips():
    rep = EdgeReport(
        slm_params=1_000_000,
        slm_size_fp32_mb=4.0,
        slm_size_bf16_mb=2.0,
        slm_size_int8_mb=1.0,
        tcn_params=50_000,
        tcn_size_fp32_mb=0.2,
        tcn_size_int8_mb=0.05,
        latency_ms_p50=20.0,
        latency_ms_p95=35.0,
        latency_ms_encoder_p50=2.0,
        latency_ms_encoder_p95=3.0,
        memory_peak_mb=200.0,
        onnx_size_mb=4.5,
        deployable_to=["RPi5", "Jetson Nano"],
        timestamp="2026-04-27",
        device="cpu",
        slm_checkpoint="/tmp/slm.pt",
        tcn_checkpoint="/tmp/tcn.pt",
    )
    d = rep.to_dict()
    assert d["slm_params"] == 1_000_000
    assert isinstance(d["deployable_to"], list)
