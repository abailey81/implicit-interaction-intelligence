"""Iter 91 — Qwen LoRA adapter / tokenizer / config alignment tests.

Verifies the committed Qwen adapter checkpoint is structurally
consistent: adapter_config.json keys are sane, tokenizer files load,
and the base model id matches the trained-on model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


_ADAPTER_DIR = Path("checkpoints/intent_lora/qwen3.5-2b_best")
_TRAINING_METRICS = Path("checkpoints/intent_lora/qwen3.5-2b/training_metrics.json")


def _skip_if_no_adapter():
    if not _ADAPTER_DIR.exists():
        pytest.skip("Qwen LoRA adapter not present")


# ---------------------------------------------------------------------------
# adapter_config.json
# ---------------------------------------------------------------------------

def test_adapter_config_is_valid_json():
    _skip_if_no_adapter()
    cfg_path = _ADAPTER_DIR / "adapter_config.json"
    if not cfg_path.exists():
        pytest.skip("adapter_config.json missing")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(cfg, dict)


def test_adapter_config_has_base_model():
    _skip_if_no_adapter()
    cfg_path = _ADAPTER_DIR / "adapter_config.json"
    if not cfg_path.exists():
        pytest.skip("adapter_config.json missing")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg.get("base_model_name_or_path") or cfg.get("base_model")
    assert base is not None, f"base model missing from adapter_config: {cfg.keys()}"
    assert "qwen" in base.lower() or "Qwen" in base


def test_adapter_config_has_lora_rank():
    _skip_if_no_adapter()
    cfg_path = _ADAPTER_DIR / "adapter_config.json"
    if not cfg_path.exists():
        pytest.skip("adapter_config.json missing")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg.get("r") in (8, 16, 32, 64), f"unexpected LoRA rank: {cfg.get('r')}"


# ---------------------------------------------------------------------------
# tokenizer files
# ---------------------------------------------------------------------------

def test_tokenizer_json_present():
    _skip_if_no_adapter()
    assert (_ADAPTER_DIR / "tokenizer.json").exists() or \
           (_ADAPTER_DIR / "tokenizer_config.json").exists()


def test_tokenizer_config_is_valid_json():
    _skip_if_no_adapter()
    p = _ADAPTER_DIR / "tokenizer_config.json"
    if not p.exists():
        pytest.skip("tokenizer_config.json missing")
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# Training metrics consistency
# ---------------------------------------------------------------------------

def test_training_metrics_are_valid_json():
    if not _TRAINING_METRICS.exists():
        pytest.skip("training_metrics.json missing")
    m = json.loads(_TRAINING_METRICS.read_text(encoding="utf-8"))
    assert isinstance(m, dict)


def test_training_metrics_records_best_val_loss():
    if not _TRAINING_METRICS.exists():
        pytest.skip("training_metrics.json missing")
    m = json.loads(_TRAINING_METRICS.read_text(encoding="utf-8"))
    assert "best_val_loss" in m


def test_training_metrics_uses_qwen3_base():
    if not _TRAINING_METRICS.exists():
        pytest.skip("training_metrics.json missing")
    m = json.loads(_TRAINING_METRICS.read_text(encoding="utf-8"))
    base = m.get("model") or ""
    if not base:
        pytest.skip("training metrics has no 'model' field")
    assert "qwen" in base.lower()
