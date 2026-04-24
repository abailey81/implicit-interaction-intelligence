"""Unit tests for ``scripts.orchestration_progress``.

Uses representative log lines pulled from live orchestration logs
(see ``reports/orchestration/*.log``) plus synthetic lines for
parsers whose stages haven't been exercised yet.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ``scripts/`` is not a Python package — load ``orchestration_progress``
# directly from its file path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE_PATH = _REPO_ROOT / "scripts" / "orchestration_progress.py"
_SPEC = importlib.util.spec_from_file_location(
    "orchestration_progress", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
progress = importlib.util.module_from_spec(_SPEC)
sys.modules["orchestration_progress"] = progress
_SPEC.loader.exec_module(progress)


# ---------------------------------------------------------------------------
# ANSI stripper
# ---------------------------------------------------------------------------


def test_strip_ansi_removes_colour_codes() -> None:
    assert progress.strip_ansi("\x1b[31merror\x1b[0m") == "error"
    assert progress.strip_ansi("\x1b[38;5;196mhex\x1b[0m") == "hex"


def test_strip_ansi_noop_on_plain_text() -> None:
    assert progress.strip_ansi("just text") == "just text"


# ---------------------------------------------------------------------------
# train-encoder
# ---------------------------------------------------------------------------


def test_parse_train_encoder_epoch_summary() -> None:
    line = (
        "2026-04-24 19:32:34,549  i3.encoder.train  INFO  "
        "Epoch   1/10  train_loss=3.1857  val_loss=2.6203  sil=0.388"
    )
    up = progress.parse_train_encoder(line)
    assert up is not None
    assert up.completed == 1
    assert up.total == 10
    assert "epoch 1/10" in up.description
    assert "train_loss=3.1857" in up.description


def test_parse_train_encoder_none_on_junk() -> None:
    assert progress.parse_train_encoder("unrelated junk line") is None


# ---------------------------------------------------------------------------
# train-slm
# ---------------------------------------------------------------------------


def test_parse_train_slm_step_line() -> None:
    line = "Step 120/2830  loss=2.4989  avg_loss=2.5112  lr=3.00e-04  grad_norm=0.12"
    up = progress.parse_train_slm(line)
    assert up is not None
    assert up.completed == 120
    assert up.total == 2830
    assert "step 120/2830" in up.description
    assert "loss=2.4989" in up.description


def test_parse_train_slm_plan_line_primes_total() -> None:
    up = progress.parse_train_slm("Converting 5 epochs to 2830 steps (566 steps/epoch)")
    assert up is not None
    assert up.total == 2830
    assert up.completed == 0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def test_parse_data_progress_line() -> None:
    up = progress.parse_data("  ... 2000 / 3200 sessions")
    assert up is not None
    assert up.completed == 2000
    assert up.total == 3200


def test_parse_data_plan_line() -> None:
    up = progress.parse_data("Generating 3200 sessions x 20 messages (window=10) ...")
    assert up is not None
    assert up.total == 3200


# ---------------------------------------------------------------------------
# dialogue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected_current",
    [
        ("Step 1: Loading raw dialogue data...", 1),
        ("Step 3: Building tokenizer vocabulary (size=4096)...", 3),
        ("Step 6: Saving processed data...", 6),
    ],
)
def test_parse_dialogue(line: str, expected_current: int) -> None:
    up = progress.parse_dialogue(line)
    assert up is not None
    assert up.completed == expected_current
    assert up.total == 6


# ---------------------------------------------------------------------------
# test (pytest/xdist)
# ---------------------------------------------------------------------------


def test_parse_test_pct_line() -> None:
    up = progress.parse_test("[gw3] [ 42%] PASSED tests/test_foo.py::test_bar")
    assert up is not None
    assert up.completed == 42
    assert up.total == 100


def test_parse_test_summary_line() -> None:
    up = progress.parse_test("====== 182 passed, 0 failed, 3 skipped in 23.45s ======")
    assert up is not None
    assert "182 passed" in up.description
    assert "0 failed" in up.description


def test_parse_test_summary_line_no_failures() -> None:
    up = progress.parse_test("====== 182 passed in 23.45s ======")
    assert up is not None
    assert "182 passed" in up.description


# ---------------------------------------------------------------------------
# lint (ruff)
# ---------------------------------------------------------------------------


def test_parse_lint_found_errors() -> None:
    up = progress.parse_lint("Found 12 errors.")
    assert up is not None
    assert "12 error" in up.description


def test_parse_lint_clean() -> None:
    up = progress.parse_lint("All checks passed!")
    assert up is not None
    assert "clean" in up.description


# ---------------------------------------------------------------------------
# typecheck (mypy)
# ---------------------------------------------------------------------------


def test_parse_typecheck_found_errors() -> None:
    up = progress.parse_typecheck(
        "Found 5 errors in 2 files (checked 184 source files)"
    )
    assert up is not None
    assert "5 error" in up.description


def test_parse_typecheck_clean() -> None:
    up = progress.parse_typecheck("Success: no issues found in 184 source files")
    assert up is not None
    assert "clean" in up.description


# ---------------------------------------------------------------------------
# security (bandit)
# ---------------------------------------------------------------------------


def test_parse_security_files_processed() -> None:
    up = progress.parse_security("\tFiles processed: 187")
    assert up is not None
    assert up.completed == 187


# ---------------------------------------------------------------------------
# redteam
# ---------------------------------------------------------------------------


def test_parse_redteam_loads_corpus_and_counts_attacks() -> None:
    # Reset internal counter between tests so ordering independence holds.
    progress._redteam_state.seen = 0
    progress._redteam_state.total = 0

    up1 = progress.parse_redteam("Loaded corpus with 55 attacks")
    assert up1 is not None
    assert up1.total == 55
    assert up1.completed == 0

    up2 = progress.parse_redteam("Attack dpi-001 raised ProviderError: ...")
    assert up2 is not None
    assert up2.completed == 1
    assert up2.total == 55


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_parse_verify_summary() -> None:
    line = "[verify_all] total=44 pass=39 fail=0 skip=5 duration=5.36s"
    up = progress.parse_verify(line)
    assert up is not None
    assert up.completed == 44
    assert up.total == 44
    assert "39/44" in up.description


# ---------------------------------------------------------------------------
# onnx-export
# ---------------------------------------------------------------------------


def test_parse_onnx_export() -> None:
    up = progress.parse_onnx_export("Exporting TCN encoder -> checkpoints/encoder/tcn.onnx")
    assert up is not None
    assert "tcn.onnx" in up.description


# ---------------------------------------------------------------------------
# docker-build (BuildKit)
# ---------------------------------------------------------------------------


def test_parse_docker_build_layer_progress() -> None:
    up = progress.parse_docker_build("#12 [ 3/14] RUN pip install --no-cache-dir .")
    assert up is not None
    assert up.completed == 3
    assert up.total == 14


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_parse_line_dispatches_by_stage_name() -> None:
    up = progress.parse_line("train-encoder", "\x1b[36mEpoch 2/10 train_loss=1.5\x1b[0m")
    assert up is not None
    assert up.completed == 2


def test_parse_line_unknown_stage_returns_none() -> None:
    assert progress.parse_line("no-such-stage", "anything") is None


def test_parse_line_strips_ansi_before_matching() -> None:
    # A non-matching-looking line once ANSI is present should still match.
    up = progress.parse_line(
        "verify",
        "\x1b[2m[verify_all] total=10 pass=9 fail=1 skip=0 duration=1.2s\x1b[0m",
    )
    assert up is not None
    assert up.total == 10
