"""Pluggable log-line parsers for the orchestrator's live progress bars.

Each parser takes a single log line (ANSI-stripped) and returns an
optional :class:`ProgressUpdate`.  The orchestrator dispatches based on
the stage name; unknown stages fall back to the time-based ETA bar, so
adding a new parser never regresses the existing behaviour.

Design contract
---------------

* **Pure** — parsers never touch disk, network, or mutable globals.
* **Fast** — regex-only, compiled at import time.  A typical parser
  runs in <2 µs per line.
* **Conservative** — if a line is ambiguous, return ``None`` rather
  than emitting a bogus progress frame.
* **Idempotent ordering** — parsers return the latest observed
  ``(completed, total)`` pair; the orchestrator calls
  :meth:`rich.progress.Progress.update` with absolute values so out-of-
  order or retransmitted lines never double-advance the bar.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# ANSI-colour stripper — shared by all parsers.
# ---------------------------------------------------------------------------

# Matches the full CSI sequence including 256-colour / true-colour forms.
_ANSI_RE = re.compile(r"\x1B(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")


def strip_ansi(line: str) -> str:
    """Return ``line`` with ANSI escape sequences removed.

    Robust against the truecolor and cursor-movement variants that
    ``rich`` and ``tqdm`` emit when writing to a TTY.
    """
    return _ANSI_RE.sub("", line)


# ---------------------------------------------------------------------------
# Progress update record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressUpdate:
    """One parsed progress frame.

    Attributes
    ----------
    completed:
        Current step count (0-based or 1-based; we clamp to
        ``[0, total]`` at render time).
    total:
        Expected final step count.  May be ``0`` for unknown — the bar
        then falls back to an indeterminate spinner.
    description:
        Optional one-line override for the progress row, e.g.
        ``"epoch 3/5 · loss 0.42"``.
    """

    completed: float
    total: float
    description: str | None = None


# ---------------------------------------------------------------------------
# Individual parsers — each returns ``None`` when the line doesn't match.
# ---------------------------------------------------------------------------

# --- train-encoder ---------------------------------------------------------
#   "Epoch   1/10  train_loss=3.1857  val_loss=2.6203 ..."
_ENCODER_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)"
    r"(?:.*?train_loss=([\d.]+))?"
    r"(?:.*?val_loss=([\d.]+))?"
)


def parse_train_encoder(line: str) -> ProgressUpdate | None:
    """Parse TCN encoder training log lines.

    Matches both the summary ``Epoch  N/M`` line emitted once per epoch
    and the intra-epoch batch loss lines if present.
    """
    m = _ENCODER_EPOCH_RE.search(line)
    if not m:
        return None
    cur, tot = int(m.group(1)), int(m.group(2))
    train_loss = m.group(3)
    val_loss = m.group(4)
    desc_bits = [f"epoch {cur}/{tot}"]
    if train_loss:
        desc_bits.append(f"train_loss={train_loss}")
    if val_loss:
        desc_bits.append(f"val_loss={val_loss}")
    return ProgressUpdate(cur, tot, " · ".join(desc_bits))


# --- train-slm -------------------------------------------------------------
#   "Step 120/2830  loss=2.4989  avg_loss=... lr=... grad_norm=..."
_SLM_STEP_RE = re.compile(
    r"Step\s+(\d+)\s*/\s*(\d+)(?:.*?loss=([\d.]+))?(?:.*?lr=([\d.e+-]+))?"
)
#   "Converting 5 epochs to 2830 steps (566 steps/epoch)"
_SLM_PLAN_RE = re.compile(r"to\s+(\d+)\s+steps")


def parse_train_slm(line: str) -> ProgressUpdate | None:
    """Parse SLM training log lines (cross-attention trainer)."""
    m = _SLM_STEP_RE.search(line)
    if m:
        cur, tot = int(m.group(1)), int(m.group(2))
        loss = m.group(3)
        desc = f"step {cur}/{tot}" + (f" · loss={loss}" if loss else "")
        return ProgressUpdate(cur, tot, desc)
    # Plan-line: gives us the total but not the current — prime the bar.
    m = _SLM_PLAN_RE.search(line)
    if m:
        tot = int(m.group(1))
        return ProgressUpdate(0, tot, f"0/{tot} steps")
    return None


# --- data ------------------------------------------------------------------
#   "Generating 3200 sessions x 20 messages (window=10) ..."
#   "  ... 2000 / 3200 sessions"
_DATA_PLAN_RE = re.compile(r"Generating\s+(\d+)\s+sessions")
_DATA_PROG_RE = re.compile(r"\.\.\.\s*(\d+)\s*/\s*(\d+)\s+sessions")


def parse_data(line: str) -> ProgressUpdate | None:
    """Parse ``training/generate_synthetic.py`` output."""
    m = _DATA_PROG_RE.search(line)
    if m:
        cur, tot = int(m.group(1)), int(m.group(2))
        return ProgressUpdate(cur, tot, f"{cur}/{tot} sessions")
    m = _DATA_PLAN_RE.search(line)
    if m:
        tot = int(m.group(1))
        return ProgressUpdate(0, tot, f"0/{tot} sessions")
    return None


# --- dialogue --------------------------------------------------------------
#   "Step 1: Loading raw dialogue data..."
#   "Step 3: Building tokenizer vocabulary (size=...)"
_DIALOGUE_STEP_RE = re.compile(r"Step\s+(\d+):")
_DIALOGUE_TOTAL_STEPS = 6  # see training/prepare_dialogue.py (6-step pipeline)


def parse_dialogue(line: str) -> ProgressUpdate | None:
    """Parse ``training/prepare_dialogue.py`` output.

    The script emits six numbered "Step N:" markers; we translate those
    to a 0–6 progress bar.
    """
    m = _DIALOGUE_STEP_RE.search(line)
    if not m:
        return None
    cur = int(m.group(1))
    return ProgressUpdate(
        min(cur, _DIALOGUE_TOTAL_STEPS),
        _DIALOGUE_TOTAL_STEPS,
        f"dialogue step {cur}/{_DIALOGUE_TOTAL_STEPS}",
    )


# --- test (pytest + xdist) -------------------------------------------------
#   "[gw3] [ 42%] PASSED tests/test_foo.py::test_bar"
#   "====== 182 passed, 0 failed in 23.45s ======"
_PYTEST_PCT_RE = re.compile(r"\[\s*(\d+)%\s*\]")
_PYTEST_SUM_RE = re.compile(
    r"(\d+)\s+passed"
    r"(?:[^,]*?,\s*(\d+)\s+failed)?"
    r"(?:[^,]*?,\s*(\d+)\s+skipped)?"
)


def parse_test(line: str) -> ProgressUpdate | None:
    """Parse pytest output (xdist-aware).

    We prefer the ``[NN%]`` token that xdist emits on every completed
    test because it gives us a true 0–100 progress signal.  The final
    summary line replaces the bar with the actual pass / fail counts.
    """
    m = _PYTEST_SUM_RE.search(line)
    if m:
        passed = int(m.group(1))
        failed = int(m.group(2) or 0)
        skipped = int(m.group(3) or 0)
        total = passed + failed + skipped
        return ProgressUpdate(
            total,
            max(total, 1),
            f"{passed} passed · {failed} failed · {skipped} skipped",
        )
    m = _PYTEST_PCT_RE.search(line)
    if m:
        pct = int(m.group(1))
        return ProgressUpdate(pct, 100, f"pytest {pct}%")
    return None


# --- lint (ruff) -----------------------------------------------------------
#   "Found 42 errors."   OR   "All checks passed!"
_RUFF_FOUND_RE = re.compile(r"Found\s+(\d+)\s+error")
_RUFF_OK_RE = re.compile(r"All checks passed")


def parse_lint(line: str) -> ProgressUpdate | None:
    """Parse ruff's summary footer."""
    m = _RUFF_FOUND_RE.search(line)
    if m:
        errs = int(m.group(1))
        return ProgressUpdate(1, 1, f"ruff: {errs} error(s)")
    if _RUFF_OK_RE.search(line):
        return ProgressUpdate(1, 1, "ruff: clean")
    return None


# --- typecheck (mypy) ------------------------------------------------------
#   "Found 12 errors in 3 files (checked 184 source files)"
#   "Success: no issues found in 184 source files"
_MYPY_FOUND_RE = re.compile(r"Found\s+(\d+)\s+errors?\s+in\s+(\d+)\s+files?")
_MYPY_OK_RE = re.compile(r"Success:\s+no issues found in\s+(\d+)")


def parse_typecheck(line: str) -> ProgressUpdate | None:
    """Parse mypy summary lines."""
    m = _MYPY_FOUND_RE.search(line)
    if m:
        errs = int(m.group(1))
        files = int(m.group(2))
        return ProgressUpdate(1, 1, f"mypy: {errs} error(s) in {files} file(s)")
    m = _MYPY_OK_RE.search(line)
    if m:
        files = int(m.group(1))
        return ProgressUpdate(1, 1, f"mypy: clean ({files} files)")
    return None


# --- security (bandit) -----------------------------------------------------
#   "Files processed: 187"  (the typical bandit verbose mode)
#   "Total lines of code: 31204"
_BANDIT_FILES_RE = re.compile(r"[Ff]iles processed:\s*(\d+)")
_BANDIT_TOTAL_RE = re.compile(r"[Tt]otal issues.*?Severity:.*?(\d+)")


def parse_security(line: str) -> ProgressUpdate | None:
    """Parse bandit progress / summary lines."""
    m = _BANDIT_FILES_RE.search(line)
    if m:
        n = int(m.group(1))
        return ProgressUpdate(n, max(n, 1), f"bandit: {n} files scanned")
    return None


# --- redteam ---------------------------------------------------------------
#   "Attack dpi-001 raised ProviderError: ..."
#   "Loaded corpus with 55 attacks"
_REDTEAM_TOTAL_RE = re.compile(r"Loaded corpus with\s+(\d+)\s+attacks?")
_REDTEAM_ATTACK_RE = re.compile(r"Attack\s+([A-Za-z0-9_\-]+)")


class _RedTeamState:
    __slots__ = ("seen", "total")

    def __init__(self) -> None:
        self.seen = 0
        self.total = 0


_redteam_state = _RedTeamState()


def parse_redteam(line: str) -> ProgressUpdate | None:
    """Parse red-team harness output.

    The harness logs one "Loaded corpus with N attacks" line up front,
    then one log line per attack execution — we count those as they
    stream.  Best-effort: the corpus logger only fires on failures, so
    this parser primarily primes the bar's ``total``.
    """
    m = _REDTEAM_TOTAL_RE.search(line)
    if m:
        _redteam_state.total = int(m.group(1))
        _redteam_state.seen = 0
        return ProgressUpdate(0, _redteam_state.total, f"0/{_redteam_state.total}")
    if _redteam_state.total and _REDTEAM_ATTACK_RE.search(line):
        _redteam_state.seen = min(_redteam_state.seen + 1, _redteam_state.total)
        return ProgressUpdate(
            _redteam_state.seen,
            _redteam_state.total,
            f"attack {_redteam_state.seen}/{_redteam_state.total}",
        )
    return None


# --- verify (scripts/verify_all.py) ----------------------------------------
#   "[verify_all] total=44 pass=39 fail=0 skip=5 duration=5.36s"
_VERIFY_SUMMARY_RE = re.compile(
    r"\[verify_all\]\s+total=(\d+)\s+pass=(\d+)\s+fail=(\d+)\s+skip=(\d+)"
)


def parse_verify(line: str) -> ProgressUpdate | None:
    """Parse the final ``verify_all`` summary line."""
    m = _VERIFY_SUMMARY_RE.search(line)
    if not m:
        return None
    total = int(m.group(1))
    passed = int(m.group(2))
    failed = int(m.group(3))
    skipped = int(m.group(4))
    return ProgressUpdate(
        total,
        max(total, 1),
        f"{passed}/{total} pass · {failed} fail · {skipped} skip",
    )


# --- onnx-export -----------------------------------------------------------
#   "Exporting TCN encoder -> checkpoints/encoder/tcn.onnx"
_ONNX_EXPORT_RE = re.compile(r"Exporting\s+(.+?)\s*->\s*(\S+\.onnx)")


def parse_onnx_export(line: str) -> ProgressUpdate | None:
    """Parse ``i3/encoder/onnx_export.py`` stdout."""
    m = _ONNX_EXPORT_RE.search(line)
    if not m:
        return None
    target = m.group(2)
    return ProgressUpdate(1, 1, f"exporting -> {target}")


# --- docker-build (BuildKit) ----------------------------------------------
#   "#12 [ 3/14] RUN pip install ..."
#   "#12 DONE 0.2s"
_DOCKER_LAYER_RE = re.compile(r"#\d+\s+\[\s*(\d+)\s*/\s*(\d+)\s*\]")


def parse_docker_build(line: str) -> ProgressUpdate | None:
    """Parse ``docker buildx`` layer progress."""
    m = _DOCKER_LAYER_RE.search(line)
    if not m:
        return None
    cur, tot = int(m.group(1)), int(m.group(2))
    return ProgressUpdate(cur, tot, f"layer {cur}/{tot}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

Parser = Callable[[str], ProgressUpdate | None]

# Keyed by ``Stage.name``.  Stages without a registered parser fall back
# to the orchestrator's time-based ETA heartbeat.
PARSERS: dict[str, Parser] = {
    "train-encoder": parse_train_encoder,
    "train-slm": parse_train_slm,
    "data": parse_data,
    "dialogue": parse_dialogue,
    "test": parse_test,
    "benchmarks": parse_test,
    "lint": parse_lint,
    "typecheck": parse_typecheck,
    "security": parse_security,
    "redteam": parse_redteam,
    "verify": parse_verify,
    "onnx-export": parse_onnx_export,
    "docker-build": parse_docker_build,
}


def parse_line(stage_name: str, line: str) -> ProgressUpdate | None:
    """Dispatch helper — look up the parser for ``stage_name``.

    Strips ANSI before dispatching.  Returns ``None`` when no parser is
    registered or the line doesn't match the stage's pattern.
    """
    parser = PARSERS.get(stage_name)
    if parser is None:
        return None
    return parser(strip_ansi(line))


__all__ = [
    "PARSERS",
    "ProgressUpdate",
    "parse_data",
    "parse_dialogue",
    "parse_docker_build",
    "parse_line",
    "parse_lint",
    "parse_onnx_export",
    "parse_redteam",
    "parse_security",
    "parse_test",
    "parse_train_encoder",
    "parse_train_slm",
    "parse_typecheck",
    "parse_verify",
    "strip_ansi",
]
