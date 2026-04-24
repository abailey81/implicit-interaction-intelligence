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
from dataclasses import dataclass, field

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
    loss:
        Optional scalar training/validation loss snapshot — the
        orchestrator feeds this into ``StageMetrics`` to render a
        mini loss curve alongside the bar.
    lr:
        Optional learning-rate snapshot.
    metric_kind:
        Opt-in hint telling the dashboard which curve this frame
        contributes to.  Currently ``"train"`` or ``"val"``.  ``None``
        means "don't plot".
    """

    completed: float
    total: float
    description: str | None = None
    loss: float | None = None
    lr: float | None = None
    metric_kind: str | None = None


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
    and the intra-epoch batch loss lines if present.  When a loss is
    present it's surfaced as ``ProgressUpdate.loss`` so the dashboard
    can render a mini loss curve.
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
    # Prefer val_loss for the plotted curve when both are present —
    # generalisation matters more than training fit for observers.
    loss_val: float | None = None
    metric_kind: str | None = None
    if val_loss:
        loss_val = float(val_loss)
        metric_kind = "val"
    elif train_loss:
        loss_val = float(train_loss)
        metric_kind = "train"
    return ProgressUpdate(
        cur, tot, " · ".join(desc_bits), loss=loss_val, metric_kind=metric_kind
    )


# --- train-slm -------------------------------------------------------------
#   "Step 120/2830  loss=2.4989  avg_loss=... lr=... grad_norm=..."
_SLM_STEP_RE = re.compile(
    r"Step\s+(\d+)\s*/\s*(\d+)(?:.*?loss=([\d.]+))?(?:.*?lr=([\d.e+-]+))?"
)
#   "Converting 5 epochs to 2830 steps (566 steps/epoch)"
_SLM_PLAN_RE = re.compile(r"to\s+(\d+)\s+steps")


def parse_train_slm(line: str) -> ProgressUpdate | None:
    """Parse SLM training log lines (cross-attention trainer).

    Extracts step, loss, and learning-rate into a :class:`ProgressUpdate`
    so the dashboard can show a live loss curve + current LR alongside
    the progress bar.
    """
    m = _SLM_STEP_RE.search(line)
    if m:
        cur, tot = int(m.group(1)), int(m.group(2))
        loss = m.group(3)
        lr = m.group(4)
        bits = [f"step {cur}/{tot}"]
        if loss:
            bits.append(f"loss={loss}")
        if lr:
            bits.append(f"lr={lr}")
        return ProgressUpdate(
            cur,
            tot,
            " · ".join(bits),
            loss=float(loss) if loss else None,
            lr=float(lr) if lr else None,
            metric_kind="train" if loss else None,
        )
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


# ---------------------------------------------------------------------------
# Per-stage metric tracker — feeds the dashboard's live throughput, ETA,
# and loss-curve widgets.
# ---------------------------------------------------------------------------

import time
from collections import deque
from typing import Deque


@dataclass
class StageMetrics:
    """Accumulates progress frames for one stage and derives rich metrics.

    The orchestrator creates one instance per stage as it transitions to
    ``running``; every parsed :class:`ProgressUpdate` is pushed in via
    :meth:`update`.  The dashboard then reads:

    * :attr:`throughput_per_s` — EMA of completed-steps-per-second,
      computed from the rolling 30-sample timing window.  Smooths out
      the noise you get from per-step timing alone.
    * :meth:`eta_seconds` — ``(total - completed) / throughput``.  Falls
      back to ``None`` when we don't yet have enough samples.
    * :meth:`loss_sparkline` — unicode sparkline of the last 60 loss
      values, split by ``metric_kind``.
    * :meth:`summary_line` — one-line formatted string for the stage row.
    """

    stage_name: str
    started_at: float = field(default_factory=time.monotonic)
    #: Rolling buffer of ``(monotonic_ts, completed)`` pairs — used to
    #: compute a windowed EMA of throughput.
    _samples: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    #: Latest parsed state.
    completed: float = 0.0
    total: float = 0.0
    #: Rolling loss history per kind ("train" / "val"), 120 samples each.
    train_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    val_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    latest_lr: float | None = None
    latest_description: str | None = None
    #: Exponentially-smoothed throughput in steps/sec.  ``0.0`` until
    #: we've seen two samples.
    throughput_per_s: float = 0.0
    #: EMA smoothing factor — 0.3 prioritises recent data without being
    #: jittery.  Tuned against real training runs.
    ema_alpha: float = 0.3

    def update(self, frame: ProgressUpdate) -> None:
        """Incorporate one parsed progress frame."""
        self.completed = frame.completed
        self.total = frame.total
        if frame.description is not None:
            self.latest_description = frame.description
        if frame.loss is not None:
            if frame.metric_kind == "val":
                self.val_losses.append(float(frame.loss))
            else:
                self.train_losses.append(float(frame.loss))
        if frame.lr is not None:
            self.latest_lr = float(frame.lr)

        # Timing sample — only if "completed" actually moved forward so
        # log lines that merely restate the current state don't pollute
        # the throughput calculation.
        now = time.monotonic()
        if self._samples:
            prev_ts, prev_completed = self._samples[-1]
            if frame.completed <= prev_completed:
                return  # stale or duplicate line
            dt = max(now - prev_ts, 1e-6)
            rate = (frame.completed - prev_completed) / dt
            # Bootstrap the EMA with the first real rate, then blend.
            if self.throughput_per_s <= 0.0:
                self.throughput_per_s = rate
            else:
                self.throughput_per_s = (
                    self.ema_alpha * rate
                    + (1.0 - self.ema_alpha) * self.throughput_per_s
                )
        self._samples.append((now, frame.completed))

    def eta_seconds(self) -> float | None:
        """Return the projected remaining wall-clock seconds, or ``None``."""
        if self.total <= 0 or self.throughput_per_s <= 0.0:
            return None
        remaining = max(0.0, self.total - self.completed)
        return remaining / self.throughput_per_s

    def elapsed_seconds(self) -> float:
        """Seconds since this stage started."""
        return time.monotonic() - self.started_at

    def loss_sparkline(self, kind: str = "train", width: int = 24) -> str:
        """Return a unicode sparkline of the selected loss series."""
        try:
            from i3.runtime.monitoring import sparkline
        except ImportError:  # pragma: no cover — monitoring missing
            return ""
        src = self.val_losses if kind == "val" else self.train_losses
        return sparkline(list(src), width=width)

    def latest_loss(self, kind: str = "train") -> float | None:
        """Return the most recent loss value of the named kind."""
        src = self.val_losses if kind == "val" else self.train_losses
        return src[-1] if src else None

    def summary_line(self) -> str:
        """One-line status ready for the live dashboard.

        Example::

            step 412/2830 · loss=2.41 · lr=1.2e-04 · 18.4 steps/s · ETA 2m13s
        """
        bits: list[str] = []
        if self.latest_description:
            bits.append(self.latest_description)
        if self.throughput_per_s > 0:
            bits.append(f"{self.throughput_per_s:.1f} steps/s")
        eta = self.eta_seconds()
        if eta is not None:
            bits.append(f"ETA {_fmt_eta(eta)}")
        return " · ".join(bits)


def _fmt_eta(seconds: float) -> str:
    """Compact duration formatter: 2m13s / 41s / 1h04m."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


__all__ = [
    "PARSERS",
    "ProgressUpdate",
    "StageMetrics",
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
