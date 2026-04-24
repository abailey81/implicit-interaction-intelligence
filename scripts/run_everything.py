"""End-to-end orchestrator for the I³ project.

One command runs every stage that turns a fresh clone into a running,
trained, verified, and served I³ system.  Every stage is streamed
through a live ``rich`` dashboard so you can see exactly what is
running, how long it has been running, how long it is projected to
take, and where the logs are being written.

Two canonical profiles
----------------------

* ``--mode fast`` — *quickstart path* (~5 minutes on a laptop).
  Installs deps, generates the ``.env`` + Fernet key, runs the
  verification harness, and launches the server on the shipped demo
  checkpoints.  Skips data generation, training, and evaluation.

* ``--mode full`` — *full end-to-end* (hours on CPU, ~15 min on a
  modern GPU).  Runs every stage: prerequisites check, dependency
  install, env setup, synthetic data, dialogue corpus prep, encoder
  training, SLM training, evaluation, demo seed, tests, type check,
  security scan, verification harness, red-team, benchmarks, ONNX
  export, docs build, and finally launches the server.

Finer-grained control
---------------------

* ``--only a,b,c`` — run exactly the named stages.
* ``--skip a,b,c`` — run everything except the named stages.
* ``--resume`` — skip stages whose on-disk outputs already exist.
* ``--no-serve`` — do not launch the HTTP server at the end.
* ``--list`` — print the stage graph for the chosen mode and exit.

Design notes
------------

* Pure stdlib + ``rich``.  No heavyweight extra deps.  If ``rich`` is
  missing the script installs it into the current interpreter on first
  run so a fresh clone just works.
* Every stage is a :class:`Stage` dataclass describing its *name*,
  *human description*, *command line*, *expected output paths*, and an
  optional *ETA seed* (seconds) used to paint a plausible ETA bar on
  the very first run.  After the first run the observed wall-clock is
  written to ``reports/orchestration.json`` and used as the EMA ETA
  for the next run, so the bars calibrate to your machine over time.
* Each stage's stdout + stderr is streamed to both the live dashboard
  tail panel *and* a timestamped log file under
  ``reports/orchestration/<stage>.log``.  A failure does not print a
  wall of noise — it points you at the log and prints the last 12
  lines.
* Stages never fail silently.  A non-zero exit code aborts the
  orchestration and surfaces the log tail in the final summary.
* Re-runnable: every stage is idempotent at the filesystem level, and
  ``--resume`` skips stages whose outputs are already in place.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# SEC: Force UTF-8 on stdout/stderr so the "I³" glyph and rich's box
# drawing characters render on Windows consoles that default to
# cp1251 / cp437 / cp1252.  Without this, rich crashes with
# ``UnicodeEncodeError: 'charmap' codec can't encode character '\\xb3'``
# the first time the banner panel is rendered.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Rich bootstrap — auto-install on first run so a fresh clone just works.
# ---------------------------------------------------------------------------

def _ensure_rich() -> None:
    try:
        import rich  # noqa: F401
        return
    except ImportError:
        pass
    print("[run_everything] 'rich' not installed; installing it now...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "rich>=13.7"]
    )


_ensure_rich()

from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402


CONSOLE = Console()

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
LOG_DIR = REPORTS_DIR / "orchestration"
STATE_FILE = REPORTS_DIR / "orchestration.json"


# ---------------------------------------------------------------------------
# Stage model
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    """One orchestrated stage."""

    name: str
    description: str
    #: Command to run as argv list.  ``None`` means the stage is a
    #: Python callable — see ``action``.
    cmd: list[str] | None = None
    #: Python callable invoked instead of a subprocess.
    action: Callable[[argparse.Namespace], None] | None = None
    #: Paths that, if they all exist, let ``--resume`` skip this stage.
    produces: list[Path] = field(default_factory=list)
    #: Seed for the ETA bar (seconds) when we have no prior
    #: observation on this machine.
    eta_seed_s: float = 10.0
    #: Set to True to tolerate non-zero exit without aborting the run.
    optional: bool = False
    #: If the stage needs ``I3_DEMO_MODE=1``, set this.
    needs_demo_mode: bool = False
    #: Skip automatically if this predicate is True.
    skip_if: Callable[[argparse.Namespace], bool] | None = None
    #: Categorisation for the summary table.
    category: str = "core"
    #: Execution wave.  Stages in the same wave run *concurrently* via
    #: a thread pool (subprocess stages) or sequentially (action stages
    #: that mutate shared Python state — they always run alone).  Waves
    #: run in ascending numeric order.  A wave gate waits for every
    #: stage in the current wave to finish before advancing.  Design
    #: contract: stages sharing a wave MUST have no mutual file /
    #: directory / port / DB conflicts.
    wave: int = 0

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.name}.log"


# ---------------------------------------------------------------------------
# Toolchain probes
# ---------------------------------------------------------------------------

def _poetry_available() -> bool:
    return shutil.which("poetry") is not None


def _make_available() -> bool:
    return shutil.which("make") is not None


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _py_cmd() -> list[str]:
    """Return the command prefix for running project Python.

    Resolution order:

    1. If a project-local virtualenv exists at ``./.venv``, use its
       Python binary directly.  This is the most reliable choice on
       machines where ``poetry`` itself might be installed under a
       different interpreter than the project's venv (e.g. Poetry
       bootstrapped via Python 3.14 while the project venv is 3.12).
    2. Otherwise, if ``poetry`` is on PATH, ``poetry run python``.
    3. Otherwise, the interpreter running *this* script.
    """
    venv_py = REPO_ROOT / ".venv" / (
        "Scripts" if sys.platform == "win32" else "bin"
    ) / ("python.exe" if sys.platform == "win32" else "python")
    if venv_py.exists():
        return [str(venv_py)]
    if _poetry_available():
        return ["poetry", "run", "python"]
    return [sys.executable]


# ---------------------------------------------------------------------------
# Stage actions (Python-native, not subprocess).
# ---------------------------------------------------------------------------

def _action_prerequisites(args: argparse.Namespace) -> None:
    """Check Python version + optional toolchain."""
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python 3.10+ required, have "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    if not _poetry_available():
        CONSOLE.print(
            "[yellow]    ⚠  poetry not on PATH — falling back to the current "
            "interpreter.  Install poetry for a reproducible env.[/yellow]"
        )
    if not _make_available():
        CONSOLE.print(
            "[yellow]    ⚠  make not on PATH — every stage will fall back "
            "to its direct Python/poetry invocation.[/yellow]"
        )


def _action_env_file(args: argparse.Namespace) -> None:
    """Copy ``.env.example`` to ``.env`` if missing; generate Fernet key."""
    src = REPO_ROOT / ".env.example"
    dst = REPO_ROOT / ".env"
    if not dst.exists():
        if not src.exists():
            raise FileNotFoundError(".env.example is missing from the repo")
        dst.write_bytes(src.read_bytes())

    # If I3_ENCRYPTION_KEY is blank, populate it.
    lines = dst.read_text(encoding="utf-8").splitlines()
    needs_key = True
    for line in lines:
        stripped = line.lstrip()
        if (
            stripped.startswith("I3_ENCRYPTION_KEY=")
            and stripped.split("=", 1)[1].strip()
        ):
            needs_key = False
            break

    if needs_key:
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            CONSOLE.print(
                "[yellow]    ⚠  cryptography not yet installed; leaving "
                "I3_ENCRYPTION_KEY blank (will regenerate on next run).[/yellow]"
            )
            return
        key = Fernet.generate_key().decode()
        new_lines: list[str] = []
        written = False
        for line in lines:
            if line.lstrip().startswith("I3_ENCRYPTION_KEY="):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}I3_ENCRYPTION_KEY={key}")
                written = True
            else:
                new_lines.append(line)
        if not written:
            new_lines.append(f"I3_ENCRYPTION_KEY={key}")
        dst.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _action_launch_server(args: argparse.Namespace) -> None:
    """Replace the orchestrator process with uvicorn."""
    cmd = _py_cmd() + [
        "-m",
        "uvicorn",
        "server.app:create_app",
        "--factory",
        "--host",
        os.environ.get("I3_HOST", "127.0.0.1"),
        "--port",
        os.environ.get("I3_PORT", "8000"),
    ]
    CONSOLE.print(
        Panel.fit(
            "[bold green]Launching server in the foreground.[/bold green]\n"
            "  Demo UI:         [cyan]http://127.0.0.1:8000/[/cyan]\n"
            "  Advanced UI:     [cyan]http://127.0.0.1:8000/advanced/[/cyan]\n"
            "  Swagger:         [cyan]http://127.0.0.1:8000/api/docs[/cyan]\n"
            "  Health:          [cyan]http://127.0.0.1:8000/api/health[/cyan]\n\n"
            "Press [bold]Ctrl-C[/bold] to stop.",
            title="I³ server",
            border_style="green",
        )
    )
    os.execvp(cmd[0], cmd)


# ---------------------------------------------------------------------------
# Stage definitions — declarative, one entry per orchestrated step.
# ---------------------------------------------------------------------------

def _build_all_stages(args: argparse.Namespace) -> list[Stage]:
    py = _py_cmd()
    stages: list[Stage] = [
        # ─────────────────────────────────────────────────────────────
        # Wave 0 — prereq check.  Fast, must run first.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="prereq",
            description="Check Python + toolchain",
            action=_action_prerequisites,
            eta_seed_s=0.5,
            category="setup",
            wave=0,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 1 — install.  Single heavy stage, blocks everything.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="install",
            description="Install dependencies (poetry install --with dev,security)",
            cmd=(
                ["poetry", "install", "--with", "dev,security"]
                if _poetry_available()
                else [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]
            ),
            produces=[REPO_ROOT / "poetry.lock"],
            eta_seed_s=240.0,
            skip_if=lambda a: a.skip_install,
            category="setup",
            wave=1,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 2 — env.  Python action; must run alone.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="env",
            description="Create .env + generate Fernet encryption key",
            action=_action_env_file,
            produces=[REPO_ROOT / ".env"],
            eta_seed_s=0.3,
            category="setup",
            wave=2,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 3 — data generation + dialogue prep.  Different inputs,
        # different output paths, completely independent.  Run
        # concurrently → roughly halves data-prep wall-clock.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="data",
            description="Generate synthetic interaction corpus",
            # ``generate_synthetic.py`` writes train/val/test ``.pt``
            # shards to ``--output-dir``.  Total sessions = the product
            # of the old (archetypes × sessions-per-archetype) knobs —
            # 8 × args.sessions_per_archetype — so behaviour matches
            # the prior orchestrator contract.
            cmd=py
            + [
                "training/generate_synthetic.py",
                "--sessions",
                str(8 * args.sessions_per_archetype),
                "--output-dir",
                "data/synthetic",
            ],
            produces=[
                REPO_ROOT / "data" / "synthetic" / "train.pt",
                REPO_ROOT / "data" / "synthetic" / "val.pt",
                REPO_ROOT / "data" / "synthetic" / "test.pt",
            ],
            eta_seed_s=30.0,
            skip_if=lambda a: a.mode == "fast",
            category="data",
            wave=3,
        ),
        Stage(
            name="dialogue",
            description="Prepare dialogue corpus + build tokenizer",
            # v1 ``prepare_dialogue.py`` is the full SLM-training
            # pipeline: clean + dedup + split + build tokenizer +
            # write ``tokenizer.json`` + train/val/test ``.pt`` shards.
            # v2 ``prepare_dialogue_v2.py`` only does cleaning, so the
            # SLM trainer (which needs the tokenizer) fails downstream.
            cmd=(
                ["make", "prepare-dialogue"]
                if _make_available()
                else py
                + [
                    "training/prepare_dialogue.py",
                    "--output-dir",
                    "data/dialogue",
                ]
            ),
            produces=[
                REPO_ROOT / "data" / "dialogue" / "tokenizer.json",
                REPO_ROOT / "data" / "dialogue" / "train.pt",
            ],
            eta_seed_s=20.0,
            optional=True,
            skip_if=lambda a: a.mode == "fast",
            category="data",
            wave=3,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 4 — TCN encoder.  Blocks SLM training (needs encoder
        # checkpoint).  Runs alone in the wave.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="train-encoder",
            description="Train TCN encoder (NT-Xent contrastive loss)",
            # The trainer writes ``best_model.pt`` inside
            # ``--checkpoint-dir`` and reads shards from
            # ``--data-dir`` produced by the data stage.
            cmd=py
            + [
                "training/train_encoder.py",
                "--config",
                "configs/default.yaml",
                "--epochs",
                str(args.encoder_epochs),
                "--data-dir",
                "data/synthetic",
                "--checkpoint-dir",
                "checkpoints/encoder",
            ],
            produces=[REPO_ROOT / "checkpoints" / "encoder" / "best_model.pt"],
            eta_seed_s=1800.0,
            skip_if=lambda a: a.mode == "fast",
            category="train",
            wave=4,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 5 — SLM training + demo seed.  SLM needs encoder, demo
        # seed is independent of everything but shouldn't race the GPU
        # with training — so it sits after training in a non-blocking
        # wave with the SLM (seed writes to SQLite, SLM writes to
        # disk, no conflict).
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="train-slm",
            description="Train SLM with cross-attention conditioning",
            # The SLM trainer pulls encoder conditioning from the
            # checkpoint dir internally; it doesn't take an explicit
            # ``--conditioning-encoder`` flag.  Output lives at
            # ``{checkpoint-dir}/best_model.pt``.
            cmd=py
            + [
                "training/train_slm.py",
                "--config",
                "configs/default.yaml",
                "--epochs",
                str(args.slm_epochs),
                "--data-dir",
                "data/dialogue",
                "--checkpoint-dir",
                "checkpoints/slm",
            ],
            produces=[REPO_ROOT / "checkpoints" / "slm" / "best_model.pt"],
            eta_seed_s=3000.0,
            skip_if=lambda a: a.mode == "fast",
            category="train",
            wave=5,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 6 — evaluation + conditioning-sensitivity + demo seed.
        # All independent: each reads checkpoints, writes its own
        # report file.  Concurrent execution trims ~1.5× off the
        # post-training bar.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="evaluate",
            description="Perplexity + conditioning sensitivity + latency",
            # ``evaluate.py`` reads one SLM checkpoint via
            # ``--checkpoint`` and writes JSON via ``--output``.
            cmd=py
            + [
                "training/evaluate.py",
                "--config",
                "configs/default.yaml",
                "--checkpoint",
                "checkpoints/slm/best_model.pt",
                "--data-dir",
                "data/dialogue",
                "--output",
                "reports/evaluation.json",
            ],
            produces=[REPORTS_DIR / "evaluation.json"],
            eta_seed_s=120.0,
            optional=True,
            skip_if=lambda a: a.mode == "fast",
            category="eval",
            wave=6,
        ),
        Stage(
            name="eval-conditioning",
            description="Cross-attention conditioning-sensitivity evaluation",
            # The canonical CLI lives at
            # ``scripts/benchmarks/evaluate_conditioning.py`` —
            # ``i3.eval.conditioning_sensitivity`` is a library module,
            # not a runnable ``__main__``.
            cmd=py
            + [
                "scripts/benchmarks/evaluate_conditioning.py",
                "--checkpoint",
                "checkpoints/slm/best_model.pt",
                "--out",
                "reports/conditioning_sensitivity.json",
                "--markdown",
                "reports/conditioning_sensitivity.md",
            ],
            produces=[REPORTS_DIR / "conditioning_sensitivity.json"],
            eta_seed_s=60.0,
            optional=True,
            skip_if=lambda a: a.mode == "fast",
            category="eval",
            wave=6,
        ),
        Stage(
            name="seed",
            description="Seed the demo database",
            cmd=py
            + [
                "demo/pre_seed.py",
                "--users",
                "3",
                "--sessions-per-user",
                "20",
            ],
            eta_seed_s=5.0,
            optional=True,
            needs_demo_mode=True,
            category="demo",
            wave=6,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 7 — code-quality + security gates.  Every stage here
        # is read-only over the source tree and writes its own
        # report; full concurrency.  Typically 4-5× wall-clock win on
        # an 8-core box since lint+typecheck+test+security+redteam are
        # bounded by the slowest (test).
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="lint",
            description="ruff lint",
            cmd=(
                ["poetry", "run", "ruff", "check", "i3", "server", "tests"]
                if _poetry_available()
                else [sys.executable, "-m", "ruff", "check", "i3", "server", "tests"]
            ),
            eta_seed_s=15.0,
            optional=True,
            skip_if=lambda a: a.mode != "full",
            category="quality",
            wave=7,
        ),
        Stage(
            name="typecheck",
            description="mypy type check",
            cmd=(
                ["poetry", "run", "mypy", "i3"]
                if _poetry_available()
                else [sys.executable, "-m", "mypy", "i3"]
            ),
            eta_seed_s=45.0,
            optional=True,
            skip_if=lambda a: a.mode != "full",
            category="quality",
            wave=7,
        ),
        Stage(
            name="test",
            # PERF: ``pytest-xdist`` fans the test run across all cores;
            # ``-n auto`` auto-detects.  Still respects the 120 s per-
            # test timeout so a hung test can't wedge the worker pool.
            description="pytest unit + integration tests (parallel -n auto)",
            cmd=(
                ["poetry", "run", "pytest", "-q", "-n", "auto", "--timeout=120"]
                if _poetry_available()
                else [sys.executable, "-m", "pytest", "-q", "-n", "auto", "--timeout=120"]
            ),
            eta_seed_s=180.0,
            optional=True,
            skip_if=lambda a: a.mode != "full",
            category="quality",
            wave=7,
        ),
        Stage(
            name="security",
            description="Bandit + pip-audit security scan",
            cmd=(
                ["make", "security-check"]
                if _make_available()
                else ["poetry", "run", "bandit", "-r", "i3", "server", "-ll", "-q"]
            ),
            eta_seed_s=30.0,
            optional=True,
            skip_if=lambda a: a.mode != "full",
            category="security",
            wave=7,
        ),
        Stage(
            name="redteam",
            description="55-attack red-team harness",
            cmd=py
            + [
                "scripts/security/run_redteam.py",
                "--out",
                "reports/redteam.json",
                "--out-md",
                "reports/redteam.md",
            ],
            eta_seed_s=45.0,
            optional=True,
            skip_if=lambda a: a.mode != "full",
            category="security",
            wave=7,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 8 — verification harness gate.  Runs after quality +
        # security gates so a failure there bubbles up first.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="verify",
            description="44-check verification harness (parallel internally)",
            cmd=py + ["scripts/verify_all.py", "--fail-on", "blocker,high"],
            eta_seed_s=5.0,
            category="verify",
            wave=8,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 9 — perf + docs + deploy artefacts.  All read-only
        # over checkpoints, each writes its own output directory:
        #   benchmarks → .benchmarks/
        #   onnx-export → checkpoints/*.onnx
        #   profile-edge → reports/edge_profile_*.md
        #   docs → site/
        #   docker-build → container registry / local daemon
        # No mutual conflicts → full concurrency.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="benchmarks",
            description="Latency + throughput micro-benchmarks (parallel -n auto)",
            cmd=(
                ["make", "benchmarks"]
                if _make_available()
                else py + ["-m", "pytest", "benchmarks/", "-q", "-n", "auto"]
            ),
            eta_seed_s=90.0,
            optional=True,
            skip_if=lambda a: a.mode != "full" or a.skip_benchmarks,
            category="perf",
            wave=9,
        ),
        Stage(
            name="onnx-export",
            description="Export TCN + SLM to ONNX",
            # ``onnx_export.py`` needs an explicit ``--checkpoint`` —
            # point it at the encoder checkpoint produced by wave 4.
            cmd=(
                ["make", "export-onnx"]
                if _make_available()
                else py
                + [
                    "i3/encoder/onnx_export.py",
                    "--checkpoint",
                    "checkpoints/encoder/best_model.pt",
                    "--output",
                    "checkpoints/encoder/tcn.onnx",
                ]
            ),
            produces=[REPO_ROOT / "checkpoints" / "encoder" / "tcn.onnx"],
            eta_seed_s=60.0,
            optional=True,
            skip_if=lambda a: a.mode != "full" or a.skip_onnx,
            category="perf",
            wave=9,
        ),
        Stage(
            name="profile-edge",
            description="Edge-feasibility profiling report",
            # Canonical CLI is under ``scripts/benchmarks/`` —
            # ``i3/profiling/report.py`` is a library module.
            cmd=(
                ["make", "profile-edge"]
                if _make_available()
                else py
                + [
                    "scripts/benchmarks/profile_edge.py",
                    "--encoder",
                    "checkpoints/encoder/best_model.pt",
                    "--slm",
                    "checkpoints/slm/best_model.pt",
                    "--output",
                    "reports/edge_profile.md",
                ]
            ),
            eta_seed_s=30.0,
            optional=True,
            skip_if=lambda a: a.mode != "full",
            category="perf",
            wave=9,
        ),
        Stage(
            name="docs",
            description="Build MkDocs site (non-strict)",
            cmd=(
                ["make", "docs-build"]
                if _make_available()
                else [sys.executable, "-m", "mkdocs", "build"]
            ),
            produces=[REPO_ROOT / "site" / "index.html"],
            eta_seed_s=45.0,
            optional=True,
            skip_if=lambda a: a.mode != "full" or a.skip_docs,
            category="docs",
            wave=9,
        ),
        Stage(
            name="docker-build",
            description="Build production Docker image",
            cmd=["docker", "build", "-t", "i3:latest", "."],
            eta_seed_s=300.0,
            optional=True,
            skip_if=lambda a: (
                a.mode != "full"
                or not a.with_docker
                or not _docker_available()
            ),
            category="deploy",
            wave=9,
        ),
        # ─────────────────────────────────────────────────────────────
        # Wave 10 — serve.  Always last, blocks forever in foreground.
        # ─────────────────────────────────────────────────────────────
        Stage(
            name="serve",
            description="Launch the FastAPI server",
            action=_action_launch_server,
            eta_seed_s=0.0,
            skip_if=lambda a: a.no_serve,
            category="serve",
            wave=10,
        ),
    ]
    return stages


# ---------------------------------------------------------------------------
# Historical ETA learning.
# ---------------------------------------------------------------------------

def _load_history() -> dict[str, float]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_history(history: dict[str, float]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Filtering.
# ---------------------------------------------------------------------------

def _filter(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    if args.only:
        names = {n.strip() for n in args.only.split(",") if n.strip()}
        stages = [s for s in stages if s.name in names]
    if args.skip:
        skip = {n.strip() for n in args.skip.split(",") if n.strip()}
        stages = [s for s in stages if s.name not in skip]
    if args.resume:
        kept: list[Stage] = []
        for s in stages:
            if s.produces and all(p.exists() for p in s.produces):
                CONSOLE.print(
                    f"[dim]  → resume: skipping [bold]{s.name}[/bold] "
                    f"({', '.join(p.name for p in s.produces)} already exist)[/dim]"
                )
                continue
            kept.append(s)
        stages = kept
    stages = [s for s in stages if not (s.skip_if and s.skip_if(args))]
    return stages


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 60:
        return f"{s:5.1f}s"
    if s < 3600:
        return f"{s / 60:5.1f}m"
    return f"{s / 3600:5.2f}h"


_CATEGORY_COLOURS = {
    "setup": "blue",
    "data": "magenta",
    "train": "yellow",
    "eval": "magenta",
    "demo": "cyan",
    "quality": "blue",
    "security": "red",
    "verify": "green",
    "perf": "yellow",
    "docs": "cyan",
    "deploy": "cyan",
    "serve": "green",
    "core": "white",
}


def _status_table(
    stages: list[Stage],
    current_idx: int,
    statuses: dict[str, str],
    timings: dict[str, float],
    history: dict[str, float],
) -> Table:
    table = Table(
        title="Pipeline status",
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Stage", style="bold")
    table.add_column("Category", width=10)
    table.add_column("Description")
    table.add_column("Status", width=14)
    table.add_column("Elapsed", justify="right", width=9)
    table.add_column("ETA", justify="right", width=9)

    for i, stage in enumerate(stages):
        status = statuses.get(stage.name, "pending")
        if i == current_idx and status == "running":
            icon = "[cyan]●[/cyan]"
            status_cell = Text("running", style="cyan")
        elif status == "done":
            icon = "[green]✓[/green]"
            status_cell = Text("done", style="green")
        elif status == "skipped":
            icon = "[yellow]•[/yellow]"
            status_cell = Text("skipped", style="yellow dim")
        elif status == "failed":
            icon = "[red]✗[/red]"
            status_cell = Text("failed", style="red")
        else:
            icon = "[dim]·[/dim]"
            status_cell = Text("pending", style="dim")

        elapsed = timings.get(stage.name)
        elapsed_cell = _fmt_seconds(elapsed) if elapsed is not None else "—"
        eta = history.get(stage.name, stage.eta_seed_s)
        eta_cell = _fmt_seconds(eta) if status == "pending" else "—"

        cat_colour = _CATEGORY_COLOURS.get(stage.category, "white")
        table.add_row(
            icon,
            stage.name,
            Text(stage.category, style=cat_colour),
            stage.description,
            status_cell,
            elapsed_cell,
            eta_cell,
        )
    return table


def _stream_subprocess(
    stage: Stage,
    extra_env: dict[str, str],
    progress: Progress,
    task_id: int,
    tail: list[str],
) -> int:
    assert stage.cmd is not None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = stage.log_path.open("w", encoding="utf-8", errors="replace")
    env = {**os.environ, **extra_env}
    proc = subprocess.Popen(
        stage.cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
            clean = line.rstrip("\n")
            tail.append(clean)
            while len(tail) > 6:
                tail.pop(0)
            progress.update(task_id, advance=1)
    finally:
        log_handle.close()
        proc.wait()
    return proc.returncode


def _run_stage(
    stage: Stage,
    args: argparse.Namespace,
    history: dict[str, float],
    live_state: dict,
) -> tuple[bool, float, str | None]:
    eta = history.get(stage.name, stage.eta_seed_s)
    extra_env: dict[str, str] = {}
    if stage.needs_demo_mode:
        extra_env["I3_DEMO_MODE"] = "1"

    tail: list[str] = []
    progress = live_state["progress"]
    task_id = progress.add_task(
        description=f"[cyan]{stage.name}[/cyan] — {stage.description}",
        total=max(10, int(eta)),
    )
    live_state["current_tail"] = tail
    live_state["current_stage"] = stage.name
    start = time.perf_counter()

    stop = threading.Event()

    def _heartbeat() -> None:
        while not stop.is_set():
            elapsed = time.perf_counter() - start
            progress.update(task_id, completed=min(elapsed, eta - 0.01))
            stop.wait(0.4)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    failure: str | None = None
    try:
        if stage.action is not None:
            stage.action(args)
            rc = 0
        else:
            rc = _stream_subprocess(stage, extra_env, progress, task_id, tail)
        if rc != 0:
            if stage.optional:
                failure = f"optional stage exited with {rc} (continuing)"
            else:
                failure = f"exited with non-zero status {rc}"
    except Exception as exc:  # pragma: no cover - surfaced via dashboard
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        hb.join(timeout=1.0)
    seconds = time.perf_counter() - start
    progress.update(task_id, completed=max(10, int(eta)))
    progress.stop_task(task_id)
    ok = failure is None or stage.optional
    return ok, seconds, failure


def _render(stages: list[Stage], state: dict) -> Group:
    table = _status_table(
        stages,
        current_idx=state["current_idx"],
        statuses=state["statuses"],
        timings=state["timings"],
        history=state["history"],
    )
    progress = state["progress"]
    tail_lines = state.get("current_tail") or []
    stage_name = state.get("current_stage", "—")
    tail_body = "\n".join(tail_lines[-6:]) if tail_lines else "(waiting for output…)"
    tail_panel = Panel(
        tail_body,
        title=f"live log tail — {stage_name}",
        border_style="dim",
        padding=(0, 1),
    )
    return Group(table, Rule(style="dim"), progress, tail_panel)


def _run(stages: list[Stage], args: argparse.Namespace) -> int:
    history = _load_history()
    statuses: dict[str, str] = {s.name: "pending" for s in stages}
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}

    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA[/dim]"),
        TimeRemainingColumn(compact=True),
        expand=True,
        transient=False,
    )
    state = {
        "progress": progress,
        "current_idx": -1,
        "current_stage": "—",
        "statuses": statuses,
        "timings": timings,
        "history": history,
        "current_tail": [],
    }

    overall_start = time.perf_counter()
    total_eta = sum(history.get(s.name, s.eta_seed_s) for s in stages)

    CONSOLE.print()
    CONSOLE.print(
        Panel.fit(
            f"[bold]I³ end-to-end orchestrator — mode={args.mode}[/bold]\n"
            f"[dim]{len(stages)} stage(s) scheduled · "
            f"estimated total: {_fmt_seconds(total_eta)} · "
            f"logs: {LOG_DIR.relative_to(REPO_ROOT)}[/dim]",
            border_style="cyan",
        )
    )
    CONSOLE.print()

    # Group stages by wave.  Waves run in ascending order; stages
    # within a wave run concurrently.  Action-only stages (pure Python
    # mutation) always run alone — grouped by themselves.
    waves: dict[int, list[Stage]] = {}
    for s in stages:
        waves.setdefault(s.wave, []).append(s)
    wave_order = sorted(waves.keys())

    # How many subprocess stages may run concurrently per wave.  Honour
    # ``--parallelism`` if supplied, else scale with core count (but at
    # least 2, at most 8 — disk + network I/O typically saturate before
    # that).
    max_parallel = getattr(args, "parallelism", 0) or max(
        2, min(8, os.cpu_count() or 4)
    )

    with Live(_render(stages, state), console=CONSOLE, refresh_per_second=6) as live:
        aborted = False
        for wave in wave_order:
            if aborted:
                break
            wave_stages = waves[wave]

            # Action stages must run alone — Python actions mutate the
            # orchestrator's own process state (the launch_server
            # action even ``os.execvp``s).  Fall back to sequential for
            # those.
            has_action = any(s.action is not None for s in wave_stages)
            if len(wave_stages) == 1 or has_action:
                # Sequential — each stage alone.
                for stage in wave_stages:
                    idx = stages.index(stage)
                    state["current_idx"] = idx
                    statuses[stage.name] = "running"
                    live.update(_render(stages, state))

                    ok, seconds, failure = _run_stage(stage, args, history, state)
                    timings[stage.name] = seconds

                    if ok and failure is None:
                        statuses[stage.name] = "done"
                        prev = history.get(stage.name, seconds)
                        history[stage.name] = 0.3 * prev + 0.7 * seconds
                    elif ok and failure is not None:
                        statuses[stage.name] = "skipped"
                        failures[stage.name] = failure
                    else:
                        statuses[stage.name] = "failed"
                        failures[stage.name] = failure or "unknown failure"
                        aborted = True
                        live.update(_render(stages, state))
                        break

                    live.update(_render(stages, state))
                continue

            # Concurrent — run the whole wave through a thread pool.
            # Each subprocess has its own PIPE, its own log file, and
            # its own progress task, so contention is only at the
            # ``rich.Live.update`` surface which is already thread-safe
            # under a single Live instance.
            CONSOLE.print(
                f"[dim]  → wave {wave}: running {len(wave_stages)} stages "
                f"concurrently (max {min(max_parallel, len(wave_stages))} at a "
                f"time)[/dim]"
            )
            for s in wave_stages:
                statuses[s.name] = "running"
            live.update(_render(stages, state))

            import concurrent.futures as _cf

            with _cf.ThreadPoolExecutor(
                max_workers=min(max_parallel, len(wave_stages)),
                thread_name_prefix=f"wave{wave}",
            ) as pool:
                future_to_stage = {
                    pool.submit(_run_stage, s, args, history, state): s
                    for s in wave_stages
                }
                for fut in _cf.as_completed(future_to_stage):
                    stage = future_to_stage[fut]
                    try:
                        ok, seconds, failure = fut.result()
                    except Exception as exc:  # pragma: no cover
                        ok, seconds, failure = False, 0.0, f"{type(exc).__name__}: {exc}"
                    timings[stage.name] = seconds
                    if ok and failure is None:
                        statuses[stage.name] = "done"
                        prev = history.get(stage.name, seconds)
                        history[stage.name] = 0.3 * prev + 0.7 * seconds
                    elif ok and failure is not None:
                        statuses[stage.name] = "skipped"
                        failures[stage.name] = failure
                    else:
                        statuses[stage.name] = "failed"
                        failures[stage.name] = failure or "unknown failure"
                        aborted = True
                    live.update(_render(stages, state))

    _save_history(history)

    total = time.perf_counter() - overall_start
    CONSOLE.print()
    CONSOLE.print(Rule(style="cyan"))
    CONSOLE.print()

    summary = Table(show_header=True, header_style="bold", expand=True)
    summary.add_column("Stage", style="bold")
    summary.add_column("Status", width=14)
    summary.add_column("Elapsed", justify="right", width=9)
    summary.add_column("Log file", style="dim")

    any_failed = False
    for s in stages:
        status = statuses[s.name]
        style = {
            "done": "green",
            "skipped": "yellow",
            "failed": "red",
            "pending": "dim",
            "running": "cyan",
        }.get(status, "white")
        if status == "failed":
            any_failed = True
        summary.add_row(
            s.name,
            Text(status, style=style),
            _fmt_seconds(timings.get(s.name)),
            str(s.log_path.relative_to(REPO_ROOT)) if s.cmd else "—",
        )

    CONSOLE.print(summary)
    CONSOLE.print()
    CONSOLE.print(f"[bold]Total wall-clock: {_fmt_seconds(total)}[/bold]")

    if failures:
        CONSOLE.print()
        for name, msg in failures.items():
            stage = next((s for s in stages if s.name == name), None)
            tail = ""
            if stage and stage.cmd and stage.log_path.exists():
                with suppress(OSError):
                    lines = stage.log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    tail = "\n".join(lines[-12:])
            CONSOLE.print(
                Panel(
                    f"[bold red]{name}[/bold red]: {msg}\n\n"
                    f"[dim]last 12 lines of log:[/dim]\n{tail or '(no log)'}",
                    title="failure",
                    border_style="red",
                )
            )

    return 1 if any_failed else 0


# ---------------------------------------------------------------------------
# --list output
# ---------------------------------------------------------------------------

def _print_stage_list(stages: list[Stage], args: argparse.Namespace) -> None:
    history = _load_history()
    table = Table(
        title=f"Stages for mode={args.mode}",
        title_style="bold",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="bold")
    table.add_column("Category", width=10)
    table.add_column("Description")
    table.add_column("ETA (seed)", justify="right", width=12)
    table.add_column("Optional", width=10)

    total_eta = 0.0
    for i, stage in enumerate(stages):
        eta = history.get(stage.name, stage.eta_seed_s)
        total_eta += eta
        cat_colour = _CATEGORY_COLOURS.get(stage.category, "white")
        table.add_row(
            str(i + 1),
            stage.name,
            Text(stage.category, style=cat_colour),
            stage.description,
            _fmt_seconds(eta),
            "yes" if stage.optional else "no",
        )
    CONSOLE.print(table)
    CONSOLE.print()
    CONSOLE.print(f"[bold]Estimated total wall-clock: {_fmt_seconds(total_eta)}[/bold]")


# ---------------------------------------------------------------------------
# Argument parser.
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end orchestrator for I³",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help=(
            "fast = quickstart (skip data/training/eval/quality/perf, use demo ckpts). "
            "full = every stage — prereq → install → env → data → dialogue → "
            "encoder → SLM → evaluate → seed → lint → typecheck → test → security "
            "→ redteam → verify → benchmarks → onnx → profile-edge → docs → serve."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Alias for --mode fast.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Alias for --mode full.",
    )
    p.add_argument(
        "--skip",
        default="",
        help="Comma-separated stage names to skip (e.g. --skip verify,evaluate).",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated stage names to run exclusively (overrides --mode).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose outputs already exist on disk.",
    )
    p.add_argument(
        "--no-serve",
        action="store_true",
        help="Run every stage except the final server launch.",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip the dependency-install step (assume environment is ready).",
    )
    p.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip the ONNX export stage (full mode only).",
    )
    p.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip the MkDocs build stage (full mode only).",
    )
    p.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip the pytest-benchmark stage (full mode only).",
    )
    p.add_argument(
        "--with-docker",
        action="store_true",
        help="Include the Docker image-build stage (full mode only).",
    )
    p.add_argument(
        "--sessions-per-archetype",
        type=int,
        default=400,
        help="Synthetic sessions per archetype (data stage).",
    )
    p.add_argument(
        "--encoder-epochs",
        type=int,
        default=10,
        help="Epochs for TCN encoder training.",
    )
    p.add_argument(
        "--slm-epochs",
        type=int,
        default=5,
        help="Epochs for SLM training.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the stage graph for the chosen mode and exit.",
    )
    p.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help=(
            "Max concurrent stages per wave.  0 (default) = "
            "min(8, cpu_count).  1 = serial execution."
        ),
    )
    args = p.parse_args(argv)
    if args.fast:
        args.mode = "fast"
    if args.full:
        args.mode = "full"
    return args


def _autoload_dotenv() -> None:
    """Load ``.env`` into ``os.environ`` so subprocess stages inherit it.

    Stages invoked via ``subprocess.Popen`` only see variables already in
    the orchestrator's environment.  Without this, ``I3_ENCRYPTION_KEY``,
    ``ANTHROPIC_API_KEY`` and friends are silently missing for every
    downstream stage even though the ``env`` stage just wrote them.

    Soft dependency on ``python-dotenv``.  If the package is not
    available (or the file is missing), do nothing — stages that need
    specific vars will still work if they've been set in the shell.
    """
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Minimal inline parser fallback so the orchestrator stays
        # functional even on a machine where python-dotenv is not yet
        # installed (e.g. the very first ``install`` stage run).
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return
    load_dotenv(dotenv_path, override=False)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _autoload_dotenv()
    stages = _build_all_stages(args)
    stages = _filter(stages, args)
    if args.list:
        _print_stage_list(stages, args)
        return 0
    if not stages:
        CONSOLE.print("[yellow]No stages selected — nothing to do.[/yellow]")
        return 0
    return _run(stages, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
