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
    # In a venv, ``pip install --user`` fails because user-site is hidden;
    # install into the active env instead.  ``sys.prefix != sys.base_prefix``
    # is the portable venv-detection idiom.
    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "rich>=13.7"]
    if not in_venv:
        cmd.insert(-1, "--user")
    subprocess.check_call(cmd)


_ensure_rich()

from rich.columns import Columns  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.layout import Layout  # noqa: E402
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


# ---------------------------------------------------------------------------
# Dashboard-support imports (sibling modules authored by this feature).
# They live alongside the orchestrator so a fresh clone needs nothing
# beyond the repo root on sys.path.
# ---------------------------------------------------------------------------

# ``scripts/`` is not a package; ``orchestration_progress`` lives next to
# this file, so we load it via a filesystem spec.
import importlib.util as _importlib_util  # noqa: E402

_PROGRESS_SPEC = _importlib_util.spec_from_file_location(
    "orchestration_progress",
    Path(__file__).resolve().parent / "orchestration_progress.py",
)
assert _PROGRESS_SPEC and _PROGRESS_SPEC.loader
orchestration_progress = _importlib_util.module_from_spec(_PROGRESS_SPEC)
sys.modules.setdefault("orchestration_progress", orchestration_progress)
_PROGRESS_SPEC.loader.exec_module(orchestration_progress)

# ``i3.runtime.monitoring`` needs the repo root on sys.path so the
# import works whether we're invoked via ``poetry run`` or the bare
# interpreter.
_REPO_ROOT_STR = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from i3.runtime.monitoring import (  # noqa: E402
        ResourceSampler,
        ResourceSnapshot,
        render_resource_panel,
    )

    _HAS_MONITORING = True
except Exception:  # noqa: BLE001 - extremely unlikely but keeps orchestrator robust
    ResourceSampler = None  # type: ignore[assignment,misc]
    ResourceSnapshot = None  # type: ignore[assignment,misc]
    render_resource_panel = None  # type: ignore[assignment]
    _HAS_MONITORING = False


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
# Post-stage health check registry.  Each function takes the completed
# ``Stage`` and returns an optional one-line badge rendered in the
# pipeline-status table (e.g. "✓ 2.5M params (16 MiB)").  A raised
# exception turns into a red "check failed" annotation but never aborts
# the run — we trust the stage's own exit code for failure gating.
# ---------------------------------------------------------------------------


@dataclass
class HealthBadge:
    """One-line post-stage annotation."""

    ok: bool
    text: str


def _fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.1f} {units[i]}"


def _check_torch_checkpoint(path: Path, required_keys: list[str]) -> HealthBadge:
    """Load a torch checkpoint and verify required keys + param count."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return HealthBadge(False, f"torch not available ({exc})")
    try:
        ckpt = torch.load(
            str(path), map_location="cpu", weights_only=False
        )  # trusted: produced by us this run
    except Exception as exc:  # noqa: BLE001
        return HealthBadge(False, f"load failed: {exc}")
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        return HealthBadge(False, f"missing keys: {','.join(missing)}")
    n_params = 0
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or {}
    if isinstance(state, dict):
        n_params = sum(
            int(v.numel()) for v in state.values() if hasattr(v, "numel")
        )
    size = path.stat().st_size
    if n_params:
        pretty = (
            f"{n_params / 1_000_000:.1f}M params"
            if n_params >= 1_000_000
            else f"{n_params / 1000:.1f}K params"
        )
        return HealthBadge(True, f"{pretty} ({_fmt_bytes(size)})")
    return HealthBadge(True, f"checkpoint {_fmt_bytes(size)}")


def _check_onnx(path: Path) -> HealthBadge:
    """Ensure the exported ONNX has a non-empty graph."""
    if not path.exists():
        return HealthBadge(False, "onnx file missing")
    try:
        import onnx  # type: ignore
    except Exception:  # noqa: BLE001
        return HealthBadge(True, f"onnx {_fmt_bytes(path.stat().st_size)} (onnx not installed)")
    try:
        model = onnx.load(str(path))
        n_nodes = len(model.graph.node)
        return HealthBadge(
            n_nodes > 0,
            f"{n_nodes} nodes ({_fmt_bytes(path.stat().st_size)})",
        )
    except Exception as exc:  # noqa: BLE001
        return HealthBadge(False, f"onnx parse failed: {exc}")


def _check_file_nonempty(path: Path) -> HealthBadge:
    if not path.exists():
        return HealthBadge(False, f"{path.name} missing")
    size = path.stat().st_size
    if size == 0:
        return HealthBadge(False, f"{path.name} empty")
    return HealthBadge(True, f"{path.name} {_fmt_bytes(size)}")


def _check_data_outputs(stage: Stage) -> HealthBadge:
    """All produced .pt shards exist + report row count if possible."""
    missing = [p for p in stage.produces if not p.exists()]
    if missing:
        return HealthBadge(False, f"missing {','.join(p.name for p in missing)}")
    total_rows = 0
    try:
        import torch  # type: ignore

        for p in stage.produces:
            if p.suffix == ".pt":
                data = torch.load(str(p), map_location="cpu", weights_only=False)
                # Shape [N, ...] or tuple of (X, y); best-effort.
                if hasattr(data, "shape"):
                    total_rows += int(data.shape[0])
                elif isinstance(data, (tuple, list)) and hasattr(data[0], "shape"):
                    total_rows += int(data[0].shape[0])
                elif isinstance(data, dict) and "sequences" in data and hasattr(data["sequences"], "shape"):
                    total_rows += int(data["sequences"].shape[0])
    except Exception:  # noqa: BLE001
        pass
    size = sum(p.stat().st_size for p in stage.produces)
    if total_rows:
        return HealthBadge(True, f"{total_rows:,} rows · {_fmt_bytes(size)}")
    return HealthBadge(True, f"{len(stage.produces)} files · {_fmt_bytes(size)}")


def _check_pytest_log(stage: Stage) -> HealthBadge:
    """Scrape pass / fail totals from the latest log tail."""
    if not stage.log_path.exists():
        return HealthBadge(False, "no log")
    import re as _re

    text = stage.log_path.read_text(encoding="utf-8", errors="replace")
    pat = _re.compile(
        r"(\d+)\s+passed"
        r"(?:[^,]*?,\s*(\d+)\s+failed)?"
        r"(?:[^,]*?,\s*(\d+)\s+skipped)?"
    )
    last: tuple[int, int, int] | None = None
    for m in pat.finditer(text):
        last = (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))
    if last is None:
        return HealthBadge(False, "no pytest summary found")
    passed, failed, skipped = last
    return HealthBadge(
        failed == 0,
        f"{passed} passed · {failed} failed · {skipped} skipped",
    )


def _check_docker_image(_stage: Stage) -> HealthBadge:
    res = subprocess.run(
        ["docker", "image", "inspect", "i3:latest"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return HealthBadge(False, "docker image missing")
    return HealthBadge(True, "i3:latest present")


def _check_verify_report(_stage: Stage) -> HealthBadge:
    """Find the most recent verification_*.md and scrape its verdict."""
    candidates = sorted(REPORTS_DIR.glob("verification_*.md"))
    if not candidates:
        return HealthBadge(False, "no verification report")
    latest = candidates[-1]
    text = latest.read_text(encoding="utf-8", errors="replace")
    # Cheap heuristic: verification reports end with PASS/FAIL.
    if "FAIL" in text.split("\n")[-5:][0:].__str__().upper() or "fail=" in text.lower() and "fail=0" not in text.lower():
        ok = False
    else:
        ok = True
    return HealthBadge(ok, f"{latest.name}")


def _check_docs_site(_stage: Stage) -> HealthBadge:
    idx = REPO_ROOT / "site" / "index.html"
    if not idx.exists() or idx.stat().st_size == 0:
        return HealthBadge(False, "site/index.html missing or empty")
    return HealthBadge(True, f"site/index.html {_fmt_bytes(idx.stat().st_size)}")


# Registry keyed by stage name.
HEALTH_CHECKS: dict[str, Callable[[Stage], HealthBadge]] = {
    "train-encoder": lambda s: _check_torch_checkpoint(
        REPO_ROOT / "checkpoints" / "encoder" / "best_model.pt",
        ["model_state_dict"],
    ),
    "train-slm": lambda s: _check_torch_checkpoint(
        REPO_ROOT / "checkpoints" / "slm" / "best_model.pt",
        ["model_state_dict"],
    ),
    "onnx-export": lambda s: _check_onnx(
        REPO_ROOT / "checkpoints" / "encoder" / "tcn.onnx"
    ),
    "data": _check_data_outputs,
    "dialogue": _check_data_outputs,
    "test": _check_pytest_log,
    "benchmarks": _check_pytest_log,
    "docker-build": _check_docker_image,
    "verify": _check_verify_report,
    "docs": _check_docs_site,
}


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
# Pre-flight checks.  Run before wave 0 so broken environments fail
# immediately with a clear diagnosis rather than exploding half-way
# through training.
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """One pre-flight line."""

    ok: bool  # True = green ✓; False + fatal=False = yellow ⚠; False + fatal=True = red ✗
    label: str
    detail: str
    fatal: bool = False


def _preflight_python() -> PreflightResult:
    v = sys.version_info
    ok = v >= (3, 12)
    return PreflightResult(
        ok,
        "Python version",
        f"{v.major}.{v.minor}.{v.micro}",
        fatal=not ok,
    )


def _preflight_disk() -> PreflightResult:
    try:
        import psutil

        path = "D:" if os.path.exists("D:") else os.getcwd()
        usage = psutil.disk_usage(path)
        gib = usage.free / (1024**3)
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(False, "Disk free", f"error: {exc}", fatal=False)
    if gib < 5:
        return PreflightResult(False, f"Disk free ({path})", f"{gib:.1f} GiB — CRITICAL", fatal=True)
    if gib < 10:
        return PreflightResult(False, f"Disk free ({path})", f"{gib:.1f} GiB — low", fatal=False)
    return PreflightResult(True, f"Disk free ({path})", f"{gib:.1f} GiB")


def _preflight_docker(args: argparse.Namespace) -> PreflightResult | None:
    if not getattr(args, "with_docker", False):
        return None
    if not _docker_available():
        return PreflightResult(False, "Docker", "docker not on PATH", fatal=True)
    try:
        res = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
        if res.returncode != 0:
            return PreflightResult(False, "Docker daemon", "not reachable", fatal=True)
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(False, "Docker daemon", f"error: {exc}", fatal=True)
    return PreflightResult(True, "Docker daemon", "reachable")


def _preflight_env_file() -> PreflightResult:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return PreflightResult(False, ".env", "missing — will be generated", fatal=False)
    text = env_path.read_text(encoding="utf-8", errors="replace")
    n_placeholders = text.count("REPLACE_ME")
    if n_placeholders:
        return PreflightResult(
            False,
            ".env",
            f"contains {n_placeholders} REPLACE_ME placeholder(s)",
            fatal=False,
        )
    return PreflightResult(True, ".env", "populated")


def _preflight_gpu() -> PreflightResult | None:
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return PreflightResult(True, "GPU", "CPU-only torch (no CUDA)", fatal=False)
    name = torch.cuda.get_device_name(0)
    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
        vram_gib = total_b / (1024**3)
    except Exception:  # noqa: BLE001
        vram_gib = 0.0
    if vram_gib and vram_gib < 4:
        return PreflightResult(False, "GPU", f"{name} · {vram_gib:.1f} GiB VRAM — tight", fatal=False)
    if vram_gib:
        return PreflightResult(True, "GPU", f"{name} · {vram_gib:.1f} GiB VRAM")
    return PreflightResult(True, "GPU", name)


def _preflight_llm_ping() -> PreflightResult | None:
    """Best-effort ping of a configured LLM provider.

    Only runs when ``I3_TEST_LIVE_PROVIDERS=1`` so we never surprise
    the user with a paid API call.
    """
    if os.environ.get("I3_TEST_LIVE_PROVIDERS") != "1":
        return None
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        from anthropic import Anthropic  # type: ignore

        client = Anthropic(api_key=key)
        client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(False, "Anthropic API", f"ping failed: {exc}", fatal=False)
    return PreflightResult(True, "Anthropic API", "reachable (1-token ping)")


def _preflight_sidecars() -> list[PreflightResult]:
    """Best-effort probe of OTel collector + MLflow if configured."""
    results: list[PreflightResult] = []
    otel = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if otel:
        import socket as _socket
        from urllib.parse import urlparse

        parsed = urlparse(otel if "://" in otel else f"http://{otel}")
        host = parsed.hostname or "localhost"
        port = parsed.port or 4317
        try:
            with _socket.create_connection((host, port), timeout=2):
                pass
            results.append(PreflightResult(True, "OTel collector", f"{host}:{port}"))
        except Exception as exc:  # noqa: BLE001
            results.append(
                PreflightResult(False, "OTel collector", f"{host}:{port} · {exc}", fatal=False)
            )
    mlflow = os.environ.get("MLFLOW_TRACKING_URI", "")
    if mlflow.startswith("http"):
        try:
            import urllib.request as _ur

            with _ur.urlopen(mlflow, timeout=2):
                pass
            results.append(PreflightResult(True, "MLflow", mlflow))
        except Exception as exc:  # noqa: BLE001
            results.append(
                PreflightResult(False, "MLflow", f"{mlflow} · {exc}", fatal=False)
            )
    return results


def _render_preflight_panel(results: list[PreflightResult]) -> Panel:
    """Render the pre-flight summary as a single collapsible panel."""
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column()
    for r in results:
        if r.ok:
            icon = Text("✓", style="green")
        elif r.fatal:
            icon = Text("✗", style="red bold")
        else:
            icon = Text("⚠", style="yellow")
        table.add_row(icon, Text(r.label, style="bold"), Text(r.detail))
    return Panel(table, title="Pre-flight", border_style="cyan", padding=(0, 1))


def _preflight_checks(args: argparse.Namespace) -> list[PreflightResult]:
    """Run all pre-flight checks in order."""
    results: list[PreflightResult] = [
        _preflight_python(),
        _preflight_disk(),
        _preflight_env_file(),
    ]
    for optional in (
        _preflight_gpu(),
        _preflight_docker(args),
        _preflight_llm_ping(),
    ):
        if optional is not None:
            results.append(optional)
    results.extend(_preflight_sidecars())
    return results


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
            # ``demo/pre_seed.py`` seeds a single user id into the
            # SQLite diary.  The older bulk-seed flags
            # (--users / --sessions-per-user) belong to a deprecated
            # variant that no longer exists.
            cmd=py
            + [
                "demo/pre_seed.py",
                "--user-id",
                "demo",
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
            # ``-p no:benchmark`` disables the pytest-benchmark plugin
            # because it self-disables under xdist and, with the
            # project's ``filterwarnings = error`` policy, its
            # disable-warning is promoted to a fatal INTERNALERROR.
            # The ``benchmarks`` stage below runs benchmarks serially.
            description="pytest unit + integration tests (parallel -n auto)",
            cmd=(
                ["poetry", "run", "pytest", "-q", "-n", "auto", "-p", "no:benchmark", "--timeout=120"]
                if _poetry_available()
                else [sys.executable, "-m", "pytest", "-q", "-n", "auto", "-p", "no:benchmark", "--timeout=120"]
            ),
            eta_seed_s=180.0,
            optional=True,
            skip_if=lambda a: a.mode != "full" or a.skip_tests,
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
            # Benchmarks MUST run serially — pytest-benchmark self-
            # disables under xdist because it relies on stable single-
            # process timing.  Running with ``-p no:xdist`` avoids the
            # plugin loading at all; no ``-n auto`` here.
            description="Latency + throughput micro-benchmarks (serial)",
            cmd=(
                ["make", "benchmarks"]
                if _make_available()
                else py + ["-m", "pytest", "benchmarks/", "-q", "-p", "no:xdist"]
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
    badges: dict[str, HealthBadge] | None = None,
) -> Table:
    badges = badges or {}
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
    table.add_column("ETA / badge", width=28)

    for i, stage in enumerate(stages):
        status = statuses.get(stage.name, "pending")
        if i == current_idx and status == "running":
            icon = "[cyan]●[/cyan]"
            status_cell = Text("running", style="cyan")
        elif status == "running":
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

        # ETA column doubles as the post-stage badge once the stage
        # has finished — "✓ 2.5M params (42 MiB)" etc.
        badge = badges.get(stage.name)
        if status == "done" and badge is not None:
            colour = "green" if badge.ok else "yellow"
            eta_cell: Text = Text(
                f"{'✓' if badge.ok else '⚠'} {badge.text}", style=colour
            )
        elif status == "pending":
            eta_cell = Text(_fmt_seconds(eta), style="dim")
        else:
            eta_cell = Text("—", style="dim")

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
    live_state: dict,
) -> int:
    """Stream a subprocess's stdout into log file, tail ring, and progress.

    For every line emitted we:

    1. Write it verbatim to the stage's log file.
    2. Append an ANSI-stripped copy to the shared rolling tail (so the
       dashboard panel stays readable).
    3. Dispatch to the stage's registered progress parser; if the
       parser returns a :class:`ProgressUpdate` the rich progress bar
       is updated with absolute ``completed`` / ``total`` values — this
       is how the bar tracks real training progress instead of the
       time-based ETA fallback.
    """
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
            raw = line.rstrip("\n")
            clean = orchestration_progress.strip_ansi(raw)
            tail.append(clean)
            while len(tail) > 6:
                tail.pop(0)

            # Stage-specific progress parser — drives the real bar.
            update = orchestration_progress.parse_line(stage.name, clean)
            if update is not None:
                total = max(int(update.total), 1)
                progress.update(
                    task_id,
                    completed=min(float(update.completed), float(total)),
                    total=float(total),
                    description=(
                        f"[cyan]{stage.name}[/cyan] — {update.description}"
                        if update.description
                        else f"[cyan]{stage.name}[/cyan] — {stage.description}"
                    ),
                )
                # Flag the stage as driven-by-parser so the heartbeat
                # stops clobbering ``completed`` with the ETA estimate.
                live_state.setdefault("parser_driven", set()).add(stage.name)
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
    """Run one stage and return ``(ok, elapsed_seconds, failure_msg)``.

    Responsibilities:

    * Allocate a progress task and drive it either via the parser
      (preferred) or via a time-based ETA heartbeat (fallback).
    * Keep a per-stage rolling log tail so the multi-pane log grid can
      render every concurrent stage independently.
    * Sample resource telemetry every 500 ms and record per-stage peaks
      for the final summary.
    * After a successful stage, run the registered health check and
      attach a one-line badge to the pipeline-status table.
    """
    eta = history.get(stage.name, stage.eta_seed_s)
    extra_env: dict[str, str] = {}
    if stage.needs_demo_mode:
        extra_env["I3_DEMO_MODE"] = "1"
    # WIN: force UTF-8 I/O in every stage subprocess.  bandit and some
    # other tools emit BOM / non-ASCII characters that a default
    # Windows cp1251 stdout can't encode, triggering spurious
    # "'charmap' codec can't encode" RuntimeErrors.  The ``UTF-8`` flag
    # (PEP 540) flips Python's stdout/stderr encoding to utf-8 and
    # disables strict-ASCII on the file system layer.
    extra_env.setdefault("PYTHONIOENCODING", "utf-8")
    extra_env.setdefault("PYTHONUTF8", "1")

    tail: list[str] = []
    progress = live_state["progress"]
    task_id = progress.add_task(
        description=f"[cyan]{stage.name}[/cyan] — {stage.description}",
        total=max(10, int(eta)),
    )
    # Register this tail in the per-stage dict so the multi-pane log
    # grid can render every active stage concurrently.
    live_state.setdefault("stage_tails", {})[stage.name] = tail
    live_state.setdefault("active_stages", set()).add(stage.name)
    # Last-stage pointer is still useful for the --quiet single-pane
    # view; mutate under a lock-free single-writer discipline.
    live_state["current_tail"] = tail
    live_state["current_stage"] = stage.name
    start = time.perf_counter()

    # Peak-resource tracking — shared sampler with the main loop.
    peaks = live_state.setdefault("peaks", {}).setdefault(
        stage.name,
        {"cpu": 0.0, "ram_mib": 0.0, "vram_mib": 0.0},
    )
    sampler: object | None = live_state.get("sampler")

    stop = threading.Event()

    def _heartbeat() -> None:
        while not stop.is_set():
            elapsed = time.perf_counter() - start
            # Only advance the bar by wall-clock when we don't have a
            # real progress parser driving it.
            if stage.name not in live_state.get("parser_driven", set()):
                progress.update(task_id, completed=min(elapsed, eta - 0.01))
            # Sample peak resources for this stage.
            if sampler is not None:
                try:
                    snap = sampler.sample()  # type: ignore[attr-defined]
                    peaks["cpu"] = max(peaks["cpu"], snap.cpu_pct)
                    peaks["ram_mib"] = max(peaks["ram_mib"], snap.ram_used_mib)
                    if snap.vram_used_mib is not None:
                        peaks["vram_mib"] = max(peaks["vram_mib"], snap.vram_used_mib)
                except Exception:  # noqa: BLE001
                    pass
            stop.wait(0.5)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    failure: str | None = None
    try:
        if stage.action is not None:
            stage.action(args)
            rc = 0
        else:
            rc = _stream_subprocess(
                stage, extra_env, progress, task_id, tail, live_state
            )
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
        live_state.get("active_stages", set()).discard(stage.name)
    seconds = time.perf_counter() - start
    progress.update(task_id, completed=max(10, int(eta)))
    progress.stop_task(task_id)
    ok = failure is None or stage.optional

    # Post-stage health check — best-effort; never turns a passing
    # stage into a failure, but the badge surfaces any anomaly.
    if ok and failure is None and stage.name in HEALTH_CHECKS:
        try:
            badge = HEALTH_CHECKS[stage.name](stage)
        except Exception as exc:  # noqa: BLE001
            badge = HealthBadge(False, f"check error: {exc}")
        live_state.setdefault("badges", {})[stage.name] = badge

    return ok, seconds, failure


def _render_log_grid(state: dict) -> Panel:
    """Render one tail panel per currently-active stage.

    During a parallel wave (e.g. wave 7 with 5 concurrent stages) we
    flow the tails into a :class:`rich.columns.Columns` layout so every
    stage gets its own visible tile rather than fighting over a single
    tail panel.  Finished stages in the same wave get a compact "done"
    summary in place of their tail.
    """
    active: set[str] = state.get("active_stages", set())
    tails: dict[str, list[str]] = state.get("stage_tails", {})
    statuses: dict[str, str] = state.get("statuses", {})
    badges: dict[str, HealthBadge] = state.get("badges", {})

    if not active and not tails:
        return Panel(
            "(waiting for first stage…)",
            title="Active stage logs",
            border_style="dim",
            padding=(0, 1),
        )

    panels: list[Panel] = []
    # Active first, then just-finished siblings from the current wave.
    ordered = sorted(active) + [n for n in tails if n not in active][-3:]
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        status = statuses.get(name, "running")
        tail = tails.get(name, [])
        if status == "done":
            badge = badges.get(name)
            body = (
                f"[green]✓ done[/green] — {badge.text}"
                if badge
                else "[green]✓ done[/green]"
            )
            border = "green"
        elif status == "failed":
            body = "[red]✗ failed[/red] — see log tail"
            border = "red"
        else:
            body = "\n".join(tail[-4:]) if tail else "(waiting for output…)"
            border = "cyan" if name in active else "dim"
        panels.append(
            Panel(
                body,
                title=name,
                border_style=border,
                padding=(0, 1),
                width=42,
                height=7,
            )
        )
    return Panel(
        Columns(panels, equal=False, expand=True),
        title="Active stage logs",
        border_style="dim",
        padding=(0, 0),
    )


def _render_quiet(stages: list[Stage], state: dict) -> Group:
    """Simplified renderer for ``--quiet``: status table + one resource line."""
    table = _status_table(
        stages,
        current_idx=state["current_idx"],
        statuses=state["statuses"],
        timings=state["timings"],
        history=state["history"],
        badges=state.get("badges", {}),
    )
    snap: ResourceSnapshot | None = state.get("last_snapshot")  # type: ignore[assignment]
    res_line = snap.format_line() if snap else ""
    return Group(table, Text(res_line, style="dim"))


def _render(stages: list[Stage], state: dict) -> Layout:
    """Compose the full dashboard as a rich :class:`Layout` tree.

    Top-down: pre-flight → resources → pipeline status table → progress
    bars → multi-pane active-stage logs.  Each section is wrapped in a
    Panel so the overall layout remains legible on a narrow terminal.
    """
    if state.get("quiet"):
        # The quiet layout is a plain group so rich can render it as a
        # sequence of lines; :class:`Live` is happy with either.
        return _render_quiet(stages, state)  # type: ignore[return-value]

    layout = Layout()
    layout.split_column(
        Layout(name="top", size=3 + max(1, len(state.get("preflight", [])))),
        Layout(name="resources", size=9),
        Layout(name="status", ratio=2, minimum_size=10),
        Layout(name="progress", size=max(3, len(stages) + 2)),
        Layout(name="logs", size=9),
    )

    # Pre-flight panel (may be empty on skip).
    preflight = state.get("preflight") or []
    layout["top"].update(
        _render_preflight_panel(preflight)
        if preflight
        else Panel(
            Text("(pre-flight skipped)", style="dim"),
            title="Pre-flight",
            border_style="dim",
            padding=(0, 1),
        )
    )

    # Resource panel — refresh-driven by the sampler in the main loop.
    snap: ResourceSnapshot | None = state.get("last_snapshot")  # type: ignore[assignment]
    if snap is not None and render_resource_panel is not None:
        layout["resources"].update(render_resource_panel(snap))
    else:
        layout["resources"].update(
            Panel(Text("(monitoring unavailable)", style="dim"), border_style="dim")
        )

    layout["status"].update(
        _status_table(
            stages,
            current_idx=state["current_idx"],
            statuses=state["statuses"],
            timings=state["timings"],
            history=state["history"],
            badges=state.get("badges", {}),
        )
    )
    layout["progress"].update(
        Panel(state["progress"], title="Live progress", border_style="dim")
    )
    layout["logs"].update(_render_log_grid(state))
    return layout


def _run(stages: list[Stage], args: argparse.Namespace) -> int:
    history = _load_history()
    statuses: dict[str, str] = {s.name: "pending" for s in stages}
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}

    # ---- Pre-flight --------------------------------------------------
    preflight_results = _preflight_checks(args)
    fatal_fails = [r for r in preflight_results if not r.ok and r.fatal]
    if fatal_fails:
        CONSOLE.print(_render_preflight_panel(preflight_results))
        CONSOLE.print(
            "[red bold]Fatal pre-flight failures:[/red bold] "
            + ", ".join(r.label for r in fatal_fails)
        )
        if not getattr(args, "yes", False):
            try:
                ans = input("Continue anyway? [y/N] ").strip().lower()
            except EOFError:
                ans = "n"
            if ans != "y":
                return 2

    # ---- Resource sampler (best-effort) ------------------------------
    sampler = ResourceSampler() if _HAS_MONITORING else None  # type: ignore[misc]

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
    state: dict = {
        "progress": progress,
        "current_idx": -1,
        "current_stage": "—",
        "statuses": statuses,
        "timings": timings,
        "history": history,
        "current_tail": [],
        "stage_tails": {},
        "active_stages": set(),
        "badges": {},
        "peaks": {},
        "parser_driven": set(),
        "preflight": preflight_results,
        "sampler": sampler,
        "last_snapshot": sampler.sample() if sampler else None,
        "quiet": getattr(args, "quiet", False),
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

    # Background resource-refresh thread.  Samples every ~500 ms so
    # the resource panel feels alive even when no stage is spewing
    # output.  Terminated cleanly via ``stop_event`` after the last
    # wave finishes.
    stop_event = threading.Event()

    def _resource_pump() -> None:
        while not stop_event.is_set() and sampler is not None:
            try:
                state["last_snapshot"] = sampler.sample()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            stop_event.wait(0.5)

    pump_thread: threading.Thread | None = None
    if sampler is not None:
        pump_thread = threading.Thread(target=_resource_pump, daemon=True)
        pump_thread.start()

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

    # Tear down the sampler pump before printing the final summary.
    stop_event.set()
    if pump_thread is not None:
        pump_thread.join(timeout=1.0)
    if sampler is not None:
        try:
            sampler.close()
        except Exception:  # noqa: BLE001
            pass

    _save_history(history)

    total = time.perf_counter() - overall_start
    CONSOLE.print()
    CONSOLE.print(Rule(style="cyan"))
    CONSOLE.print()

    summary = Table(
        title="Run summary",
        title_style="bold",
        show_header=True,
        header_style="bold",
        expand=True,
    )
    summary.add_column("Stage", style="bold")
    summary.add_column("Status", width=10)
    summary.add_column("Elapsed", justify="right", width=9)
    summary.add_column("Peak CPU", justify="right", width=9)
    summary.add_column("Peak RAM", justify="right", width=11)
    summary.add_column("Peak VRAM", justify="right", width=11)
    summary.add_column("Health", width=34)
    summary.add_column("Log", style="dim")

    any_failed = False
    peaks_map: dict[str, dict[str, float]] = state.get("peaks", {})
    badges_map: dict[str, HealthBadge] = state.get("badges", {})
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
        p = peaks_map.get(s.name, {})
        peak_cpu = f"{p['cpu']:.0f}%" if p.get("cpu") else "—"
        peak_ram = f"{p['ram_mib'] / 1024:.1f} GiB" if p.get("ram_mib") else "—"
        peak_vram = (
            f"{p['vram_mib'] / 1024:.1f} GiB" if p.get("vram_mib") else "—"
        )
        badge = badges_map.get(s.name)
        badge_cell = (
            Text(
                f"{'✓' if badge.ok else '⚠'} {badge.text}",
                style="green" if badge.ok else "yellow",
            )
            if badge
            else Text("—", style="dim")
        )
        summary.add_row(
            s.name,
            Text(status, style=style),
            _fmt_seconds(timings.get(s.name)),
            peak_cpu,
            peak_ram,
            peak_vram,
            badge_cell,
            str(s.log_path.relative_to(REPO_ROOT)) if s.cmd else "—",
        )

    CONSOLE.print(summary)
    CONSOLE.print()

    # Artefact inventory + report file links.
    artefact_patterns = [
        REPO_ROOT / "checkpoints" / "encoder" / "best_model.pt",
        REPO_ROOT / "checkpoints" / "slm" / "best_model.pt",
        REPO_ROOT / "checkpoints" / "encoder" / "tcn.onnx",
        REPO_ROOT / "data" / "synthetic" / "train.pt",
        REPO_ROOT / "data" / "synthetic" / "val.pt",
        REPO_ROOT / "data" / "synthetic" / "test.pt",
        REPO_ROOT / "data" / "dialogue" / "train.pt",
        REPO_ROOT / "data" / "dialogue" / "tokenizer.json",
        REPO_ROOT / "site" / "index.html",
    ]
    artefact_size = 0
    existing_artefacts: list[Path] = []
    for p in artefact_patterns:
        if p.exists():
            artefact_size += p.stat().st_size
            existing_artefacts.append(p)

    report_files = sorted(REPORTS_DIR.glob("*.json")) + sorted(
        REPORTS_DIR.glob("*.md")
    )
    if existing_artefacts or report_files:
        CONSOLE.print(
            Panel.fit(
                f"[bold]Artefacts:[/bold] {len(existing_artefacts)} files · "
                f"total {_fmt_bytes(artefact_size)}\n"
                + "\n".join(
                    f"  [dim]•[/dim] {p.relative_to(REPO_ROOT)}"
                    for p in existing_artefacts[:10]
                )
                + (
                    f"\n\n[bold]Reports[/bold] (latest 6):\n"
                    + "\n".join(
                        f"  [dim]•[/dim] {p.relative_to(REPO_ROOT)}"
                        for p in report_files[-6:]
                    )
                    if report_files
                    else ""
                ),
                title="Outputs",
                border_style="green" if not any_failed else "yellow",
            )
        )
        CONSOLE.print()

    # One-liner suitable for Slack / email.
    done_count = sum(1 for v in statuses.values() if v == "done")
    failed_count = sum(1 for v in statuses.values() if v == "failed")
    skipped_count = sum(1 for v in statuses.values() if v == "skipped")
    oneliner = (
        f"I³ orchestrator [{args.mode}] — "
        f"{done_count} ok, {failed_count} failed, {skipped_count} skipped · "
        f"{_fmt_seconds(total)} · "
        f"{len(existing_artefacts)} artefacts ({_fmt_bytes(artefact_size)})"
    )
    CONSOLE.print(Text(oneliner, style="bold magenta"))
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
        "--skip-tests",
        action="store_true",
        help=(
            "Skip the pytest stage.  Useful when one test is hanging "
            "a worker and blocking the downstream waves — the rest of "
            "the quality gates (lint, typecheck, security, redteam) "
            "still run because they don't share a pytest process."
        ),
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
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-confirm any interactive prompts (e.g. pre-flight warnings).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Simplified dashboard: status table + one resource line only "
            "(useful for CI or long-running terminals)."
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
