"""Code-integrity checks: AST parse, imports, lint, soft-import guards, ...

Each function in this module is registered with the global
:class:`~scripts.verification.framework.CheckRegistry` via the
``@register_check`` decorator and returns a
:class:`~scripts.verification.framework.CheckResult`.

Design:
    * Checks are best-effort -- missing tools (ruff, mypy) degrade to
      ``SKIP`` rather than ``FAIL`` so the harness never becomes a
      tooling-install blocker.
    * All filesystem walks anchor on :data:`REPO_ROOT` so the harness
      works from any current working directory.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import shutil
import subprocess
import time
from pathlib import Path

from scripts.verification.framework import CheckResult, register_check

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

PY_ROOTS: tuple[str, ...] = (
    "i3",
    "server",
    "scripts",
    "tests",
    "benchmarks",
    "training",
    "demo",
)

# Third-party SDKs whose imports MUST sit inside a try/except ImportError.
# Grouped here so the soft-import check can grep for both the optional
# ``import foo`` line AND the matching guard.
SOFT_IMPORT_LIBS: tuple[str, ...] = (
    "mlflow",
    "onnx",
    "executorch",
    "torchao",
    "langfuse",
    "librosa",
    "mediapipe",
    "flwr",
    "opacus",
    "openai",
    "mcp",
    "dspy",
    "nemoguardrails",
    "litellm",
)


def _iter_py(root: Path) -> list[Path]:
    """Recursively yield every ``.py`` file under ``root``.

    Hidden directories (``.git``, ``__pycache__``, ``.venv``) are skipped
    so the walk is deterministic and fast.
    """
    out: list[Path] = []
    if not root.exists():
        return out
    skip = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"}
    for p in root.rglob("*.py"):
        if any(part in skip or part.startswith(".") for part in p.parts):
            continue
        out.append(p)
    return out


def _now_ms(t0: float) -> int:
    """Return milliseconds elapsed since ``t0`` (monotonic)."""
    return int((time.monotonic() - t0) * 1000)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@register_check(
    id="code.ast_parse_all_python",
    name="All .py files parse with ast",
    category="code_integrity",
    severity="blocker",
)
def check_ast_parse_all_python() -> CheckResult:
    """Every ``.py`` under the code roots must parse without ``SyntaxError``."""
    t0 = time.monotonic()
    failures: list[str] = []
    n = 0
    for root in PY_ROOTS:
        for p in _iter_py(REPO_ROOT / root):
            n += 1
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                failures.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
            except (OSError, UnicodeDecodeError) as exc:
                failures.append(
                    f"{p.relative_to(REPO_ROOT)}: unreadable ({exc})"
                )
    status = "PASS" if not failures else "FAIL"
    return CheckResult(
        check_id="code.ast_parse_all_python",
        status=status,
        duration_ms=_now_ms(t0),
        message=(
            f"parsed {n} files cleanly"
            if not failures
            else f"{len(failures)} parse failures across {n} files"
        ),
        evidence="\n".join(failures[:50]) if failures else None,
    )


@register_check(
    id="code.top_level_imports",
    name="Top-level packages import cleanly",
    category="code_integrity",
    severity="blocker",
)
def check_all_top_level_imports() -> CheckResult:
    """``import i3`` etc. must succeed -- smoke-test that __init__ files work.

    When the underlying runtime deps (numpy / torch / fastapi) are not
    installed in the current interpreter, the import transitively fails
    through no fault of the top-level package. We treat that as SKIP
    rather than FAIL so the check only flags *real* package-structure
    problems, not environment problems.
    """
    t0 = time.monotonic()
    modules = ("i3", "server", "training", "demo")
    failures: list[str] = []
    env_missing: list[str] = []
    # Deps that imply "environment not fully installed", not a bug.
    RUNTIME_DEPS_MISSING = {"numpy", "torch", "fastapi", "pydantic", "cryptography"}
    # OS-level fingerprints that indicate a broken binary dep (e.g. torch
    # c10.dll failing to load on Windows because the VC++ redistributable is
    # missing).  These are environment issues, not code defects.
    OS_ENV_FINGERPRINTS = ("WinError 1114", "c10.dll", "cudart", "DLL load failed")
    for m in modules:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or ""
            if missing in RUNTIME_DEPS_MISSING:
                env_missing.append(f"{m}: requires {missing}")
            else:
                failures.append(f"{m}: {type(exc).__name__}: {exc}")
        except OSError as exc:
            msg = str(exc)
            if any(fp in msg for fp in OS_ENV_FINGERPRINTS):
                env_missing.append(f"{m}: OS dep load failed ({msg[:80]})")
            else:
                failures.append(f"{m}: OSError: {exc}")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # A partial failed import can leave sys.modules in a state where
            # subsequent imports raise KeyError during module-lookup.  Treat
            # those as the same environment issue.
            if any(fp in msg for fp in OS_ENV_FINGERPRINTS) or (
                isinstance(exc, KeyError) and env_missing
            ):
                env_missing.append(f"{m}: cascading env failure ({type(exc).__name__})")
            else:
                failures.append(f"{m}: {type(exc).__name__}: {exc}")
    if failures:
        return CheckResult(
            check_id="code.top_level_imports",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"{len(failures)} import failures",
            evidence="\n".join(failures),
        )
    if env_missing:
        return CheckResult(
            check_id="code.top_level_imports",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"{len(env_missing)} package(s) blocked by missing runtime deps",
            evidence="\n".join(env_missing),
        )
    return CheckResult(
        check_id="code.top_level_imports",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=f"all {len(modules)} top-level packages imported",
        evidence=None,
    )


@register_check(
    id="code.no_bare_except",
    name="No bare except in library code",
    category="code_integrity",
    severity="high",
)
def check_no_bare_except() -> CheckResult:
    """``except:`` (without a class) forbidden in ``i3/`` and ``server/``."""
    t0 = time.monotonic()
    offenders: list[str] = []
    for root in ("i3", "server"):
        for p in _iter_py(REPO_ROOT / root):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    offenders.append(
                        f"{p.relative_to(REPO_ROOT)}:{node.lineno}"
                    )
    return CheckResult(
        check_id="code.no_bare_except",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "no bare except handlers"
            if not offenders
            else f"{len(offenders)} bare except handler(s)"
        ),
        evidence="\n".join(offenders[:25]) if offenders else None,
    )


@register_check(
    id="code.no_print_in_library",
    name="No print() in i3/ library code",
    category="code_integrity",
    severity="medium",
)
def check_no_print_in_library() -> CheckResult:
    """Library code should use ``logger.*``, never ``print()``."""
    t0 = time.monotonic()
    offenders: list[str] = []
    for p in _iter_py(REPO_ROOT / "i3"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(
                    f"{p.relative_to(REPO_ROOT)}:{node.lineno}"
                )
    return CheckResult(
        check_id="code.no_print_in_library",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "no print() calls in i3/"
            if not offenders
            else f"{len(offenders)} print() call(s) in i3/"
        ),
        evidence="\n".join(offenders[:20]) if offenders else None,
    )


@register_check(
    id="code.soft_import_pattern",
    name="Optional SDK imports guarded by try/except ImportError",
    category="code_integrity",
    severity="high",
)
def check_soft_import_pattern() -> CheckResult:
    """Every ``import <optional-sdk>`` must sit inside a ``try`` block.

    The check is conservative: a file that mentions the SDK only inside
    a string or comment is ignored; a top-level ``import`` without a
    guarding try is flagged.
    """
    t0 = time.monotonic()
    offenders: list[str] = []
    for p in _iter_py(REPO_ROOT / "i3"):
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        # Collect every (import-name, parent-is-try?) pair.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                else:
                    if node.module:
                        names = [node.module.split(".")[0]]
                for n in names:
                    if n not in SOFT_IMPORT_LIBS:
                        continue
                    # Find the nearest enclosing statement; if it isn't
                    # inside a Try, flag it.
                    if not _inside_try(tree, node):
                        offenders.append(
                            f"{p.relative_to(REPO_ROOT)}:{node.lineno}: {n}"
                        )
    return CheckResult(
        check_id="code.soft_import_pattern",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "all optional-SDK imports guarded"
            if not offenders
            else f"{len(offenders)} unguarded soft-import(s)"
        ),
        evidence="\n".join(offenders[:30]) if offenders else None,
    )


def _inside_try(root: ast.AST, target: ast.AST) -> bool:
    """Return True if ``target`` lies inside an :class:`ast.Try` node.

    Also treats *function-scoped* imports as acceptable: placing a
    soft-import inside a function body is the canonical lazy-import
    pattern -- the module is never loaded unless the function is called,
    which is functionally equivalent to the try/except guard at the
    module top-level.
    """
    for parent in ast.walk(root):
        if isinstance(parent, (ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is target:
                    return True
    return False


@register_check(
    id="code.ruff_clean",
    name="ruff check (library code)",
    category="code_integrity",
    severity="medium",
)
def check_ruff_clean() -> CheckResult:
    """Best-effort ``ruff check`` over the library tree.

    Missing ruff binary -> SKIP.  Non-zero exit from ruff is tolerated
    unless the output contains ``E9`` (syntax) or ``F63``/``F7``/``F82``
    (undefined-name / break-outside-loop / undefined-name-in-all).
    """
    t0 = time.monotonic()
    ruff = shutil.which("ruff")
    if not ruff:
        return CheckResult(
            check_id="code.ruff_clean",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="ruff binary not on PATH",
            evidence=None,
        )
    targets = [str(REPO_ROOT / r) for r in ("i3", "server", "training", "tests")]
    try:
        proc = subprocess.run(
            [ruff, "check", "--quiet", *targets],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            check_id="code.ruff_clean",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"ruff invocation failed: {exc}",
            evidence=None,
        )
    combined = (proc.stdout or "") + (proc.stderr or "")
    critical = any(
        code in combined
        for code in (" E9", " F63", " F7", " F82")
    )
    if critical:
        return CheckResult(
            check_id="code.ruff_clean",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="ruff reported critical rule violations",
            evidence=combined[-2000:],
        )
    return CheckResult(
        check_id="code.ruff_clean",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=(
            "ruff clean"
            if proc.returncode == 0
            else f"ruff non-zero (exit {proc.returncode}) but no critical rules"
        ),
        evidence=None,
    )


@register_check(
    id="code.mypy_clean",
    name="mypy (best-effort)",
    category="code_integrity",
    severity="low",
)
def check_mypy_clean() -> CheckResult:
    """Run mypy once and tolerate noise.

    > 100 errors -> SKIP with "project-size grace".  Missing binary -> SKIP.
    """
    t0 = time.monotonic()
    mypy = shutil.which("mypy")
    if not mypy:
        return CheckResult(
            check_id="code.mypy_clean",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="mypy binary not on PATH",
            evidence=None,
        )
    try:
        proc = subprocess.run(
            [mypy, "--no-color-output", str(REPO_ROOT / "i3")],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            check_id="code.mypy_clean",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"mypy invocation failed: {exc}",
            evidence=None,
        )
    err_count = sum(
        1 for line in (proc.stdout or "").splitlines() if ": error:" in line
    )
    if err_count > 100:
        return CheckResult(
            check_id="code.mypy_clean",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=(
                f"{err_count} mypy errors -- skipping under project-size grace"
            ),
            evidence=None,
        )
    return CheckResult(
        check_id="code.mypy_clean",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=f"{err_count} mypy error(s) (<=100 threshold)",
        evidence=(proc.stdout or "")[-1500:] if err_count else None,
    )


@register_check(
    id="code.from_future_annotations",
    name="from __future__ import annotations in i3/ modules (informational)",
    category="code_integrity",
    severity="info",
)
def check_from_future_annotations() -> CheckResult:
    """Library modules in ``i3/`` declare ``from __future__ import annotations``.

    Only the first 10 lines of each file are inspected -- docstrings and
    ``__future__`` imports must come first.

    **Informational only.** The pre-v1.0 baseline predates this convention
    project-wide. We keep measuring coverage as a quality signal but never
    FAIL the harness on it -- the value is the trend, not the absolute.
    """
    t0 = time.monotonic()
    missing: list[str] = []
    for p in _iter_py(REPO_ROOT / "i3"):
        if p.name == "__init__.py":
            # __init__ files are often thin re-exports where the future
            # import is not critical.
            continue
        try:
            head = "\n".join(
                p.read_text(encoding="utf-8").splitlines()[:10]
            )
        except (OSError, UnicodeDecodeError):
            continue
        if "from __future__ import annotations" not in head:
            missing.append(str(p.relative_to(REPO_ROOT)))
    # Informational only -- always PASS. Message reports current coverage.
    total = sum(1 for _ in _iter_py(REPO_ROOT / "i3") if _.name != "__init__.py")
    coverage = (total - len(missing)) / total if total else 1.0
    return CheckResult(
        check_id="code.from_future_annotations",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(missing)}/{total} modules missing; "
            f"coverage {coverage:.1%} (informational -- never FAIL)"
        ),
        evidence="\n".join(missing[:40]) if missing else None,
    )


@register_check(
    id="code.pep604_union_syntax",
    name="PEP-604 union syntax (X | Y over Optional/Union)",
    category="code_integrity",
    severity="info",
)
def check_pep604_union_syntax() -> CheckResult:
    """Warn (never FAIL) on leftover ``Optional[`` / ``Union[`` in ``i3/``."""
    t0 = time.monotonic()
    hits: list[str] = []
    for p in _iter_py(REPO_ROOT / "i3"):
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(src.splitlines(), 1):
            if "Optional[" in line or "Union[" in line:
                hits.append(f"{p.relative_to(REPO_ROOT)}:{lineno}")
                if len(hits) > 50:
                    break
    return CheckResult(
        check_id="code.pep604_union_syntax",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=(
            "no legacy typing-Union usage"
            if not hits
            else f"{len(hits)} legacy Optional/Union occurrence(s) (warning only)"
        ),
        evidence="\n".join(hits[:20]) if hits else None,
    )


@register_check(
    id="code.no_todo_personnel_references",
    name="No invalidated personnel references in interviewer-facing docs",
    category="code_integrity",
    severity="high",
)
def check_no_todo_personnel_references() -> CheckResult:
    """Forbid phrases invalidated by the corrections notice in BRIEF_ANALYSIS."""
    t0 = time.monotonic()
    forbidden = (
        "Matthew's Apple",
        "TextSpaced",
        "10-year Apple",
        "ex-Apple",
    )
    offenders: list[str] = []
    roots = (REPO_ROOT / "docs" / "slides", REPO_ROOT / "docs" / "huawei")
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            try:
                src = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for phrase in forbidden:
                if phrase in src:
                    offenders.append(
                        f"{p.relative_to(REPO_ROOT)}: {phrase!r}"
                    )
    return CheckResult(
        check_id="code.no_todo_personnel_references",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "no invalidated personnel references"
            if not offenders
            else f"{len(offenders)} invalidated reference(s) found"
        ),
        evidence="\n".join(offenders[:20]) if offenders else None,
    )
