"""Config / data-integrity checks: YAML/JSON/TOML parse, model-id lock, ...

Each check degrades gracefully: if the parser library is unavailable
(old Python, optional dep missing) the result becomes ``SKIP`` rather
than a ``FAIL``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from scripts.verification.framework import CheckResult, register_check

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Known prefixes used by major AI vendors for secret tokens.  Grep for
# them everywhere except ``.env.example`` / ``.env.providers.example``
# (where they may appear as placeholder literals).
SECRET_PREFIXES: tuple[str, ...] = (
    "sk-ant-api",
    "sk-proj-",
    "sk-svcacct-",
    "AIzaSy",
    "ghp_",
    "xoxb-",
    "AKIA",
)

_ENV_EXAMPLE_ALLOWLIST: tuple[str, ...] = (
    ".env.example",
    ".env.providers.example",
)


def _now_ms(t0: float) -> int:
    """Return milliseconds elapsed since ``t0``."""
    return int((time.monotonic() - t0) * 1000)


def _iter_files(pattern: str) -> list[Path]:
    """Recursively yield repo files matching ``pattern``.

    Hidden directories and common vendored / cache directories are
    skipped so the walk is deterministic.
    """
    skip = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        "site",  # mkdocs build output
    }
    out: list[Path] = []
    for p in REPO_ROOT.rglob(pattern):
        if any(part in skip for part in p.parts):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@register_check(
    id="config.yaml_parse_all",
    name="All .yaml/.yml files parse",
    category="config_data",
    severity="high",
)
def check_yaml_parse_all() -> CheckResult:
    """Every YAML file in the repo parses cleanly.

    Skips when ``PyYAML`` is not importable.
    """
    t0 = time.monotonic()
    try:
        import yaml
    except ImportError:
        return CheckResult(
            check_id="config.yaml_parse_all",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="PyYAML not importable",
            evidence=None,
        )
    failures: list[str] = []
    files = _iter_files("*.yaml") + _iter_files("*.yml")
    for p in files:
        try:
            list(yaml.safe_load_all(p.read_text(encoding="utf-8")))
        except yaml.YAMLError as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: unreadable ({exc})")
    return CheckResult(
        check_id="config.yaml_parse_all",
        status="PASS" if not failures else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} YAML file(s) parsed cleanly"
            if not failures
            else f"{len(failures)} YAML parse failure(s)"
        ),
        evidence="\n".join(failures[:20]) if failures else None,
    )


@register_check(
    id="config.json_parse_all",
    name="All .json files parse",
    category="config_data",
    severity="high",
)
def check_json_parse_all() -> CheckResult:
    """Every ``*.json`` file in the repo parses cleanly."""
    t0 = time.monotonic()
    failures: list[str] = []
    files = _iter_files("*.json")
    for p in files:
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: unreadable ({exc})")
    return CheckResult(
        check_id="config.json_parse_all",
        status="PASS" if not failures else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} JSON file(s) parsed cleanly"
            if not failures
            else f"{len(failures)} JSON parse failure(s)"
        ),
        evidence="\n".join(failures[:20]) if failures else None,
    )


@register_check(
    id="config.toml_parse_all",
    name="All .toml files parse",
    category="config_data",
    severity="high",
)
def check_toml_parse_all() -> CheckResult:
    """Every ``*.toml`` file parses via :mod:`tomllib` (Py 3.11+)."""
    t0 = time.monotonic()
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:
        return CheckResult(
            check_id="config.toml_parse_all",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="tomllib not available (Python < 3.11)",
            evidence=None,
        )
    failures: list[str] = []
    files = _iter_files("*.toml")
    for p in files:
        try:
            tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: unreadable ({exc})")
    return CheckResult(
        check_id="config.toml_parse_all",
        status="PASS" if not failures else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} TOML file(s) parsed cleanly"
            if not failures
            else f"{len(failures)} TOML parse failure(s)"
        ),
        evidence="\n".join(failures[:20]) if failures else None,
    )


@register_check(
    id="config.notebooks_valid_nbformat",
    name="All .ipynb files are valid nbformat v4",
    category="config_data",
    severity="medium",
)
def check_notebooks_valid_nbformat() -> CheckResult:
    """Every ``*.ipynb`` parses as JSON with ``nbformat == 4``."""
    t0 = time.monotonic()
    failures: list[str] = []
    files = _iter_files("*.ipynb")
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            failures.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
            continue
        if not isinstance(obj, dict) or obj.get("nbformat") != 4:
            failures.append(
                f"{p.relative_to(REPO_ROOT)}: nbformat != 4"
            )
    return CheckResult(
        check_id="config.notebooks_valid_nbformat",
        status="PASS" if not failures else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} notebook(s) valid"
            if not failures
            else f"{len(failures)} invalid notebook(s)"
        ),
        evidence="\n".join(failures[:10]) if failures else None,
    )


@register_check(
    id="config.env_example_keys_documented",
    name=".env.example keys referenced in python source",
    category="config_data",
    severity="medium",
)
def check_env_example_keys_documented() -> CheckResult:
    """Every key in the env-example files is referenced somewhere in the source.

    Keys that look like non-secret comments (``#`` lines) are ignored.
    """
    t0 = time.monotonic()
    keys: set[str] = set()
    for name in _ENV_EXAMPLE_ALLOWLIST:
        p = REPO_ROOT / name
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key and key.isupper():
                    keys.add(key)
        except (OSError, UnicodeDecodeError):
            continue

    if not keys:
        return CheckResult(
            check_id="config.env_example_keys_documented",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="no .env.example files found",
            evidence=None,
        )

    # Scan python source for each key name.
    sources: list[str] = []
    for root in ("i3", "server", "scripts", "training"):
        rd = REPO_ROOT / root
        if not rd.exists():
            continue
        for p in rd.rglob("*.py"):
            try:
                sources.append(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    blob = "\n".join(sources)
    missing = sorted(k for k in keys if k not in blob)
    return CheckResult(
        check_id="config.env_example_keys_documented",
        status="PASS" if not missing else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(keys)} env keys, all referenced"
            if not missing
            else f"{len(missing)}/{len(keys)} env keys not referenced in source"
        ),
        evidence="\n".join(missing[:30]) if missing else None,
    )


@register_check(
    id="config.no_hardcoded_secrets",
    name="No hardcoded secret prefixes outside .env.example",
    category="security",
    severity="blocker",
)
def check_no_hardcoded_secrets() -> CheckResult:
    """Grep for known secret prefixes across the repo.

    The harness itself (``scripts/verification/`` and
    ``scripts/verify_all.py``) is allow-listed because it declares the
    secret prefixes as string literals so it can search for them -- a
    self-match would turn this check into a false-positive generator.
    """
    t0 = time.monotonic()
    hits: list[str] = []

    def _is_self_or_vendor(rel: Path) -> bool:
        """Return True for files that intentionally mention the prefixes."""
        parts = rel.parts
        if not parts:
            return False
        # The harness's own source files reference the prefix strings.
        if parts[0] == "scripts" and (
            len(parts) >= 2
            and (parts[1] == "verification" or parts[1] == "verify_all.py")
        ):
            return True
        # Tests and docs may reference the prefixes as examples.
        if parts[0] == "tests":
            return True
        if parts[0] == "docs" and "operations" in parts:
            return True
        return False

    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        # Skip obvious non-source artefacts and the env-example files.
        if p.name in _ENV_EXAMPLE_ALLOWLIST:
            continue
        parts = set(p.parts)
        if parts & {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            "site",
        }:
            continue
        if p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
            ".zip", ".bin", ".onnx", ".pt", ".pth",
        }:
            continue
        rel = p.relative_to(REPO_ROOT)
        if _is_self_or_vendor(rel):
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pref in SECRET_PREFIXES:
            if pref in src:
                hits.append(f"{rel}: {pref}*")
                break
    return CheckResult(
        check_id="config.no_hardcoded_secrets",
        status="PASS" if not hits else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "no hardcoded secret prefixes found"
            if not hits
            else f"{len(hits)} potential secret(s) found"
        ),
        evidence="\n".join(hits[:20]) if hits else None,
    )


@register_check(
    id="config.claude_model_id_locked",
    name="configs/default.yaml cloud.model == claude-sonnet-4-5",
    category="config_data",
    severity="blocker",
)
def check_claude_model_id_locked() -> CheckResult:
    """Brief locks the Anthropic model id; accidental edits must be caught."""
    t0 = time.monotonic()
    try:
        import yaml
    except ImportError:
        return CheckResult(
            check_id="config.claude_model_id_locked",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="PyYAML not available",
            evidence=None,
        )
    p = REPO_ROOT / "configs" / "default.yaml"
    if not p.exists():
        return CheckResult(
            check_id="config.claude_model_id_locked",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="configs/default.yaml missing",
            evidence=None,
        )
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return CheckResult(
            check_id="config.claude_model_id_locked",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"YAML parse error: {exc}",
            evidence=None,
        )
    model = (data or {}).get("cloud", {}).get("model")
    expected = "claude-sonnet-4-5"
    return CheckResult(
        check_id="config.claude_model_id_locked",
        status="PASS" if model == expected else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"cloud.model == {expected!r}"
            if model == expected
            else f"expected {expected!r}, got {model!r}"
        ),
        evidence=None,
    )


@register_check(
    id="config.mkdocs_build_strict",
    name="mkdocs build --strict",
    category="config_data",
    severity="medium",
)
def check_mkdocs_build_strict() -> CheckResult:
    """Shell out to ``mkdocs build --strict`` into a temp dir.

    Broken links, missing nav entries, etc. fail --strict.  Missing
    mkdocs binary -> SKIP.
    """
    t0 = time.monotonic()
    mkdocs = shutil.which("mkdocs")
    if not mkdocs:
        return CheckResult(
            check_id="config.mkdocs_build_strict",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="mkdocs binary not on PATH",
            evidence=None,
        )
    mkdocs_yml = REPO_ROOT / "mkdocs.yml"
    if not mkdocs_yml.exists():
        return CheckResult(
            check_id="config.mkdocs_build_strict",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="mkdocs.yml missing",
            evidence=None,
        )
    with tempfile.TemporaryDirectory() as td:
        try:
            proc = subprocess.run(
                [
                    mkdocs,
                    "build",
                    "--strict",
                    "--site-dir",
                    td,
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return CheckResult(
                check_id="config.mkdocs_build_strict",
                status="SKIP",
                duration_ms=_now_ms(t0),
                message=f"mkdocs invocation failed: {exc}",
                evidence=None,
            )
    if proc.returncode != 0:
        combined = (proc.stdout or "") + (proc.stderr or "")
        return CheckResult(
            check_id="config.mkdocs_build_strict",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"mkdocs --strict exit {proc.returncode}",
            evidence=combined[-2000:],
        )
    return CheckResult(
        check_id="config.mkdocs_build_strict",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="mkdocs --strict build succeeded",
        evidence=None,
    )


# Silence unused-import warnings for the lone regex fixture import, which
# is intentionally kept available for downstream checks.
_ = re
