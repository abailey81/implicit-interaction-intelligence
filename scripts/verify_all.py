"""I3 verification-harness master CLI.

Usage::

    python scripts/verify_all.py --strict
    python scripts/verify_all.py --categories code,runtime,interview
    python scripts/verify_all.py --out reports/out.json --out-md reports/out.md

The script discovers every check registered by the ``checks_*.py``
modules of :mod:`scripts.verification` at import time, runs them via
:meth:`CheckRegistry.run_all`, emits a JSON blob plus a human-readable
Markdown report, and returns an exit code of 0 (all clean) / 1 (failures
above the ``--fail-on`` threshold) / 2 (harness crash).

This script is the tool G10 will use to iteratively verify the repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the package importable when invoked as a bare script.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verification.framework import (  # noqa: E402
    VALID_CATEGORIES,
    CheckRegistry,
    VerificationReport,
)

# Category aliases for the --categories CLI flag.
_CATEGORY_ALIAS: dict[str, str] = {
    "code": "code_integrity",
    "config": "config_data",
    "data": "config_data",
    "runtime": "architecture_runtime",
    "arch": "architecture_runtime",
    "providers": "providers",
    "infra": "infrastructure",
    "infrastructure": "infrastructure",
    "interview": "interview_readiness",
    "security": "security",
}

# Severity ordering (worst-first) used for the --fail-on filter.
_SEVERITY_ORDER = ("blocker", "high", "medium", "low", "info")


def _import_all_check_modules() -> None:
    """Force-import every ``checks_*.py`` so their decorators register.

    This side-effect style of registration keeps per-check files
    self-contained.
    """
    pkg = importlib.import_module("scripts.verification")
    for mod_name in (
        "scripts.verification.checks_code",
        "scripts.verification.checks_config",
        "scripts.verification.checks_runtime",
        "scripts.verification.checks_providers",
        "scripts.verification.checks_infrastructure",
        "scripts.verification.checks_interview",
    ):
        importlib.import_module(mod_name)
    _ = pkg  # keep the reference to silence linters


def _resolve_categories(raw: str | None) -> set[str] | None:
    """Translate ``--categories code,runtime`` to the internal set."""
    if not raw:
        return None
    resolved: set[str] = set()
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key in VALID_CATEGORIES:
            resolved.add(key)
        elif key in _CATEGORY_ALIAS:
            resolved.add(_CATEGORY_ALIAS[key])
        else:
            raise SystemExit(
                f"Unknown category {key!r}; valid: "
                f"{sorted(set(VALID_CATEGORIES) | set(_CATEGORY_ALIAS))}"
            )
    return resolved


def _ts_suffix() -> str:
    """UTC timestamp suitable for file names."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _status_emoji(status: str) -> str:
    """Return a harmless glyph for table rendering (ASCII-compatible)."""
    return {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, status)


def render_markdown(report: VerificationReport, checks_by_id: dict[str, tuple[str, str, str]]) -> str:
    """Render a Markdown verification report.

    Args:
        report: The completed :class:`VerificationReport`.
        checks_by_id: Mapping from check id to ``(name, category, severity)``.

    Returns:
        A string suitable for writing to a ``.md`` file or echoing to
        ``$GITHUB_STEP_SUMMARY``.
    """
    lines: list[str] = []
    lines.append("# I3 Verification Report")
    lines.append("")
    lines.append(f"- Run at: `{report.run_at.isoformat()}`")
    lines.append(f"- Git SHA: `{report.git_sha}`")
    lines.append(f"- Python: `{report.python_version}`")
    lines.append(f"- Platform: `{report.platform}`")
    lines.append(f"- Duration: `{report.duration_s:.2f}s`")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"- Total: **{report.total}** | "
        f"PASS: **{report.passed}** | "
        f"FAIL: **{report.failed}** | "
        f"SKIP: **{report.skipped}** | "
        f"Pass-rate (excluding skips): **{report.pass_rate:.1%}**"
    )
    lines.append("")

    # Per-category
    lines.append("### Per-category summary")
    lines.append("")
    lines.append("| Category | Total | PASS | FAIL | SKIP |")
    lines.append("|---|---|---|---|---|")
    for cat, counts in sorted(report.per_category_summary.items()):
        lines.append(
            f"| `{cat}` | {counts['total']} | {counts['passed']} | "
            f"{counts['failed']} | {counts['skipped']} |"
        )
    lines.append("")

    # Full table
    lines.append("## Full Results")
    lines.append("")
    lines.append("| Status | Severity | Check | Duration | Message |")
    lines.append("|---|---|---|---|---|")
    for r in report.results:
        name, _cat, sev = checks_by_id.get(r.check_id, (r.check_id, "?", "?"))
        msg = r.message.replace("|", "\\|")
        lines.append(
            f"| {_status_emoji(r.status)} | `{sev}` | `{r.check_id}` "
            f"({name}) | {r.duration_ms} ms | {msg} |"
        )
    lines.append("")

    # Failures with evidence
    failures = [r for r in report.results if r.status == "FAIL"]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            name, _cat, sev = checks_by_id.get(
                r.check_id, (r.check_id, "?", "?")
            )
            lines.append(f"### `{r.check_id}` - {name} (`{sev}`)")
            lines.append("")
            lines.append(f"- Message: {r.message}")
            if r.evidence:
                lines.append("")
                lines.append("```")
                lines.append(r.evidence[:4000])
                lines.append("```")
            lines.append("")

    # Skipped
    skipped = [r for r in report.results if r.status == "SKIP"]
    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for r in skipped:
            name, _cat, _sev = checks_by_id.get(
                r.check_id, (r.check_id, "?", "?")
            )
            lines.append(f"- `{r.check_id}` ({name}): {r.message}")
        lines.append("")

    # Environment block
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Git SHA: `{report.git_sha}`")
    lines.append(f"- Python: `{report.python_version}`")
    lines.append(f"- Platform: `{report.platform}`")
    lines.append(f"- Run at: `{report.run_at.isoformat()}`")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Exit-code logic
# ---------------------------------------------------------------------------


def _compute_exit_code(
    report: VerificationReport,
    *,
    fail_on: set[str],
    strict: bool,
    checks_by_id: dict[str, tuple[str, str, str]],
) -> int:
    """Return 0/1 based on ``--fail-on`` / ``--strict`` semantics.

    Args:
        report: Completed report.
        fail_on: Severity strings (e.g. {"blocker", "high"}) that cause
            a non-zero exit when at least one matching check FAILed.
        strict: If ``True``, any FAIL regardless of severity -> non-zero.
        checks_by_id: Registration metadata lookup.
    """
    failed = [r for r in report.results if r.status == "FAIL"]
    if not failed:
        return 0
    if strict:
        return 1
    # Respect fail_on
    for r in failed:
        _name, _cat, sev = checks_by_id.get(r.check_id, ("", "", "info"))
        if sev in fail_on:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Build the argparse namespace."""
    parser = argparse.ArgumentParser(
        prog="verify_all",
        description="Run the I3 verification harness.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated subset (code,config,runtime,providers,infra,"
            "interview,security). Default: all."
        ),
    )
    ts = _ts_suffix()
    parser.add_argument(
        "--out",
        default=f"reports/verification_{ts}.json",
        help="Path to the JSON report (default: reports/verification_<ts>.json).",
    )
    parser.add_argument(
        "--out-md",
        default=f"reports/verification_{ts}.md",
        help="Path to the Markdown report.",
    )
    parser.add_argument(
        "--timeout", type=int, default=60, help="Per-check timeout (s)."
    )
    parser.add_argument(
        "--parallelism", type=int, default=4, help="Thread-pool workers."
    )
    parser.add_argument(
        "--fail-on",
        default="blocker,high",
        help="Severities that flip the exit code on FAIL (default: blocker,high).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Any FAIL -> non-zero exit (overrides --fail-on).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        ``0`` on success, ``1`` when failures exceed ``--fail-on``, ``2``
        when the harness itself crashed.
    """
    try:
        args = _parse_args(argv)
        _import_all_check_modules()

        categories = _resolve_categories(args.categories)
        fail_on = {s.strip() for s in args.fail_on.split(",") if s.strip()}

        checks_by_id: dict[str, tuple[str, str, str]] = {
            c.id: (c.name, c.category, c.severity)
            for c in CheckRegistry.all_checks()
        }

        report = CheckRegistry.run_all(
            timeout_s=args.timeout,
            parallelism=args.parallelism,
            categories=categories,
        )

        # JSON
        out_json = Path(args.out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        # Markdown
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md = render_markdown(report, checks_by_id)
        out_md.write_text(md, encoding="utf-8")

        # Human console summary
        sys.stdout.write(
            f"[verify_all] total={report.total} "
            f"pass={report.passed} fail={report.failed} "
            f"skip={report.skipped} duration={report.duration_s:.2f}s\n"
            f"[verify_all] JSON  -> {out_json}\n"
            f"[verify_all] MD    -> {out_md}\n"
        )

        return _compute_exit_code(
            report,
            fail_on=fail_on,
            strict=args.strict,
            checks_by_id=checks_by_id,
        )
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - top-level harness guard
        sys.stderr.write("[verify_all] HARNESS CRASH:\n")
        sys.stderr.write(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
