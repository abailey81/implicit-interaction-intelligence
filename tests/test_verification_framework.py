"""Unit tests for the verification-harness framework.

These tests exercise the framework itself (:mod:`scripts.verification.framework`)
and the master CLI (:mod:`scripts.verify_all`) without running the full
check battery.  Each test clears the registry first so tests cannot
contaminate each other.
"""

from __future__ import annotations

import time
from typing import Iterator

import pytest
from pydantic import ValidationError

from scripts.verification.framework import (
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    Check,
    CheckRegistry,
    CheckResult,
    VerificationReport,
    register_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Ensure each test starts from an empty registry."""
    CheckRegistry.clear()
    yield
    CheckRegistry.clear()


# ---------------------------------------------------------------------------
# Check / CheckResult validation
# ---------------------------------------------------------------------------


def test_check_model_accepts_valid_fields() -> None:
    """A :class:`Check` with all expected fields validates."""
    c = Check(
        id="x.y",
        name="test",
        category="code_integrity",
        severity="blocker",
    )
    assert c.id == "x.y"
    assert c.category in VALID_CATEGORIES
    assert c.severity in VALID_SEVERITIES


def test_check_rejects_invalid_category() -> None:
    """Unknown category strings must raise :class:`ValidationError`."""
    with pytest.raises(ValidationError):
        Check(
            id="x",
            name="n",
            category="not_a_category",  # type: ignore[arg-type]
            severity="blocker",
        )


def test_check_rejects_invalid_severity() -> None:
    """Unknown severity strings must raise :class:`ValidationError`."""
    with pytest.raises(ValidationError):
        Check(
            id="x",
            name="n",
            category="code_integrity",
            severity="urgent",  # type: ignore[arg-type]
        )


def test_check_result_rejects_negative_duration() -> None:
    """``duration_ms`` must be >= 0."""
    with pytest.raises(ValidationError):
        CheckResult(
            check_id="x",
            status="PASS",
            duration_ms=-1,
            message="bad",
            evidence=None,
        )


def test_check_result_collapses_multiline_message() -> None:
    """Multi-line messages are collapsed so table rendering stays honest."""
    r = CheckResult(
        check_id="x",
        status="PASS",
        duration_ms=1,
        message="line1\nline2\r\nline3",
        evidence=None,
    )
    assert "\n" not in r.message
    assert "\r" not in r.message


def test_check_result_rejects_unknown_status() -> None:
    """Only PASS/FAIL/SKIP are permitted."""
    with pytest.raises(ValidationError):
        CheckResult(
            check_id="x",
            status="WARN",  # type: ignore[arg-type]
            duration_ms=0,
            message="",
            evidence=None,
        )


# ---------------------------------------------------------------------------
# register_check
# ---------------------------------------------------------------------------


def test_register_check_decorator_registers_callable() -> None:
    """A decorated callable appears in ``CheckRegistry.all_checks()``."""

    @register_check(
        id="unit.example",
        name="example",
        category="code_integrity",
        severity="info",
    )
    def _ex() -> CheckResult:
        return CheckResult(
            check_id="unit.example",
            status="PASS",
            duration_ms=0,
            message="ok",
            evidence=None,
        )

    ids = {c.id for c in CheckRegistry.all_checks()}
    assert "unit.example" in ids


def test_register_check_rejects_duplicate_ids() -> None:
    """Two checks with the same id must raise."""

    @register_check(
        id="unit.dup",
        name="one",
        category="code_integrity",
        severity="info",
    )
    def _a() -> CheckResult:
        return CheckResult(
            check_id="unit.dup", status="PASS", duration_ms=0, message=""
        )

    with pytest.raises(ValueError):

        @register_check(
            id="unit.dup",
            name="two",
            category="code_integrity",
            severity="info",
        )
        def _b() -> CheckResult:
            return CheckResult(
                check_id="unit.dup", status="PASS", duration_ms=0, message=""
            )


# ---------------------------------------------------------------------------
# run_all: timeout + exception trap
# ---------------------------------------------------------------------------


def test_run_all_times_out_slow_check() -> None:
    """A check exceeding the per-check timeout becomes a FAIL."""

    @register_check(
        id="unit.slow",
        name="slow",
        category="code_integrity",
        severity="info",
    )
    def _slow() -> CheckResult:
        time.sleep(3)
        return CheckResult(
            check_id="unit.slow",
            status="PASS",
            duration_ms=3000,
            message="should not happen",
        )

    report = CheckRegistry.run_all(timeout_s=1, parallelism=1)
    assert report.failed == 1
    assert report.results[0].status == "FAIL"
    assert "timeout" in report.results[0].message.lower()


def test_run_all_catches_exceptions() -> None:
    """A raising check produces FAIL with traceback evidence, not a crash."""

    @register_check(
        id="unit.boom",
        name="boom",
        category="code_integrity",
        severity="info",
    )
    def _boom() -> CheckResult:
        raise RuntimeError("explode")

    report = CheckRegistry.run_all(timeout_s=5, parallelism=1)
    assert report.failed == 1
    assert report.results[0].status == "FAIL"
    assert "RuntimeError" in (report.results[0].evidence or "")


def test_run_all_pass_rate_excludes_skips() -> None:
    """``pass_rate`` uses only PASS + FAIL as denominator."""

    @register_check(
        id="unit.ok",
        name="ok",
        category="code_integrity",
        severity="info",
    )
    def _ok() -> CheckResult:
        return CheckResult(
            check_id="unit.ok", status="PASS", duration_ms=0, message=""
        )

    @register_check(
        id="unit.skipped",
        name="skipped",
        category="code_integrity",
        severity="info",
    )
    def _skipped() -> CheckResult:
        return CheckResult(
            check_id="unit.skipped",
            status="SKIP",
            duration_ms=0,
            message="no sdk",
        )

    report = CheckRegistry.run_all(timeout_s=5, parallelism=1)
    assert report.passed == 1
    assert report.skipped == 1
    assert report.pass_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# per_category_summary
# ---------------------------------------------------------------------------


def test_per_category_summary_totals_match_full_counts() -> None:
    """Each category bucket's totals sum to the top-level counters."""

    @register_check(
        id="unit.a",
        name="a",
        category="code_integrity",
        severity="info",
    )
    def _a() -> CheckResult:
        return CheckResult(
            check_id="unit.a", status="PASS", duration_ms=0, message=""
        )

    @register_check(
        id="unit.b",
        name="b",
        category="config_data",
        severity="info",
    )
    def _b() -> CheckResult:
        return CheckResult(
            check_id="unit.b", status="FAIL", duration_ms=0, message="nope"
        )

    @register_check(
        id="unit.c",
        name="c",
        category="interview_readiness",
        severity="info",
    )
    def _c() -> CheckResult:
        return CheckResult(
            check_id="unit.c",
            status="SKIP",
            duration_ms=0,
            message="skip",
        )

    report = CheckRegistry.run_all(timeout_s=5, parallelism=1)
    summed_pass = sum(
        c["passed"] for c in report.per_category_summary.values()
    )
    summed_fail = sum(
        c["failed"] for c in report.per_category_summary.values()
    )
    summed_skip = sum(
        c["skipped"] for c in report.per_category_summary.values()
    )
    summed_total = sum(
        c["total"] for c in report.per_category_summary.values()
    )
    assert summed_pass == report.passed
    assert summed_fail == report.failed
    assert summed_skip == report.skipped
    assert summed_total == report.total


# ---------------------------------------------------------------------------
# --strict exit-code logic
# ---------------------------------------------------------------------------


def test_strict_exit_code_flips_on_any_failure() -> None:
    """`--strict` turns a single low-severity FAIL into a non-zero exit."""
    from scripts.verify_all import _compute_exit_code

    @register_check(
        id="unit.low_fail",
        name="low_fail",
        category="code_integrity",
        severity="low",
    )
    def _lf() -> CheckResult:
        return CheckResult(
            check_id="unit.low_fail",
            status="FAIL",
            duration_ms=0,
            message="fail",
        )

    report = CheckRegistry.run_all(timeout_s=5, parallelism=1)
    checks_by_id = {
        c.id: (c.name, c.category, c.severity)
        for c in CheckRegistry.all_checks()
    }
    # Without strict and with fail_on={blocker,high}, low-severity failure
    # does NOT flip the exit code.
    tolerant = _compute_exit_code(
        report,
        fail_on={"blocker", "high"},
        strict=False,
        checks_by_id=checks_by_id,
    )
    assert tolerant == 0

    # With strict=True, any FAIL -> 1.
    strict = _compute_exit_code(
        report,
        fail_on={"blocker", "high"},
        strict=True,
        checks_by_id=checks_by_id,
    )
    assert strict == 1


# ---------------------------------------------------------------------------
# VerificationReport schema sanity
# ---------------------------------------------------------------------------


def test_verification_report_rejects_pass_rate_above_one() -> None:
    """The schema enforces ``0 <= pass_rate <= 1``."""
    from datetime import datetime, timezone

    with pytest.raises(ValidationError):
        VerificationReport(
            total=1,
            passed=1,
            failed=0,
            skipped=0,
            duration_s=0.1,
            pass_rate=1.5,  # out of range
            per_category_summary={},
            results=[],
            run_at=datetime.now(timezone.utc),
            git_sha="abc",
            python_version="3.11",
            platform="linux",
        )
