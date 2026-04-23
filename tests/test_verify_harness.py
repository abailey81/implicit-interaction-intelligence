"""Tests for the :mod:`scripts.verification` harness framework.

The harness is the project's third verification layer (after pytest
and the red-team suite).  Its own correctness is load-bearing — a
silently-broken check would hide real regressions.  These tests cover:

* Registry behaviour (registration, duplicate detection).
* ``CheckResult`` validation (message collapsing, status enum).
* ``_run_one`` exception safety and timeout.
* The environment-issue detectors (OSError DLL, cascading KeyError,
  torch AttributeError) — these were added during the 2026-04-23
  audit to prevent false FAILs on Windows boxes with broken torch.
* Report assembly (per-category counters, pass-rate math).
"""

from __future__ import annotations

import time

import pytest

from scripts.verification.framework import (
    Check,
    CheckRegistry,
    CheckResult,
    VerificationReport,
    _run_one,
    register_check,
)


# ---------------------------------------------------------------------------
# CheckResult validation
# ---------------------------------------------------------------------------


def test_check_result_valid():
    r = CheckResult(
        check_id="x.ok", status="PASS", duration_ms=5, message="ok"
    )
    assert r.status == "PASS"
    assert r.duration_ms == 5


def test_check_result_collapses_multiline_message():
    """Multi-line messages are flattened so the Markdown table renders."""
    r = CheckResult(
        check_id="x.long",
        status="FAIL",
        duration_ms=1,
        message="line one\nline two\rline three",
    )
    assert "\n" not in r.message and "\r" not in r.message


def test_check_result_clamps_message_length():
    r = CheckResult(
        check_id="x.big",
        status="PASS",
        duration_ms=0,
        message="x" * 10_000,
    )
    assert len(r.message) <= 500


def test_check_result_rejects_negative_duration():
    with pytest.raises(Exception):
        CheckResult(check_id="x", status="PASS", duration_ms=-1, message="")


def test_check_result_rejects_bad_status():
    with pytest.raises(Exception):
        CheckResult(check_id="x", status="MAYBE", duration_ms=0, message="")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_and_list(monkeypatch):
    # Snapshot current registry size so registering our probe is the
    # only delta, even when the full suite has already loaded.
    before = len(CheckRegistry.all_checks())

    @register_check(
        id="probe.always_pass", name="probe", category="code_integrity",
        severity="info",
    )
    def _probe() -> CheckResult:
        return CheckResult(
            check_id="probe.always_pass", status="PASS", duration_ms=0,
            message="ok",
        )

    after = len(CheckRegistry.all_checks())
    assert after == before + 1
    ids = {c.id for c in CheckRegistry.all_checks()}
    assert "probe.always_pass" in ids


def test_registry_rejects_duplicate_id():

    @register_check(
        id="probe.dup", name="dup", category="code_integrity",
        severity="info",
    )
    def _first() -> CheckResult:
        return CheckResult(
            check_id="probe.dup", status="PASS", duration_ms=0, message=""
        )

    with pytest.raises(ValueError):

        @register_check(
            id="probe.dup", name="dup2", category="code_integrity",
            severity="info",
        )
        def _second() -> CheckResult:  # pragma: no cover - not reached
            return CheckResult(
                check_id="probe.dup", status="PASS", duration_ms=0, message=""
            )


# ---------------------------------------------------------------------------
# _run_one — exception safety + timeout
# ---------------------------------------------------------------------------


def _check(id_: str = "probe") -> Check:
    return Check(
        id=id_, name=id_, category="code_integrity", severity="info"
    )


def test_run_one_converts_exception_to_fail():
    def boom() -> CheckResult:
        raise ValueError("boom")

    result = _run_one(_check(), boom, timeout_s=5)
    assert result.status == "FAIL"
    assert "ValueError" in result.message
    assert "boom" in result.message


def test_run_one_times_out():
    def slow() -> CheckResult:
        time.sleep(5.0)
        return CheckResult(
            check_id="probe", status="PASS", duration_ms=5000, message=""
        )

    result = _run_one(_check(), slow, timeout_s=1)
    assert result.status == "FAIL"
    assert "timeout" in result.message.lower()


def test_run_one_rejects_non_check_result_return():
    def bad() -> CheckResult:
        return "not a CheckResult"  # type: ignore[return-value]

    result = _run_one(_check(), bad, timeout_s=5)
    assert result.status == "FAIL"


def test_run_one_passes_through_successful_result():
    def good() -> CheckResult:
        return CheckResult(
            check_id="probe", status="PASS", duration_ms=7, message="fine"
        )

    result = _run_one(_check(), good, timeout_s=5)
    assert result.status == "PASS"
    assert result.duration_ms == 7


# ---------------------------------------------------------------------------
# Environment-issue detectors (added during the 2026-04-23 audit)
# ---------------------------------------------------------------------------


def test_is_os_env_issue_recognises_dll_load_failure():
    from scripts.verification.checks_runtime import _is_os_env_issue

    err = OSError(
        "[WinError 1114] A dynamic link library (DLL) initialization "
        "routine failed. Error loading "
        r"'D:\venv\Lib\site-packages\torch\lib\c10.dll'"
    )
    assert _is_os_env_issue(err) is True


def test_is_os_env_issue_ignores_unrelated_oserror():
    from scripts.verification.checks_runtime import _is_os_env_issue

    assert _is_os_env_issue(OSError("disk full")) is False


def test_is_os_env_issue_recognises_torch_attribute_error():
    """Partial torch import leaves a stub with missing symbols."""
    from scripts.verification.checks_runtime import _is_os_env_issue

    err = AttributeError("module 'torch' has no attribute 'manual_seed'")
    assert _is_os_env_issue(err) is True


def test_env_missing_result_skips_missing_runtime_dep():
    from scripts.verification.checks_providers import _env_missing_result

    # ModuleNotFoundError with .name in _RUNTIME_DEPS → SKIP
    err = ModuleNotFoundError("No module named 'torch'")
    err.name = "torch"
    result = _env_missing_result("probe", t0=time.monotonic(), exc=err)
    assert result is not None
    assert result.status == "SKIP"


def test_env_missing_result_none_on_real_failure():
    from scripts.verification.checks_providers import _env_missing_result

    err = ModuleNotFoundError("No module named 'some_random_package'")
    err.name = "some_random_package"
    # Not in _RUNTIME_DEPS → the caller should FAIL, not SKIP.
    assert _env_missing_result("probe", t0=time.monotonic(), exc=err) is None


def test_env_missing_result_skips_windows_dll_oserror():
    from scripts.verification.checks_providers import _env_missing_result

    err = OSError(
        "[WinError 1114] ... Error loading "
        r"'D:\venv\Lib\site-packages\torch\lib\c10.dll'"
    )
    result = _env_missing_result("probe", t0=time.monotonic(), exc=err)
    assert result is not None and result.status == "SKIP"


def test_env_missing_result_skips_cascading_keyerror():
    """A prior failed binary import leaves sys.modules partially loaded."""
    from scripts.verification.checks_providers import _env_missing_result

    err = KeyError("i3")
    result = _env_missing_result("probe", t0=time.monotonic(), exc=err)
    assert result is not None and result.status == "SKIP"


# ---------------------------------------------------------------------------
# VerificationReport — Pydantic validation
# ---------------------------------------------------------------------------


def test_verification_report_basic_shape():
    from datetime import datetime, timezone

    r = VerificationReport(
        total=2, passed=1, failed=1, skipped=0,
        duration_s=0.1, pass_rate=0.5,
        per_category_summary={},
        results=[
            CheckResult(check_id="a", status="PASS", duration_ms=1, message=""),
            CheckResult(check_id="b", status="FAIL", duration_ms=1, message=""),
        ],
        run_at=datetime.now(timezone.utc),
    )
    assert r.total == 2


def test_verification_report_rejects_out_of_range_pass_rate():
    from datetime import datetime, timezone

    with pytest.raises(Exception):
        VerificationReport(
            total=1, passed=1, failed=0, skipped=0,
            duration_s=0.0, pass_rate=1.5,
            per_category_summary={},
            results=[],
            run_at=datetime.now(timezone.utc),
        )
