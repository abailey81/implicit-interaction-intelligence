"""Verification-harness framework: :class:`Check`, :class:`CheckResult`,
:class:`CheckRegistry`, and :class:`VerificationReport`.

The harness is a thin, Pydantic-typed layer over a mutable registry of
callables.  Each callable returns a :class:`CheckResult`.  Checks are
registered with a decorator and executed in a bounded thread-pool with
a per-check timeout and exception trap -- a broken or slow check
produces a FAIL rather than crashing the whole harness.

Categories are fixed (see :data:`VALID_CATEGORIES`) so reports can be
grouped deterministically.  Severities are also fixed and drive the
``--fail-on`` filter in :mod:`scripts.verify_all`.

Example
-------
>>> from scripts.verification.framework import CheckRegistry, CheckResult, register_check
>>> @register_check(
...     id="demo.ok",
...     name="Demo always-pass",
...     category="code_integrity",
...     severity="info",
... )
... def _demo() -> CheckResult:
...     return CheckResult(
...         check_id="demo.ok",
...         status="PASS",
...         duration_ms=0,
...         message="ok",
...         evidence=None,
...     )
>>> report = CheckRegistry.run_all(timeout_s=5, parallelism=1)
"""

from __future__ import annotations

import concurrent.futures as _futures
import logging
import platform as _platform
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

Severity = Literal["blocker", "high", "medium", "low", "info"]
Status = Literal["PASS", "FAIL", "SKIP"]
Category = Literal[
    "code_integrity",
    "config_data",
    "architecture_runtime",
    "providers",
    "infrastructure",
    "interview_readiness",
    "security",
]

VALID_CATEGORIES: tuple[str, ...] = (
    "code_integrity",
    "config_data",
    "architecture_runtime",
    "providers",
    "infrastructure",
    "interview_readiness",
    "security",
)

VALID_SEVERITIES: tuple[str, ...] = (
    "blocker",
    "high",
    "medium",
    "low",
    "info",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Check(BaseModel):
    """Descriptor for a single registered check.

    Attributes:
        id: Stable, dot-separated identifier (e.g. ``code.ast_parse_all``).
        name: Human-readable short name for the report.
        category: One of :data:`VALID_CATEGORIES`.
        severity: One of :data:`VALID_SEVERITIES`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: Category
    severity: Severity


class CheckResult(BaseModel):
    """Outcome of a single check invocation.

    Attributes:
        check_id: The :attr:`Check.id` this result belongs to.
        status: ``"PASS"``, ``"FAIL"``, or ``"SKIP"``.
        duration_ms: Wall-clock duration of the check in milliseconds.
        message: Short, single-line human-readable summary.
        evidence: Optional multi-line evidence block (stack trace,
            command output, diff, ...).  Rendered verbatim in the
            Markdown report under the Failures section.
    """

    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1)
    status: Status
    duration_ms: int = Field(ge=0)
    message: str = ""
    evidence: str | None = None

    @field_validator("message")
    @classmethod
    def _single_line(cls, v: str) -> str:
        """Collapse the message to a single line for table rendering."""
        return v.replace("\r", " ").replace("\n", " ")[:500]


class VerificationReport(BaseModel):
    """Aggregate of every :class:`CheckResult` produced by a run.

    Attributes:
        total: Number of checks attempted.
        passed: Count of ``PASS`` results.
        failed: Count of ``FAIL`` results.
        skipped: Count of ``SKIP`` results.
        duration_s: Wall-clock duration of the whole run.
        pass_rate: ``passed / (passed + failed)`` as a fraction in [0, 1].
        per_category_summary: Mapping from category name to a dict with
            ``total``/``passed``/``failed``/``skipped`` counters.
        results: Full list of :class:`CheckResult` in registration order.
        run_at: UTC timestamp of the run start.
        git_sha: Current git HEAD SHA (``"unknown"`` if git unavailable).
        python_version: ``sys.version.split()[0]``.
        platform: ``platform.platform()``.
    """

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    duration_s: float = Field(ge=0.0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    per_category_summary: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    results: list[CheckResult] = Field(default_factory=list)
    run_at: datetime
    git_sha: str = "unknown"
    python_version: str = ""
    platform: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CheckCallable = Callable[[], CheckResult]


class _RegistryImpl:
    """Mutable, thread-safe registry backing :class:`CheckRegistry`."""

    def __init__(self) -> None:
        self._checks: dict[str, tuple[Check, CheckCallable]] = {}
        self._lock = threading.Lock()

    def register(self, check: Check, fn: CheckCallable) -> None:
        """Register ``fn`` under ``check.id``.

        Args:
            check: The :class:`Check` descriptor.
            fn: Zero-arg callable returning a :class:`CheckResult`.

        Raises:
            ValueError: If ``check.id`` is already registered.
        """
        with self._lock:
            if check.id in self._checks:
                raise ValueError(
                    f"duplicate check id: {check.id!r}"
                )
            self._checks[check.id] = (check, fn)

    def all(self) -> list[tuple[Check, CheckCallable]]:
        """Return a snapshot of all registered checks."""
        with self._lock:
            return list(self._checks.values())

    def clear(self) -> None:
        """Remove every registered check (test-only helper)."""
        with self._lock:
            self._checks.clear()


_IMPL = _RegistryImpl()


def register_check(
    *,
    id: str,  # noqa: A002 - shadowing is intentional; this is a kwarg name
    name: str,
    category: Category,
    severity: Severity,
) -> Callable[[CheckCallable], CheckCallable]:
    """Decorator that registers a zero-arg check callable.

    The wrapped callable MUST return a :class:`CheckResult`; this wrapper
    does not catch exceptions -- that happens inside :func:`_run_one` so
    the exception text becomes evidence on the FAIL.

    Args:
        id: Stable dotted identifier.
        name: Human-readable short name.
        category: One of :data:`VALID_CATEGORIES`.
        severity: One of :data:`VALID_SEVERITIES`.

    Returns:
        The original callable (registration has a side effect only).
    """

    def _decorator(fn: CheckCallable) -> CheckCallable:
        check = Check(id=id, name=name, category=category, severity=severity)
        _IMPL.register(check, fn)
        return fn

    return _decorator


def _git_sha() -> str:
    """Return the short git SHA of HEAD or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _run_one(
    check: Check, fn: CheckCallable, timeout_s: int
) -> CheckResult:
    """Execute a single check with a per-call timeout and exception trap.

    Any exception becomes a ``FAIL`` with the traceback in ``evidence``.
    A :class:`concurrent.futures.TimeoutError` becomes a ``FAIL`` with a
    ``"timeout after Ns"`` message.

    Args:
        check: The :class:`Check` descriptor (for ``id``).
        fn: The zero-arg callable.
        timeout_s: Per-check timeout in seconds.

    Returns:
        A :class:`CheckResult` -- always, never raises.
    """
    t0 = time.monotonic()
    with _futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            result = future.result(timeout=timeout_s)
        except _futures.TimeoutError:
            future.cancel()
            return CheckResult(
                check_id=check.id,
                status="FAIL",
                duration_ms=int((time.monotonic() - t0) * 1000),
                message=f"timeout after {timeout_s}s",
                evidence=None,
            )
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                check_id=check.id,
                status="FAIL",
                duration_ms=int((time.monotonic() - t0) * 1000),
                message=f"exception: {type(exc).__name__}: {exc}",
                evidence=traceback.format_exc(limit=20),
            )
    # Guarantee the duration field reflects the wall-clock elapsed even
    # if the check misreports it.
    if not isinstance(result, CheckResult):  # pragma: no cover - defensive
        return CheckResult(
            check_id=check.id,
            status="FAIL",
            duration_ms=int((time.monotonic() - t0) * 1000),
            message=f"check returned {type(result).__name__}, not CheckResult",
            evidence=None,
        )
    return result


class CheckRegistry:
    """Class-level facade over the global :class:`_RegistryImpl`."""

    @classmethod
    def register(cls, check: Check, fn: CheckCallable) -> None:
        """Register ``fn`` under ``check.id``.  See :func:`register_check`."""
        _IMPL.register(check, fn)

    @classmethod
    def all_checks(cls) -> list[Check]:
        """Return a list of every registered :class:`Check`."""
        return [c for c, _ in _IMPL.all()]

    @classmethod
    def clear(cls) -> None:
        """Remove every registered check (tests only)."""
        _IMPL.clear()

    @classmethod
    def run_all(
        cls,
        *,
        timeout_s: int = 60,
        parallelism: int = 4,
        categories: set[str] | None = None,
    ) -> VerificationReport:
        """Execute every registered check concurrently.

        Args:
            timeout_s: Per-check timeout (seconds).
            parallelism: Thread-pool worker count for the outer scheduler.
            categories: If set, only run checks whose category is in this
                set.  ``None`` means all categories.

        Returns:
            A populated :class:`VerificationReport`.
        """
        all_entries = _IMPL.all()
        if categories is not None:
            all_entries = [
                (c, fn) for c, fn in all_entries if c.category in categories
            ]

        t_start = time.monotonic()
        run_at = datetime.now(timezone.utc)

        results: list[CheckResult] = []
        # Preserve registration order by indexing the futures.
        with _futures.ThreadPoolExecutor(
            max_workers=max(1, parallelism)
        ) as pool:
            futures: list[_futures.Future[CheckResult]] = []
            for check, fn in all_entries:
                futures.append(
                    pool.submit(_run_one, check, fn, timeout_s)
                )
            for check, future in zip([c for c, _ in all_entries], futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - final safety net
                    results.append(
                        CheckResult(
                            check_id=check.id,
                            status="FAIL",
                            duration_ms=0,
                            message=f"outer scheduler exception: {exc}",
                            evidence=traceback.format_exc(limit=20),
                        )
                    )

        passed = sum(1 for r in results if r.status == "PASS")
        failed = sum(1 for r in results if r.status == "FAIL")
        skipped = sum(1 for r in results if r.status == "SKIP")
        total = len(results)
        denom = passed + failed
        pass_rate = (passed / denom) if denom > 0 else 1.0

        # Per-category summary.
        by_cat: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
        )
        id_to_cat = {c.id: c.category for c, _ in all_entries}
        for r in results:
            cat = id_to_cat.get(r.check_id, "code_integrity")
            entry = by_cat[cat]
            entry["total"] += 1
            if r.status == "PASS":
                entry["passed"] += 1
            elif r.status == "FAIL":
                entry["failed"] += 1
            else:
                entry["skipped"] += 1

        return VerificationReport(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_s=time.monotonic() - t_start,
            pass_rate=pass_rate,
            per_category_summary=dict(by_cat),
            results=results,
            run_at=run_at,
            git_sha=_git_sha(),
            python_version=sys.version.split()[0],
            platform=_platform.platform(),
        )
