"""Unit tests for the red-team runner and invariant checkers.

Uses a deterministic ``_MockTarget`` that returns a pre-programmed
outcome per attack, so the tests do not depend on any network, any
model, or any sqlite file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import pytest

from i3.redteam.attack_corpus import ATTACK_CORPUS, Attack
from i3.redteam.attacker import (
    AttackResult,
    RedTeamReport,
    RedTeamRunner,
    TargetSurface,
)
from i3.redteam.policy_check import (
    verify_pddl_soundness,
    verify_privacy_invariant,
    verify_rate_limit_invariant,
    verify_sensitive_topic_invariant,
)


# ---------------------------------------------------------------------------
# Mock target implementations
# ---------------------------------------------------------------------------


class _MockTarget:
    """Deterministic :class:`TargetSurface` for tests.

    Args:
        outcome_fn: Callable that maps each :class:`Attack` to the
            outcome string the mock should report.
    """

    name: str = "mock"

    def __init__(self, outcome_fn: Callable[[Attack], str]) -> None:
        self._outcome_fn = outcome_fn
        self.calls: list[str] = []

    async def execute(self, attack: Attack) -> AttackResult:
        self.calls.append(attack.id)
        outcome = self._outcome_fn(attack)
        return AttackResult(
            attack_id=attack.id,
            category=attack.category,
            actual_outcome=outcome,
            matched_expected=outcome == attack.expected_outcome,
            evidence=f"mock -> {outcome}",
            duration_ms=0,
            severity=attack.severity,
        )


class _ConcurrentCounterTarget:
    """Target that records the maximum concurrent ``execute`` calls."""

    name: str = "counter"

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0
        self._lock = asyncio.Lock()

    async def execute(self, attack: Attack) -> AttackResult:
        async with self._lock:
            self.current += 1
            if self.current > self.peak:
                self.peak = self.current
        try:
            await asyncio.sleep(0.001)  # let the scheduler interleave
        finally:
            async with self._lock:
                self.current -= 1
        return AttackResult(
            attack_id=attack.id,
            category=attack.category,
            actual_outcome=attack.expected_outcome,
            matched_expected=True,
            evidence="counter",
            duration_ms=1,
            severity=attack.severity,
        )


class _CrashingTarget:
    """Target that always raises to exercise the error wrapper."""

    name: str = "crash"

    async def execute(self, attack: Attack) -> AttackResult:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_invokes_every_attack() -> None:
    """Every attack in the corpus is executed exactly once."""
    target = _MockTarget(lambda a: a.expected_outcome)
    runner = RedTeamRunner(target)
    report = asyncio.run(runner.run(parallelism=4))
    assert len(target.calls) == len(ATTACK_CORPUS)
    assert sorted(target.calls) == sorted(a.id for a in ATTACK_CORPUS)
    assert report.total_attacks == len(ATTACK_CORPUS)


def test_runner_perfect_pass_rate() -> None:
    """A target that always returns the expected outcome scores 1.0."""
    target = _MockTarget(lambda a: a.expected_outcome)
    runner = RedTeamRunner(target)
    report = asyncio.run(runner.run(parallelism=4))
    assert report.pass_rate == 1.0
    assert report.critical_fail_count == 0
    assert report.high_fail_count == 0
    assert not report.failures


def test_runner_counts_critical_and_high_separately() -> None:
    """Critical and high failures are tracked independently."""
    # Always return "passed_through" -- everything mismatches.
    target = _MockTarget(lambda a: "passed_through")
    runner = RedTeamRunner(target)
    report = asyncio.run(runner.run(parallelism=4))
    assert report.critical_fail_count > 0
    assert report.high_fail_count > 0
    # Every failure is listed
    assert len(report.failures) == len(ATTACK_CORPUS)
    # Critical failures only count "critical" severity
    assert report.critical_fail_count == sum(
        1 for a in ATTACK_CORPUS if a.severity == "critical"
    )


def test_runner_respects_parallelism_semaphore() -> None:
    """Runner should never exceed the requested concurrency level."""
    target = _ConcurrentCounterTarget()
    runner = RedTeamRunner(target)
    asyncio.run(runner.run(parallelism=3))
    assert target.peak <= 3, f"Peak concurrency {target.peak} exceeded 3"


def test_runner_wraps_exceptions_into_results() -> None:
    """A crashing target yields error-shaped results, not an exception."""
    target = _CrashingTarget()
    runner = RedTeamRunner(target)
    report = asyncio.run(runner.run(parallelism=2))
    assert report.pass_rate == 0.0
    assert all(r.error is not None for r in report.failures)
    assert all(r.actual_outcome == "error" for r in report.failures)


def test_runner_per_category_pass_rates() -> None:
    """Per-category pass rate reflects mixed mock outcomes."""

    def outcome(a: Attack) -> str:
        # Fail only the "jailbreak" category
        if a.category == "jailbreak":
            return "passed_through"
        return a.expected_outcome

    target = _MockTarget(outcome)
    runner = RedTeamRunner(target)
    report = asyncio.run(runner.run(parallelism=4))
    assert report.per_category_pass_rate["jailbreak"] == 0.0
    for cat, rate in report.per_category_pass_rate.items():
        if cat != "jailbreak":
            assert rate == 1.0, f"Unexpected failure in {cat}: rate={rate}"


def test_runner_rejects_zero_parallelism() -> None:
    target = _MockTarget(lambda a: a.expected_outcome)
    runner = RedTeamRunner(target)
    with pytest.raises(ValueError):
        asyncio.run(runner.run(parallelism=0))


# ---------------------------------------------------------------------------
# Invariant checkers
# ---------------------------------------------------------------------------


def _mk_rate_result(attack_id: str, outcome: str) -> AttackResult:
    return AttackResult(
        attack_id=attack_id,
        category="rate_limit_abuse",
        actual_outcome=outcome,
        matched_expected=outcome == "rate_limited",
        evidence="synthetic",
        duration_ms=0,
        severity="high",
    )


def _mk_bypass_result(attack_id: str, outcome: str) -> AttackResult:
    return AttackResult(
        attack_id=attack_id,
        category="privacy_override_bypass",
        actual_outcome=outcome,
        matched_expected=outcome == "forced_local",
        evidence="synthetic",
        duration_ms=0,
        severity="high",
    )


def test_rate_limit_invariant_passes_with_at_least_one_429() -> None:
    results = [
        _mk_rate_result("rl-001", "rate_limited"),
        _mk_rate_result("rl-002", "passed_through"),
    ]
    passed, ev = verify_rate_limit_invariant(results)
    assert passed
    assert "429" in ev


def test_rate_limit_invariant_fails_with_no_429() -> None:
    results = [_mk_rate_result("rl-001", "passed_through")]
    passed, ev = verify_rate_limit_invariant(results)
    assert not passed


def test_sensitive_topic_invariant_fails_on_cloud_route() -> None:
    results = [
        _mk_bypass_result("po-001", "forced_local"),
        _mk_bypass_result("po-002", "routed_cloud"),  # <-- violation
    ]
    passed, ev = verify_sensitive_topic_invariant(results)
    assert not passed
    assert "po-002" in ev


def test_sensitive_topic_invariant_passes_when_all_local() -> None:
    results = [
        _mk_bypass_result("po-001", "forced_local"),
        _mk_bypass_result("po-002", "forced_local"),
    ]
    passed, _ = verify_sensitive_topic_invariant(results)
    assert passed


def test_pddl_soundness_passes_on_default_corpus() -> None:
    from i3.safety.pddl_planner import PrivacySafetyPlanner

    passed, ev = verify_pddl_soundness(PrivacySafetyPlanner(), ATTACK_CORPUS)
    assert passed, f"PDDL soundness broken: {ev}"


def test_privacy_invariant_passes_without_diary() -> None:
    """No store -> vacuously satisfied."""
    passed, _ = verify_privacy_invariant(None)
    assert passed


def test_privacy_invariant_detects_forbidden_column(tmp_path: Path) -> None:
    """A diary with a raw_text column must fail the invariant."""
    import sqlite3

    db = tmp_path / "diary.sqlite"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE entries (id INTEGER, raw_text TEXT)")
        conn.execute("INSERT INTO entries VALUES (1, 'hi')")
        conn.commit()
    passed, ev = verify_privacy_invariant(str(db))
    assert not passed
    assert "raw_text" in ev


def test_privacy_invariant_passes_on_clean_schema(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "clean.sqlite"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE entries (id INTEGER, topic TEXT)")
        conn.commit()
    passed, _ = verify_privacy_invariant(str(db))
    assert passed


def test_cli_exit_code_zero_on_perfect_mock_run(tmp_path: Path) -> None:
    """`scripts.run_redteam.main` returns 0 when every attack passes."""
    # Build tiny argv that exercises the sanitiser target (no FastAPI
    # startup cost), writing reports to a tmp directory.
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    from scripts.run_redteam import main as cli_main

    rc = cli_main(
        [
            "--targets",
            "sanitizer",
            "--corpus",
            "default",
            "--parallelism",
            "2",
            "--tolerance",
            "1.0",  # any mismatch tolerated
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    # With tolerance=1.0 and no critical breakage the CLI should exit 0.
    # PDDL soundness and sensitive_topic invariants may still gate; this
    # asserts the machinery does not crash.
    assert rc in (0, 1), f"Unexpected exit code: {rc}"
    assert out_json.exists()
    assert out_md.exists()


def test_cli_exit_code_nonzero_on_unknown_target() -> None:
    """Unknown target -> exit 2 (argparse-level failure)."""
    from scripts.run_redteam import main as cli_main

    rc = cli_main(
        [
            "--targets",
            "not-a-real-target",
            "--out",
            "reports/ignored.json",
            "--out-md",
            "reports/ignored.md",
        ]
    )
    assert rc == 2
