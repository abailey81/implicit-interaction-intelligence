"""Iter 60 — multi-user concurrent emulation as a regression test.

Validates that per-user state stays isolated when multiple users
share a single ``ResponsePostProcessor`` (the lone shared component
in the pipeline).  Specifically pins:

  * Each user's cognitive_load reflects their OWN typing pattern,
    not the cross-user mean.
  * Different archetypes produce different reply lengths.
  * Within-user smoothing is stable across many interleaved turns.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pytest

from i3.adaptation.controller import AdaptationController
from i3.cloud.postprocess import ResponsePostProcessor
from i3.config import AdaptationConfig
from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.user_model.types import DeviationMetrics


CANONICAL = (
    "Sure! Absolutely happy to help. "
    "You might want to consider that perhaps approximately five different "
    "perspectives could provide additional context on this complex topic."
)


@dataclass
class _User:
    user_id: str
    archetype: str
    iki_ms: float
    iki_std: float
    edits: int
    msg: str
    extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    baseline: BaselineTracker = field(
        default_factory=lambda: BaselineTracker(warmup=2)
    )
    controller: AdaptationController = field(
        default_factory=lambda: AdaptationController(AdaptationConfig())
    )
    cl_history: list[float] = field(default_factory=list)
    len_history: list[int] = field(default_factory=list)


@pytest.fixture
def shared_pp() -> ResponsePostProcessor:
    return ResponsePostProcessor()


@pytest.fixture
def users() -> list[_User]:
    return [
        _User("u_alice", "fast_typer", 70, 8, 0, "next song"),
        _User("u_bob", "thoughtful", 145, 18, 1,
              "explain the typing rhythm signal please"),
        _User("u_carol", "anxious", 200, 55, 5, "wait did i mess that up"),
        _User("u_dave", "formal", 120, 14, 0,
              "kindly provide a comprehensive summary regarding the matter"),
        _User("u_eve", "stressed", 260, 70, 7, "ugh nothing works"),
    ]


def _drive_one_turn(user: _User, pp: ResponsePostProcessor) -> None:
    km = {
        "mean_iki_ms": user.iki_ms,
        "std_iki_ms": user.iki_std,
        "mean_burst_length": 8.0,
        "mean_pause_duration_ms": 200.0,
        "backspace_ratio": min(1.0, user.edits / max(len(user.msg), 10)),
        "composition_speed_cps": max(0.5, len(user.msg) / 4.0),
        "pause_before_send_ms": 300.0,
        "editing_effort": min(1.0, user.edits / 10.0),
    }
    fv = user.extractor.extract(
        keystroke_metrics=km, message_text=user.msg, history=[],
        baseline=user.baseline, session_start_ts=0.0, current_ts=30.0,
    )
    user.baseline.update(fv)
    dev = DeviationMetrics(
        current_vs_baseline=0.0, current_vs_session=0.0,
        engagement_score=0.5, magnitude=0.0,
        iki_deviation=fv.iki_deviation,
        length_deviation=fv.length_deviation,
        vocab_deviation=fv.vocab_deviation,
        formality_deviation=fv.formality_deviation,
        speed_deviation=fv.speed_deviation,
        engagement_deviation=fv.engagement_deviation,
        complexity_deviation=fv.complexity_deviation,
        pattern_deviation=fv.pattern_deviation,
    )
    av = user.controller.compute(fv, dev)
    shaped, _ = pp.adapt_with_log(CANONICAL, av)
    user.cl_history.append(av.cognitive_load)
    user.len_history.append(len(shaped))


def _interleaved_run(users: list[_User], pp: ResponsePostProcessor,
                      turns: int = 10) -> None:
    """10 interleaved turns per user — the order matters: it forces
    any cross-user state to surface as inconsistent within-user
    smoothing."""
    for _ in range(turns):
        for user in users:
            _drive_one_turn(user, pp)


def test_archetypes_produce_distinct_cl_means(users, shared_pp) -> None:
    """Different archetypes typing in interleaved order must produce
    visibly different cognitive_load means.  If state leaks between
    users, all five would converge to a similar mid-cl value."""
    _interleaved_run(users, shared_pp)
    cl_means = [statistics.mean(u.cl_history) for u in users]
    spread = max(cl_means) - min(cl_means)
    assert spread >= 0.30, (
        f"cl_mean spread {spread:.3f} across 5 archetypes is too tight; "
        f"per-user state may be leaking. cl_means={cl_means}"
    )


def test_archetype_ordering_makes_intuitive_sense(users, shared_pp) -> None:
    """Fast typer alice < anxious carol < stressed eve in cognitive_load.
    If state leaks, any of these orderings could break."""
    _interleaved_run(users, shared_pp)
    cls = {u.archetype: statistics.mean(u.cl_history) for u in users}
    assert cls["fast_typer"] < cls["anxious"], (
        f"fast_typer ({cls['fast_typer']}) should be < "
        f"anxious ({cls['anxious']})"
    )
    assert cls["anxious"] < cls["stressed"], (
        f"anxious ({cls['anxious']}) should be < "
        f"stressed ({cls['stressed']})"
    )


def test_within_user_smoothing_is_stable(users, shared_pp) -> None:
    """Each user types the same message with the same rhythm 10 times.
    Their per-turn cl should converge — pstdev across turns should be
    small.  If interleaved turns from other users leaked into a user's
    state, that user's cl would oscillate."""
    _interleaved_run(users, shared_pp)
    for user in users:
        std = statistics.pstdev(user.cl_history)
        assert std < 0.20, (
            f"{user.user_id} ({user.archetype}): cl_std={std:.3f} "
            f"is too high — interleaved turns from other users may be "
            f"corrupting this user's per-turn smoothing."
        )


def test_distinct_reply_lengths_across_users(users, shared_pp) -> None:
    """Different cognitive_load means should produce different shaped
    reply lengths.  At least 3 of the 5 users should have a distinct
    mean reply length."""
    _interleaved_run(users, shared_pp)
    distinct = len({
        round(statistics.mean(u.len_history))
        for u in users
    })
    assert distinct >= 3, (
        f"only {distinct}/5 distinct mean reply lengths — "
        f"adaptation isn't differentiating per user."
    )
