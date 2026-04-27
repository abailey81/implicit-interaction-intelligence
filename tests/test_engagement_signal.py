"""Iter 69 — EngagementSignal / EngagementEstimator invariants.

The engagement composite drives the LinUCB router's reward, so its
behaviour under hostile input (negative latency, NaN, huge ratios)
must be predictable.  These tests pin the bounds + a few intuitive
monotonicity properties.
"""
from __future__ import annotations

import math

import pytest

from i3.pipeline.types import EngagementEstimator, EngagementSignal


def _signal(**overrides):
    base = dict(
        continued_conversation=True,
        response_latency_ms=2_000.0,
        response_length_ratio=1.0,
        topic_continuity=0.7,
        sentiment_shift=0.1,
    )
    base.update(overrides)
    return EngagementSignal(**base)


def test_score_in_unit_interval_default():
    s = _signal()
    assert 0.0 <= s.score <= 1.0


@pytest.mark.parametrize("k,v", [
    ("response_latency_ms", -100.0),
    ("response_latency_ms", 0.0),
    ("response_latency_ms", 1e9),
    ("response_length_ratio", -2.5),
    ("response_length_ratio", 1e9),
    ("topic_continuity", -1.0),
    ("topic_continuity", 7.0),
    ("sentiment_shift", -1e9),
    ("sentiment_shift", 1e9),
])
def test_score_robust_to_hostile_value(k, v):
    s = _signal(**{k: v})
    sc = s.score
    assert 0.0 <= sc <= 1.0
    assert not math.isnan(sc)


def test_continuation_dominates_when_other_signals_weak():
    """Continuing the conversation must lift the score over not-continuing
    when all other dimensions are equal."""
    s_yes = _signal(continued_conversation=True,
                    response_length_ratio=0.0, topic_continuity=0.0,
                    sentiment_shift=-1.0, response_latency_ms=30_000.0)
    s_no = _signal(continued_conversation=False,
                   response_length_ratio=0.0, topic_continuity=0.0,
                   sentiment_shift=-1.0, response_latency_ms=30_000.0)
    assert s_yes.score > s_no.score


def test_lower_latency_means_higher_score():
    s_fast = _signal(response_latency_ms=500.0)
    s_slow = _signal(response_latency_ms=15_000.0)
    assert s_fast.score > s_slow.score


def test_higher_topic_continuity_helps():
    a = _signal(topic_continuity=0.1)
    b = _signal(topic_continuity=0.9)
    assert b.score > a.score


def test_estimator_returns_valid_signal():
    est = EngagementEstimator()
    s = est.compute(
        continued=True, response_latency_ms=1500.0,
        user_msg_length=10, ai_msg_length=20,
        topic_continuity=0.6, sentiment_shift=0.2,
    )
    assert isinstance(s, EngagementSignal)
    assert 0.0 <= s.score <= 1.0


def test_estimator_clamps_topic_and_sentiment():
    est = EngagementEstimator()
    s = est.compute(
        continued=True, response_latency_ms=1000.0,
        user_msg_length=5, ai_msg_length=5,
        topic_continuity=99.0, sentiment_shift=99.0,
    )
    assert 0.0 <= s.score <= 1.0
    assert s.topic_continuity <= 1.0
    assert s.sentiment_shift <= 1.0


def test_zero_ai_length_does_not_divide_by_zero():
    est = EngagementEstimator()
    s = est.compute(
        continued=True, response_latency_ms=500.0,
        user_msg_length=10, ai_msg_length=0,
        topic_continuity=0.5, sentiment_shift=0.0,
    )
    assert not math.isnan(s.score)
    assert not math.isinf(s.score)
