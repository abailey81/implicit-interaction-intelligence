"""Iter 117 — ExplainPlan + SubAnswer dataclass contract tests."""
from __future__ import annotations

import json

import pytest

from i3.pipeline.explain_decomposer import ExplainPlan, SubAnswer


def test_subanswer_defaults():
    sa = SubAnswer(question="what?")
    assert sa.question == "what?"
    assert sa.source == "unknown"
    assert sa.text == ""
    assert sa.confidence == 0.0


def test_subanswer_full_construction():
    sa = SubAnswer(question="why?", source="retrieval",
                   text="Because reasons.", confidence=0.85)
    assert sa.confidence == 0.85
    assert sa.source == "retrieval"


def test_explain_plan_minimal():
    p = ExplainPlan(topic="photosynthesis")
    assert p.topic == "photosynthesis"
    assert p.sub_questions == []
    assert p.sub_answers == []
    assert p.composite_answer == ""


def test_explain_plan_to_dict_shape():
    p = ExplainPlan(
        topic="entropy",
        sub_questions=["What is entropy?", "Why does it increase?"],
        sub_answers=[
            SubAnswer(question="What is entropy?", source="kg_overview",
                      text="Disorder.", confidence=0.9),
        ],
        composite_answer="Entropy is disorder.",
    )
    d = p.to_dict()
    assert d["topic"] == "entropy"
    assert len(d["sub_questions"]) == 2
    assert len(d["sub_answers"]) == 1
    assert d["sub_answers"][0]["question"] == "What is entropy?"


def test_explain_plan_to_dict_round_trips_through_json():
    p = ExplainPlan(
        topic="x",
        sub_questions=["q1"],
        sub_answers=[SubAnswer(question="q1", text="a1", confidence=0.5)],
        composite_answer="composite",
    )
    s = json.dumps(p.to_dict())
    parsed = json.loads(s)
    assert parsed["topic"] == "x"


def test_explain_plan_default_factory_isolation():
    a = ExplainPlan(topic="x")
    b = ExplainPlan(topic="y")
    a.sub_questions.append("q")
    assert b.sub_questions == []
