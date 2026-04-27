"""Iter 126 — ElasticWeightConsolidation construction + state invariants."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from i3.continual.ewc import ElasticWeightConsolidation


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))


def test_lambda_ewc_must_be_non_negative():
    m = _tiny_model()
    with pytest.raises(ValueError):
        ElasticWeightConsolidation(m, lambda_ewc=-1.0)


def test_fisher_samples_must_be_positive():
    m = _tiny_model()
    with pytest.raises(ValueError):
        ElasticWeightConsolidation(m, fisher_estimation_samples=0)


def test_fisher_epsilon_must_be_non_negative():
    m = _tiny_model()
    with pytest.raises(ValueError):
        ElasticWeightConsolidation(m, fisher_epsilon=-0.001)


def test_initial_state_empty():
    m = _tiny_model()
    ewc = ElasticWeightConsolidation(m)
    assert ewc.fisher == {}
    assert ewc.star_params == {}
    assert ewc.num_tasks_consolidated == 0


def test_penalty_loss_zero_with_no_consolidation():
    m = _tiny_model()
    ewc = ElasticWeightConsolidation(m)
    loss = ewc.penalty_loss()
    assert loss.item() == pytest.approx(0.0)


def test_reset_clears_state():
    m = _tiny_model()
    ewc = ElasticWeightConsolidation(m)
    # Manually mark a fake consolidation to verify reset.
    ewc._num_tasks = 5
    ewc._fisher = {"fake": torch.tensor([1.0])}
    ewc._star_params = {"fake": torch.tensor([0.0])}
    ewc.reset()
    assert ewc.num_tasks_consolidated == 0
    assert ewc.fisher == {}
    assert ewc.star_params == {}


def test_state_dict_round_trips():
    m = _tiny_model()
    ewc = ElasticWeightConsolidation(m)
    sd = ewc.state_dict()
    assert isinstance(sd, dict)
    assert "fisher" in sd
    assert "star_params" in sd


def test_model_property_returns_wrapped():
    m = _tiny_model()
    ewc = ElasticWeightConsolidation(m)
    assert ewc.model is m
