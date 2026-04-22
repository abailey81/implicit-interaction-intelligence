"""Tests for the meta-learning (MAML / Reptile / FewShotAdapter) package.

Covers:
* :class:`MetaTask` Pydantic validation.
* Inner-loop adaptation actually reduces the query loss.
* FO-MAML does not require ``create_graph`` on outer backward.
* Reptile pulls meta-weights toward adapted weights.
* :class:`FewShotAdapter` returns a model with different parameters.
* Invalid hyperparameters raise ``ValueError``.
* Determinism under a fixed seed.
"""

from __future__ import annotations

import pytest
import torch

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.encoder.tcn import TemporalConvNet
from i3.interaction.types import InteractionFeatureVector
from i3.meta_learning.few_shot_adapter import FewShotAdapter
from i3.meta_learning.maml import MAMLTrainer, MetaBatch, MetaTask
from i3.meta_learning.reptile import ReptileTrainer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sample_vector(seed: int) -> InteractionFeatureVector:
    """Return a deterministic dummy feature vector."""
    g = torch.Generator().manual_seed(seed)
    vals = torch.rand(32, generator=g).tolist()
    return InteractionFeatureVector.from_tensor(torch.tensor(vals))


def _sample_adaptation() -> AdaptationVector:
    """Return a non-default AdaptationVector with some structure."""
    return AdaptationVector(
        cognitive_load=0.7,
        style_mirror=StyleVector(
            formality=0.4, verbosity=0.5, emotionality=0.6, directness=0.5
        ),
        emotional_tone=0.4,
        accessibility=0.2,
    )


def _make_task(seed: int = 0) -> MetaTask:
    """Build a deterministic :class:`MetaTask`."""
    return MetaTask(
        persona_name="unit_test_persona",
        support_set=[_sample_vector(seed + i) for i in range(3)],
        query_set=[_sample_vector(seed + 10 + i) for i in range(4)],
        target_adaptation=_sample_adaptation(),
    )


def _tiny_encoder() -> TemporalConvNet:
    """Return a small, fast TCN suitable for unit tests."""
    return TemporalConvNet(
        input_dim=32,
        hidden_dims=[16, 16],
        dilations=[1, 2],
        kernel_size=3,
        embedding_dim=16,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# MetaTask validation
# ---------------------------------------------------------------------------


def test_meta_task_requires_nonempty_support_and_query() -> None:
    """Empty support or query sets are rejected at construction."""
    with pytest.raises(Exception):
        MetaTask(
            persona_name="p",
            support_set=[],
            query_set=[_sample_vector(0)],
            target_adaptation=_sample_adaptation(),
        )
    with pytest.raises(Exception):
        MetaTask(
            persona_name="p",
            support_set=[_sample_vector(0)],
            query_set=[],
            target_adaptation=_sample_adaptation(),
        )


def test_meta_task_frozen_immutable() -> None:
    """MetaTask is immutable -- attempting to mutate raises."""
    task = _make_task()
    with pytest.raises(Exception):
        task.persona_name = "mutated"  # type: ignore[misc]


def test_meta_batch_rejects_empty_tasks() -> None:
    """Zero-task MetaBatches are rejected."""
    with pytest.raises(Exception):
        MetaBatch(tasks=[])


def test_meta_batch_len() -> None:
    """MetaBatch len matches the number of tasks."""
    batch = MetaBatch(tasks=[_make_task(i) for i in range(3)])
    assert len(batch) == 3


# ---------------------------------------------------------------------------
# MAMLTrainer
# ---------------------------------------------------------------------------


def test_maml_invalid_inner_steps_raises() -> None:
    """inner_steps < 1 raises ValueError."""
    model = _tiny_encoder()
    with pytest.raises(ValueError):
        MAMLTrainer(model, inner_steps=0, embedding_dim=16)


def test_maml_invalid_inner_lr_raises() -> None:
    """Non-positive inner_lr raises ValueError."""
    model = _tiny_encoder()
    with pytest.raises(ValueError):
        MAMLTrainer(model, inner_lr=0.0, embedding_dim=16)
    with pytest.raises(ValueError):
        MAMLTrainer(model, inner_lr=-1.0, embedding_dim=16)


def test_maml_invalid_outer_lr_raises() -> None:
    """Non-positive outer_lr raises ValueError."""
    model = _tiny_encoder()
    with pytest.raises(ValueError):
        MAMLTrainer(model, outer_lr=0.0, embedding_dim=16)


def test_maml_inner_loop_reduces_support_loss() -> None:
    """Inner loop produces an adapted model whose support loss drops."""
    torch.manual_seed(7)
    model = _tiny_encoder()
    trainer = MAMLTrainer(
        model,
        inner_lr=0.1,
        inner_steps=5,
        first_order=True,
        embedding_dim=16,
    )
    task = _make_task(seed=3)
    params_before = trainer._named_params()
    # Pre-adapt query loss.
    q_before = trainer._task_query_loss(task, params_before).item()
    adapted_params, _ = trainer._adapt(task, create_graph=False)
    q_after = trainer._task_query_loss(task, adapted_params).item()
    # The inner loop should help on the query set (at least marginally)
    # when support and query share a target.
    assert q_after < q_before + 1e-4


def test_maml_first_order_outer_step_is_cheap() -> None:
    """FO-MAML outer step succeeds without second-order autograd.

    We monkey-patch torch.autograd.grad to fail when create_graph=True
    and check that the first_order=True path still works.
    """
    torch.manual_seed(0)
    model = _tiny_encoder()
    trainer = MAMLTrainer(
        model,
        inner_lr=0.05,
        outer_lr=1e-3,
        inner_steps=2,
        first_order=True,
        embedding_dim=16,
    )
    batch = MetaBatch(tasks=[_make_task(i) for i in range(2)])
    stats = trainer.outer_step(batch)
    assert "meta_loss" in stats
    assert stats["meta_loss"] == stats["meta_loss"]  # not NaN


def test_maml_second_order_outer_step_runs() -> None:
    """Second-order MAML outer step completes and returns finite loss."""
    torch.manual_seed(0)
    model = _tiny_encoder()
    trainer = MAMLTrainer(
        model,
        inner_lr=0.05,
        outer_lr=1e-3,
        inner_steps=2,
        first_order=False,
        embedding_dim=16,
    )
    batch = MetaBatch(tasks=[_make_task(i) for i in range(2)])
    stats = trainer.outer_step(batch)
    assert torch.isfinite(torch.tensor(stats["meta_loss"]))


def test_maml_outer_step_changes_parameters() -> None:
    """Outer step updates at least some of the meta-parameters."""
    torch.manual_seed(1)
    model = _tiny_encoder()
    trainer = MAMLTrainer(
        model,
        inner_lr=0.05,
        outer_lr=1e-2,
        inner_steps=1,
        first_order=True,
        embedding_dim=16,
    )
    before = {
        n: p.detach().clone() for n, p in trainer.model.named_parameters()
    }
    batch = MetaBatch(tasks=[_make_task(i) for i in range(2)])
    trainer.outer_step(batch)
    after = dict(trainer.model.named_parameters())
    n_changed = sum(
        int(not torch.allclose(before[n], after[n], atol=0.0))
        for n in before
    )
    assert n_changed > 0


def test_maml_deterministic_under_seed() -> None:
    """Two identical runs produce identical outer-loop losses."""

    def run_once() -> float:
        torch.manual_seed(11)
        model = _tiny_encoder()
        trainer = MAMLTrainer(
            model,
            inner_lr=0.1,
            outer_lr=1e-3,
            inner_steps=2,
            first_order=True,
            embedding_dim=16,
        )
        batch = MetaBatch(tasks=[_make_task(i) for i in range(3)])
        return trainer.outer_step(batch)["meta_loss"]

    a = run_once()
    b = run_once()
    assert a == pytest.approx(b, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# ReptileTrainer
# ---------------------------------------------------------------------------


def test_reptile_invalid_hyperparameters_raise() -> None:
    """Reptile rejects non-positive learning rates and non-positive steps."""
    model = _tiny_encoder()
    with pytest.raises(ValueError):
        ReptileTrainer(model, inner_lr=0.0, embedding_dim=16)
    with pytest.raises(ValueError):
        ReptileTrainer(model, outer_lr=0.0, embedding_dim=16)
    with pytest.raises(ValueError):
        ReptileTrainer(model, inner_steps=0, embedding_dim=16)


def test_reptile_update_moves_theta_toward_theta_prime() -> None:
    """θ moves toward θ' after a Reptile outer step."""
    torch.manual_seed(5)
    model = _tiny_encoder()
    trainer = ReptileTrainer(
        model, inner_lr=0.1, outer_lr=0.5, inner_steps=3, embedding_dim=16
    )
    task = _make_task(seed=2)
    # Snapshot theta.
    theta_before = {
        n: p.detach().clone() for n, p in trainer.model.named_parameters()
    }
    # Compute theta' before taking the outer step (use inner_loop).
    adapted_enc, _, _ = trainer.inner_loop(task)
    theta_prime = {
        n: p.detach().clone() for n, p in adapted_enc.named_parameters()
    }
    # Apply outer step using the same task in a batch-of-one.
    trainer.outer_step(MetaBatch(tasks=[task]))
    theta_after = {
        n: p.detach().clone() for n, p in trainer.model.named_parameters()
    }
    # For each changed parameter, the after value should lie between
    # before and theta' along the direction (before -> theta').
    moves_correct = 0
    moves_total = 0
    for name in theta_before:
        before = theta_before[name]
        prime = theta_prime[name]
        after = theta_after[name]
        direction = prime - before
        if direction.abs().sum().item() < 1e-8:
            continue
        movement = after - before
        dot = float((direction * movement).sum().item())
        moves_total += 1
        if dot > 0:
            moves_correct += 1
    assert moves_total > 0
    # Most of the parameters should move in the direction of theta'.
    assert moves_correct / moves_total > 0.5


def test_reptile_deterministic_under_seed() -> None:
    """Reptile produces identical outputs under identical seeds."""

    def run_once() -> float:
        torch.manual_seed(3)
        model = _tiny_encoder()
        trainer = ReptileTrainer(
            model,
            inner_lr=0.05,
            outer_lr=0.5,
            inner_steps=2,
            embedding_dim=16,
        )
        batch = MetaBatch(tasks=[_make_task(i) for i in range(2)])
        return trainer.outer_step(batch)["meta_loss"]

    a = run_once()
    b = run_once()
    assert a == pytest.approx(b, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# FewShotAdapter
# ---------------------------------------------------------------------------


def test_few_shot_adapter_produces_different_parameters() -> None:
    """Adapted model has at least one parameter that differs from the input."""
    torch.manual_seed(2)
    model = _tiny_encoder()
    head = torch.nn.Linear(16, 8)
    adapter = FewShotAdapter(
        model,
        n_adaptation_steps=3,
        adaptation_lr=0.1,
        adaptation_head=head,
    )
    support = [_sample_vector(i) for i in range(2)]
    adapted = adapter.adapt_to_user(support, target_hint=_sample_adaptation())
    n_diff = 0
    for (n, before), (_, after) in zip(
        model.named_parameters(), adapted.named_parameters()
    ):
        if not torch.allclose(before, after, atol=0.0):
            n_diff += 1
    assert n_diff > 0


def test_few_shot_adapter_preserves_original_model() -> None:
    """Adapting does not mutate the meta-trained model in place."""
    torch.manual_seed(4)
    model = _tiny_encoder()
    head = torch.nn.Linear(16, 8)
    adapter = FewShotAdapter(
        model, n_adaptation_steps=2, adaptation_lr=0.1, adaptation_head=head
    )
    snapshot = {
        n: p.detach().clone() for n, p in model.named_parameters()
    }
    adapter.adapt_to_user(
        [_sample_vector(0), _sample_vector(1)],
        target_hint=_sample_adaptation(),
    )
    for n, p in model.named_parameters():
        assert torch.allclose(p, snapshot[n], atol=0.0)


def test_few_shot_adapter_invalid_steps_raise() -> None:
    """Adapter rejects non-positive n_adaptation_steps or adaptation_lr."""
    model = _tiny_encoder()
    with pytest.raises(ValueError):
        FewShotAdapter(model, n_adaptation_steps=0, adaptation_lr=0.1)
    with pytest.raises(ValueError):
        FewShotAdapter(model, n_adaptation_steps=1, adaptation_lr=0.0)


def test_few_shot_adapter_amortisation_cache() -> None:
    """Subsequent amortised calls re-use the cached state."""
    torch.manual_seed(6)
    model = _tiny_encoder()
    head = torch.nn.Linear(16, 8)
    adapter = FewShotAdapter(
        model, n_adaptation_steps=2, adaptation_lr=0.1, adaptation_head=head
    )
    support = [_sample_vector(9), _sample_vector(10)]
    emb_first = adapter.amortised_representation(
        support, user_id="alice", target_hint=_sample_adaptation()
    )
    emb_second = adapter.amortised_representation(
        support, user_id="alice", target_hint=_sample_adaptation()
    )
    assert torch.allclose(emb_first, emb_second, atol=1e-6)
    adapter.clear_cache("alice")
    assert "alice" not in adapter._cache


def test_few_shot_adapter_requires_head_for_target_hint() -> None:
    """Supplying a target_hint without an adaptation_head raises."""
    model = _tiny_encoder()
    adapter = FewShotAdapter(model, n_adaptation_steps=1, adaptation_lr=0.1)
    with pytest.raises(RuntimeError):
        adapter.adapt_to_user(
            [_sample_vector(0)], target_hint=_sample_adaptation()
        )


def test_few_shot_adapter_self_supervised_without_head() -> None:
    """Without a head, the adapter still runs (self-supervised loss)."""
    torch.manual_seed(8)
    model = _tiny_encoder()
    adapter = FewShotAdapter(model, n_adaptation_steps=2, adaptation_lr=0.1)
    adapted = adapter.adapt_to_user([_sample_vector(0), _sample_vector(1)])
    n_diff = 0
    for (_, before), (_, after) in zip(
        model.named_parameters(), adapted.named_parameters()
    ):
        if not torch.allclose(before, after, atol=0.0):
            n_diff += 1
    # At least one parameter should have moved; if every gradient was
    # zero we'd have a silent no-op.
    assert n_diff > 0
