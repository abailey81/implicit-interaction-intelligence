"""Tests for the continual-learning sub-package.

Covers:
* :class:`i3.continual.ewc.ElasticWeightConsolidation` -- Fisher shape,
  non-negativity of the penalty, zero-penalty identity, monotonicity in
  parameter shift, and reset semantics.
* :class:`i3.continual.ewc.OnlineEWC` -- running Fisher that
  incrementally updates instead of resetting between tasks.
* :class:`i3.continual.replay_buffer.ReservoirReplayBuffer` -- uniform
  sampling invariant (empirical Monte Carlo test).
* :class:`i3.continual.drift_detector.ConceptDriftDetector` -- fires on
  known shift, stays silent on stationary streams.
* :class:`i3.continual.ewc_user_model.EWCUserModel` -- does not mutate
  the wrapped :class:`UserModel`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from i3.config import UserModelConfig
from i3.continual.drift_detector import ConceptDriftDetector
from i3.continual.ewc import (
    ElasticWeightConsolidation,
    OnlineEWC,
    build_ewc_for_encoder,
)
from i3.continual.ewc_user_model import EWCUserModel
from i3.continual.replay_buffer import (
    ExperienceReplay,
    ReplaySample,
    ReservoirReplayBuffer,
)
from i3.user_model.model import UserModel
from i3.user_model.types import UserProfile


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class TinyRegressor(nn.Module):
    """Small MLP used across the EWC tests."""

    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return a regressed tensor."""
        return self.fc2(torch.relu(self.fc1(x)))


def _make_loader(n: int = 32, in_dim: int = 4, out_dim: int = 2, seed: int = 0) -> DataLoader:
    """Synthetic regression dataloader."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    y = torch.randn(n, out_dim, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def _mse_closure(model: nn.Module, batch: tuple) -> torch.Tensor:
    """MSE loss closure suitable for the EWC Fisher estimator."""
    x, y = batch
    out = model(x)
    return torch.nn.functional.mse_loss(out, y)


# ---------------------------------------------------------------------------
# ElasticWeightConsolidation -- core API
# ---------------------------------------------------------------------------


class TestElasticWeightConsolidation:
    """Tests for :class:`ElasticWeightConsolidation`."""

    def test_estimate_fisher_returns_dict_with_expected_shapes(self) -> None:
        """Fisher dict keys must match trainable params and shapes must agree."""
        torch.manual_seed(0)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(model, loss_closure=_mse_closure)
        loader = _make_loader()
        fisher = ewc.estimate_fisher(loader)
        trainable = dict(model.named_parameters())
        assert set(fisher.keys()) == set(trainable.keys())
        for name, tensor in fisher.items():
            assert tensor.shape == trainable[name].shape
            assert torch.all(tensor >= 0), (
                f"Fisher entry {name} has a negative component"
            )

    def test_penalty_loss_non_negative_after_consolidation(self) -> None:
        """The EWC penalty is always non-negative."""
        torch.manual_seed(1)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(model, loss_closure=_mse_closure)
        ewc.consolidate(_make_loader())
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        penalty = ewc.penalty_loss()
        assert penalty.dim() == 0
        assert penalty.item() >= 0.0

    def test_identical_params_zero_penalty(self) -> None:
        """If params have not changed since consolidate(), penalty == 0."""
        torch.manual_seed(2)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(model, loss_closure=_mse_closure)
        ewc.consolidate(_make_loader())
        # No parameter change.
        penalty = ewc.penalty_loss()
        assert penalty.item() == pytest.approx(0.0, abs=1e-8)

    def test_larger_shift_larger_penalty_monotonic(self) -> None:
        """Monotonic: bigger parameter shift -> bigger penalty."""
        torch.manual_seed(3)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(
            model, lambda_ewc=100.0, loss_closure=_mse_closure,
        )
        ewc.consolidate(_make_loader())
        star = {k: v.detach().clone() for k, v in model.named_parameters()}

        # Scale away from θ* by increasing magnitudes.
        penalties: list[float] = []
        for scale in (0.01, 0.05, 0.2, 0.5):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(star[name] + scale)
            penalties.append(ewc.penalty_loss().item())
        # Strictly monotonic non-decreasing -- allow tiny numerical slack.
        for prev, nxt in zip(penalties, penalties[1:]):
            assert nxt >= prev - 1e-7, (
                f"Penalty should be monotonic, got {penalties}"
            )
        # And actually grow by orders of magnitude.
        assert penalties[-1] > penalties[0] * 5.0

    def test_reset_clears_state(self) -> None:
        """reset() drops stored Fisher and star params."""
        torch.manual_seed(4)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(model, loss_closure=_mse_closure)
        ewc.consolidate(_make_loader())
        assert ewc.num_tasks_consolidated == 1
        assert len(ewc.fisher) > 0
        assert len(ewc.star_params) > 0
        ewc.reset()
        assert ewc.num_tasks_consolidated == 0
        assert ewc.fisher == {}
        assert ewc.star_params == {}
        # Penalty returns a scalar zero even with no state.
        assert ewc.penalty_loss().item() == pytest.approx(0.0)

    def test_penalty_is_differentiable(self) -> None:
        """The penalty must propagate gradients back to the model."""
        torch.manual_seed(5)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(
            model, lambda_ewc=10.0, loss_closure=_mse_closure,
        )
        ewc.consolidate(_make_loader())
        # Perturb params.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1 * torch.randn_like(p))
        loss = ewc.penalty_loss()
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_state_dict_roundtrip(self) -> None:
        """Fisher and star params survive state_dict save/load."""
        torch.manual_seed(6)
        model = TinyRegressor()
        ewc = ElasticWeightConsolidation(model, loss_closure=_mse_closure)
        ewc.consolidate(_make_loader())
        payload = ewc.state_dict()

        # New instance.
        model2 = TinyRegressor()
        ewc2 = ElasticWeightConsolidation(model2, loss_closure=_mse_closure)
        ewc2.load_state_dict(payload)
        assert ewc2.num_tasks_consolidated == 1
        assert set(ewc2.fisher.keys()) == set(ewc.fisher.keys())
        for k in ewc.fisher:
            assert torch.allclose(ewc.fisher[k], ewc2.fisher[k])


# ---------------------------------------------------------------------------
# OnlineEWC
# ---------------------------------------------------------------------------


class TestOnlineEWC:
    """Tests for :class:`OnlineEWC` running-Fisher behaviour."""

    def test_running_fisher_accumulates_not_resets(self) -> None:
        """Successive consolidate() calls must update the running Fisher."""
        torch.manual_seed(10)
        model = TinyRegressor()
        ewc = OnlineEWC(
            model, lambda_ewc=100.0, gamma=0.9, loss_closure=_mse_closure,
        )
        ewc.consolidate(_make_loader(seed=0))
        fisher_after_one = {k: v.detach().clone() for k, v in ewc.fisher.items()}
        ewc.consolidate(_make_loader(seed=1))
        fisher_after_two = ewc.fisher

        # Every entry after two consolidations should be at least
        # gamma * (first entry) since a new Fisher (≥0) was added.
        for k in fisher_after_one:
            assert torch.all(
                fisher_after_two[k] + 1e-8 >= 0.9 * fisher_after_one[k]
            )

    def test_builder_returns_correct_class(self) -> None:
        """build_ewc_for_encoder should respect the ``online`` flag."""
        model = TinyRegressor()
        vanilla = build_ewc_for_encoder(model, online=False)
        online = build_ewc_for_encoder(model, online=True, gamma=0.8)
        assert type(vanilla) is ElasticWeightConsolidation
        assert isinstance(online, OnlineEWC)
        assert online.gamma == pytest.approx(0.8)

    def test_invalid_gamma_rejected(self) -> None:
        """gamma must lie in (0, 1]."""
        model = TinyRegressor()
        with pytest.raises(ValueError):
            OnlineEWC(model, gamma=0.0)
        with pytest.raises(ValueError):
            OnlineEWC(model, gamma=1.5)


# ---------------------------------------------------------------------------
# Reservoir replay
# ---------------------------------------------------------------------------


class TestReservoirReplayBuffer:
    """Tests for :class:`ReservoirReplayBuffer`."""

    def test_capacity_respected(self) -> None:
        """The buffer never exceeds ``capacity``."""
        buf = ReservoirReplayBuffer(capacity=32, seed=0)
        for i in range(1000):
            buf.add(ReplaySample(input_tensor=torch.tensor([float(i)])))
        assert len(buf) == 32
        assert buf.observed == 1000

    def test_uniform_sampling_invariant(self) -> None:
        """Empirical retention frequency ≈ capacity / n."""
        capacity = 20
        n = 400
        trials = 400
        counts = torch.zeros(n)
        for trial in range(trials):
            buf = ReservoirReplayBuffer(capacity=capacity, seed=trial)
            for i in range(n):
                buf.add(ReplaySample(input_tensor=torch.tensor([float(i)])))
            for stored in buf:
                idx = int(stored.input_tensor.item())
                counts[idx] += 1
        freq = counts / float(trials)
        expected = capacity / n
        # Each index should have roughly ``expected`` probability.
        mean_abs_error = (freq - expected).abs().mean().item()
        assert mean_abs_error < expected * 0.35, (
            f"uniform sampling violated: mae={mean_abs_error:.4f}"
        )

    def test_sample_returns_up_to_batch_size(self) -> None:
        """sample() returns min(batch_size, len(buffer)) items."""
        buf = ReservoirReplayBuffer(capacity=8, seed=0)
        for i in range(4):
            buf.add(ReplaySample(input_tensor=torch.tensor([float(i)])))
        assert len(buf.sample(2)) == 2
        assert len(buf.sample(10)) == 4  # clipped to buffer length

    def test_experience_replay_integrates_loss(self) -> None:
        """ExperienceReplay sums task + α · replay loss."""
        buf = ReservoirReplayBuffer(capacity=4, seed=0)
        for i in range(4):
            buf.add(ReplaySample(input_tensor=torch.tensor([float(i)])))
        replay = ExperienceReplay(buf, replay_batch_size=2, replay_weight=1.0)

        def loss_fn(sample: ReplaySample) -> torch.Tensor:
            # Scalar proportional to the stored value.
            return sample.input_tensor.sum()

        task_loss = torch.tensor(1.0)
        combined = replay.integrate_into_training(loss_fn, task_loss=task_loss)
        assert combined.item() >= task_loss.item()

    def test_empty_buffer_yields_zero_replay(self) -> None:
        """Empty buffer must contribute zero replay loss."""
        buf = ReservoirReplayBuffer(capacity=4, seed=0)
        replay = ExperienceReplay(buf, replay_batch_size=2, replay_weight=1.0)
        result = replay.integrate_into_training(
            lambda s: torch.tensor(1.0), task_loss=torch.tensor(5.0)
        )
        assert result.item() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Drift detector
# ---------------------------------------------------------------------------


class TestConceptDriftDetectorSmoke:
    """Light detector tests; heavier tests live in test_drift_detector.py."""

    def test_fires_on_shift(self) -> None:
        """Detector raises drift after a clear distribution shift."""
        det = ConceptDriftDetector(
            delta=0.05, min_sub_window=8, max_window=256,
        )
        for _ in range(48):
            det.update(0.1)
        fired = False
        for _ in range(96):
            result = det.update(5.0)
            if result.drift_detected:
                fired = True
                break
        assert fired

    def test_silent_on_stationary(self) -> None:
        """No drift is reported on a stationary stream."""
        det = ConceptDriftDetector(
            delta=0.002, min_sub_window=8, max_window=256, cooldown=0,
        )
        torch.manual_seed(0)
        stream = 0.1 + 0.01 * torch.randn(256)
        any_drift = False
        for v in stream.tolist():
            if det.update(v).drift_detected:
                any_drift = True
                break
        assert not any_drift


# ---------------------------------------------------------------------------
# EWCUserModel composition
# ---------------------------------------------------------------------------


def _make_user_model(user_id: str = "u1") -> UserModel:
    """Build a minimal :class:`UserModel` instance."""
    config = UserModelConfig()
    now = datetime.now(timezone.utc)
    profile = UserProfile(
        user_id=user_id,
        baseline_embedding=None,
        baseline_features_mean=None,
        baseline_features_std=None,
        total_sessions=0,
        total_messages=0,
        relationship_strength=0.0,
        long_term_style={},
        created_at=now,
        updated_at=now,
        baseline_established=False,
    )
    return UserModel(user_id=user_id, config=config, profile=profile)


class TestEWCUserModel:
    """Tests for the composed :class:`EWCUserModel`."""

    def test_wraps_without_mutating_inner(self) -> None:
        """Constructing EWCUserModel must not touch the inner model."""
        inner = _make_user_model()
        inner_state_before = (
            inner.user_id,
            inner.profile.total_sessions,
            inner.profile.total_messages,
            inner.current_session,
            inner.current_state,
        )
        encoder = TinyRegressor()
        wrapped = EWCUserModel(
            inner, encoder, lambda_ewc=100.0, replay_capacity=16,
        )
        # Every inner-state field unchanged.
        assert wrapped.inner is inner
        assert (
            inner.user_id,
            inner.profile.total_sessions,
            inner.profile.total_messages,
            inner.current_session,
            inner.current_state,
        ) == inner_state_before
        # EWC surface present.
        assert wrapped.penalty_loss().item() == pytest.approx(0.0)
        assert wrapped.ewc.num_tasks_consolidated == 0
        assert wrapped.replay is not None
        assert wrapped.drift_detector is not None

    def test_penalty_loss_delegates_to_ewc(self) -> None:
        """After consolidate(), penalty_loss() on the wrapper is non-zero."""
        inner = _make_user_model()
        encoder = TinyRegressor()
        wrapped = EWCUserModel(inner, encoder, lambda_ewc=100.0)
        wrapped.consolidate(_make_loader())
        with torch.no_grad():
            for p in encoder.parameters():
                p.add_(0.1 * torch.randn_like(p))
        assert wrapped.penalty_loss().item() > 0.0

    def test_replay_capacity_zero_disables_replay(self) -> None:
        """replay is None when capacity == 0."""
        inner = _make_user_model()
        encoder = TinyRegressor()
        wrapped = EWCUserModel(inner, encoder, replay_capacity=0)
        assert wrapped.replay is None
