"""Unit tests for :mod:`i3.router.preference_learning`.

Covers dataset round-trip, Bradley-Terry training, active selection
ranking, DPO fit reporting, deterministic seeding, and invalid inputs.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from i3.router.preference_learning import (
    ActivePreferenceSelector,
    BradleyTerryRewardModel,
    DPOFitReport,
    DPOPreferenceOptimizer,
    PreferenceDataset,
    PreferencePair,
    build_response_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_pair(
    *,
    winner: str = "a",
    ctx_scale: float = 1.0,
    a_bias: float = 0.0,
    b_bias: float = 1.0,
    seed: int | None = None,
) -> PreferencePair:
    """Build a small synthetic :class:`PreferencePair`."""
    rng = np.random.default_rng(seed)
    ctx = (rng.uniform(0.0, 1.0, size=12) * ctx_scale).tolist()
    feat_a = (rng.uniform(0.0, 1.0, size=12) + a_bias).tolist()
    feat_b = (rng.uniform(0.0, 1.0, size=12) + b_bias).tolist()
    return PreferencePair(
        prompt="test prompt",
        response_a="resp a",
        response_b="resp b",
        winner=winner,
        context=ctx,
        response_a_features=feat_a,
        response_b_features=feat_b,
        user_id="test",
    )


def _easy_dataset(n: int = 64, seed: int = 0) -> PreferenceDataset:
    """A dataset where the winner is trivially inferable from the features."""
    rng = np.random.default_rng(seed)
    ds = PreferenceDataset()
    for i in range(n):
        ctx = rng.uniform(0.0, 1.0, size=12).tolist()
        # Make A always better on feature[0]; winner always "a".
        feat_a = [0.9] + rng.uniform(0.0, 0.2, size=11).tolist()
        feat_b = [0.1] + rng.uniform(0.0, 0.2, size=11).tolist()
        # 90% of the time A wins, 10% of the time we flip (label noise).
        winner = "a" if rng.random() < 0.9 else "b"
        ds.append(
            PreferencePair(
                prompt=f"p{i}",
                response_a="ra",
                response_b="rb",
                winner=winner,
                context=ctx,
                response_a_features=feat_a,
                response_b_features=feat_b,
                user_id="test",
            )
        )
    return ds


# ---------------------------------------------------------------------------
# PreferencePair validation
# ---------------------------------------------------------------------------


class TestPreferencePairValidation:
    """Invalid inputs to :class:`PreferencePair` raise :class:`ValueError`."""

    def test_invalid_winner_raises(self) -> None:
        with pytest.raises(ValueError, match="winner"):
            PreferencePair(
                prompt="p",
                response_a="a",
                response_b="b",
                winner="maybe",  # invalid
                context=[0.0] * 12,
                response_a_features=[0.0] * 12,
                response_b_features=[0.0] * 12,
            ).validate()

    def test_non_finite_context_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            PreferencePair(
                prompt="p",
                response_a="a",
                response_b="b",
                winner="a",
                context=[float("nan")] * 12,
                response_a_features=[0.0] * 12,
                response_b_features=[0.0] * 12,
            ).validate()

    def test_empty_feature_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            PreferencePair(
                prompt="p",
                response_a="a",
                response_b="b",
                winner="a",
                context=[0.0] * 12,
                response_a_features=[],
                response_b_features=[0.0] * 12,
            ).validate()

    def test_prompt_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt"):
            PreferencePair(
                prompt="x" * (9 * 1024),
                response_a="a",
                response_b="b",
                winner="a",
                context=[0.0] * 12,
                response_a_features=[0.0] * 12,
                response_b_features=[0.0] * 12,
            ).validate()


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------


class TestPreferenceDataset:
    """In-memory append, len, and filter-by-user."""

    def test_append_round_trip(self) -> None:
        ds = PreferenceDataset()
        pair = _mk_pair(seed=0)
        ds.append(pair)
        assert len(ds) == 1
        all_pairs = ds.all()
        assert len(all_pairs) == 1
        assert all_pairs[0].winner == "a"

    def test_filter_by_user_isolates(self) -> None:
        ds = PreferenceDataset()
        a = _mk_pair(seed=0)
        b = PreferencePair(
            prompt="p",
            response_a="a",
            response_b="b",
            winner="b",
            context=[0.0] * 12,
            response_a_features=[0.0] * 12,
            response_b_features=[0.0] * 12,
            user_id="someone_else",
        )
        ds.append(a)
        ds.append(b)
        assert len(ds.filter_by_user("test")) == 1
        assert len(ds.filter_by_user("someone_else")) == 1

    def test_append_rejects_invalid(self) -> None:
        ds = PreferenceDataset()
        bad = PreferencePair(
            prompt="p",
            response_a="a",
            response_b="b",
            winner="zzz",
            context=[0.0] * 12,
            response_a_features=[0.0] * 12,
            response_b_features=[0.0] * 12,
        )
        with pytest.raises(ValueError):
            ds.append(bad)
        assert len(ds) == 0

    @pytest.mark.asyncio
    async def test_sqlite_persist_roundtrip(self, tmp_path: Path) -> None:
        """Round-trip through aiosqlite persistence when available."""
        try:
            import aiosqlite  # noqa: F401  # availability check
        except ImportError:
            pytest.skip("aiosqlite not installed")
        db_path = tmp_path / "prefs.sqlite"
        ds = PreferenceDataset(db_path=db_path)
        ds.append(_mk_pair(seed=1, winner="a"))
        ds.append(_mk_pair(seed=2, winner="b"))
        await ds.persist()
        reopened = PreferenceDataset(db_path=db_path)
        await reopened.load()
        assert len(reopened) == 2


# ---------------------------------------------------------------------------
# Bradley-Terry reward model
# ---------------------------------------------------------------------------


class TestBradleyTerryRewardModel:
    """Constructor, forward shape, deterministic init."""

    def test_constructor_rejects_bad_dims(self) -> None:
        with pytest.raises(ValueError):
            BradleyTerryRewardModel(context_dim=0)
        with pytest.raises(ValueError):
            BradleyTerryRewardModel(response_dim=0)
        with pytest.raises(ValueError):
            BradleyTerryRewardModel(hidden_dim=0)

    def test_forward_shape(self) -> None:
        model = BradleyTerryRewardModel()
        ctx = torch.zeros(4, 12)
        resp = torch.zeros(4, 12)
        out = model(ctx, resp)
        assert out.shape == (4,)

    def test_forward_rejects_wrong_last_dim(self) -> None:
        model = BradleyTerryRewardModel()
        with pytest.raises(ValueError):
            model(torch.zeros(2, 5), torch.zeros(2, 12))
        with pytest.raises(ValueError):
            model(torch.zeros(2, 12), torch.zeros(2, 7))

    def test_score_is_scalar_float(self) -> None:
        model = BradleyTerryRewardModel()
        s = model.score([0.1] * 12, [0.2] * 12)
        assert isinstance(s, float)
        # clipped to [-10, 10]
        assert -10.0 - 1e-3 <= s <= 10.0 + 1e-3

    def test_trains_to_high_accuracy(self) -> None:
        """After training on an easy-decision dataset, accuracy > 0.8."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        ds = _easy_dataset(n=80, seed=42)
        model = BradleyTerryRewardModel()
        optim = DPOPreferenceOptimizer(model, learning_rate=1e-2)
        report = optim.fit(ds, n_epochs=60, seed=42)
        assert isinstance(report, DPOFitReport)
        assert report.val_accuracy >= 0.8

    def test_deterministic_under_fixed_seed(self) -> None:
        """Two training runs with the same seed yield the same weights."""
        ds = _easy_dataset(n=32, seed=7)

        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)
        m1 = BradleyTerryRewardModel()
        DPOPreferenceOptimizer(m1, learning_rate=1e-2).fit(ds, n_epochs=10, seed=7)
        w1 = m1.head.weight.detach().clone()

        torch.manual_seed(7)
        np.random.seed(7)
        random.seed(7)
        m2 = BradleyTerryRewardModel()
        DPOPreferenceOptimizer(m2, learning_rate=1e-2).fit(ds, n_epochs=10, seed=7)
        w2 = m2.head.weight.detach().clone()

        assert torch.allclose(w1, w2, atol=1e-6)


# ---------------------------------------------------------------------------
# DPOPreferenceOptimizer
# ---------------------------------------------------------------------------


class TestDPOPreferenceOptimizer:
    """Constructor validation and fit-report structure."""

    def test_invalid_kl_coef_raises(self) -> None:
        model = BradleyTerryRewardModel()
        with pytest.raises(ValueError):
            DPOPreferenceOptimizer(model, reference_policy_kl_coef=-0.1)

    def test_invalid_learning_rate_raises(self) -> None:
        model = BradleyTerryRewardModel()
        with pytest.raises(ValueError):
            DPOPreferenceOptimizer(model, learning_rate=0.0)

    def test_empty_dataset_raises(self) -> None:
        model = BradleyTerryRewardModel()
        optim = DPOPreferenceOptimizer(model)
        with pytest.raises(ValueError, match="empty"):
            optim.fit(PreferenceDataset(), n_epochs=1)

    def test_fit_returns_populated_report(self) -> None:
        model = BradleyTerryRewardModel()
        optim = DPOPreferenceOptimizer(model, learning_rate=5e-3)
        ds = _easy_dataset(n=32, seed=3)
        report = optim.fit(ds, n_epochs=5, seed=3)
        d = report.to_dict()
        assert d["n_pairs"] == 32
        assert d["epochs_run"] == 5
        assert d["elapsed_seconds"] >= 0.0
        # train_loss is finite after a few epochs
        assert np.isfinite(d["train_loss"])

    def test_zero_epochs_raises(self) -> None:
        model = BradleyTerryRewardModel()
        with pytest.raises(ValueError, match="n_epochs"):
            DPOPreferenceOptimizer(model).fit(_easy_dataset(n=4), n_epochs=0)


# ---------------------------------------------------------------------------
# ActivePreferenceSelector
# ---------------------------------------------------------------------------


class TestActivePreferenceSelector:
    """D-optimal ranking and Fisher update behaviour."""

    def test_select_rejects_invalid_n(self) -> None:
        model = BradleyTerryRewardModel()
        selector = ActivePreferenceSelector(model)
        with pytest.raises(ValueError, match="n"):
            selector.select_next_query([_mk_pair(seed=0)], n=0)

    def test_empty_candidates_returns_empty(self) -> None:
        selector = ActivePreferenceSelector(BradleyTerryRewardModel())
        assert selector.select_next_query([], n=1) == []

    def test_picks_higher_uncertainty_pair(self) -> None:
        """Pair with a *larger* last-layer feature gap wins.

        With an un-trained Bradley-Terry model the last-layer
        difference is essentially proportional to the raw feature
        difference, so a high-contrast pair dominates a low-contrast
        one.
        """
        torch.manual_seed(123)
        np.random.seed(123)
        model = BradleyTerryRewardModel()
        selector = ActivePreferenceSelector(model)

        # Low-contrast pair: responses are near-identical.
        low = PreferencePair(
            prompt="p",
            response_a="a",
            response_b="b",
            winner="tie",
            context=[0.5] * 12,
            response_a_features=[0.5] * 12,
            response_b_features=[0.5001] * 12,
        )
        # High-contrast pair: responses are far apart.
        high = PreferencePair(
            prompt="p",
            response_a="a",
            response_b="b",
            winner="tie",
            context=[0.5] * 12,
            response_a_features=[1.0] * 12,
            response_b_features=[0.0] * 12,
        )
        chosen = selector.select_next_query([low, high], n=1)
        assert len(chosen) == 1
        assert chosen[0] is high

    def test_register_labelled_updates_fisher(self) -> None:
        """After registering a label, the same pair has a *lower* score."""
        torch.manual_seed(5)
        model = BradleyTerryRewardModel()
        selector = ActivePreferenceSelector(model)
        pair = PreferencePair(
            prompt="p",
            response_a="a",
            response_b="b",
            winner="a",
            context=[0.5] * 12,
            response_a_features=[1.0] * 12,
            response_b_features=[0.0] * 12,
        )
        before = selector.score_pair(pair)
        # Register many copies to make the effect visible.
        for _ in range(10):
            selector.register_labelled(pair)
        after = selector.score_pair(pair)
        assert after < before

    def test_information_gain_threshold(self) -> None:
        selector = ActivePreferenceSelector(BradleyTerryRewardModel())
        pair = _mk_pair(seed=11)
        # With the initial ridge-only Fisher any non-trivial pair exceeds 0.
        assert selector.information_gain_threshold(pair, threshold=0.0)
        # A huge threshold should be unreachable.
        assert not selector.information_gain_threshold(pair, threshold=1e12)
        with pytest.raises(ValueError):
            selector.information_gain_threshold(pair, threshold=-1.0)


# ---------------------------------------------------------------------------
# build_response_features
# ---------------------------------------------------------------------------


class TestBuildResponseFeatures:
    """Length/latency/confidence are normalised to [0, 1]."""

    def test_normalisation(self) -> None:
        v = build_response_features(
            length_tokens=10_000.0,
            latency_ms=1_000_000.0,
            model_confidence=2.0,
        )
        assert len(v) == 12
        assert v[0] == 1.0  # length clipped
        assert v[1] == 1.0  # latency clipped
        assert v[2] == 1.0  # confidence clamped
        assert all(v[i] == 0.0 for i in range(3, 12))

    def test_negative_inputs_clamp_to_zero(self) -> None:
        v = build_response_features(
            length_tokens=-50.0,
            latency_ms=-5.0,
            model_confidence=-0.3,
        )
        assert v[0] == 0.0
        assert v[1] == 0.0
        assert v[2] == 0.0

    def test_rejects_zero_dim(self) -> None:
        with pytest.raises(ValueError):
            build_response_features(
                length_tokens=0.0,
                latency_ms=0.0,
                model_confidence=0.0,
                response_dim=0,
            )
