"""Snapshot tests for the adaptation controller output.

These tests pin the *exact* AdaptationVector the controller produces for
a handful of canonical input fixtures.  A change in controller behaviour
(even a small one) shows up immediately as a snapshot diff, forcing a
deliberate review rather than silently drifting.

Run ``pytest --snapshot-update tests/snapshot/test_adaptation_snapshots.py``
to regenerate the committed snapshots after a legitimate behavioural
change.
"""

from __future__ import annotations

import pytest


syrupy = pytest.importorskip("syrupy")


# ─────────────────────────────────────────────────────────────────────────
#  Deterministic fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Snapshot reproducibility requires seeded RNGs."""
    import random

    import numpy as np
    import torch

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)


@pytest.fixture
def controller():
    try:
        from i3.adaptation.controller import AdaptationController
        from i3.config import load_config
    except Exception as exc:
        pytest.skip(f"Controller import failed: {exc}")
    cfg = load_config("configs/default.yaml", set_seeds=False)
    return AdaptationController(cfg.adaptation)


@pytest.fixture
def baseline_features():
    """Neutral feature vector — all zeros."""
    from i3.interaction.types import InteractionFeatureVector

    return InteractionFeatureVector.zeros()


@pytest.fixture
def engaged_features():
    """Linguistically sophisticated, calm typing."""
    from i3.interaction.types import InteractionFeatureVector

    return InteractionFeatureVector(
        mean_iki=0.3,
        std_iki=0.1,
        mean_burst_length=0.6,
        mean_pause_duration=0.2,
        backspace_ratio=0.05,
        composition_speed=0.7,
        pause_before_send=0.1,
        editing_effort=0.05,
        message_length=0.5,
        type_token_ratio=0.7,
        mean_word_length=0.6,
        flesch_kincaid=0.5,
        question_ratio=0.2,
        formality=0.6,
        emoji_density=0.0,
        sentiment_valence=0.3,
    )


@pytest.fixture
def distressed_features():
    """High edit rate, slow typing, negative sentiment."""
    from i3.interaction.types import InteractionFeatureVector

    return InteractionFeatureVector(
        mean_iki=0.8,
        std_iki=0.4,
        mean_burst_length=0.2,
        mean_pause_duration=0.7,
        backspace_ratio=0.4,
        composition_speed=0.2,
        pause_before_send=0.6,
        editing_effort=0.5,
        message_length=0.3,
        sentiment_valence=-0.5,
    )


@pytest.fixture
def zero_deviation():
    from i3.user_model.types import DeviationMetrics

    return DeviationMetrics(
        current_vs_baseline=0.0,
        current_vs_session=0.0,
        engagement_score=0.5,
        magnitude=0.0,
        iki_deviation=0.0,
        length_deviation=0.0,
        vocab_deviation=0.0,
        formality_deviation=0.0,
        speed_deviation=0.0,
        engagement_deviation=0.0,
        complexity_deviation=0.0,
        pattern_deviation=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Adaptation-vector snapshots
# ─────────────────────────────────────────────────────────────────────────


def _serialise(vec) -> dict:
    """Produce a stable, float-rounded dict for snapshotting."""
    d = vec.to_dict()
    # Round every nested float so CPU non-determinism in the 15th digit
    # cannot flake the snapshot.
    def _round(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        return obj

    return _round(d)


class TestAdaptationSnapshots:
    def test_baseline_adaptation(
        self, snapshot, controller, baseline_features, zero_deviation
    ) -> None:
        vec = controller.compute(baseline_features, zero_deviation)
        assert _serialise(vec) == snapshot

    def test_engaged_adaptation(
        self, snapshot, controller, engaged_features, zero_deviation
    ) -> None:
        vec = controller.compute(engaged_features, zero_deviation)
        assert _serialise(vec) == snapshot

    def test_distressed_adaptation(
        self, snapshot, controller, distressed_features, zero_deviation
    ) -> None:
        vec = controller.compute(distressed_features, zero_deviation)
        assert _serialise(vec) == snapshot

    def test_repeated_calls_are_stable(
        self, snapshot, controller, baseline_features, zero_deviation
    ) -> None:
        """The controller is stateful (style mirror is smoothed) — pin the
        trajectory across three calls so drift is caught immediately."""
        trajectory: list[dict] = []
        for _ in range(3):
            v = controller.compute(baseline_features, zero_deviation)
            trajectory.append(_serialise(v))
        assert trajectory == snapshot
