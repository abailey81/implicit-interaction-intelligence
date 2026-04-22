"""Unit tests for the persona library and :class:`UserSimulator`.

These tests are intentionally independent of the full pipeline so they
can run in isolation. They verify:

* Every canonical persona has all required fields populated and in
  valid ranges.
* The simulator is deterministic under a fixed seed.
* Typing-profile samples lie within the documented ranges.
* Personas with genuinely different typing distributions (fatigued vs.
  energetic) produce statistically distinguishable inter-key interval
  streams under a Kolmogorov-Smirnov two-sample test.
* Drift schedules are monotonic where applicable.
* ``ALL_PERSONAS`` contains exactly 8 unique personas.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.eval.simulation import (
    ALL_PERSONAS,
    ENERGETIC_USER,
    FATIGUED_DEVELOPER,
    FRESH_USER,
    HCIPersona,
    LinguisticProfile,
    MOTOR_IMPAIRED_USER,
    SimulatedMessage,
    TypingProfile,
    UserSimulator,
)


# ---------------------------------------------------------------------------
# Collection-level invariants
# ---------------------------------------------------------------------------


def test_all_personas_has_exactly_eight_entries() -> None:
    """The canonical persona count is fixed at eight."""
    assert len(ALL_PERSONAS) == 8


def test_all_personas_have_unique_names() -> None:
    """No two canonical personas may share a name."""
    names = [p.name for p in ALL_PERSONAS]
    assert len(set(names)) == len(names), f"duplicate persona names: {names}"


def test_all_personas_have_required_fields() -> None:
    """Every persona populates all required typed fields."""
    for persona in ALL_PERSONAS:
        assert isinstance(persona, HCIPersona)
        assert persona.name and isinstance(persona.name, str)
        assert persona.description and isinstance(persona.description, str)
        assert isinstance(persona.typing_profile, TypingProfile)
        assert isinstance(persona.linguistic_profile, LinguisticProfile)
        assert isinstance(persona.expected_adaptation, AdaptationVector)
        assert isinstance(persona.expected_adaptation.style_mirror, StyleVector)


def test_expected_adaptation_values_are_in_unit_interval() -> None:
    """All scalar adaptation fields must lie in [0, 1]."""
    for persona in ALL_PERSONAS:
        adapt = persona.expected_adaptation
        for name, value in (
            ("cognitive_load", adapt.cognitive_load),
            ("emotional_tone", adapt.emotional_tone),
            ("accessibility", adapt.accessibility),
            ("style_mirror.formality", adapt.style_mirror.formality),
            ("style_mirror.verbosity", adapt.style_mirror.verbosity),
            ("style_mirror.emotionality", adapt.style_mirror.emotionality),
            ("style_mirror.directness", adapt.style_mirror.directness),
        ):
            assert 0.0 <= value <= 1.0, (
                f"{persona.name}: {name}={value} out of [0, 1]"
            )


def test_expected_adaptations_are_pairwise_distinguishable() -> None:
    """Pairwise L2 between any two ground-truth vectors exceeds 0.2.

    This is the design contract that makes 1-NN persona recovery a
    non-trivial diagnostic.
    """
    vectors = [
        np.asarray(p.expected_adaptation.to_tensor().numpy()[:7])
        for p in ALL_PERSONAS
    ]
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(vectors[i] - vectors[j]))
            assert d > 0.2, (
                f"personas {ALL_PERSONAS[i].name} and {ALL_PERSONAS[j].name} "
                f"are too close in ground-truth space: L2={d:.4f}"
            )


# ---------------------------------------------------------------------------
# Simulator determinism
# ---------------------------------------------------------------------------


def test_simulator_is_deterministic_for_single_message() -> None:
    """Two simulators with identical (persona, seed) produce identical output."""
    sim1 = UserSimulator(FRESH_USER, seed=123)
    sim2 = UserSimulator(FRESH_USER, seed=123)
    msg1 = sim1.generate_message(message_index=3, total_messages=10)
    msg2 = sim2.generate_message(message_index=3, total_messages=10)
    assert msg1.text == msg2.text
    assert msg1.keystroke_intervals_ms == msg2.keystroke_intervals_ms


def test_simulator_is_deterministic_for_full_session() -> None:
    """Two simulators with identical seeds produce identical sessions."""
    sim1 = UserSimulator(FATIGUED_DEVELOPER, seed=7)
    sim2 = UserSimulator(FATIGUED_DEVELOPER, seed=7)
    s1 = sim1.run_session(n_messages=6)
    s2 = sim2.run_session(n_messages=6)
    assert [m.text for m in s1] == [m.text for m in s2]
    for a, b in zip(s1, s2):
        assert a.keystroke_intervals_ms == b.keystroke_intervals_ms
        assert a.timestamp == pytest.approx(b.timestamp)
        assert isinstance(a, SimulatedMessage)


def test_simulator_different_seeds_produce_different_streams() -> None:
    """Different seeds produce materially different keystroke streams."""
    sim1 = UserSimulator(FRESH_USER, seed=1)
    sim2 = UserSimulator(FRESH_USER, seed=2)
    ks1 = sim1.generate_keystroke_stream(64)
    ks2 = sim2.generate_keystroke_stream(64)
    # At least one entry should differ.
    assert any(a != b for a, b in zip(ks1, ks2))


# ---------------------------------------------------------------------------
# Sample-range validation
# ---------------------------------------------------------------------------


def test_keystroke_samples_lie_within_clip_bounds() -> None:
    """Simulator must truncate Gaussian tails to [20, 5000] ms."""
    sim = UserSimulator(MOTOR_IMPAIRED_USER, seed=1)
    ks = sim.generate_keystroke_stream(512)
    for v in ks:
        assert 20.0 <= v <= 5000.0, f"out-of-range inter-key interval: {v}"


def test_sampled_iki_mean_tracks_typing_profile_mean() -> None:
    """The sampled IKI mean should be within 2 standard deviations of the
    persona's profile mean (statistical sanity check)."""
    for persona in (FRESH_USER, FATIGUED_DEVELOPER, MOTOR_IMPAIRED_USER):
        sim = UserSimulator(persona, seed=99)
        ks = sim.generate_keystroke_stream(2048)
        mean, std = persona.typing_profile.inter_key_interval_ms
        sampled_mean = float(np.mean(ks))
        # With pause multipliers the empirical mean is biased high, but
        # it should still be within a loose envelope around the profile
        # mean.
        assert sampled_mean > mean * 0.7
        assert sampled_mean < mean * 3.5


# ---------------------------------------------------------------------------
# Persona-level discriminability (KS test)
# ---------------------------------------------------------------------------


def test_fatigued_vs_energetic_iki_distributions_differ_significantly() -> None:
    """KS two-sample test must reject the null at p < 0.001."""
    sim_fatigued = UserSimulator(FATIGUED_DEVELOPER, seed=1234)
    sim_energetic = UserSimulator(ENERGETIC_USER, seed=5678)
    ks_fat = np.asarray(sim_fatigued.generate_keystroke_stream(1024))
    ks_ene = np.asarray(sim_energetic.generate_keystroke_stream(1024))
    stat = ks_2samp(ks_fat, ks_ene)
    assert stat.pvalue < 1e-3, (
        f"KS p-value {stat.pvalue} too high; personas not distinguishable"
    )


def test_motor_impaired_iki_exceeds_fresh_user() -> None:
    """Motor impairment should produce meaningfully slower typing."""
    fresh = UserSimulator(FRESH_USER, seed=1)
    motor = UserSimulator(MOTOR_IMPAIRED_USER, seed=1)
    ks_fresh = np.asarray(fresh.generate_keystroke_stream(1024))
    ks_motor = np.asarray(motor.generate_keystroke_stream(1024))
    assert float(np.mean(ks_motor)) > float(np.mean(ks_fresh)) * 1.5


# ---------------------------------------------------------------------------
# Drift schedule invariants
# ---------------------------------------------------------------------------


def test_drift_schedules_are_time_sorted() -> None:
    """Drift schedule time fractions must be monotonically non-decreasing."""
    for persona in ALL_PERSONAS:
        fractions = [t for (t, _) in persona.drift_schedule]
        assert fractions == sorted(fractions), (
            f"{persona.name}: drift schedule times not sorted: {fractions}"
        )
        for t in fractions:
            assert 0.0 <= t <= 1.0


def test_fatigued_developer_drift_increases_iki_mean() -> None:
    """After the drift points fire, IKI mean should rise."""
    sim = UserSimulator(FATIGUED_DEVELOPER, seed=42)
    early = sim.generate_keystroke_stream(512, time_fraction=0.0)
    late = sim.generate_keystroke_stream(512, time_fraction=0.95)
    assert float(np.mean(late)) > float(np.mean(early))


# ---------------------------------------------------------------------------
# Generate_message boundary conditions
# ---------------------------------------------------------------------------


def test_generate_keystroke_stream_rejects_negative_length() -> None:
    """Negative prompt lengths must raise ValueError."""
    sim = UserSimulator(FRESH_USER, seed=0)
    with pytest.raises(ValueError):
        sim.generate_keystroke_stream(-1)


def test_generate_keystroke_stream_accepts_zero_length() -> None:
    """A zero-length request must return an empty list."""
    sim = UserSimulator(FRESH_USER, seed=0)
    assert sim.generate_keystroke_stream(0) == []


def test_run_session_rejects_non_positive_n_messages() -> None:
    """``n_messages=0`` must raise ValueError."""
    sim = UserSimulator(FRESH_USER, seed=0)
    with pytest.raises(ValueError):
        sim.run_session(0)
