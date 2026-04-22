"""Tests for :class:`PersonaTaskGenerator`.

Covers:
* Validation of constructor arguments.
* Determinism under a fixed seed.
* Support + query sizes match the requested cardinalities.
* Ground-truth target matches the persona's expected_adaptation.
* Different personas produce statistically-distinguishable IKI features.
* Round-robin sampling covers all personas.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from i3.eval.simulation.personas import (
    ALL_PERSONAS,
    ENERGETIC_USER,
    FATIGUED_DEVELOPER,
    FRESH_USER,
)
from i3.meta_learning.maml import MetaBatch, MetaTask
from i3.meta_learning.task_generator import PersonaTaskGenerator


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_empty_personas_raises() -> None:
    """An empty persona list raises ``ValueError``."""
    with pytest.raises(ValueError):
        PersonaTaskGenerator(personas=[])


def test_invalid_support_size_raises() -> None:
    """support_size < 1 raises ``ValueError``."""
    with pytest.raises(ValueError):
        PersonaTaskGenerator(
            personas=[FRESH_USER], support_size=0, query_size=1
        )


def test_invalid_query_size_raises() -> None:
    """query_size < 1 raises ``ValueError``."""
    with pytest.raises(ValueError):
        PersonaTaskGenerator(
            personas=[FRESH_USER], support_size=1, query_size=0
        )


def test_invalid_meta_batch_size_raises() -> None:
    """generate_batch rejects meta_batch_size < 1."""
    gen = PersonaTaskGenerator(personas=[FRESH_USER])
    with pytest.raises(ValueError):
        gen.generate_batch(meta_batch_size=0)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_generate_task_sizes() -> None:
    """Generated task has the requested support and query cardinalities."""
    gen = PersonaTaskGenerator(
        personas=[FRESH_USER], support_size=2, query_size=4, seed=0
    )
    task = gen.generate_task()
    assert isinstance(task, MetaTask)
    assert len(task.support_set) == 2
    assert len(task.query_set) == 4


def test_generate_task_target_matches_persona() -> None:
    """The ground-truth target equals the persona's expected_adaptation."""
    gen = PersonaTaskGenerator(personas=[FATIGUED_DEVELOPER], seed=0)
    task = gen.generate_task()
    assert task.target_adaptation == FATIGUED_DEVELOPER.expected_adaptation


def test_generate_batch_size() -> None:
    """generate_batch returns a MetaBatch of the requested size."""
    gen = PersonaTaskGenerator(personas=ALL_PERSONAS, seed=0)
    batch = gen.generate_batch(meta_batch_size=5)
    assert isinstance(batch, MetaBatch)
    assert len(batch) == 5


def test_round_robin_covers_all_personas() -> None:
    """Successive tasks cycle through every persona in order."""
    gen = PersonaTaskGenerator(personas=ALL_PERSONAS, seed=0)
    seen: list[str] = []
    for _ in range(len(ALL_PERSONAS)):
        seen.append(gen.generate_task().persona_name)
    assert set(seen) == {p.name for p in ALL_PERSONAS}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_under_fixed_seed() -> None:
    """Two generators with the same seed produce identical task streams."""
    a = PersonaTaskGenerator(
        personas=ALL_PERSONAS, support_size=2, query_size=2, seed=42
    )
    b = PersonaTaskGenerator(
        personas=ALL_PERSONAS, support_size=2, query_size=2, seed=42
    )
    for _ in range(3):
        task_a = a.generate_task()
        task_b = b.generate_task()
        assert task_a.persona_name == task_b.persona_name
        # Feature vectors should match bit-for-bit.
        for fv_a, fv_b in zip(task_a.support_set, task_b.support_set):
            assert fv_a.to_tensor().tolist() == fv_b.to_tensor().tolist()


def test_reset_rewinds_counter() -> None:
    """reset() causes the next task to match the generator's first output."""
    gen = PersonaTaskGenerator(personas=ALL_PERSONAS, seed=1)
    first = gen.generate_task()
    gen.generate_task()
    gen.reset()
    again = gen.generate_task()
    assert first.persona_name == again.persona_name
    for fv_a, fv_b in zip(first.support_set, again.support_set):
        assert fv_a.to_tensor().tolist() == fv_b.to_tensor().tolist()


# ---------------------------------------------------------------------------
# Distributional test: different personas -> different IKI marginals.
# ---------------------------------------------------------------------------


def test_different_personas_have_distinguishable_iki() -> None:
    """Energetic vs Fatigued tasks produce distinguishable mean_iki features.

    KS test on the generator's ``mean_iki`` feature across a batch of
    tasks. Personas chosen for maximal separability in IKI space.
    """
    gen_energetic = PersonaTaskGenerator(
        personas=[ENERGETIC_USER],
        support_size=5,
        query_size=5,
        seed=7,
    )
    gen_fatigued = PersonaTaskGenerator(
        personas=[FATIGUED_DEVELOPER],
        support_size=5,
        query_size=5,
        seed=7,
    )
    energetic_ikis: list[float] = []
    fatigued_ikis: list[float] = []
    for _ in range(8):
        t_e = gen_energetic.generate_task()
        t_f = gen_fatigued.generate_task()
        energetic_ikis.extend(fv.mean_iki for fv in t_e.support_set)
        fatigued_ikis.extend(fv.mean_iki for fv in t_f.support_set)
    stat, p = ks_2samp(np.array(energetic_ikis), np.array(fatigued_ikis))
    # The personas have an ~2.5x IKI gap so the KS test should be
    # comfortably significant.
    assert p < 0.05


def test_iterator_yields_meta_batches() -> None:
    """Iterating a generator yields MetaBatch objects sized by persona list."""
    gen = PersonaTaskGenerator(
        personas=[FRESH_USER, ENERGETIC_USER],
        support_size=1,
        query_size=1,
        seed=0,
    )
    it = iter(gen)
    batch = next(it)
    assert isinstance(batch, MetaBatch)
    assert len(batch) == 2
