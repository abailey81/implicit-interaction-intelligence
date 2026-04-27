"""Iter 125 — SecureAggregator FedAvg + masking invariants."""
from __future__ import annotations

import numpy as np
import pytest

from i3.federated.aggregator import (
    MaskedUpdate,
    SecureAggregator,
    generate_shared_seed,
)


def test_mask_update_round_trips_under_pairwise_cancellation():
    """Two clients with a shared seed: each masks, server sums →
    masks cancel, leaving the unmasked sum."""
    params_a = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    params_b = [np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    seed = 12345

    masked_a = SecureAggregator.mask_update(
        client_id=0, parameters=params_a, peer_seeds={1: seed},
    )
    masked_b = SecureAggregator.mask_update(
        client_id=1, parameters=params_b, peer_seeds={0: seed},
    )

    # Server sums the masked updates.  With equal weight (each has
    # num_examples=1), the aggregator computes weighted mean.
    agg = SecureAggregator()
    agg.submit(MaskedUpdate(client_id=0, masked_parameters=masked_a,
                             num_examples=1, peer_seeds={1: seed}))
    agg.submit(MaskedUpdate(client_id=1, masked_parameters=masked_b,
                             num_examples=1, peer_seeds={0: seed}))
    out = agg.aggregate()
    expected_mean = (params_a[0] + params_b[0]) / 2.0
    np.testing.assert_allclose(out[0], expected_mean, atol=1e-5)


def test_aggregate_weights_by_num_examples_unmasked():
    """Unmasked update with equal weights is a vanilla weighted mean."""
    params_a = [np.array([0.0], dtype=np.float32)]
    params_b = [np.array([10.0], dtype=np.float32)]
    agg = SecureAggregator()
    agg.submit(MaskedUpdate(client_id=0, masked_parameters=params_a,
                             num_examples=1))
    agg.submit(MaskedUpdate(client_id=1, masked_parameters=params_b,
                             num_examples=9))
    out = agg.aggregate()
    # Weighted mean: (1*0 + 9*10) / 10 = 9.0
    np.testing.assert_allclose(out[0], np.array([9.0]), atol=1e-5)


def test_aggregate_raises_on_no_updates():
    agg = SecureAggregator()
    with pytest.raises(ValueError):
        agg.aggregate()


def test_aggregate_raises_on_zero_total_examples():
    params = [np.zeros(3, dtype=np.float32)]
    agg = SecureAggregator()
    agg.submit(MaskedUpdate(client_id=0, masked_parameters=params,
                             num_examples=0))
    with pytest.raises(ValueError):
        agg.aggregate()


def test_aggregate_raises_on_dropouts_in_sketch():
    params = [np.zeros(3, dtype=np.float32)]
    agg = SecureAggregator()
    agg.submit(MaskedUpdate(client_id=0, masked_parameters=params,
                             num_examples=1))
    with pytest.raises(NotImplementedError):
        agg.aggregate(dropped_clients={1})


def test_clear_resets_pending():
    params = [np.zeros(3, dtype=np.float32)]
    agg = SecureAggregator()
    agg.submit(MaskedUpdate(client_id=0, masked_parameters=params,
                             num_examples=1))
    agg.clear()
    with pytest.raises(ValueError):
        agg.aggregate()


def test_generate_shared_seed_returns_int():
    s = generate_shared_seed()
    assert isinstance(s, int)


def test_generate_shared_seed_unique_with_high_probability():
    seeds = {generate_shared_seed() for _ in range(50)}
    # Should be 50 distinct integers
    assert len(seeds) == 50
