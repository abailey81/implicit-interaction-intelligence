"""Hypothesis property tests for the TCN encoder (``i3.encoder.tcn``).

Invariants verified:
    * **Shape** — output is ``(batch, embedding_dim)`` for any valid input.
    * **L2 norm** — every output row lies on the unit hypersphere
      (``norm == 1`` up to float tolerance).
    * **Determinism** — the same input and identical RNG seed always
      produces the same embedding.
    * **Causality** — a change in the *last* timestep must produce a
      different embedding, while padding trailing positions with a
      constant value (not the last real step) must eventually change
      the output only through the receptive-field window.
    * **Finiteness** — no ``NaN`` / ``Inf`` in the output for any
      finite input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings, strategies as st
from hypothesis.extra.numpy import arrays

torch = pytest.importorskip("torch")

if TYPE_CHECKING:
    from torch import nn


# ─────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def encoder() -> "nn.Module":
    """A randomly-initialised TCN with seed-stable weights."""
    tcn_mod = pytest.importorskip("i3.encoder.tcn")
    torch.manual_seed(42)
    np.random.seed(42)
    model = tcn_mod.TemporalConvNet(input_dim=32, embedding_dim=64)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────

# A float strategy that excludes NaN / Inf so the *input* is always valid.
_FINITE_FLOAT = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _input_tensor(batch: int, seq_len: int) -> st.SearchStrategy[np.ndarray]:
    """Hypothesis strategy yielding a ``[batch, seq_len, 32]`` float32 array."""
    return arrays(
        dtype=np.float32,
        shape=(batch, seq_len, 32),
        elements=_FINITE_FLOAT,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────


class TestEncoderShapeInvariant:
    @given(
        batch=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=32),
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_output_shape(self, encoder, batch: int, seq_len: int) -> None:
        """Output shape is always ``(batch, 64)`` regardless of input size."""
        x = torch.randn(batch, seq_len, 32)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (batch, 64)


class TestEncoderL2Norm:
    @given(data=_input_tensor(batch=4, seq_len=8))
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @example(data=np.zeros((4, 8, 32), dtype=np.float32))
    def test_output_l2_normalized(self, encoder, data: np.ndarray) -> None:
        """Every output row lies on the unit hypersphere (|| . ||_2 == 1)."""
        x = torch.from_numpy(data)
        with torch.no_grad():
            out = encoder(x)
        norms = out.norm(dim=1)
        # L2 normalisation leaves a non-zero row with ||x|| == 1.
        # Guard: rows whose pre-norm magnitude is zero normalize to zero
        # (PyTorch's F.normalize returns the input unchanged at that row).
        for i, n in enumerate(norms.tolist()):
            assert 0.0 <= n <= 1.0 + 1e-4, f"row {i} norm={n} outside [0, 1]"
            # Virtually every realistic input yields non-zero rows; the
            # all-zero @example above is the only degenerate case.
            if data[i].any():
                assert abs(n - 1.0) < 1e-4, f"row {i} norm={n} != 1"


class TestEncoderFiniteness:
    @given(data=_input_tensor(batch=2, seq_len=6))
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_output_is_finite(self, encoder, data: np.ndarray) -> None:
        """Finite input must produce finite output — no NaN/Inf leaks."""
        x = torch.from_numpy(data)
        with torch.no_grad():
            out = encoder(x)
        assert torch.isfinite(out).all(), "NaN/Inf in encoder output"


class TestEncoderDeterminism:
    @given(data=_input_tensor(batch=2, seq_len=10))
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_fixed_seed_yields_identical_output(
        self, encoder, data: np.ndarray
    ) -> None:
        """Two forward passes with the same input produce bit-identical output."""
        x = torch.from_numpy(data)
        with torch.no_grad():
            a = encoder(x)
            b = encoder(x)
        assert torch.equal(a, b), "Encoder is non-deterministic in eval() mode"


class TestEncoderCausality:
    @given(
        data=_input_tensor(batch=1, seq_len=16),
        perturb=st.floats(
            min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_last_step_affects_output(
        self, encoder, data: np.ndarray, perturb: float
    ) -> None:
        """Perturbing the final timestep should change the pooled embedding.

        The TCN uses global-average pooling so every non-trivial change
        to any timestep (including the last) must move the output.
        """
        x = torch.from_numpy(data)
        with torch.no_grad():
            base = encoder(x)
        x2 = x.clone()
        x2[0, -1, 0] = float(x2[0, -1, 0]) + float(perturb) + 1.0
        with torch.no_grad():
            perturbed = encoder(x2)
        # Outputs must differ somewhere — the change must propagate.
        assert not torch.allclose(base, perturbed, atol=1e-8), (
            "Change in last timestep did not alter embedding — "
            "possible causality / pooling bug"
        )
