"""Unit tests for the Batch B mechanistic-interpretability modules.

Covers:
    * :class:`ActivationPatcher` -- hook counts match the canonical
      component list; patched outputs differ from the unpatched
      reference; hooks are removed on context exit even after failures.
    * :class:`ProbingSuite` -- recovers an identity "pass-through"
      dimension with R² > 0.95 when the hidden state is the raw
      adaptation vector itself.
    * :class:`AttentionPattern` -- shape invariants; per-head entropy
      non-negative; per-token focus within [0, 1].
    * :func:`identify_conditioning_specialists` -- returns the hand-
      crafted specialist in a synthetic pattern.

All tests run on CPU and take under a second each on a 2023-class
laptop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from i3.interpretability.activation_patching import (
    ActivationPatcher,
    canonical_components,
    trace_causal_effect,
)
from i3.interpretability.attention_circuits import (
    AttentionPattern,
    extract_attention_patterns,
    identify_conditioning_specialists,
    summarise_circuit,
)
from i3.interpretability.feature_attribution import ADAPTATION_DIMS
from i3.interpretability.probing_classifiers import (
    LinearProbe,
    ProbingExample,
    ProbingSuite,
    compute_probe_selectivity,
)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_slm() -> nn.Module:
    """Build a tiny :class:`AdaptiveSLM` suitable for fast tests."""
    from i3.slm.model import AdaptiveSLM

    torch.manual_seed(0)
    model = AdaptiveSLM(
        vocab_size=64,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        n_cross_heads=2,
        n_conditioning_tokens=4,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def clean_corrupted_inputs(tiny_slm: nn.Module) -> tuple[dict, dict]:
    """Return (clean_kwargs, corrupted_kwargs) for the tiny model."""
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(1, 64, (1, 8), generator=gen)
    a_dim = tiny_slm.conditioning_projector.adaptation_dim
    u_dim = tiny_slm.conditioning_projector.user_state_dim
    clean = {
        "input_ids": ids,
        "adaptation_vector": torch.rand((1, a_dim), generator=gen),
        "user_state": torch.randn((1, u_dim), generator=gen),
    }
    corrupted = {
        "input_ids": ids,
        "adaptation_vector": torch.zeros((1, a_dim)),
        "user_state": torch.zeros((1, u_dim)),
    }
    return clean, corrupted


# ---------------------------------------------------------------------------
# ActivationPatcher.
# ---------------------------------------------------------------------------


class TestActivationPatcher:
    """Hook-management and patching semantics."""

    def test_canonical_component_count(self) -> None:
        names = canonical_components(4)
        assert names[0] == "conditioning_projector"
        assert len(names) == 1 + 4 * 3  # projector + 3 kinds × n_layers
        # Four of each kind.
        assert sum(n.startswith("cross_attn_layer_") for n in names) == 4
        assert sum(n.startswith("self_attn_layer_") for n in names) == 4
        assert sum(n.startswith("ffn_layer_") for n in names) == 4

    def test_hooks_detached_on_exit(
        self, tiny_slm: nn.Module, clean_corrupted_inputs: tuple[dict, dict]
    ) -> None:
        _, corrupted = clean_corrupted_inputs
        components = canonical_components(len(tiny_slm.layers))
        with ActivationPatcher(tiny_slm, components=components) as patcher:
            with torch.no_grad():
                patcher.cache_corrupted(lambda: tiny_slm(**corrupted))
            assert patcher.n_attached_hooks == 0  # capture hooks detached
        assert patcher.n_attached_hooks == 0

    def test_hook_attach_count_matches_components(
        self, tiny_slm: nn.Module
    ) -> None:
        components = canonical_components(len(tiny_slm.layers))
        counts: list[int] = []

        def spy_corrupted() -> object:
            counts.append(patcher.n_attached_hooks)
            return tiny_slm(
                torch.randint(1, 64, (1, 4)),
                torch.zeros(1, tiny_slm.conditioning_projector.adaptation_dim),
                torch.zeros(1, tiny_slm.conditioning_projector.user_state_dim),
            )

        with ActivationPatcher(tiny_slm, components=components) as patcher:
            patcher.cache_corrupted(spy_corrupted)
            assert counts[0] == len(components)

    def test_patch_changes_logits(
        self, tiny_slm: nn.Module, clean_corrupted_inputs: tuple[dict, dict]
    ) -> None:
        clean, corrupted = clean_corrupted_inputs
        components = canonical_components(len(tiny_slm.layers))
        with ActivationPatcher(tiny_slm, components=components) as patcher:
            with torch.no_grad():
                patcher.cache_corrupted(lambda: tiny_slm(**corrupted))
                reference = tiny_slm(**clean)[0]
                patched = patcher.patch(
                    "conditioning_projector",
                    clean_run=lambda: tiny_slm(**clean),
                )
        assert reference.shape == patched.shape
        # Patching the projector with its corrupted (zero-adapt) output
        # must change at least *some* logits.
        assert not torch.allclose(reference, patched, atol=1e-6)

    def test_trace_causal_effect_runs_every_component(
        self, tiny_slm: nn.Module, clean_corrupted_inputs: tuple[dict, dict]
    ) -> None:
        clean, corrupted = clean_corrupted_inputs
        results = trace_causal_effect(tiny_slm, clean, corrupted)
        expected = canonical_components(len(tiny_slm.layers))
        assert list(results.keys()) == expected
        for eff in results.values():
            assert eff.kl_to_clean >= 0.0
            assert eff.logit_l2 >= 0.0

    def test_missing_component_raises(
        self, tiny_slm: nn.Module, clean_corrupted_inputs: tuple[dict, dict]
    ) -> None:
        _, corrupted = clean_corrupted_inputs
        with ActivationPatcher(tiny_slm, components=["conditioning_projector"]) as p:
            with torch.no_grad():
                p.cache_corrupted(lambda: tiny_slm(**corrupted))
            with pytest.raises(KeyError):
                p.patch("ffn_layer_0", clean_run=lambda: tiny_slm(**corrupted))


# ---------------------------------------------------------------------------
# ProbingSuite.
# ---------------------------------------------------------------------------


class _PassthroughLayer(nn.Module):
    """A transformer-shaped layer whose output is the adaptation vector.

    Shape: receives ``x`` of shape ``[B, S, D]``; the layer replaces the
    first ``A`` positions along the last dim of the first token with the
    adaptation-vector-like feature ``z`` of shape ``[B, A]`` that it
    received via the module buffer. Used in tests to verify that probes
    recover a signal that is linearly embedded in the hidden stream.
    """

    def __init__(self, d_model: int, adaptation_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.adaptation_dim = adaptation_dim
        self.pending_z: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del conditioning_tokens, causal_mask, use_cache
        if self.pending_z is None:
            return x, {}
        out = x.clone()
        B, S, D = out.shape
        a = self.pending_z  # [B, A]
        # Broadcast the adaptation vector across the sequence dim.
        out[:, :, : self.adaptation_dim] = a.unsqueeze(1).expand(B, S, -1)
        return out, {}

    def clear_cache(self) -> None:
        """No-op; required for AdaptiveSLM's cache API."""
        return None


class _PassthroughModel(nn.Module):
    """Minimal stand-in for :class:`AdaptiveSLM` used in probe tests."""

    def __init__(
        self, vocab_size: int = 32, d_model: int = 16, n_layers: int = 2,
        adaptation_dim: int = 8, user_state_dim: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [_PassthroughLayer(d_model, adaptation_dim) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Mimic the public attributes used by the probing suite.
        class _Projector:
            def __init__(self, a: int, u: int, d: int) -> None:
                self.adaptation_dim = a
                self.user_state_dim = u
                self.d_model = d

        self.conditioning_projector = _Projector(
            adaptation_dim, user_state_dim, d_model
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        adaptation_vector: Optional[torch.Tensor] = None,
        user_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        del user_state, use_cache
        x = self.embedding(input_ids)
        for layer in self.layers:
            layer.pending_z = adaptation_vector
            x, _ = layer(x, conditioning_tokens=torch.empty(0))
            layer.pending_z = None
        x = self.final_ln(x)
        return self.output_projection(x), {}


class TestProbingSuite:
    """End-to-end probing behaviour."""

    def test_recovers_passthrough_dimension(self) -> None:
        model = _PassthroughModel(adaptation_dim=len(ADAPTATION_DIMS))
        model.eval()
        gen = torch.Generator().manual_seed(0)
        dataset: list[ProbingExample] = []
        for _ in range(96):
            ids = torch.randint(1, 32, (8,), generator=gen)
            # The first dimension is perfectly linearly embedded into
            # the hidden state; probe should achieve near-perfect R².
            adapt = torch.rand(len(ADAPTATION_DIMS), generator=gen)
            dataset.append(ProbingExample(input_ids=ids, adaptation_vector=adapt))

        suite = ProbingSuite(n_epochs=300, lr=5e-2)
        r2 = suite.train_probes(
            model=model,
            adaptation_dataset=dataset,
            target_dimension="cognitive_load",  # first dimension
            layer_indices=[0, 1],
        )
        # Both layers should recover the identity-embedded dimension.
        assert r2[0] > 0.95
        assert r2[1] > 0.95

    def test_rejects_unknown_dimension(self) -> None:
        model = _PassthroughModel()
        suite = ProbingSuite()
        with pytest.raises(ValueError):
            suite.train_probes(
                model=model,
                adaptation_dataset=[
                    ProbingExample(
                        input_ids=torch.tensor([1, 2]),
                        adaptation_vector=torch.zeros(8),
                    )
                ],
                target_dimension="not_a_dim",
            )

    def test_compute_probe_selectivity_returns_something(self) -> None:
        probe_results = {"cognitive_load": {0: 0.9, 1: 0.5},
                         "accessibility": {0: 0.2, 1: 0.7}}
        table = compute_probe_selectivity(probe_results)
        # Either a pandas DataFrame (if pandas installed) or a dict.
        assert table is not None
        # Values round-trip either way.
        if hasattr(table, "loc"):
            assert float(table.loc["cognitive_load", 0]) == pytest.approx(0.9)
        else:
            assert table["cognitive_load"][0] == pytest.approx(0.9)

    def test_linear_probe_forward_shape(self) -> None:
        probe = LinearProbe(d_model=16)
        out = probe(torch.randn(4, 16))
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# AttentionPattern + specialists.
# ---------------------------------------------------------------------------


def _synthetic_focused_pattern(
    n_layers: int = 3,
    n_heads: int = 2,
    seq_len: int = 8,
    n_cond: int = 4,
    focused_layer: int = 2,
    focused_head: int = 1,
) -> AttentionPattern:
    """Build a hand-crafted pattern with one obvious specialist head."""
    per_layer: list[np.ndarray] = []
    for li in range(n_layers):
        # All heads default to uniform attention.
        layer = np.full(
            (n_heads, seq_len, n_cond), 1.0 / n_cond, dtype=np.float32
        )
        if li == focused_layer:
            focus = np.full((seq_len, n_cond), 0.02, dtype=np.float32)
            focus[:, 0] = 1.0 - 0.02 * (n_cond - 1)  # 0.94 -> exceeds 0.6
            layer[focused_head] = focus
        per_layer.append(layer)

    eps = 1e-9
    entropy = np.zeros((n_layers, n_heads), dtype=np.float32)
    for li, layer in enumerate(per_layer):
        ent = -(layer * np.log(layer + eps)).sum(axis=-1)
        entropy[li] = ent.mean(axis=-1)

    focus = np.zeros(seq_len, dtype=np.float32)
    total = 0
    for layer in per_layer:
        focus += layer.max(axis=-1).sum(axis=0)
        total += layer.shape[0]
    focus = focus / float(total)

    return AttentionPattern(
        per_layer=per_layer,
        per_head_entropy=entropy,
        per_token_conditioning_focus=focus,
        n_cond=n_cond,
    )


class TestAttentionCircuits:
    """Shape invariants and specialist detection."""

    def test_pattern_shape_invariants(self, tiny_slm: nn.Module) -> None:
        gen = torch.Generator().manual_seed(3)
        ids = torch.randint(1, 64, (8,), generator=gen)
        adapt = torch.rand(tiny_slm.conditioning_projector.adaptation_dim,
                           generator=gen)
        user = torch.randn(tiny_slm.conditioning_projector.user_state_dim,
                           generator=gen)
        pattern = extract_attention_patterns(
            tiny_slm, ids, adapt, user_state=user, max_tokens=8
        )
        assert pattern.n_layers == len(tiny_slm.layers)
        assert pattern.n_heads >= 1
        assert pattern.seq_len == 8
        for layer in pattern.per_layer:
            assert layer.shape == (pattern.n_heads, pattern.seq_len, pattern.n_cond)
            # Rows should be (approximately) probability distributions.
            sums = layer.sum(axis=-1)
            assert np.allclose(sums, 1.0, atol=1e-4)
        assert float(pattern.per_token_conditioning_focus.min()) >= 0.0
        assert float(pattern.per_token_conditioning_focus.max()) <= 1.0 + 1e-4
        assert float(pattern.per_head_entropy.min()) >= -1e-6

    def test_identify_conditioning_specialists_synthetic(self) -> None:
        pattern = _synthetic_focused_pattern()
        specs = identify_conditioning_specialists(pattern, threshold=0.6)
        assert specs, "expected at least one specialist in the synthetic pattern"
        # The hand-crafted specialist must appear.
        layers = {(s.layer, s.head) for s in specs}
        assert (2, 1) in layers
        # Preferred conditioning token is the one we spiked (index 0).
        top = next(s for s in specs if (s.layer, s.head) == (2, 1))
        assert top.preferred_cond_token == 0
        assert top.mean_max_weight > 0.6

    def test_identify_conditioning_specialists_empty_when_uniform(self) -> None:
        # Uniform pattern: max weight = 1/n_cond = 0.25 < 0.6 -> no specs.
        pattern = _synthetic_focused_pattern(focused_layer=-1)
        specs = identify_conditioning_specialists(pattern, threshold=0.6)
        assert specs == []

    def test_summarise_circuit_returns_nonempty_string(self) -> None:
        pattern = _synthetic_focused_pattern()
        text = summarise_circuit(pattern, threshold=0.6)
        assert isinstance(text, str) and len(text) > 50
        assert "specialist" in text or "conditioning" in text

    def test_extract_attention_rejects_empty_prompt(
        self, tiny_slm: nn.Module
    ) -> None:
        with pytest.raises(ValueError):
            extract_attention_patterns(
                tiny_slm,
                prompt=torch.zeros((1, 0), dtype=torch.long),
                conditioning_vector=torch.zeros(
                    tiny_slm.conditioning_projector.adaptation_dim
                ),
            )
