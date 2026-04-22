"""Shared helpers for the three ImplicitAdaptBench baselines.

Centralises:

* lazy construction of an :class:`AdaptiveSLM` + :class:`SimpleTokenizer`;
* a reusable ``verbalise_adaptation`` fallback when ``i3.eval`` is not
  importable;
* per-record latency measurement with :func:`time.perf_counter`.

All three baselines use the same harness so they measure the same thing
modulo the conditioning path.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from benchmarks.implicit_adapt_bench.data_schema import BenchmarkRecord

logger = logging.getLogger(__name__)


try:
    from i3.eval.ablation_experiment import verbalise_adaptation as _i3_verbalise
except ImportError as exc:  # pragma: no cover - I³ not importable
    logger.info(
        "i3.eval.ablation_experiment.verbalise_adaptation not importable (%s); "
        "falling back to the in-module verbaliser.",
        exc,
    )
    _i3_verbalise = None  # type: ignore[assignment]


def verbalise(archetype_name: str, adaptation_vector: Sequence[float]) -> str:
    """Return a short system-prompt prefix for the prompt baseline.

    Args:
        archetype_name: Canonical archetype name.
        adaptation_vector: 8-dim AdaptationVector.

    Returns:
        A ``"[System: ...] "``-style prefix.
    """
    # Use the production I³ verbaliser when it's importable so the benchmark
    # and the Batch A ablation stay consistent.
    if _i3_verbalise is not None:
        from i3.adaptation.types import AdaptationVector, StyleVector

        av = AdaptationVector(
            cognitive_load=float(adaptation_vector[0]),
            style_mirror=StyleVector(
                formality=float(adaptation_vector[1]),
                verbosity=float(adaptation_vector[2]),
                emotionality=float(adaptation_vector[3]),
                directness=float(adaptation_vector[4]),
            ),
            emotional_tone=float(adaptation_vector[5]),
            accessibility=float(adaptation_vector[6]),
        )
        return str(_i3_verbalise(archetype_name, av))

    # -- Local fallback ----------------------------------------------------
    cognitive_load = float(adaptation_vector[0])
    formality = float(adaptation_vector[1])
    verbosity = float(adaptation_vector[2])
    tone = float(adaptation_vector[5])
    accessibility = float(adaptation_vector[6])

    f_label = "formal" if formality > 0.6 else ("casual" if formality < 0.4 else "neutral")
    v_label = "elaborate" if verbosity > 0.6 else ("concise" if verbosity < 0.4 else "balanced")
    t_label = "warmly" if tone < 0.35 else ("neutrally" if tone < 0.65 else "objectively")
    access = " Use accessible language." if accessibility > 0.6 else ""
    load = (
        "simply"
        if cognitive_load < 0.4
        else ("with technical depth" if cognitive_load > 0.6 else "")
    )
    return (
        f"[System: respond {v_label}, {f_label}, {t_label}"
        + (f", {load}" if load else "")
        + f".{access}] "
    )


@dataclass
class _GenerationHarness:
    """A thin wrapper bundling an SLM with a tokenizer.

    The harness is intentionally kept tiny — the benchmark does not need a
    full :class:`i3.slm.generate.SLMGenerator` for the "responsiveness" story;
    a single forward pass + greedy decode is enough to demonstrate that the
    conditioning signal does or does not shape the output.

    Attributes:
        model: The underlying :class:`AdaptiveSLM`.
        tokenizer: The :class:`SimpleTokenizer` fitted to the prompt set.
        device: Torch device string.
        max_new_tokens: Generation budget per record.
    """

    model: object
    tokenizer: object
    device: str = "cpu"
    max_new_tokens: int = 32

    def generate(
        self,
        prompt: str,
        adaptation_vector: torch.Tensor | None,
        user_state: torch.Tensor | None,
    ) -> tuple[str, float, float]:
        """Run a short deterministic generation and measure latency.

        Args:
            prompt: Prompt text.
            adaptation_vector: Optional ``[1, 8]`` adaptation tensor.
            user_state: Optional ``[1, 64]`` user-state tensor.

        Returns:
            Tuple ``(generated_text, latency_p50_ms, latency_p95_ms)``.
            The p50 / p95 are measured over three short repeats.
        """
        from i3.slm.generate import SLMGenerator

        gen = SLMGenerator(self.model, self.tokenizer, device=self.device)  # type: ignore[arg-type]

        latencies: list[float] = []
        text = ""
        for _ in range(3):
            t0 = time.perf_counter()
            text = gen.generate(
                prompt=prompt,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,  # greedy = deterministic
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latencies.sort()
        # For 3 samples, index 1 is the median; index 2 is the max (proxy for p95).
        p50 = latencies[1]
        p95 = latencies[-1]
        return text, p50, p95


def build_harness(
    prompts: Sequence[str],
    *,
    device: str = "cpu",
    max_new_tokens: int = 32,
    seed: int = 42,
) -> _GenerationHarness:
    """Build a random-init :class:`AdaptiveSLM` + tokenizer for benchmarking.

    Args:
        prompts: Prompts used to fit the tokenizer vocabulary.
        device: Torch device string.
        max_new_tokens: Generation budget per record.
        seed: Seed passed to ``torch.manual_seed`` so the random-init
            weights are reproducible across baselines.

    Returns:
        A fully-built :class:`_GenerationHarness`.
    """
    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer

    torch.manual_seed(seed)

    tok = SimpleTokenizer(vocab_size=4000)
    tok.build_vocab(list(prompts))

    model = AdaptiveSLM(vocab_size=tok.vocab_size, max_seq_len=128)
    model.to(device).eval()
    return _GenerationHarness(
        model=model, tokenizer=tok, device=device, max_new_tokens=max_new_tokens
    )


def neutral_adaptation(device: str = "cpu") -> torch.Tensor:
    """Zero-filled 1x8 adaptation tensor on ``device``."""
    return torch.zeros(1, 8, device=device)


def neutral_user_state(device: str = "cpu", dim: int = 64) -> torch.Tensor:
    """Zero-filled 1x``dim`` user-state tensor on ``device``."""
    return torch.zeros(1, dim, device=device)


def adaptation_tensor(
    adaptation_vector: Sequence[float], device: str = "cpu"
) -> torch.Tensor:
    """Materialise an 8-dim list as a ``[1, 8]`` float32 tensor."""
    if len(adaptation_vector) != 8:
        raise ValueError(
            f"adaptation_vector must have 8 elements, got {len(adaptation_vector)}."
        )
    return torch.tensor(
        [float(x) for x in adaptation_vector], dtype=torch.float32, device=device
    ).unsqueeze(0)


def extract_user_state(
    record: BenchmarkRecord, device: str = "cpu", dim: int = 64
) -> torch.Tensor:
    """Project the 32-dim behavioural feature vector into a ``[1, 64]`` tensor.

    We use a simple tile-and-truncate projection rather than an ML-trained
    TCN so the baselines remain runnable on a laptop without the user-state
    encoder checkpoint. This is intentional: the benchmark measures
    responsiveness of the generator to the implicit signal, not the
    reconstruction quality of the encoder.

    Args:
        record: The benchmark record supplying the 32-dim feature vector.
        device: Torch device string.
        dim: Output dimensionality (default 64).

    Returns:
        A ``[1, dim]`` float32 tensor.
    """
    fv = record.behavioural_window.feature_vector
    if not fv:
        return torch.zeros(1, dim, device=device)
    # Tile to at least ``dim`` elements, then truncate.
    repeats = (dim + len(fv) - 1) // len(fv)
    tiled = (fv * repeats)[:dim]
    return torch.tensor(tiled, dtype=torch.float32, device=device).unsqueeze(0)
