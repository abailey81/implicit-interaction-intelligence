"""Cross-attention conditioning ablation experiment (Batch A).

This module implements the pre-registered ablation study described in
``docs/experiments/preregistration.md``. It measures *responsiveness* of the
next-token probability distribution to changes in the input
``AdaptationVector`` across three conditions:

* ``none`` — no conditioning (neutral zero vector).
* ``prompt`` — natural-language description of the adaptation is prepended
  to the prompt; the architectural conditioning path is held neutral.
* ``cross_attn`` — the adaptation flows through
  :class:`~i3.slm.cross_attention.ConditioningProjector` into every
  cross-attention sub-layer of the :class:`~i3.slm.model.AdaptiveSLM`.

The experiment is runnable on a **random-init** model: we are measuring the
*architectural* effect of the conditioning path, not the quality of the
generated text. A checkpoint may be supplied via ``AblationExperiment``'s
``checkpoint_path`` argument (or the ``I3_CHECKPOINT_PATH`` environment
variable when invoked from ``scripts/run_ablation_study.py``), but this is
not required.

Usage:
    >>> from i3.eval.ablation_experiment import AblationExperiment
    >>> exp = AblationExperiment(seed=42, n_prompts=50)
    >>> result = exp.run()
    >>> print(result.condition_kl_means)
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.eval.ablation_statistics import (
    bootstrap_ci,
    cohens_d,
    effect_size_interpretation,
    paired_sign_test,
)
from i3.slm.model import AdaptiveSLM
from i3.slm.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)

_EPS: float = 1e-12
_USER_STATE_DIM: int = 64


# ---------------------------------------------------------------------------
# Canonical test set (hard-coded for reproducibility)
# ---------------------------------------------------------------------------


# 50 fixed prompts spanning conversational, technical, emotional-support, and
# task-oriented registers. Keep the list here (not in a config file) so that
# the pre-registration commits to an exact experimental grid.
_CANONICAL_PROMPTS: tuple[str, ...] = (
    # Conversational (1-13)
    "Tell me about your weekend.",
    "What is your favourite season, and why?",
    "How have you been feeling lately?",
    "Do you enjoy reading novels?",
    "What did you have for breakfast today?",
    "If you could travel anywhere, where would you go?",
    "Describe your ideal Sunday morning.",
    "What music do you listen to when working?",
    "Tell me a short story about a cat and a lighthouse.",
    "What makes you laugh?",
    "Who is someone you admire, and why?",
    "Describe a place that feels calm to you.",
    "What is the best meal you have had recently?",
    # Technical (14-26)
    "Explain how a transformer self-attention layer works.",
    "What is the difference between variance and standard deviation?",
    "Summarise the key idea behind reinforcement learning.",
    "What is the purpose of layer normalisation in a neural network?",
    "Explain what a closure is in Python.",
    "What is the time complexity of merge sort?",
    "Why is cross-validation important in machine learning?",
    "How does a hash table resolve collisions?",
    "Explain gradient descent to a beginner.",
    "What is the CAP theorem?",
    "How does TCP differ from UDP?",
    "Describe how a B-tree index speeds up database lookups.",
    "What is overfitting, and how can it be mitigated?",
    # Emotional support (27-38)
    "I feel overwhelmed by work lately.",
    "I cannot seem to focus today.",
    "I had an argument with a close friend.",
    "I am anxious about an upcoming interview.",
    "I lost a pet recently and I miss them.",
    "I am struggling to sleep this week.",
    "I feel guilty for taking a day off.",
    "I doubt my own decisions a lot.",
    "I am lonely even when people are around.",
    "I feel stuck in my current job.",
    "I worry that I am disappointing my family.",
    "I keep procrastinating on something important.",
    # Task-oriented (39-50)
    "Please write a two-sentence apology email.",
    "Draft a polite request to reschedule a meeting.",
    "Give me three tips for writing cleaner code.",
    "List four healthy lunch ideas.",
    "Help me plan a 30-minute home workout.",
    "Outline the agenda for a project kick-off meeting.",
    "Write a thank-you note for a job interview.",
    "Suggest a safe icebreaker for a new team.",
    "Summarise the key steps of the scientific method.",
    "Generate a packing list for a three-day work trip.",
    "Draft a bullet-point weekly status update.",
    "Write a short review of a book you enjoyed.",
)


def canonical_prompts() -> tuple[str, ...]:
    """Return the immutable 50-prompt evaluation set.

    Returns:
        Tuple of 50 prompt strings. The order is fixed and must not be
        changed without updating the pre-registration.
    """
    return _CANONICAL_PROMPTS


# 8 archetype AdaptationVectors spanning the corners of the adaptation space.
_CANONICAL_ARCHETYPES: dict[str, AdaptationVector] = {
    "neutral": AdaptationVector.default(),
    "low_load_warm": AdaptationVector(
        cognitive_load=0.1,
        style_mirror=StyleVector(
            formality=0.2, verbosity=0.3, emotionality=0.8, directness=0.3
        ),
        emotional_tone=0.1,  # most supportive
        accessibility=0.6,
    ),
    "high_load_technical": AdaptationVector(
        cognitive_load=0.9,
        style_mirror=StyleVector(
            formality=0.8, verbosity=0.8, emotionality=0.2, directness=0.8
        ),
        emotional_tone=0.7,
        accessibility=0.1,
    ),
    "urgent_formal": AdaptationVector(
        cognitive_load=0.6,
        style_mirror=StyleVector(
            formality=0.9, verbosity=0.2, emotionality=0.2, directness=0.95
        ),
        emotional_tone=0.5,
        accessibility=0.2,
    ),
    "accessible_simple": AdaptationVector(
        cognitive_load=0.15,
        style_mirror=StyleVector(
            formality=0.3, verbosity=0.3, emotionality=0.6, directness=0.5
        ),
        emotional_tone=0.3,
        accessibility=0.95,
    ),
    "casual_verbose": AdaptationVector(
        cognitive_load=0.4,
        style_mirror=StyleVector(
            formality=0.1, verbosity=0.9, emotionality=0.7, directness=0.4
        ),
        emotional_tone=0.2,
        accessibility=0.3,
    ),
    "direct_terse": AdaptationVector(
        cognitive_load=0.5,
        style_mirror=StyleVector(
            formality=0.5, verbosity=0.1, emotionality=0.2, directness=0.95
        ),
        emotional_tone=0.6,
        accessibility=0.2,
    ),
    "reflective_neutral": AdaptationVector(
        cognitive_load=0.5,
        style_mirror=StyleVector(
            formality=0.5, verbosity=0.7, emotionality=0.5, directness=0.3
        ),
        emotional_tone=0.4,
        accessibility=0.4,
    ),
}


def canonical_archetypes() -> dict[str, AdaptationVector]:
    """Return the 8 archetype AdaptationVectors used by Batch A.

    Returns:
        Ordered dict-like mapping archetype label → AdaptationVector.
    """
    return dict(_CANONICAL_ARCHETYPES)


# ---------------------------------------------------------------------------
# Prompt-style verbalisation for the ``prompt`` condition
# ---------------------------------------------------------------------------


def verbalise_adaptation(name: str, vec: AdaptationVector) -> str:
    """Convert an ``AdaptationVector`` into a short prompt prefix.

    The prefix is prepended to the user's prompt under the ``prompt``
    condition. It encodes the same information that the ``cross_attn``
    condition receives architecturally, giving a fair comparison.

    Args:
        name: Human-readable archetype label (e.g., ``"high_load_technical"``).
        vec: The 8-dimensional adaptation vector.

    Returns:
        A short system-prompt-style sentence.
    """
    verbosity = "elaborate" if vec.style_mirror.verbosity > 0.6 else (
        "concise" if vec.style_mirror.verbosity < 0.4 else "balanced"
    )
    formality = "formal" if vec.style_mirror.formality > 0.6 else (
        "casual" if vec.style_mirror.formality < 0.4 else "neutral"
    )
    load = "simply" if vec.cognitive_load < 0.4 else (
        "with technical depth" if vec.cognitive_load > 0.6 else ""
    )
    tone = "warmly" if vec.emotional_tone < 0.35 else (
        "neutrally" if vec.emotional_tone < 0.65 else "objectively"
    )
    access = " Use accessible language." if vec.accessibility > 0.6 else ""
    return (
        f"[System: respond {verbosity}, {formality}, {tone}"
        + (f", {load}" if load else "")
        + f".{access}] "
    )


# ---------------------------------------------------------------------------
# Result schema (Pydantic v2)
# ---------------------------------------------------------------------------


class PairRecord(BaseModel):
    """One row of the per-pair ablation dataframe."""

    model_config = ConfigDict(frozen=True)

    condition: str
    prompt_index: int
    archetype_i: str
    archetype_j: str
    kl_sym: float
    latency_ms_i: float
    latency_ms_j: float


class AblationResult(BaseModel):
    """Complete Pydantic record of an ablation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conditions: list[str] = Field(default_factory=list)
    archetype_labels: list[str] = Field(default_factory=list)
    n_prompts: int = 0
    n_pairs_per_condition: int = 0

    condition_kl_means: dict[str, float] = Field(default_factory=dict)
    condition_kl_cis: dict[str, tuple[float, float]] = Field(default_factory=dict)
    condition_style_fidelity: dict[str, float] = Field(default_factory=dict)
    condition_style_fidelity_cis: dict[str, tuple[float, float]] = Field(
        default_factory=dict
    )
    condition_latency_ms_p50: dict[str, float] = Field(default_factory=dict)
    condition_latency_ms_p95: dict[str, float] = Field(default_factory=dict)
    condition_latency_ms_p99: dict[str, float] = Field(default_factory=dict)

    pairwise_cohens_d: dict[str, float] = Field(default_factory=dict)
    pairwise_sign_p: dict[str, float] = Field(default_factory=dict)
    pairwise_effect_label: dict[str, str] = Field(default_factory=dict)

    per_pair_records: list[PairRecord] = Field(default_factory=list)

    def to_dataframe_rows(self) -> list[dict[str, Any]]:
        """Dump per-pair records as a list of plain dicts (pandas-friendly).

        Returns:
            List of row dicts suitable for ``pandas.DataFrame(rows)``.
        """
        return [rec.model_dump() for rec in self.per_pair_records]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class AblationExperiment:
    """Runnable pre-registered ablation of cross-attention conditioning.

    The experiment instantiates (or loads) an :class:`AdaptiveSLM`, a small
    in-memory tokenizer, and runs three conditions over every
    ``(prompt, archetype)`` cell, computing the pairwise KL responsiveness
    matrix and downstream statistics.

    Attributes:
        seed: Global deterministic seed (default 42).
        n_prompts: Number of prompts to draw from
            :func:`canonical_prompts`; capped at 50.
        device: Torch device string (``"cpu"`` is recommended for the
            latency claim in H3).
        model: The :class:`AdaptiveSLM` under test.
        tokenizer: A :class:`SimpleTokenizer` fitted to the prompt set.
        archetypes: The 8 :class:`AdaptationVector` archetypes.
    """

    def __init__(
        self,
        seed: int = 42,
        n_prompts: int = 50,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Initialise the experiment.

        Args:
            seed: Global seed for all random sources.
            n_prompts: Number of prompts to use (1–50). Values > 50 are
                clamped to 50 with a warning.
            device: Torch device string (``"cpu"`` or ``"cuda"``).
            checkpoint_path: Optional path to an ``AdaptiveSLM`` state dict
                produced by ``training/`` scripts. When ``None`` the model
                is kept at random-init weights.
        """
        if n_prompts < 1:
            raise ValueError(f"n_prompts must be >= 1, got {n_prompts}")
        if n_prompts > len(_CANONICAL_PROMPTS):
            logger.warning(
                "n_prompts=%d exceeds canonical set of %d; clamping.",
                n_prompts,
                len(_CANONICAL_PROMPTS),
            )
            n_prompts = len(_CANONICAL_PROMPTS)

        self.seed: int = seed
        self.n_prompts: int = n_prompts
        self.device: str = device
        self.prompts: tuple[str, ...] = _CANONICAL_PROMPTS[:n_prompts]
        self.archetypes: dict[str, AdaptationVector] = canonical_archetypes()
        self._checkpoint_path: Path | None = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )

        self._seed_all(seed)
        self.tokenizer: SimpleTokenizer = self._build_tokenizer(self.prompts)
        self.model: AdaptiveSLM = self._build_model(
            self.tokenizer.vocab_size, self._checkpoint_path
        )
        self.model.to(device).eval()

    # -- initialisation helpers ------------------------------------------------

    @staticmethod
    def _seed_all(seed: int) -> None:
        """Seed all known random sources.

        Args:
            seed: The integer seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _build_tokenizer(prompts: Sequence[str]) -> SimpleTokenizer:
        """Build a small vocabulary from the prompt set plus English filler.

        Args:
            prompts: The prompts the experiment will encode.

        Returns:
            A :class:`SimpleTokenizer` with a fitted vocabulary.
        """
        tok = SimpleTokenizer(vocab_size=8000)
        # Provide enough corpus for build_vocab to yield a useful vocab.
        corpus = list(prompts)
        # Add the verbalised adaptation prefixes so the prompt condition has
        # tokens for its own special words.
        for label, vec in _CANONICAL_ARCHETYPES.items():
            corpus.append(verbalise_adaptation(label, vec))
        tok.build_vocab(corpus)
        return tok

    @staticmethod
    def _build_model(
        vocab_size: int, checkpoint_path: Path | None
    ) -> AdaptiveSLM:
        """Instantiate :class:`AdaptiveSLM`, optionally loading a checkpoint.

        Args:
            vocab_size: Vocabulary size for the token-embedding layer.
            checkpoint_path: Optional path to a ``.pt`` state dict.

        Returns:
            An :class:`AdaptiveSLM` instance.
        """
        model = AdaptiveSLM(vocab_size=vocab_size)
        if checkpoint_path is None:
            logger.info(
                "No checkpoint supplied — running on RANDOM-INIT weights. "
                "This is expected for the Batch A responsiveness measurement."
            )
            return model
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint_path}"
            )
        # SEC: torch>=2.6 requires weights_only=True for untrusted state dicts.
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        # Handle either a bare state dict or a {"model": ...} wrapper.
        if isinstance(state, dict) and "model" in state and isinstance(
            state["model"], dict
        ):
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("missing keys in checkpoint: %s", missing)
        if unexpected:
            logger.warning("unexpected keys in checkpoint: %s", unexpected)
        return model

    # -- per-condition distribution helpers -----------------------------------

    def _encode(self, text: str, max_length: int = 32) -> torch.Tensor:
        """Encode ``text`` to a padded ``[1, seq_len]`` LongTensor.

        Args:
            text: Prompt text.
            max_length: Maximum sequence length (truncate/pad).

        Returns:
            A LongTensor on ``self.device``.
        """
        ids = self.tokenizer.encode(
            text, add_special=True, max_length=max_length, padding=True
        )
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _neutral_adapt(self) -> torch.Tensor:
        """Return a neutral ``[1, 8]`` adaptation vector (zeros)."""
        return torch.zeros(1, 8, device=self.device)

    def _neutral_user_state(self) -> torch.Tensor:
        """Return a neutral ``[1, 64]`` user-state embedding (zeros)."""
        return torch.zeros(1, _USER_STATE_DIM, device=self.device)

    def _next_token_probs(
        self,
        input_ids: torch.Tensor,
        adaptation_vector: torch.Tensor,
        user_state: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass and return ``[vocab]`` softmax probabilities.

        Args:
            input_ids: ``[1, seq_len]`` token ids.
            adaptation_vector: ``[1, 8]`` adaptation tensor.
            user_state: ``[1, 64]`` user-state tensor.

        Returns:
            A 1-D probability tensor over the vocabulary (last position).
        """
        with torch.inference_mode():
            logits, _ = self.model(
                input_ids=input_ids,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
            )
        last_logits = logits[0, -1, :]
        return F.softmax(last_logits, dim=-1)

    @staticmethod
    def _symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> float:
        """Symmetric KL divergence in nats.

        Args:
            p: First probability tensor.
            q: Second probability tensor.

        Returns:
            Non-negative float KL value.
        """
        p = p.clamp_min(_EPS)
        q = q.clamp_min(_EPS)
        kl_pq = float(torch.sum(p * (torch.log(p) - torch.log(q))).item())
        kl_qp = float(torch.sum(q * (torch.log(q) - torch.log(p))).item())
        return 0.5 * (kl_pq + kl_qp)

    # -- per-condition runners -------------------------------------------------

    def _run_condition_none(
        self, prompt: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Compute next-token dists under the *no-conditioning* condition.

        Every archetype is collapsed to a neutral zero adaptation vector and
        zero user state, so all distributions are identical modulo floating-
        point noise. This is the H1 lower bound.

        Args:
            prompt: The user prompt.

        Returns:
            Tuple ``(dists, latencies_ms)``: a dict mapping archetype label
            to a next-token probability tensor, and a dict mapping archetype
            label to the forward-pass latency in milliseconds.
        """
        input_ids = self._encode(prompt)
        dists: dict[str, torch.Tensor] = {}
        latencies: dict[str, float] = {}
        for label in self.archetypes:
            t0 = time.perf_counter()
            dists[label] = self._next_token_probs(
                input_ids,
                adaptation_vector=self._neutral_adapt(),
                user_state=self._neutral_user_state(),
            )
            latencies[label] = (time.perf_counter() - t0) * 1000.0
        return dists, latencies

    def _run_condition_prompt(
        self, prompt: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Compute next-token dists under the *prompt-based* condition.

        The adaptation is verbalised via :func:`verbalise_adaptation` and
        prepended to the prompt. The architectural conditioning path is
        held neutral (zero adaptation vector, zero user state).

        Args:
            prompt: The user prompt.

        Returns:
            Tuple ``(dists, latencies_ms)`` keyed by archetype label.
        """
        dists: dict[str, torch.Tensor] = {}
        latencies: dict[str, float] = {}
        for label, vec in self.archetypes.items():
            prefix = verbalise_adaptation(label, vec)
            full_prompt = prefix + prompt
            input_ids = self._encode(full_prompt)
            t0 = time.perf_counter()
            dists[label] = self._next_token_probs(
                input_ids,
                adaptation_vector=self._neutral_adapt(),
                user_state=self._neutral_user_state(),
            )
            latencies[label] = (time.perf_counter() - t0) * 1000.0
        return dists, latencies

    def _run_condition_crossattn(
        self, prompt: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Compute next-token dists under the *cross-attention* condition.

        The architectural conditioning path is fully engaged: each archetype
        produces its own 8-dim adaptation tensor routed through the
        :class:`~i3.slm.cross_attention.ConditioningProjector`.

        Args:
            prompt: The user prompt.

        Returns:
            Tuple ``(dists, latencies_ms)`` keyed by archetype label.
        """
        input_ids = self._encode(prompt)
        dists: dict[str, torch.Tensor] = {}
        latencies: dict[str, float] = {}
        for label, vec in self.archetypes.items():
            adapt_t = vec.to_tensor().unsqueeze(0).to(self.device)
            t0 = time.perf_counter()
            dists[label] = self._next_token_probs(
                input_ids,
                adaptation_vector=adapt_t,
                user_state=self._neutral_user_state(),
            )
            latencies[label] = (time.perf_counter() - t0) * 1000.0
        return dists, latencies

    # -- style fidelity --------------------------------------------------------

    def _style_fidelity(
        self,
        prompt: str,
        vec: AdaptationVector,
        condition: str,
        generate_tokens: int = 16,
    ) -> float:
        """Measure style fidelity of a short greedy decode.

        We greedily decode ``generate_tokens`` tokens under the given
        condition and compute a length-distribution fidelity score:
        ``-(L_target - L_actual)^2 / (2 * sigma^2)`` where ``L_target``
        is a target token count derived from ``vec.style_mirror.verbosity``
        and ``L_actual`` is the number of non-pad tokens produced before
        the first ``[EOS]``.

        Args:
            prompt: The user prompt.
            vec: The archetype adaptation vector.
            condition: One of ``"none"``, ``"prompt"``, ``"cross_attn"``.
            generate_tokens: Max new tokens to decode.

        Returns:
            A scalar fidelity score in ``(-inf, 0]``; higher is better.
        """
        # Target length as a linear function of verbosity (terse=4, verbose=16).
        target_len = 4.0 + 12.0 * float(vec.style_mirror.verbosity)
        sigma = 4.0  # tolerance

        if condition == "prompt":
            text = verbalise_adaptation("archetype", vec) + prompt
            adapt_t = self._neutral_adapt()
        else:
            text = prompt
            adapt_t = (
                vec.to_tensor().unsqueeze(0).to(self.device)
                if condition == "cross_attn"
                else self._neutral_adapt()
            )
        input_ids = self._encode(text)

        new_tokens: list[int] = []
        eos_id = self.tokenizer.EOS_ID
        pad_id = self.tokenizer.PAD_ID
        with torch.inference_mode():
            for _ in range(generate_tokens):
                logits, _ = self.model(
                    input_ids=input_ids,
                    adaptation_vector=adapt_t,
                    user_state=self._neutral_user_state(),
                )
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                new_tokens.append(next_id)
                if next_id in (eos_id, pad_id):
                    break
                # Extend input_ids with the sampled id.
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [[next_id]], dtype=torch.long, device=self.device
                        ),
                    ],
                    dim=1,
                )
        actual_len = float(
            sum(1 for t in new_tokens if t not in (eos_id, pad_id))
        )
        return float(-((target_len - actual_len) ** 2) / (2.0 * sigma**2))

    # -- main entrypoint -------------------------------------------------------

    def run(self) -> AblationResult:
        """Execute the full 1200-run grid and return an :class:`AblationResult`.

        Returns:
            A populated :class:`AblationResult` ready for JSON / Markdown
            serialisation.
        """
        conditions: list[str] = ["none", "prompt", "cross_attn"]
        runners = {
            "none": self._run_condition_none,
            "prompt": self._run_condition_prompt,
            "cross_attn": self._run_condition_crossattn,
        }

        labels: list[str] = list(self.archetypes.keys())
        n = len(labels)
        pair_indices: list[tuple[int, int]] = [
            (i, j) for i in range(n) for j in range(i + 1, n)
        ]

        per_pair: list[PairRecord] = []
        per_condition_kl: dict[str, list[float]] = {c: [] for c in conditions}
        per_condition_lat: dict[str, list[float]] = {c: [] for c in conditions}
        per_condition_style: dict[str, list[float]] = {c: [] for c in conditions}

        logger.info(
            "Ablation grid: %d prompts × %d archetype pairs × %d conditions = %d cells",
            self.n_prompts,
            len(pair_indices),
            len(conditions),
            self.n_prompts * len(pair_indices) * len(conditions),
        )

        for p_idx, prompt in enumerate(self.prompts):
            for condition in conditions:
                dists, latencies = runners[condition](prompt)
                # Pairwise KL
                for (i, j) in pair_indices:
                    lab_i = labels[i]
                    lab_j = labels[j]
                    kl = self._symmetric_kl(dists[lab_i], dists[lab_j])
                    per_pair.append(
                        PairRecord(
                            condition=condition,
                            prompt_index=p_idx,
                            archetype_i=lab_i,
                            archetype_j=lab_j,
                            kl_sym=kl,
                            latency_ms_i=latencies[lab_i],
                            latency_ms_j=latencies[lab_j],
                        )
                    )
                    per_condition_kl[condition].append(kl)
                per_condition_lat[condition].extend(latencies.values())
                # Style fidelity: one score per (prompt, archetype, condition)
                for lab, vec in self.archetypes.items():
                    per_condition_style[condition].append(
                        self._style_fidelity(prompt, vec, condition)
                    )

        # -- summary stats ----------------------------------------------------
        rng = np.random.default_rng(self.seed)
        kl_means: dict[str, float] = {}
        kl_cis: dict[str, tuple[float, float]] = {}
        style_means: dict[str, float] = {}
        style_cis: dict[str, tuple[float, float]] = {}
        lat_p50: dict[str, float] = {}
        lat_p95: dict[str, float] = {}
        lat_p99: dict[str, float] = {}
        for condition in conditions:
            kl_arr = np.asarray(per_condition_kl[condition], dtype=np.float64)
            kl_means[condition] = float(kl_arr.mean())
            kl_cis[condition] = bootstrap_ci(kl_arr, rng=rng)
            style_arr = np.asarray(
                per_condition_style[condition], dtype=np.float64
            )
            style_means[condition] = float(style_arr.mean())
            style_cis[condition] = bootstrap_ci(style_arr, rng=rng)
            lat_arr = np.asarray(per_condition_lat[condition], dtype=np.float64)
            lat_p50[condition] = float(np.percentile(lat_arr, 50))
            lat_p95[condition] = float(np.percentile(lat_arr, 95))
            lat_p99[condition] = float(np.percentile(lat_arr, 99))

        # -- pairwise comparisons --------------------------------------------
        pairs = [
            ("cross_attn", "prompt"),
            ("cross_attn", "none"),
            ("prompt", "none"),
        ]
        cohens: dict[str, float] = {}
        sign_p: dict[str, float] = {}
        effect_label: dict[str, str] = {}
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            d = cohens_d(per_condition_kl[a], per_condition_kl[b])
            cohens[key] = d
            effect_label[key] = effect_size_interpretation(d)
            # Paired sign test: the per-pair KL vectors are aligned by index
            # of emission (prompt × archetype-pair) across conditions.
            sign_p[key] = paired_sign_test(
                per_condition_kl[a], per_condition_kl[b]
            )

        return AblationResult(
            conditions=conditions,
            archetype_labels=labels,
            n_prompts=self.n_prompts,
            n_pairs_per_condition=len(pair_indices) * self.n_prompts,
            condition_kl_means=kl_means,
            condition_kl_cis=kl_cis,
            condition_style_fidelity=style_means,
            condition_style_fidelity_cis=style_cis,
            condition_latency_ms_p50=lat_p50,
            condition_latency_ms_p95=lat_p95,
            condition_latency_ms_p99=lat_p99,
            pairwise_cohens_d=cohens,
            pairwise_sign_p=sign_p,
            pairwise_effect_label=effect_label,
            per_pair_records=per_pair,
        )


__all__ = [
    "AblationExperiment",
    "AblationResult",
    "PairRecord",
    "canonical_prompts",
    "canonical_archetypes",
    "verbalise_adaptation",
]
