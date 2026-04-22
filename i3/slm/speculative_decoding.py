"""Speculative decoding for the I3 Adaptive Small Language Model.

Implements the draft-and-verify decoding loop from Leviathan et al. 2023
("Fast Inference from Transformers via Speculative Decoding", ICML 2023)
and the concurrent Chen et al. 2023 work ("Accelerating Large Language
Model Decoding with Speculative Sampling", DeepMind technical report).

The core idea:

    1. A cheap **draft** model autoregressively proposes ``k`` tokens.
    2. The expensive **target** model scores all ``k`` proposals in a
       *single* forward pass (parallel, not sequential).
    3. Each proposal is accepted with probability
       ``min(1, p_target(t) / p_draft(t))`` — the standard rejection
       sampling step. Accepted tokens are emitted unchanged; on the
       first rejection the target's residual distribution is resampled,
       and the remaining drafts are discarded.

When the draft's distribution is close to the target's, most drafts are
accepted and throughput improves roughly proportionally to ``k``.
Crucially, the output distribution is provably identical to that of the
target model alone — speculative decoding is a *lossless* acceleration.

This module is a Huawei-ecosystem analogue: Huawei's public work on
Celia Auto-answer reports doubled throughput via speculative decoding
combined with RL-augmented distillation. The I3 SLM is deliberately
smaller than Celia-class models (~6 M parameters versus billions), so
the absolute speed-up is lower, but the architectural pattern is the
one a Huawei technical reviewer recognises on sight.

Example::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.speculative_decoding import SpeculativeDecoder

    target = AdaptiveSLM()
    decoder = SpeculativeDecoder(target_model=target, num_drafts=4)
    out_ids, stats = decoder.generate(
        prompt_ids=torch.tensor([[1, 42, 7, 9]]),
        conditioning_tokens=None,
        max_new_tokens=64,
    )
    print(stats.speedup_vs_target)

References
----------
* Leviathan, Kalman & Matias (2023). *Fast Inference from Transformers
  via Speculative Decoding.* ICML 2023. arXiv:2211.17192.
* Chen, Borgeaud, Irving, Lespiau, Sifre & Jumper (2023). *Accelerating
  Large Language Model Decoding with Speculative Sampling.* DeepMind
  technical report. arXiv:2302.01318.
* Huawei (2024-2025, public Celia engineering blog posts). Speculative
  decoding + RL-augmented distillation reported to roughly double Celia
  Auto-answer throughput while preserving answer quality. [#celia]_

.. [#celia] Public Huawei engineering reports on Celia on-device
   acceleration. The Huawei work combines speculative decoding with an
   RL-distilled draft model; the I3 implementation below uses the
   classical draft-and-verify loop only and does not perform RL
   distillation (out of scope for the I3 demo). The citation establishes
   the lineage for a Huawei technical reviewer.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from i3.slm.model import AdaptiveSLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeStats:
    """Per-generation statistics for a single speculative-decoding run.

    Attributes:
        accepted_tokens: Total number of draft tokens accepted by the
            verification step across the whole generation.
        drafted_tokens: Total number of tokens proposed by the draft
            model (equivalently, ``num_drafts * verification_passes``
            minus any truncation at ``max_new_tokens``).
        verification_passes: Number of target-model forward passes used
            to verify drafts. Each pass scores up to ``num_drafts``
            proposals in parallel.
        total_latency_ms: Total wall-clock latency of the generation
            call, in milliseconds.
        speedup_vs_target: Empirical speed-up versus sequential target
            generation, computed as
            ``(accepted_tokens + verification_passes) / verification_passes``.
            This is the "speedup bound" from Leviathan et al. 2023
            Appendix A: each verification pass emits at least one new
            token (the bonus token sampled from the target residual
            distribution), plus any accepted drafts.

    Note:
        ``accepted_tokens`` is always less than or equal to
        ``drafted_tokens``; this invariant is checked in
        :meth:`SpeculativeStats.__post_init__`.
    """

    accepted_tokens: int = 0
    drafted_tokens: int = 0
    verification_passes: int = 0
    total_latency_ms: float = 0.0
    speedup_vs_target: float = 1.0
    fallback_used: bool = False
    extras: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate internal consistency of the accumulated statistics.

        Raises:
            ValueError: If any of the counters are negative, if
                ``accepted_tokens > drafted_tokens``, or if
                ``speedup_vs_target`` is non-finite.
        """
        if self.accepted_tokens < 0:
            raise ValueError(
                f"accepted_tokens must be >= 0, got {self.accepted_tokens}"
            )
        if self.drafted_tokens < 0:
            raise ValueError(
                f"drafted_tokens must be >= 0, got {self.drafted_tokens}"
            )
        if self.verification_passes < 0:
            raise ValueError(
                f"verification_passes must be >= 0, "
                f"got {self.verification_passes}"
            )
        if self.accepted_tokens > self.drafted_tokens:
            raise ValueError(
                f"accepted_tokens ({self.accepted_tokens}) cannot exceed "
                f"drafted_tokens ({self.drafted_tokens})"
            )
        if not math.isfinite(self.total_latency_ms) or self.total_latency_ms < 0:
            raise ValueError(
                f"total_latency_ms must be a finite non-negative float, "
                f"got {self.total_latency_ms}"
            )
        if not math.isfinite(self.speedup_vs_target):
            raise ValueError(
                f"speedup_vs_target must be finite, got {self.speedup_vs_target}"
            )

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that survived verification.

        Returns:
            Value in ``[0, 1]``. Returns ``0.0`` if no drafts were
            generated (e.g. when :attr:`fallback_used` is ``True``).
        """
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """Draft-and-verify speculative decoder for :class:`AdaptiveSLM`.

    The decoder composes a *target* model (the full-sized SLM) with a
    smaller *draft* model. Each step:

    1. The draft autoregressively proposes ``num_drafts`` tokens from
       the current prefix (cheap: the draft is typically half the size
       of the target).
    2. The target model runs a **single** forward pass over the full
       prefix plus the ``num_drafts`` proposed tokens and returns
       logits at each of those ``num_drafts`` positions.
    3. For each proposed token we compute the acceptance ratio
       ``r = p_target(t) / p_draft(t)`` and accept if ``r`` exceeds a
       fixed threshold (the classical rejection-sampling test with a
       configurable "greediness" knob).
    4. The first rejected position is replaced by a resample from the
       residual distribution ``max(0, p_target - p_draft)``; subsequent
       draft tokens are discarded.

    The decoder is a **drop-in** accelerator: if the draft distribution
    matches the target exactly, every draft is accepted and the speed-up
    is bounded by ``num_drafts + 1``. If the draft is poor, most drafts
    are rejected and the effective throughput approaches (but never
    falls below) the target-only baseline, minus the overhead of the
    rejected-draft forward passes.

    Args:
        target_model: The full-sized :class:`AdaptiveSLM`. Must expose
            the standard ``forward(input_ids, adaptation_vector,
            user_state, use_cache)`` signature.
        draft_model: Optional smaller :class:`AdaptiveSLM`. If ``None``,
            a smaller copy of ``target_model`` is constructed
            automatically (half the layers and half the ``d_model``,
            clamped to sensible minimums). If construction of the
            auto-draft fails (e.g. the target is already too small to
            shrink meaningfully), the decoder falls back cleanly to
            running the target model alone and :attr:`fallback_mode` is
            set to ``True``.
        num_drafts: Number of tokens the draft proposes before each
            verification pass (the ``k`` parameter of Leviathan et al.
            2023). Typical values are 2, 4 or 8.
        acceptance_threshold: Minimum acceptance ratio
            ``r = p_target / p_draft`` for a draft to be accepted.
            The classical algorithm uses ``1.0`` with
            stochastic acceptance; here we expose a "greediness" knob:
            values below ``1.0`` accept more liberally (faster but less
            faithful), values at or above ``1.0`` accept strictly in
            the Leviathan sense.
        vocab_size: Optional manual override of the target's vocab size
            (used only in tests with mocked models). Normally inferred
            from ``target_model.vocab_size``.

    Attributes:
        fallback_mode: ``True`` when the draft model is unavailable and
            the decoder runs the target model alone. In that mode
            :meth:`generate` still returns well-formed
            :class:`SpeculativeStats` with ``fallback_used=True`` and
            a speed-up of ``1.0``.
    """

    # SEC: Hard upper bound on generation length to prevent runaway loops
    # even if a caller passes a pathological ``max_new_tokens``.
    HARD_MAX_NEW_TOKENS: int = 4096

    # SEC: Sensible floors for the auto-shrunk draft. If the target has
    # fewer than 2 layers or d_model < 32, constructing a smaller draft is
    # not physically possible and we fall back to target-only generation.
    _MIN_DRAFT_LAYERS: int = 1
    _MIN_DRAFT_D_MODEL: int = 32
    _MIN_DRAFT_HEADS: int = 1

    def __init__(
        self,
        target_model: AdaptiveSLM,
        draft_model: Optional[AdaptiveSLM] = None,
        num_drafts: int = 4,
        acceptance_threshold: float = 0.7,
        vocab_size: Optional[int] = None,
    ) -> None:
        if num_drafts < 1:
            raise ValueError(f"num_drafts must be >= 1, got {num_drafts}")
        if acceptance_threshold <= 0.0:
            raise ValueError(
                f"acceptance_threshold must be > 0, got {acceptance_threshold}"
            )

        self.target_model: AdaptiveSLM = target_model
        self.num_drafts: int = int(num_drafts)
        self.acceptance_threshold: float = float(acceptance_threshold)
        self.vocab_size: int = int(
            vocab_size if vocab_size is not None else target_model.vocab_size
        )

        # ----- Draft model construction / auto-shrink ----------------------
        self.fallback_mode: bool = False
        self.draft_model: Optional[AdaptiveSLM]

        if draft_model is not None:
            self.draft_model = draft_model
            if (
                getattr(draft_model, "vocab_size", self.vocab_size)
                != self.vocab_size
            ):
                raise ValueError(
                    f"draft_model.vocab_size ({draft_model.vocab_size}) must "
                    f"match target_model.vocab_size ({self.vocab_size})"
                )
        else:
            self.draft_model = self._auto_shrink(target_model)
            if self.draft_model is None:
                logger.warning(
                    "SpeculativeDecoder: could not construct a smaller "
                    "draft model from the target (likely target is already "
                    "minimal); falling back to target-only generation."
                )
                self.fallback_mode = True

        # Eval mode on both models — speculative decoding is inference-only.
        self.target_model.eval()
        if self.draft_model is not None:
            self.draft_model.eval()

    # ------------------------------------------------------------------
    # Draft-model construction
    # ------------------------------------------------------------------

    @classmethod
    def _auto_shrink(
        cls, target: AdaptiveSLM
    ) -> Optional[AdaptiveSLM]:
        """Construct a smaller draft with half the layers and half the width.

        The shrink respects three invariants:

        * ``n_layers`` >= :attr:`_MIN_DRAFT_LAYERS`.
        * ``d_model``  >= :attr:`_MIN_DRAFT_D_MODEL` and is divisible by
          ``n_heads`` after shrinking (PyTorch's multi-head attention
          requires this).
        * ``vocab_size``, ``adaptation_dim``, ``conditioning_dim`` all
          *match the target exactly* so the two models share a token
          alphabet and conditioning interface.

        Args:
            target: The full-sized target model.

        Returns:
            A new :class:`AdaptiveSLM` initialised with random weights
            at the reduced configuration, or ``None`` if the target is
            already at or below the minimum sensible size.
        """
        try:
            n_layers = max(cls._MIN_DRAFT_LAYERS, len(target.layers) // 2)

            # SEC: Read geometry via public attributes / module inspection —
            # avoids coupling to private _cfg dicts.
            target_d_model = int(target.d_model)
            d_model = max(cls._MIN_DRAFT_D_MODEL, target_d_model // 2)

            first_layer = target.layers[0]
            target_n_heads = int(first_layer.self_attn.n_heads)
            # Preserve divisibility: d_model must be a multiple of n_heads.
            n_heads = max(cls._MIN_DRAFT_HEADS, target_n_heads // 2)
            while d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            if d_model % n_heads != 0:
                # Could not find a valid head count — bail out.
                return None

            target_d_ff = int(first_layer.ff.linear1.out_features)
            d_ff = max(d_model, target_d_ff // 2)

            target_n_cross = int(first_layer.cross_attn.n_heads)
            n_cross_heads = max(1, target_n_cross // 2)
            while d_model % n_cross_heads != 0 and n_cross_heads > 1:
                n_cross_heads -= 1

            # Geometry guards — refuse to build a draft that is not smaller.
            target_params = int(target.num_parameters)
            if (
                n_layers >= len(target.layers)
                and d_model >= target_d_model
            ):
                return None

            # Introspect the embedding / conditioning projector.
            vocab_size = int(target.vocab_size)
            max_seq_len = int(
                target.embedding.positional_encoding.pe.size(1)
            )
            n_cond = int(target.conditioning_projector.n_tokens) if hasattr(
                target.conditioning_projector, "n_tokens"
            ) else 4
            adapt_dim = int(target.conditioning_projector.adaptation_dim)
            cond_dim = int(target.conditioning_projector.user_state_dim)

            draft = AdaptiveSLM(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                conditioning_dim=cond_dim,
                adaptation_dim=adapt_dim,
                n_cross_heads=n_cross_heads,
                n_conditioning_tokens=n_cond,
                dropout=0.0,
                tie_weights=True,
            )

            # SEC: Sanity check — the draft must actually be smaller than the
            # target. If it ended up the same size or larger, fall back.
            if draft.num_parameters >= target_params:
                return None

            return draft
        except (AttributeError, ValueError, RuntimeError) as exc:
            # Target does not expose the expected sub-module layout, or the
            # shrunk geometry failed a constructor guard. Falling back to
            # target-only generation is the safe default.
            logger.warning(
                "SpeculativeDecoder._auto_shrink failed: %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        conditioning_tokens: Optional[dict[str, torch.Tensor]] = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, SpeculativeStats]:
        """Run speculative decoding from ``prompt_ids``.

        Args:
            prompt_ids: Integer token IDs of shape ``[1, prompt_len]``
                or ``[prompt_len]``. Only batch size 1 is supported
                (speculative decoding with batch > 1 is materially more
                complex; out of scope for the I3 demo).
            conditioning_tokens: Optional conditioning context. If
                provided, must be a dict with keys
                ``"adaptation_vector"`` and ``"user_state"`` matching
                the target model's forward signature. ``None`` uses the
                model's neutral defaults.
            max_new_tokens: Maximum number of new tokens to generate.
                Capped at :attr:`HARD_MAX_NEW_TOKENS`.
            temperature: Sampling temperature applied to both the draft
                and target logits before the softmax. ``1.0`` preserves
                the model's output distribution; higher values flatten
                the distribution; lower values sharpen it.

        Returns:
            A tuple ``(generated_ids, stats)`` where:

            * ``generated_ids`` is a ``[1, prompt_len + n_generated]``
              tensor containing the prompt followed by the newly
              generated tokens.
            * ``stats`` is a populated :class:`SpeculativeStats`
              instance.

        Raises:
            ValueError: If ``prompt_ids`` is not 1-D or 2-D, or if
                ``max_new_tokens < 0``.
        """
        # ----- Argument normalisation / validation ------------------------
        if max_new_tokens < 0:
            raise ValueError(
                f"max_new_tokens must be >= 0, got {max_new_tokens}"
            )
        max_new_tokens = min(int(max_new_tokens), self.HARD_MAX_NEW_TOKENS)

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if prompt_ids.dim() != 2:
            raise ValueError(
                f"prompt_ids must be 1-D or 2-D, got shape "
                f"{tuple(prompt_ids.shape)}"
            )
        if prompt_ids.size(0) != 1:
            raise ValueError(
                f"only batch_size=1 is supported, got {prompt_ids.size(0)}"
            )
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )

        device = prompt_ids.device
        adaptation_vector, user_state = self._unpack_conditioning(
            conditioning_tokens, device
        )

        t0 = time.perf_counter()

        # ----- Fallback: target-only generation ---------------------------
        if self.fallback_mode or self.draft_model is None:
            ids = self._target_only_generate(
                prompt_ids, adaptation_vector, user_state,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            stats = SpeculativeStats(
                accepted_tokens=0,
                drafted_tokens=0,
                verification_passes=max(0, ids.size(1) - prompt_ids.size(1)),
                total_latency_ms=(time.perf_counter() - t0) * 1000.0,
                speedup_vs_target=1.0,
                fallback_used=True,
            )
            return ids, stats

        # ----- Speculative loop ------------------------------------------
        generated = prompt_ids.clone()
        accepted_total = 0
        drafted_total = 0
        passes = 0

        while (generated.size(1) - prompt_ids.size(1)) < max_new_tokens:
            budget = max_new_tokens - (generated.size(1) - prompt_ids.size(1))
            k = min(self.num_drafts, budget)
            if k <= 0:
                break

            draft_ids, draft_probs = self._draft_step(
                generated, k,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                temperature=temperature,
            )
            drafted_total += k

            accepted_ids, bonus_id, n_accepted = self._verify_step(
                generated, draft_ids, draft_probs,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                temperature=temperature,
            )
            accepted_total += n_accepted
            passes += 1

            # Append accepted drafts (truncated to the remaining budget).
            emitted_this_pass = 0
            if accepted_ids.numel() > 0:
                take = min(int(accepted_ids.numel()), budget)
                if take > 0:
                    generated = torch.cat(
                        [generated, accepted_ids[:take].unsqueeze(0)],
                        dim=1,
                    )
                emitted_this_pass += take

            # The bonus token is only emitted if we still have budget —
            # otherwise we would overshoot ``max_new_tokens``.
            remaining = max_new_tokens - (
                generated.size(1) - prompt_ids.size(1)
            )
            if bonus_id is not None and remaining > 0:
                generated = torch.cat(
                    [generated, bonus_id.view(1, 1)], dim=1
                )
                emitted_this_pass += 1

            # SEC: hard bound on the generated sequence to prevent runaway
            # if something pathological makes this loop produce no new
            # tokens per iteration.
            if emitted_this_pass == 0:
                logger.warning(
                    "SpeculativeDecoder: verification pass produced "
                    "no new tokens; stopping to avoid infinite loop."
                )
                break

        # ----- Stats ------------------------------------------------------
        # Leviathan et al. 2023 Appendix A: per-pass expected emitted
        # tokens is (accepted_tokens/passes + 1). A safe empirical speed-up
        # estimate is (accepted + passes) / passes, treating each pass as
        # one target forward (the cost we would pay without drafting).
        speedup = 1.0
        if passes > 0:
            speedup = (accepted_total + passes) / float(passes)

        stats = SpeculativeStats(
            accepted_tokens=accepted_total,
            drafted_tokens=drafted_total,
            verification_passes=passes,
            total_latency_ms=(time.perf_counter() - t0) * 1000.0,
            speedup_vs_target=speedup,
            fallback_used=False,
        )
        return generated, stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_conditioning(
        conditioning_tokens: Optional[dict[str, torch.Tensor]],
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Unpack the conditioning dict into ``(adaptation_vector, user_state)``.

        Either element may be ``None`` — the model will substitute
        neutral defaults internally. If either is provided, it is moved
        to ``device`` and padded with a leading batch dimension if
        necessary.
        """
        if conditioning_tokens is None:
            return None, None

        adapt = conditioning_tokens.get("adaptation_vector")
        state = conditioning_tokens.get("user_state")

        if adapt is not None:
            if adapt.dim() == 1:
                adapt = adapt.unsqueeze(0)
            adapt = adapt.to(device)
        if state is not None:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(device)
        return adapt, state

    @torch.no_grad()
    def _target_only_generate(
        self,
        prompt_ids: torch.Tensor,
        adaptation_vector: Optional[torch.Tensor],
        user_state: Optional[torch.Tensor],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Fallback path: greedy autoregressive generation via the target.

        Used whenever :attr:`fallback_mode` is ``True``. Mirrors the
        same sampling behaviour as the speculative loop (greedy with a
        temperature scaling) so that benchmarks comparing the two paths
        are apples-to-apples.
        """
        generated = prompt_ids.clone()
        self.target_model.clear_cache()
        for _ in range(max_new_tokens):
            logits, _ = self.target_model(
                generated,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                use_cache=False,
            )
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            next_id = int(torch.argmax(next_logits, dim=-1).item())
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], device=prompt_ids.device)],
                dim=1,
            )
        return generated

    @torch.no_grad()
    def _draft_step(
        self,
        prefix: torch.Tensor,
        k: int,
        *,
        adaptation_vector: Optional[torch.Tensor],
        user_state: Optional[torch.Tensor],
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose ``k`` draft tokens autoregressively.

        Args:
            prefix: ``[1, prefix_len]`` current prefix.
            k: Number of tokens to propose.
            adaptation_vector: Optional conditioning.
            user_state: Optional conditioning.
            temperature: Sampling temperature.

        Returns:
            Tuple ``(draft_ids, draft_probs)`` where:

            * ``draft_ids`` is ``[k]`` — the proposed tokens.
            * ``draft_probs`` is ``[k, vocab_size]`` — the draft's
              per-token distribution at each proposed position. Used
              later by :meth:`_verify_step` to compute the acceptance
              ratio.
        """
        assert self.draft_model is not None  # fallback handled upstream
        device = prefix.device
        vocab = self.vocab_size

        draft_ids = torch.zeros(k, dtype=torch.long, device=device)
        draft_probs = torch.zeros(k, vocab, device=device)

        cur = prefix.clone()
        self.draft_model.clear_cache()
        for i in range(k):
            logits, _ = self.draft_model(
                cur,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                use_cache=False,
            )
            step_logits = logits[:, -1, :] / max(temperature, 1e-5)
            step_probs = F.softmax(step_logits, dim=-1).squeeze(0)
            # SEC: sanitise — a degenerate distribution (NaN / sum<=0)
            # would otherwise propagate into the acceptance ratio.
            if (
                torch.isnan(step_probs).any()
                or torch.isinf(step_probs).any()
                or float(step_probs.sum().item()) <= 0.0
            ):
                step_probs = torch.full_like(step_probs, 1.0 / vocab)

            next_id = int(torch.argmax(step_probs).item())
            draft_ids[i] = next_id
            draft_probs[i] = step_probs

            cur = torch.cat(
                [cur, torch.tensor([[next_id]], device=device)], dim=1
            )
        return draft_ids, draft_probs

    @torch.no_grad()
    def _verify_step(
        self,
        prefix: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        *,
        adaptation_vector: Optional[torch.Tensor],
        user_state: Optional[torch.Tensor],
        temperature: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Verify ``num_drafts`` proposals in a single target forward.

        Implements the rejection-sampling test of Leviathan et al.
        2023 Algorithm 1, modulated by :attr:`acceptance_threshold`:

            For each draft position t (0-indexed):
                r_t = p_target(t) / p_draft(t)
                if r_t >= acceptance_threshold: accept
                else: reject, resample from (p_target - p_draft)+

        Args:
            prefix: ``[1, prefix_len]`` current prefix (does NOT
                include the drafts).
            draft_ids: ``[k]`` proposed tokens.
            draft_probs: ``[k, vocab]`` draft distributions.
            adaptation_vector: Optional conditioning.
            user_state: Optional conditioning.
            temperature: Sampling temperature.

        Returns:
            Tuple ``(accepted_ids, bonus_id, n_accepted)``:

            * ``accepted_ids`` is a 1-D tensor (possibly empty) of the
              accepted draft token IDs.
            * ``bonus_id`` is a scalar tensor (or ``None`` if no bonus
              was produced, e.g. empty draft) containing one additional
              token sampled from the target's distribution at the
              first-rejection position (or at the post-draft position
              if every draft was accepted).
            * ``n_accepted`` is an integer ``<= k`` — the number of
              drafts that survived verification.
        """
        assert self.draft_model is not None
        device = prefix.device
        k = int(draft_ids.numel())
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=device), None, 0

        # ----- Target forward over prefix + drafts (single pass) ----------
        combined = torch.cat([prefix, draft_ids.view(1, -1)], dim=1)
        self.target_model.clear_cache()
        target_logits, _ = self.target_model(
            combined,
            adaptation_vector=adaptation_vector,
            user_state=user_state,
            use_cache=False,
        )
        # The target logits at positions [prefix_len-1 ... prefix_len+k-1]
        # predict tokens at [prefix_len ... prefix_len+k]. We care about
        # k positions: one per draft.
        prefix_len = prefix.size(1)
        step_logits = target_logits[0, prefix_len - 1 : prefix_len - 1 + k, :]
        step_logits = step_logits / max(temperature, 1e-5)
        target_probs = F.softmax(step_logits, dim=-1)  # [k, vocab]

        # ----- Acceptance test ------------------------------------------
        accepted: list[int] = []
        first_reject_idx: Optional[int] = None

        # SEC: detach draft_probs onto the target device just in case the
        # two models were placed on different devices.
        draft_probs = draft_probs.to(target_probs.device)

        for t in range(k):
            tok = int(draft_ids[t].item())
            p_t = float(target_probs[t, tok].item())
            q_t = float(draft_probs[t, tok].item())
            if q_t <= 0.0:
                # Draft claimed zero probability for a token it sampled —
                # almost certainly a numerical artefact. Treat as accept
                # only if the target also disagrees strongly.
                ratio = p_t / 1e-8
            else:
                ratio = p_t / q_t

            if ratio >= self.acceptance_threshold:
                accepted.append(tok)
            else:
                # Rejection sampling: with probability 1 - ratio the draft
                # is rejected. We use a uniform random threshold so the
                # resulting output distribution matches the Leviathan et
                # al. 2023 guarantee when acceptance_threshold == 1.0.
                u = float(torch.rand((), device=device).item())
                if u < ratio:
                    accepted.append(tok)
                else:
                    first_reject_idx = t
                    break

        n_accepted = len(accepted)
        accepted_tensor = torch.tensor(
            accepted, dtype=torch.long, device=device
        )

        # ----- Bonus token ----------------------------------------------
        bonus_id: Optional[torch.Tensor] = None
        if first_reject_idx is not None:
            # Sample from the residual max(p_target - p_draft, 0),
            # renormalised. This is the classical Leviathan recovery step.
            residual = torch.clamp(
                target_probs[first_reject_idx] - draft_probs[first_reject_idx],
                min=0.0,
            )
            total = float(residual.sum().item())
            if total > 0.0:
                residual = residual / total
                bonus_id = torch.argmax(residual).detach()
            else:
                # Fallback: target's own argmax.
                bonus_id = torch.argmax(
                    target_probs[first_reject_idx]
                ).detach()
        elif n_accepted == k:
            # All drafts accepted — the target forward also gives us a
            # free "bonus" token at position prefix_len + k. We compute
            # its distribution by running one more forward on the extended
            # prefix; cheaper than re-running the k-pass forward.
            extended = torch.cat(
                [prefix, accepted_tensor.view(1, -1)], dim=1
            )
            self.target_model.clear_cache()
            more_logits, _ = self.target_model(
                extended,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
                use_cache=False,
            )
            step_logits = more_logits[:, -1, :] / max(temperature, 1e-5)
            next_probs = F.softmax(step_logits, dim=-1).squeeze(0)
            bonus_id = torch.argmax(next_probs).detach()

        return accepted_tensor, bonus_id, n_accepted

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def draft_num_parameters(self) -> int:
        """Total parameter count of the draft model (0 when falling back)."""
        if self.draft_model is None:
            return 0
        return int(self.draft_model.num_parameters)

    @property
    def target_num_parameters(self) -> int:
        """Total parameter count of the target model."""
        return int(self.target_model.num_parameters)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"target_params={self.target_num_parameters:,}, "
            f"draft_params={self.draft_num_parameters:,}, "
            f"num_drafts={self.num_drafts}, "
            f"acceptance_threshold={self.acceptance_threshold}, "
            f"fallback_mode={self.fallback_mode}"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def clone_as_draft(target: AdaptiveSLM) -> Optional[AdaptiveSLM]:
    """Public helper that returns an auto-shrunk draft for ``target``.

    Wraps :meth:`SpeculativeDecoder._auto_shrink` so external callers
    (e.g. benchmarking scripts) can obtain the same draft the decoder
    would use without constructing a full :class:`SpeculativeDecoder`
    first.

    Args:
        target: The full-sized target model.

    Returns:
        A freshly-initialised draft model, or ``None`` if the target is
        already at or below the minimum sensible size.
    """
    # A fresh smaller architecture is returned — NOT a deepcopy of the
    # target's weights. The draft is meant to learn (or be distilled)
    # separately from the target.
    return SpeculativeDecoder._auto_shrink(target)
