"""Autoregressive text generation with sampling for the I3 Small Language Model.

Built from scratch -- no HuggingFace generate() or transformers library.
Implements temperature scaling, top-k filtering, top-p (nucleus) sampling,
repetition penalty, and KV caching for efficient token-by-token generation.

The generator supports conditioning on an AdaptationVector (8-dim behavioural
targets) and UserStateEmbedding (64-dim temporal context from the TCN encoder),
enabling the model to modulate style, complexity, and tone throughout the
generated response.

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer
    from i3.slm.generate import SLMGenerator

    model = AdaptiveSLM()
    tokenizer = SimpleTokenizer.load("models/slm/tokenizer.json")
    generator = SLMGenerator(model, tokenizer)

    text = generator.generate("Hello, how are you?", temperature=0.8)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from i3.runtime.device import pick_device

if TYPE_CHECKING:
    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class SLMGenerator:
    """Autoregressive text generation with temperature, top-k, top-p sampling.

    Supports:
    - Temperature scaling for controlling randomness
    - Top-k filtering to restrict to the k most likely tokens
    - Top-p (nucleus) sampling to dynamically limit the candidate set
    - Repetition penalty to discourage token repetition
    - KV caching for efficient autoregressive generation
    - Conditioning on AdaptationVector + UserStateEmbedding

    Parameters
    ----------
    model : AdaptiveSLM
        The language model to generate from.
    tokenizer : SimpleTokenizer
        Tokenizer for encoding prompts and decoding outputs.
    device : str
        Device to run generation on (default ``"cpu"``).
    """

    def __init__(
        self,
        model: AdaptiveSLM,
        tokenizer: SimpleTokenizer,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        # PERF: pick_device promotes to CUDA/MPS when available; callers
        # that explicitly passed "cpu" keep CPU behaviour unchanged.
        self.device = pick_device(device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    # SEC: Hard upper bound on generation length to prevent runaway loops
    # even if a caller passes a pathologically large max_new_tokens.
    HARD_MAX_NEW_TOKENS: int = 4096

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        adaptation_vector: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        stop_tokens: list[int] | None = None,
    ) -> str:
        """Generate text autoregressively from a prompt.

        Parameters
        ----------
        prompt : str
            Input text to condition generation on.
        adaptation_vector : torch.Tensor, optional
            AdaptationVector of shape ``[1, 8]`` or ``[8]``.
            If None, a neutral default is used.
        user_state : torch.Tensor, optional
            UserStateEmbedding of shape ``[1, 64]`` or ``[64]``.
            If None, zeros are used.
        max_new_tokens : int
            Maximum number of new tokens to generate (default 100).
        temperature : float
            Sampling temperature. Higher = more random, lower = more
            deterministic. Must be > 0 (default 0.8).
        top_k : int
            Number of highest-probability tokens to keep. Set to 0 to
            disable top-k filtering (default 50).
        top_p : float
            Cumulative probability threshold for nucleus sampling.
            Set to 1.0 to disable (default 0.9).
        repetition_penalty : float
            Penalty factor for previously generated tokens. 1.0 = no
            penalty. Values > 1.0 discourage repetition (default 1.2).
        stop_tokens : list[int], optional
            Token IDs that trigger generation to stop. Defaults to
            ``[EOS_ID]``.

        Returns
        -------
        str
            The generated text (prompt + continuation), decoded with
            special tokens removed.
        """
        # SEC: Validate sampling parameters to fail-fast on caller errors.
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {repetition_penalty}"
            )
        # SEC: Hard cap max_new_tokens to prevent infinite generation if
        # a caller passes an absurd value or the model never emits EOS.
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        max_new_tokens = min(max_new_tokens, self.HARD_MAX_NEW_TOKENS)

        # SEC: Treat near-zero temperature as greedy decoding rather than
        # dividing by ~0 (which would produce inf logits and NaN softmax).
        # 1e-5 matches the precision threshold used by major libraries.
        greedy = temperature <= 1e-5

        # SEC: Always clear cache on entry AND in a finally block on exit
        # so a previous session's KV state cannot leak into this generation
        # and cannot persist after a crash mid-generation.
        self.model.clear_cache()

        try:
            # 1. Encode prompt
            input_ids = self.tokenizer.encode(prompt, add_special=True)
            input_tensor = torch.tensor([input_ids], device=self.device)

            # 2. Default stop tokens
            if stop_tokens is None:
                stop_tokens = [self.tokenizer.EOS_ID]
            # SEC: Use a set for O(1) membership and to dedupe input.
            stop_set: set[int] = set(stop_tokens)
            # SEC: Identify EOS/PAD/BOS so the repetition penalty does NOT
            # discourage them — penalising EOS makes the model unable to
            # terminate, which is the most common cause of runaway gen.
            protected_ids: set[int] = {
                self.tokenizer.PAD_ID,
                self.tokenizer.BOS_ID,
                self.tokenizer.EOS_ID,
            }

            # 3. Prepare conditioning tensors
            if adaptation_vector is not None:
                if adaptation_vector.dim() == 1:
                    adaptation_vector = adaptation_vector.unsqueeze(0)
                adaptation_vector = adaptation_vector.to(self.device)

            if user_state is not None:
                if user_state.dim() == 1:
                    user_state = user_state.unsqueeze(0)
                user_state = user_state.to(self.device)

            generated_ids: list[int] = list(input_ids)

            # SEC: Resolve the model's positional encoding limit so we can
            # bound the cache and refuse to generate past it. Falls back to
            # the attention layer's MAX_CACHE_LEN if the model lacks the
            # standard embedding structure.
            try:
                pos_max = self.model.embedding.positional_encoding.pe.size(1)
            except AttributeError:
                pos_max = 2048

            for _ in range(max_new_tokens):
                # SEC: Refuse to generate past the positional encoding limit.
                # Without this guard, the next forward pass would fall over
                # inside SinusoidalPositionalEncoding (or silently produce
                # nonsense if its bounds check were ever relaxed).
                if len(generated_ids) >= pos_max:
                    logger.warning(
                        "Reached positional encoding limit (%d); stopping.",
                        pos_max,
                    )
                    break

                # Forward pass with cache
                logits, _ = self.model(
                    input_tensor,
                    adaptation_vector,
                    user_state,
                    use_cache=True,
                )
                next_logits = logits[:, -1, :]  # [1, vocab_size]

                # Apply repetition penalty (skipping protected ids)
                next_logits = self._apply_repetition_penalty(
                    next_logits,
                    generated_ids,
                    repetition_penalty,
                    protected_ids=protected_ids,
                )

                if greedy:
                    # SEC: Greedy decode — argmax instead of sampling.
                    next_token = int(torch.argmax(next_logits, dim=-1).item())
                else:
                    # Temperature scaling
                    next_logits = next_logits / temperature

                    # Top-k filtering
                    if top_k > 0:
                        next_logits = self._top_k_filter(next_logits, top_k)

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        next_logits = self._top_p_filter(next_logits, top_p)

                    # Sample from the filtered distribution
                    probs = F.softmax(next_logits, dim=-1)

                    # SEC: Degenerate-distribution guard. If filtering and
                    # numerical issues collapse probs to all-NaN or all-zero
                    # (sum<=0), torch.multinomial will crash or return junk.
                    # Fall back to uniform over the vocabulary.
                    if (
                        torch.isnan(probs).any()
                        or torch.isinf(probs).any()
                        or probs.sum().item() <= 0.0
                    ):
                        logger.warning(
                            "Degenerate sampling distribution; "
                            "falling back to uniform."
                        )
                        probs = torch.full_like(
                            probs, 1.0 / probs.size(-1)
                        )

                    next_token = int(
                        torch.multinomial(probs, num_samples=1).item()
                    )

                # Check stop condition
                if next_token in stop_set:
                    break

                generated_ids.append(next_token)
                input_tensor = torch.tensor(
                    [[next_token]], device=self.device
                )

            return self.tokenizer.decode(generated_ids, skip_special=True)

        finally:
            # SEC: Double-cleanup — guarantees cache is cleared even on
            # exception so subsequent generations start fresh.
            self.model.clear_cache()

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        adaptation_vectors: list[torch.Tensor] | None = None,
        user_states: list[torch.Tensor] | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> list[str]:
        """Generate text for multiple prompts.

        Each prompt is generated independently (no batched KV cache sharing)
        to keep the implementation simple and correct. For the I3 system this
        is acceptable because generation is typically single-user.

        Parameters
        ----------
        prompts : list[str]
            List of input prompts.
        adaptation_vectors : list[torch.Tensor], optional
            Per-prompt adaptation vectors. If None, neutral defaults are used.
        user_states : list[torch.Tensor], optional
            Per-prompt user state embeddings. If None, zeros are used.
        max_new_tokens : int
            Maximum new tokens per generation.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k filtering parameter.
        top_p : float
            Nucleus sampling threshold.
        repetition_penalty : float
            Repetition penalty factor.

        Returns
        -------
        list[str]
            Generated texts, one per prompt.
        """
        results: list[str] = []

        for i, prompt in enumerate(prompts):
            av = adaptation_vectors[i] if adaptation_vectors else None
            us = user_states[i] if user_states else None

            text = self.generate(
                prompt=prompt,
                adaptation_vector=av,
                user_state=us,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            results.append(text)

        return results

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_confidence(
        self,
        prompt: str,
        generated_text: str,
        adaptation_vector: torch.Tensor | None = None,
        user_state: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Estimate the model's confidence in a generated sequence.

        Runs a single forward pass over the full sequence (prompt +
        generation) and computes the average token-level probability and
        perplexity for the generated portion.

        Parameters
        ----------
        prompt : str
            The original prompt.
        generated_text : str
            The full generated text (including prompt).
        adaptation_vector : torch.Tensor, optional
            Adaptation vector used during generation.
        user_state : torch.Tensor, optional
            User state embedding used during generation.

        Returns
        -------
        dict[str, float]
            Dictionary with keys:

            - ``"mean_probability"`` -- average probability of generated tokens
            - ``"min_probability"`` -- minimum probability among generated tokens
            - ``"perplexity"`` -- perplexity over the generated tokens
            - ``"num_tokens"`` -- number of generated tokens measured
        """
        # Encode prompt and full text
        prompt_ids = self.tokenizer.encode(prompt, add_special=True)
        full_ids = self.tokenizer.encode(generated_text, add_special=True)

        prompt_len = len(prompt_ids)
        if len(full_ids) <= prompt_len:
            return {
                "mean_probability": 1.0,
                "min_probability": 1.0,
                "perplexity": 1.0,
                "num_tokens": 0,
            }

        # Prepare conditioning
        if adaptation_vector is not None:
            if adaptation_vector.dim() == 1:
                adaptation_vector = adaptation_vector.unsqueeze(0)
            adaptation_vector = adaptation_vector.to(self.device)
        if user_state is not None:
            if user_state.dim() == 1:
                user_state = user_state.unsqueeze(0)
            user_state = user_state.to(self.device)

        # SEC: Wrap in try/finally to guarantee cache cleanup even on
        # exception so subsequent calls start with a clean state.
        self.model.clear_cache()
        try:
            input_tensor = torch.tensor([full_ids], device=self.device)

            logits, _ = self.model(
                input_tensor, adaptation_vector, user_state, use_cache=False
            )

            # Compute per-token probabilities for the generated portion
            # logits[:, t, :] predicts token at position t+1
            probs = F.softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

            token_probs: list[float] = []
            vocab_size = probs.size(-1)
            for t in range(prompt_len - 1, len(full_ids) - 1):
                target_token = full_ids[t + 1]
                # SEC: Out-of-range token id guard — if the saved text was
                # encoded with a different tokenizer, target_token could be
                # out of range and crash the indexing op.
                if not 0 <= target_token < vocab_size:
                    continue
                prob = probs[0, t, target_token].item()
                token_probs.append(prob)

            if not token_probs:
                return {
                    "mean_probability": 1.0,
                    "min_probability": 1.0,
                    "perplexity": 1.0,
                    "num_tokens": 0,
                }

            mean_prob = sum(token_probs) / len(token_probs)
            min_prob = min(token_probs)

            # Perplexity = exp(mean negative log-likelihood)
            log_probs = [math.log(max(p, 1e-10)) for p in token_probs]
            avg_nll = -sum(log_probs) / len(log_probs)
            perplexity = math.exp(avg_nll)

            return {
                "mean_probability": mean_prob,
                "min_probability": min_prob,
                "perplexity": perplexity,
                "num_tokens": len(token_probs),
            }
        finally:
            self.model.clear_cache()

    # ------------------------------------------------------------------
    # Sampling helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float,
        protected_ids: set[int] | None = None,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits for previously generated tokens.

        For each token that has already been generated:
        - If its logit is positive, divide by the penalty factor
        - If its logit is negative, multiply by the penalty factor

        This makes repeated tokens less likely regardless of whether their
        raw logit is positive or negative.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits of shape ``[1, vocab_size]``.
        generated_ids : list[int]
            List of all token IDs generated so far.
        penalty : float
            Penalty factor (1.0 = no penalty).
        protected_ids : set[int], optional
            Token IDs that must NOT be penalised (typically EOS / PAD / BOS).
            Penalising EOS in particular makes the model unable to terminate.

        Returns
        -------
        torch.Tensor
            Modified logits with the same shape.
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        # SEC: Drop protected ids (EOS/PAD/BOS) from the penalty set so the
        # model can still terminate naturally.
        unique_ids = set(generated_ids)
        if protected_ids:
            unique_ids -= protected_ids
        if not unique_ids:
            return logits

        # SEC: Clamp ids to vocab range to avoid IndexError on bogus inputs.
        vocab_size = logits.size(-1)
        valid_ids = [i for i in unique_ids if 0 <= i < vocab_size]
        if not valid_ids:
            return logits

        prev_tokens = torch.tensor(
            valid_ids,
            dtype=torch.long,
            device=logits.device,
        )

        # Gather logits for previously generated tokens
        prev_logits = logits[0, prev_tokens]

        # Apply penalty: divide positive logits, multiply negative logits
        penalised = torch.where(
            prev_logits > 0,
            prev_logits / penalty,
            prev_logits * penalty,
        )

        # Scatter back
        logits = logits.clone()
        logits[0, prev_tokens] = penalised

        return logits

    @staticmethod
    def _top_k_filter(
        logits: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Filter logits to keep only the top-k highest values.

        All logits below the k-th highest value are set to a large negative
        value so that they receive ~zero probability after softmax.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits of shape ``[1, vocab_size]``.
        k : int
            Number of top tokens to keep.

        Returns
        -------
        torch.Tensor
            Filtered logits with the same shape.
        """
        # SEC: Edge cases — k <= 0 disables top-k; k >= vocab_size keeps all.
        # k == 1 collapses to greedy at this layer (still safe).
        if k <= 0 or k >= logits.size(-1):
            return logits

        # Find the k-th largest value as the threshold
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)  # [1, 1]

        # SEC: Use a large finite negative instead of literal -inf so that
        # subsequent additions / temperature scaling cannot produce NaN
        # via -inf * 0 or -inf + finite. softmax of -1e9 is effectively 0.
        filtered = logits.clone()
        filtered[filtered < threshold] = -1.0e9

        return filtered

    @staticmethod
    def _top_p_filter(
        logits: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits.

        Sorts tokens by probability in descending order, computes the
        cumulative probability, and masks out all tokens whose cumulative
        probability exceeds the threshold ``p``. The first token is always
        kept to ensure at least one valid option.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits of shape ``[1, vocab_size]``.
        p : float
            Cumulative probability threshold in (0, 1].

        Returns
        -------
        torch.Tensor
            Filtered logits with the same shape.
        """
        # SEC: Edge cases. p >= 1 disables filtering; p <= 0 collapses to
        # the single most likely token (still 1 valid option, never empty).
        if p >= 1.0:
            return logits
        if p <= 0.0:
            # Keep only the argmax — guaranteed at least one valid token.
            top_idx = torch.argmax(logits, dim=-1, keepdim=True)
            filtered = torch.full_like(logits, -1.0e9)
            filtered.scatter_(1, top_idx, logits.gather(1, top_idx))
            return filtered

        # SEC: Operate on a clone so we never mutate the caller's tensor
        # in-place via the sorted view (sorted_logits shares storage with
        # the result of torch.sort but writing through it via scatter_ is
        # safer with an explicit clone).
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        # Compute cumulative probabilities from sorted logits
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # SEC: Create mask — True for tokens to REMOVE (cumulative prob > p).
        # Use (cumulative - prob > p) so the FIRST token whose cumulative
        # crosses p is still kept. This guarantees at least one token
        # survives even if a single token already exceeds p.
        sorted_mask = (cumulative_probs - sorted_probs) > p
        # SEC: Belt-and-braces — force the first ranked token to always
        # remain valid even under odd numerics (NaN comparisons, etc.).
        sorted_mask[..., 0] = False

        # SEC: Use a large finite negative instead of literal -inf
        # (avoids NaN propagation via subsequent arithmetic).
        sorted_logits = sorted_logits.masked_fill(sorted_mask, -1.0e9)

        # Scatter back to original ordering
        filtered = torch.empty_like(logits)
        filtered.scatter_(1, sorted_indices, sorted_logits)

        return filtered
