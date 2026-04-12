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
from typing import Optional

import torch
import torch.nn.functional as F

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
        model: "AdaptiveSLM",
        tokenizer: "SimpleTokenizer",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        adaptation_vector: Optional[torch.Tensor] = None,
        user_state: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        stop_tokens: Optional[list[int]] = None,
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
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        # 1. Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # 2. Default stop tokens
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.EOS_ID]

        # 3. Prepare conditioning tensors
        if adaptation_vector is not None:
            if adaptation_vector.dim() == 1:
                adaptation_vector = adaptation_vector.unsqueeze(0)
            adaptation_vector = adaptation_vector.to(self.device)

        if user_state is not None:
            if user_state.dim() == 1:
                user_state = user_state.unsqueeze(0)
            user_state = user_state.to(self.device)

        # 4. Clear KV cache for fresh generation
        self.model.clear_cache()

        generated_ids: list[int] = list(input_ids)

        for _ in range(max_new_tokens):
            # Forward pass with cache
            logits, _ = self.model(
                input_tensor,
                adaptation_vector,
                user_state,
                use_cache=True,
            )
            next_logits = logits[:, -1, :]  # [1, vocab_size]

            # Apply repetition penalty
            next_logits = self._apply_repetition_penalty(
                next_logits, generated_ids, repetition_penalty
            )

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
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Check stop condition
            if next_token in stop_tokens:
                break

            generated_ids.append(next_token)
            input_tensor = torch.tensor([[next_token]], device=self.device)

        # 5. Clean up cache
        self.model.clear_cache()

        return self.tokenizer.decode(generated_ids, skip_special=True)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        adaptation_vectors: Optional[list[torch.Tensor]] = None,
        user_states: Optional[list[torch.Tensor]] = None,
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
        adaptation_vector: Optional[torch.Tensor] = None,
        user_state: Optional[torch.Tensor] = None,
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

        # Forward pass over the full sequence (no cache needed for scoring)
        self.model.clear_cache()
        input_tensor = torch.tensor([full_ids], device=self.device)

        logits, _ = self.model(
            input_tensor, adaptation_vector, user_state, use_cache=False
        )

        # Compute per-token probabilities for the generated portion
        # logits[:, t, :] predicts token at position t+1
        probs = F.softmax(logits, dim=-1)  # [1, seq_len, vocab_size]

        token_probs: list[float] = []
        for t in range(prompt_len - 1, len(full_ids) - 1):
            target_token = full_ids[t + 1]
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

        self.model.clear_cache()

        return {
            "mean_probability": mean_prob,
            "min_probability": min_prob,
            "perplexity": perplexity,
            "num_tokens": len(token_probs),
        }

    # ------------------------------------------------------------------
    # Sampling helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float,
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

        Returns
        -------
        torch.Tensor
            Modified logits with the same shape.
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        # Get unique previously generated token IDs
        prev_tokens = torch.tensor(
            list(set(generated_ids)),
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

        All logits below the k-th highest value are set to ``-inf`` so that
        they receive zero probability after softmax.

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
        if k <= 0 or k >= logits.size(-1):
            return logits

        # Find the k-th largest value as the threshold
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)  # [1, 1]

        # Mask everything below the threshold
        filtered = logits.clone()
        filtered[filtered < threshold] = float("-inf")

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
        if p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        # Compute cumulative probabilities from sorted logits
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask: True for tokens to REMOVE (cumulative prob > p)
        # Shift right by 1 so the first token exceeding p is still kept
        sorted_mask = cumulative_probs - sorted_probs > p

        # Set masked logits to -inf
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original ordering
        filtered = logits.clone()
        filtered.scatter_(1, sorted_indices, sorted_logits)

        return filtered
