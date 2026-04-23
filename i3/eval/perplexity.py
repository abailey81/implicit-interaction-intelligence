"""Held-out perplexity evaluation for the Adaptive SLM.

Computes token-level perplexity (``exp(mean negative log-likelihood)``)
over a plain-text corpus. The evaluator slides a fixed-length
context window over the corpus and aggregates the cross-entropy
loss at non-padded positions.

Works with any tokenizer that exposes an ``encode(text) -> list[int]``
method returning token ids. The default implementation targets
:class:`i3.slm.tokenizer.SimpleTokenizer`.

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer
    from i3.eval.perplexity import compute_perplexity

    model = AdaptiveSLM().eval()
    tokenizer = SimpleTokenizer.load("models/slm/tokenizer.json")
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        corpus_path=Path("data/val.txt"),
        context_len=128,
    )
    print(f"perplexity = {ppl:.3f}")
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_corpus(tokenizer: Any, text: str) -> list[int]:
    """Tokenise *text* using the supplied tokenizer.

    Supports the I3 :class:`SimpleTokenizer` (``encode(text)``) and
    any object whose ``encode`` method returns a sequence of ints.
    """
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose an encode() method")
    ids = encode(text)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


def _chunks(
    ids: list[int], context_len: int, stride: int | None = None
) -> Iterable[list[int]]:
    """Yield overlapping chunks of length ``context_len + 1``.

    We need one extra token so the target shift (``labels = input_ids[1:]``)
    is well-defined for the final position.
    """
    stride = stride or context_len
    if len(ids) < 2:
        return
    window = context_len + 1
    for start in range(0, len(ids) - 1, stride):
        chunk = ids[start : start + window]
        if len(chunk) < 2:
            break
        yield chunk


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    corpus_path: Path,
    *,
    context_len: int = 128,
    stride: int | None = None,
    device: str = "cpu",
    adaptation_vector: torch.Tensor | None = None,
    user_state: torch.Tensor | None = None,
    pad_id: int = 0,
) -> float:
    """Return held-out perplexity over *corpus_path*.

    The method streams overlapping windows of length
    ``context_len + 1`` through the SLM, computes the per-position
    cross-entropy against the shifted labels, and returns
    ``exp(mean_nll)``.

    Args:
        model: The :class:`AdaptiveSLM` (or compatible) model. Will be
            moved to *device* and placed in ``eval`` mode.
        tokenizer: Object with an ``encode(text) -> list[int]`` method.
        corpus_path: Plain-text UTF-8 file to evaluate on.
        context_len: Context window length in tokens.
        stride: Step size between consecutive windows; defaults to
            ``context_len`` (non-overlapping).
        device: Torch device string.
        adaptation_vector: Optional ``[1, 8]`` conditioning vector
            reused for every window. Defaults to the model's neutral
            conditioning.
        user_state: Optional ``[1, 64]`` user-state embedding.
        pad_id: Token id used for padding; positions equal to this id
            are excluded from the NLL aggregate.

    Returns:
        The perplexity as a Python float. Returns ``float('inf')`` if
        the corpus is too short or produced no valid tokens.

    Raises:
        FileNotFoundError: If *corpus_path* does not exist.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        return float("inf")

    ids = _encode_corpus(tokenizer, text)
    if len(ids) < 2:
        return float("inf")

    model = model.to(device).eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for chunk in _chunks(ids, context_len=context_len, stride=stride):
            input_ids = torch.tensor(
                chunk[:-1], dtype=torch.long, device=device
            ).unsqueeze(0)
            labels = torch.tensor(
                chunk[1:], dtype=torch.long, device=device
            ).unsqueeze(0)

            try:
                logits, _ = model(
                    input_ids=input_ids,
                    adaptation_vector=adaptation_vector,
                    user_state=user_state,
                )
            except TypeError:
                # Fallback for models that return just logits.
                logits = model(input_ids)
            # logits: [1, seq_len, vocab]; labels: [1, seq_len]
            loss_mask = (labels != pad_id).float()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            nll_sum = (nll * loss_mask).sum().item()
            n_tokens = int(loss_mask.sum().item())
            total_nll += nll_sum
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    mean_nll = total_nll / float(total_tokens)
    return math.exp(min(mean_nll, 50.0))


__all__ = ["compute_perplexity"]
