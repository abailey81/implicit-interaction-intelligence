"""Conditioning-sensitivity test for the Adaptive SLM.

This is the **novelty evaluation** for the I3 cross-attention
conditioning mechanism. The idea is simple: if cross-attention is
actually steering generation, then feeding the *same* prompt with
*different* AdaptationVectors should yield *different* next-token
probability distributions. We quantify the difference using the
symmetric Kullback-Leibler divergence between successive
distributions.

Reporting:

* Per-prompt KL matrix (``n_vectors x n_vectors``) in JSON.
* Summary statistics — mean / median / max pairwise KL per prompt,
  and an aggregate mean-of-means.
* Optional Markdown table suitable for pasting into a report.

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.tokenizer import SimpleTokenizer
    from i3.eval.conditioning_sensitivity import (
        measure_conditioning_sensitivity,
        standard_adaptation_vectors,
        report_markdown,
    )

    model = AdaptiveSLM().eval()
    tokenizer = SimpleTokenizer.load(...)
    results = measure_conditioning_sensitivity(
        model=model,
        tokenizer=tokenizer,
        prompts=["Tell me about machine learning.", "How are you feeling?"],
        adaptation_vectors=standard_adaptation_vectors(),
    )
    print(report_markdown(results))
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Standard adaptation vectors
# ---------------------------------------------------------------------------


def standard_adaptation_vectors() -> dict[str, torch.Tensor]:
    """Return the canonical four-corner AdaptationVector fixtures.

    Dimensions (as documented in :mod:`i3.slm.model`):

    0. cognitive_load
    1. verbosity
    2. technicality
    3. formality
    4. reading_level
    5. emotional_tone
    6. urgency
    7. accessibility

    Returns:
        A dict mapping label → ``[1, 8]`` FP32 tensor.
    """
    return {
        "neutral": torch.tensor(
            [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32
        ),
        "low_cognitive_warm": torch.tensor(
            [[0.1, 0.3, 0.2, 0.2, 0.2, 0.9, 0.2, 0.9]], dtype=torch.float32
        ),
        "high_cognitive_technical": torch.tensor(
            [[0.9, 0.8, 0.9, 0.8, 0.9, 0.4, 0.5, 0.3]], dtype=torch.float32
        ),
        "urgent_formal": torch.tensor(
            [[0.6, 0.3, 0.5, 0.9, 0.6, 0.3, 0.95, 0.5]], dtype=torch.float32
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TypeError("tokenizer must expose an encode() method")
    ids = encode(prompt)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids] or [getattr(tokenizer, "BOS_ID", 1)]


def _next_token_distribution(
    model: nn.Module,
    input_ids: torch.Tensor,
    adaptation_vector: torch.Tensor,
    user_state: torch.Tensor | None,
) -> torch.Tensor:
    """Return the next-token probability distribution ``[vocab_size]``."""
    with torch.inference_mode():
        try:
            logits, _ = model(
                input_ids=input_ids,
                adaptation_vector=adaptation_vector,
                user_state=user_state,
            )
        except TypeError:
            logits = model(input_ids)
    last_logits = logits[0, -1, :]
    return F.softmax(last_logits, dim=-1)


def _symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """Return the symmetric KL divergence in nats."""
    p = p.clamp_min(_EPS)
    q = q.clamp_min(_EPS)
    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q))).item()
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p))).item()
    return 0.5 * (kl_pq + kl_qp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def measure_conditioning_sensitivity(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    adaptation_vectors: dict[str, torch.Tensor],
    *,
    device: str = "cpu",
    user_state: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute pairwise KL divergences between conditioning variants.

    Args:
        model: The :class:`AdaptiveSLM` (or compatible) under test.
        tokenizer: Object with an ``encode(text) -> list[int]`` method.
        prompts: Prompts to evaluate; each produces a KL matrix.
        adaptation_vectors: Mapping label → ``[1, 8]`` conditioning
            tensor. Order matters — the output matrix rows/cols are
            indexed by the insertion order of the dict.
        device: Torch device string.
        user_state: Optional fixed user-state embedding.

    Returns:
        A dict with the following shape::

            {
              "labels": ["neutral", ...],
              "prompts": [
                {
                  "prompt": "...",
                  "kl_matrix": [[0.0, x, ...], ...],
                  "mean_kl": float,
                  "max_kl": float,
                },
                ...
              ],
              "aggregate_mean_kl": float,
              "aggregate_max_kl": float,
            }
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if not adaptation_vectors:
        raise ValueError("adaptation_vectors must be non-empty")

    model = model.to(device).eval()
    labels = list(adaptation_vectors.keys())
    vectors = [
        adaptation_vectors[k].to(device).float() for k in labels
    ]

    prompt_results: list[dict[str, Any]] = []
    all_kls: list[float] = []

    for prompt in prompts:
        ids = _encode_prompt(tokenizer, prompt)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        distributions: list[torch.Tensor] = []
        for vec in vectors:
            dist = _next_token_distribution(
                model,
                input_ids=input_ids,
                adaptation_vector=vec,
                user_state=user_state,
            )
            distributions.append(dist)

        n = len(distributions)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        pairwise: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                kl = _symmetric_kl(distributions[i], distributions[j])
                matrix[i][j] = kl
                matrix[j][i] = kl
                pairwise.append(kl)

        mean_kl = statistics.fmean(pairwise) if pairwise else 0.0
        max_kl = max(pairwise) if pairwise else 0.0
        all_kls.extend(pairwise)
        prompt_results.append(
            {
                "prompt": prompt,
                "kl_matrix": matrix,
                "mean_kl": mean_kl,
                "max_kl": max_kl,
            }
        )

    return {
        "labels": labels,
        "prompts": prompt_results,
        "aggregate_mean_kl": (
            statistics.fmean(all_kls) if all_kls else 0.0
        ),
        "aggregate_max_kl": max(all_kls) if all_kls else 0.0,
    }


def report_markdown(results: dict[str, Any]) -> str:
    """Render *results* as a Markdown summary table.

    Produces one table per prompt with KL values in nats.
    """
    lines: list[str] = []
    labels: list[str] = list(results.get("labels", []))
    if not labels:
        return "_no results_"
    header_cells = ["prompt / pair"] + labels
    for entry in results.get("prompts", []):
        prompt_preview = entry["prompt"][:40].replace("\n", " ")
        lines.append(f"### Prompt: `{prompt_preview}`")
        lines.append("")
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
        matrix: list[list[float]] = entry["kl_matrix"]
        for i, row_label in enumerate(labels):
            row = [row_label] + [f"{matrix[i][j]:.4f}" for j in range(len(labels))]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            f"**mean_kl** = {entry['mean_kl']:.4f}, "
            f"**max_kl** = {entry['max_kl']:.4f}"
        )
        lines.append("")
    lines.append(
        f"**aggregate_mean_kl** = {results.get('aggregate_mean_kl', 0.0):.4f}, "
        f"**aggregate_max_kl** = {results.get('aggregate_max_kl', 0.0):.4f}"
    )
    return "\n".join(lines)


def save_report(results: dict[str, Any], out_path: Path) -> Path:
    """Write *results* as JSON to *out_path* and return the path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "measure_conditioning_sensitivity",
    "report_markdown",
    "save_report",
    "standard_adaptation_vectors",
]
