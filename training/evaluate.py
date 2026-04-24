"""Evaluation script for the Adaptive Small Language Model.

Computes a comprehensive set of metrics for assessing SLM quality:

1. **Perplexity** -- standard language modelling metric on the validation set.
2. **Conditioning sensitivity** -- measures how much the model's output
   changes when the same input is paired with different AdaptationVectors.
3. **Diversity** -- Distinct-1 and Distinct-2 ratios (unique unigrams/bigrams
   relative to total) following Li et al., 2016.
4. **Length compliance** -- tests whether cognitive_load conditioning
   correlates with response length as expected.
5. **Sample generation** -- produces example outputs for manual inspection.

Usage::

    python -m training.evaluate --checkpoint models/slm/best_model.pt
    python -m training.evaluate --checkpoint models/slm/best_model.pt \\
           --data-dir data/processed/dialogue --num-samples 20

All metrics are logged and optionally saved to a JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on sys.path for absolute imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from i3.config import load_config
from i3.runtime.device import enable_cuda_optimizations, pick_device
from i3.slm.model import AdaptiveSLM
from i3.slm.tokenizer import SimpleTokenizer
from i3.slm.generate import SLMGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset class (same structure as train_slm.py)
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """Evaluation dataset loaded from a ``.pt`` file."""

    def __init__(self, path: str | Path) -> None:
        # Training/eval data `.pt` files are plain dict[str, Tensor]
        # produced by our own preprocessing pipeline; safe to load.
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.input_ids: torch.Tensor = data["input_ids"]
        self.target_ids: torch.Tensor = data["target_ids"]
        self.conditioning: torch.Tensor = data["conditioning"]
        self.user_state: torch.Tensor = data["user_state"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "target_ids": self.target_ids[idx],
            "conditioning": self.conditioning[idx],
            "user_state": self.user_state[idx],
        }


# ---------------------------------------------------------------------------
# 1. Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: AdaptiveSLM,
    data_loader: DataLoader,
    device: torch.device,
    pad_id: int = 0,
) -> dict[str, float]:
    """Compute perplexity on a dataset.

    Parameters
    ----------
    model : AdaptiveSLM
        The model to evaluate.
    data_loader : DataLoader
        Data loader yielding batch dicts.
    device : torch.device
        Device to run on.
    pad_id : int
        Padding token ID for masking.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``"perplexity"`` -- exp(avg cross-entropy loss)
        - ``"avg_loss"`` -- average cross-entropy loss
        - ``"num_tokens"`` -- total non-padding tokens evaluated
        - ``"num_batches"`` -- number of batches
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        conditioning = batch["conditioning"].to(device)
        user_state = batch["user_state"].to(device)

        logits, _ = model(input_ids, conditioning, user_state, use_cache=False)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()

        # Per-token loss (unreduced)
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_targets.view(-1),
            ignore_index=pad_id,
            reduction="sum",
        )

        # Count non-padding tokens
        non_pad = (shift_targets != pad_id).sum().item()
        total_loss += loss_per_token.item()
        total_tokens += non_pad
        n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100.0))

    result = {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "num_tokens": total_tokens,
        "num_batches": n_batches,
    }

    logger.info(
        "Perplexity: %.2f (avg_loss=%.4f, tokens=%d)",
        perplexity, avg_loss, total_tokens,
    )

    return result


# ---------------------------------------------------------------------------
# 2. Conditioning sensitivity test
# ---------------------------------------------------------------------------

@torch.no_grad()
def conditioning_sensitivity(
    model: AdaptiveSLM,
    tokenizer: SimpleTokenizer,
    test_prompts: list[str],
    device: torch.device,
    n_conditioning_variants: int = 5,
) -> dict[str, Any]:
    """Test conditioning sensitivity: same input, different adaptation vectors.

    For each prompt, generates responses with different conditioning
    vectors and measures how much the logit distributions diverge.

    Parameters
    ----------
    model : AdaptiveSLM
        The model to evaluate.
    tokenizer : SimpleTokenizer
        Tokenizer for encoding.
    test_prompts : list[str]
        List of prompts to test.
    device : torch.device
        Device to run on.
    n_conditioning_variants : int
        Number of different conditioning vectors to test per prompt.

    Returns
    -------
    dict[str, Any]
        Results with keys:
        - ``"mean_kl_divergence"`` -- average KL divergence between
          conditioning variants
        - ``"mean_cosine_distance"`` -- average cosine distance
        - ``"per_prompt"`` -- detailed per-prompt results
    """
    model.eval()

    # Define conditioning variants spanning the range
    variants: list[torch.Tensor] = []
    for i in range(n_conditioning_variants):
        t = i / max(n_conditioning_variants - 1, 1)
        vec = torch.zeros(8)
        vec[0] = t          # cognitive_load: 0 -> 1
        vec[1] = 1.0 - t    # formality: 1 -> 0
        vec[2] = t          # verbosity: 0 -> 1
        vec[3] = t * 0.8    # emotionality: 0 -> 0.8
        vec[4] = 0.5        # directness: constant
        vec[5] = 0.5        # emotional_tone: neutral
        vec[6] = 1.0 - t    # accessibility: 1 -> 0
        vec[7] = 0.0        # reserved
        variants.append(vec)

    per_prompt_results: list[dict[str, Any]] = []
    all_kl_divs: list[float] = []
    all_cosine_dists: list[float] = []

    for prompt in test_prompts:
        ids = tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([ids], device=device)

        logit_distributions: list[torch.Tensor] = []

        for variant in variants:
            av = variant.unsqueeze(0).to(device)
            logits, _ = model(input_tensor, av, use_cache=False)
            # Take the last token's distribution
            last_logits = logits[0, -1, :]  # [vocab_size]
            logit_distributions.append(last_logits)

        # Compute pairwise KL divergence and cosine distance
        kl_divs: list[float] = []
        cos_dists: list[float] = []

        for i in range(len(logit_distributions)):
            for j in range(i + 1, len(logit_distributions)):
                # SEC/numerical: use log_softmax instead of softmax().log()
                # to avoid log(0) underflow on extreme logits, and pass the
                # log-probabilities directly into kl_div as documented.
                log_p = F.log_softmax(logit_distributions[i], dim=-1)
                log_q = F.log_softmax(logit_distributions[j], dim=-1)
                p = log_p.exp()
                q = log_q.exp()

                # Symmetrised KL divergence.
                kl_pq = F.kl_div(log_q, p, reduction="sum").item()
                kl_qp = F.kl_div(log_p, q, reduction="sum").item()
                kl_sym = (kl_pq + kl_qp) / 2.0
                kl_divs.append(kl_sym)

                # Cosine distance
                cos_sim = F.cosine_similarity(
                    logit_distributions[i].unsqueeze(0),
                    logit_distributions[j].unsqueeze(0),
                ).item()
                cos_dists.append(1.0 - cos_sim)

        mean_kl = sum(kl_divs) / max(len(kl_divs), 1)
        mean_cos = sum(cos_dists) / max(len(cos_dists), 1)

        all_kl_divs.append(mean_kl)
        all_cosine_dists.append(mean_cos)

        per_prompt_results.append({
            "prompt": prompt[:80],
            "mean_kl_divergence": mean_kl,
            "mean_cosine_distance": mean_cos,
            "n_pairs": len(kl_divs),
        })

    overall_kl = sum(all_kl_divs) / max(len(all_kl_divs), 1)
    overall_cos = sum(all_cosine_dists) / max(len(all_cosine_dists), 1)

    result = {
        "mean_kl_divergence": overall_kl,
        "mean_cosine_distance": overall_cos,
        "per_prompt": per_prompt_results,
    }

    logger.info(
        "Conditioning sensitivity: KL_div=%.4f, cosine_dist=%.4f",
        overall_kl, overall_cos,
    )

    return result


# ---------------------------------------------------------------------------
# 3. Diversity metrics: Distinct-1, Distinct-2
# ---------------------------------------------------------------------------

def compute_diversity(
    generated_texts: list[str],
) -> dict[str, float]:
    """Compute Distinct-1 and Distinct-2 diversity metrics.

    Distinct-N is the ratio of unique N-grams to total N-grams across
    all generated texts. Higher values indicate more diverse generation.

    Reference: Li et al., "A Diversity-Promoting Objective Function for
    Neural Conversation Models", NAACL 2016.

    Parameters
    ----------
    generated_texts : list[str]
        List of generated response texts.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``"distinct_1"`` -- unique unigrams / total unigrams
        - ``"distinct_2"`` -- unique bigrams / total bigrams
        - ``"total_unigrams"`` -- total unigram count
        - ``"unique_unigrams"`` -- unique unigram count
        - ``"total_bigrams"`` -- total bigram count
        - ``"unique_bigrams"`` -- unique bigram count
        - ``"avg_length"`` -- average response length in words
    """
    all_unigrams: list[str] = []
    all_bigrams: list[tuple[str, str]] = []
    lengths: list[int] = []

    for text in generated_texts:
        words = text.lower().split()
        lengths.append(len(words))
        all_unigrams.extend(words)

        for i in range(len(words) - 1):
            all_bigrams.append((words[i], words[i + 1]))

    unique_unigrams = len(set(all_unigrams))
    total_unigrams = max(len(all_unigrams), 1)
    unique_bigrams = len(set(all_bigrams))
    total_bigrams = max(len(all_bigrams), 1)

    distinct_1 = unique_unigrams / total_unigrams
    distinct_2 = unique_bigrams / total_bigrams
    avg_length = sum(lengths) / max(len(lengths), 1)

    result = {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "total_unigrams": total_unigrams,
        "unique_unigrams": unique_unigrams,
        "total_bigrams": total_bigrams,
        "unique_bigrams": unique_bigrams,
        "avg_length": avg_length,
    }

    logger.info(
        "Diversity: distinct-1=%.4f, distinct-2=%.4f, avg_len=%.1f words",
        distinct_1, distinct_2, avg_length,
    )

    return result


# ---------------------------------------------------------------------------
# 4. Length compliance (cognitive_load -> response length correlation)
# ---------------------------------------------------------------------------

def length_compliance(
    model: AdaptiveSLM,
    tokenizer: SimpleTokenizer,
    test_prompts: list[str],
    device: torch.device,
    n_levels: int = 5,
    max_new_tokens: int = 100,
) -> dict[str, Any]:
    """Test whether cognitive_load conditioning correlates with response length.

    Generates responses at different cognitive_load levels and checks if
    higher cognitive load produces longer, more complex responses.

    Parameters
    ----------
    model : AdaptiveSLM
        The model to evaluate.
    tokenizer : SimpleTokenizer
        Tokenizer for encoding/decoding.
    test_prompts : list[str]
        Test prompts.
    device : torch.device
        Device to run on.
    n_levels : int
        Number of cognitive_load levels to test.
    max_new_tokens : int
        Maximum tokens per generation.

    Returns
    -------
    dict[str, Any]
        Results with keys:
        - ``"correlation"`` -- Pearson correlation between
          cognitive_load and response length
        - ``"per_level"`` -- average length at each level
        - ``"monotonic"`` -- whether lengths increase monotonically
    """
    generator = SLMGenerator(model, tokenizer, device=str(device))

    level_lengths: dict[float, list[int]] = {}

    for level_idx in range(n_levels):
        cog_load = level_idx / max(n_levels - 1, 1)
        level_lengths[cog_load] = []

        av = torch.zeros(8)
        av[0] = cog_load    # cognitive_load
        av[1] = 0.5         # formality
        av[2] = cog_load    # verbosity (correlated with cognitive load)
        av[5] = 0.5         # emotional_tone

        for prompt in test_prompts:
            text = generator.generate(
                prompt,
                adaptation_vector=av,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
            )
            # Count words in the response (after prompt)
            prompt_words = len(prompt.split())
            response_words = len(text.split()) - prompt_words
            level_lengths[cog_load].append(max(response_words, 0))

    # Compute per-level averages
    per_level: list[dict[str, float]] = []
    all_loads: list[float] = []
    all_lengths: list[float] = []

    for cog_load in sorted(level_lengths.keys()):
        lengths = level_lengths[cog_load]
        avg_len = sum(lengths) / max(len(lengths), 1)
        per_level.append({
            "cognitive_load": cog_load,
            "avg_length": avg_len,
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
        })
        all_loads.append(cog_load)
        all_lengths.append(avg_len)

    # Pearson correlation
    correlation = _pearson_correlation(all_loads, all_lengths)

    # Check monotonicity
    monotonic = all(
        all_lengths[i] <= all_lengths[i + 1]
        for i in range(len(all_lengths) - 1)
    )

    result = {
        "correlation": correlation,
        "per_level": per_level,
        "monotonic": monotonic,
    }

    logger.info(
        "Length compliance: correlation=%.4f, monotonic=%s",
        correlation, monotonic,
    )
    for entry in per_level:
        logger.info(
            "  cognitive_load=%.2f -> avg_length=%.1f words",
            entry["cognitive_load"], entry["avg_length"],
        )

    return result


# ---------------------------------------------------------------------------
# 5. Sample generation for manual inspection
# ---------------------------------------------------------------------------

def generate_samples(
    model: AdaptiveSLM,
    tokenizer: SimpleTokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
) -> list[dict[str, str]]:
    """Generate sample outputs for manual inspection.

    For each prompt, generates responses with three conditioning
    profiles: casual, formal, and empathetic.

    Parameters
    ----------
    model : AdaptiveSLM
        The model to evaluate.
    tokenizer : SimpleTokenizer
        Tokenizer.
    prompts : list[str]
        Test prompts.
    device : torch.device
        Device.
    max_new_tokens : int
        Maximum tokens per generation.
    temperature : float
        Sampling temperature.

    Returns
    -------
    list[dict[str, str]]
        List of dicts with keys: ``"prompt"``, ``"casual"``,
        ``"formal"``, ``"empathetic"``.
    """
    generator = SLMGenerator(model, tokenizer, device=str(device))

    # Conditioning profiles
    profiles = {
        "casual": torch.tensor([0.3, 0.2, 0.4, 0.3, 0.7, 0.5, 0.7, 0.0]),
        "formal": torch.tensor([0.7, 0.9, 0.6, 0.2, 0.5, 0.5, 0.3, 0.0]),
        "empathetic": torch.tensor([0.4, 0.4, 0.5, 0.8, 0.4, 0.8, 0.5, 0.0]),
    }

    samples: list[dict[str, str]] = []

    for prompt in prompts:
        sample: dict[str, str] = {"prompt": prompt}

        for profile_name, av in profiles.items():
            text = generator.generate(
                prompt,
                adaptation_vector=av,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
            )
            sample[profile_name] = text

        samples.append(sample)

        logger.info("--- Prompt: %s", prompt[:60])
        for pname in profiles:
            logger.info("  [%s]: %s", pname, sample[pname][:100])

    return samples


# ---------------------------------------------------------------------------
# Utility: Pearson correlation
# ---------------------------------------------------------------------------

def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two lists.

    Parameters
    ----------
    x, y : list[float]
        Equal-length lists of values.

    Returns
    -------
    float
        Pearson r in [-1, 1]. Returns 0.0 if standard deviation is zero.
    """
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0

    return cov / (std_x * std_y)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: str,
    data_dir: str = "data/processed/dialogue",
    config_path: str = "configs/default.yaml",
    num_samples: int = 10,
    batch_size: int = 32,
    output_path: Optional[str] = None,
    device_str: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full evaluation suite.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint.
    data_dir : str
        Directory containing processed dialogue data.
    config_path : str
        Path to config YAML.
    num_samples : int
        Number of prompts for generation tests.
    batch_size : int
        Batch size for perplexity evaluation.
    output_path : str, optional
        If provided, save results to this JSON file.
    device_str : str, optional
        Device override.

    Returns
    -------
    dict[str, Any]
        Complete evaluation results.
    """
    # --- Device ---
    # PERF: flip on cuDNN benchmark + TF32 matmul (CUDA-only, safe no-op
    # on CPU) before the first forward pass.
    enable_cuda_optimizations()
    device = pick_device(device_str)

    logger.info("Using device: %s", device)

    # --- Load config ---
    # SEC: resolve config path; YAML is parsed via yaml.safe_load (no exec).
    cfg_path = Path(config_path).resolve()
    if cfg_path.exists() and cfg_path.is_file():
        config = load_config(cfg_path)
    else:
        from i3.config import Config
        config = Config()

    # --- Load tokenizer ---
    # SEC: resolve all input paths so log messages and downstream I/O see
    # absolute, normalised paths.
    data_path = Path(data_dir).resolve()
    tok_path = data_path / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path("models/slm/tokenizer.json").resolve()

    tokenizer = SimpleTokenizer.load(str(tok_path))
    logger.info("Loaded tokenizer: %d tokens", len(tokenizer))

    # --- Build and load model ---
    model = AdaptiveSLM(
        vocab_size=len(tokenizer),
        d_model=config.slm.d_model,
        n_heads=config.slm.n_heads,
        n_layers=config.slm.n_layers,
        d_ff=config.slm.d_ff,
        max_seq_len=config.slm.max_seq_len,
        conditioning_dim=config.slm.conditioning_dim,
        adaptation_dim=config.slm.adaptation_dim,
        n_cross_heads=config.slm.cross_attention_heads,
        dropout=0.0,  # No dropout at eval time
        tie_weights=config.slm.tie_weights,
    )

    # SEC: evaluation loads are effectively inference-mode, so we use
    # weights_only=True to prevent pickled-object code execution. We also
    # resolve and validate the path before reading it so that we never
    # call torch.load on a directory or a path that does not exist.
    ckpt_path_resolved = Path(checkpoint_path).resolve()
    if not ckpt_path_resolved.exists() or not ckpt_path_resolved.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found or not a regular file: {ckpt_path_resolved}"
        )
    checkpoint = torch.load(
        str(ckpt_path_resolved), map_location=device, weights_only=True
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters (%.2f MB)", n_params, n_params * 4 / 1e6)

    # --- Load validation data ---
    val_path = data_path / "val.pt"
    if not val_path.exists():
        val_path = data_path / "test.pt"

    val_dataset = EvalDataset(val_path)
    # PERF: prefetch with a modest worker pool.  Eval is usually shorter
    # than training, but workers still help when the dataset is on a
    # slow disk.
    import os as _os
    _nw = max(0, min(4, (_os.cpu_count() or 2) // 2))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=_nw,
        persistent_workers=_nw > 0,
    )

    # --- Test prompts ---
    test_prompts = [
        "Hello, how are you doing today?",
        "Can you help me with something?",
        "I'm feeling really stressed out.",
        "What do you think about this idea?",
        "Tell me something interesting.",
        "I need some advice about a problem.",
        "How does this work exactly?",
        "I'm not sure I understand.",
        "That's really exciting news!",
        "I've been having a tough day.",
    ][:num_samples]

    results: dict[str, Any] = {}

    # --- 1. Perplexity ---
    logger.info("=" * 50)
    logger.info("1. PERPLEXITY EVALUATION")
    logger.info("=" * 50)
    results["perplexity"] = compute_perplexity(
        model, val_loader, device, pad_id=tokenizer.PAD_ID
    )

    # --- 2. Conditioning sensitivity ---
    logger.info("=" * 50)
    logger.info("2. CONDITIONING SENSITIVITY")
    logger.info("=" * 50)
    results["conditioning_sensitivity"] = conditioning_sensitivity(
        model, tokenizer, test_prompts, device
    )

    # --- 3. Diversity ---
    logger.info("=" * 50)
    logger.info("3. DIVERSITY METRICS")
    logger.info("=" * 50)
    generator = SLMGenerator(model, tokenizer, device=str(device))
    generated_texts = generator.generate_batch(
        test_prompts, max_new_tokens=80, temperature=0.8
    )
    results["diversity"] = compute_diversity(generated_texts)

    # --- 4. Length compliance ---
    logger.info("=" * 50)
    logger.info("4. LENGTH COMPLIANCE")
    logger.info("=" * 50)
    results["length_compliance"] = length_compliance(
        model, tokenizer, test_prompts[:5], device,
        n_levels=5, max_new_tokens=80,
    )

    # --- 5. Sample generation ---
    logger.info("=" * 50)
    logger.info("5. SAMPLE GENERATION")
    logger.info("=" * 50)
    results["samples"] = generate_samples(
        model, tokenizer, test_prompts[:5], device,
        max_new_tokens=80,
    )

    # --- Summary ---
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info("Perplexity:              %.2f", results["perplexity"]["perplexity"])
    logger.info(
        "Conditioning KL div:     %.4f",
        results["conditioning_sensitivity"]["mean_kl_divergence"],
    )
    logger.info("Distinct-1:              %.4f", results["diversity"]["distinct_1"])
    logger.info("Distinct-2:              %.4f", results["diversity"]["distinct_2"])
    logger.info(
        "Length correlation:       %.4f",
        results["length_compliance"]["correlation"],
    )
    logger.info(
        "Length monotonic:         %s",
        results["length_compliance"]["monotonic"],
    )

    # --- Save results ---
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Make results JSON-serializable
        serializable = _make_serializable(results)
        with open(out, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Results saved to: %s", out)

    return results


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types.

    Handles torch.Tensor, numpy arrays, and other common types.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        JSON-serializable version.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Adaptive SLM.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/dialogue",
        help="Directory with processed dialogue data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test prompts (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for perplexity eval (default: 32).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device override ('auto', 'cpu', 'cuda', 'cuda:N', 'mps'). "
            "Default 'auto' picks CUDA when available, else MPS, else CPU."
        ),
    )
    args = parser.parse_args()

    # SEC: bound-check numeric arguments before launching the model.
    if args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    args = parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        config_path=args.config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_path=args.output,
        device_str=args.device,
    )
