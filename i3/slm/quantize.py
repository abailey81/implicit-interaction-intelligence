"""INT8 dynamic quantization and profiling for edge deployment.

Built from scratch -- no HuggingFace optimum or external quantization
libraries. Uses PyTorch's built-in dynamic quantization to convert FP32
Linear layers to INT8 for reduced memory footprint and faster inference
on CPU-based edge devices (Kirin SoCs, wearables, IoT).

Dynamic quantization quantizes weights statically to INT8 at save time
and quantizes activations dynamically at runtime, requiring no calibration
dataset. This makes it ideal for the I3 deployment scenario where the
model must work immediately on a new device without a calibration step.

Usage::

    from i3.slm.model import AdaptiveSLM
    from i3.slm.quantize import SLMQuantizer

    model = AdaptiveSLM()
    quantized = SLMQuantizer.quantize_dynamic(model)
    sizes = SLMQuantizer.compare_sizes(model, quantized)
    SLMQuantizer.save_quantized(quantized, "models/slm/quantized.pt")
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SLMQuantizer:
    """INT8 dynamic quantization utilities for the Adaptive SLM.

    All methods are static -- no instance state is needed. This class
    serves as a namespace for quantization, profiling, validation, and
    persistence operations.
    """

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------

    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """Apply INT8 dynamic quantization to all Linear layers.

        Dynamic quantization converts ``nn.Linear`` weights from FP32 to
        INT8 at model-save time and quantizes activations dynamically
        during inference. This typically yields a ~2-4x reduction in model
        size with minimal quality loss for transformer architectures.

        Parameters
        ----------
        model : nn.Module
            The FP32 model to quantize.

        Returns
        -------
        nn.Module
            A new model with Linear layers replaced by their INT8
            dynamically-quantized equivalents. The original model is
            not modified.

        Notes
        -----
        The model must be on CPU before quantization. If the model is on
        GPU, it is moved to CPU first.
        """
        model_cpu = model.cpu()
        model_cpu.eval()

        quantized = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8,
        )

        n_quantized = sum(
            1 for m in quantized.modules()
            if type(m).__name__ == "DynamicQuantizedLinear"
        )
        logger.info(
            "Dynamic INT8 quantization complete: %d Linear layers quantized",
            n_quantized,
        )

        return quantized

    # ------------------------------------------------------------------
    # Size comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_sizes(
        original: nn.Module,
        quantized: nn.Module,
    ) -> dict[str, Any]:
        """Compare FP32 vs INT8 model sizes.

        Saves both models to temporary files and measures the on-disk
        size to give an accurate comparison of memory footprint.

        Parameters
        ----------
        original : nn.Module
            The original FP32 model.
        quantized : nn.Module
            The quantized INT8 model.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``"original_size_mb"`` -- FP32 model size in MB
            - ``"quantized_size_mb"`` -- INT8 model size in MB
            - ``"compression_ratio"`` -- original / quantized
            - ``"size_reduction_pct"`` -- percentage size reduction
            - ``"original_params"`` -- total parameter count (original)
            - ``"quantized_params"`` -- total parameter count (quantized)
        """
        original_size = SLMQuantizer._measure_model_size(original)
        quantized_size = SLMQuantizer._measure_model_size(quantized)

        original_params = sum(p.numel() for p in original.parameters())
        quantized_params = sum(p.numel() for p in quantized.parameters())

        compression = original_size / max(quantized_size, 1e-10)
        reduction = (1.0 - quantized_size / max(original_size, 1e-10)) * 100.0

        result = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression,
            "size_reduction_pct": reduction,
            "original_params": original_params,
            "quantized_params": quantized_params,
        }

        logger.info(
            "Size comparison: FP32=%.2f MB, INT8=%.2f MB, "
            "compression=%.2fx, reduction=%.1f%%",
            original_size,
            quantized_size,
            compression,
            reduction,
        )

        return result

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_quantized(
        original: nn.Module,
        quantized: nn.Module,
        tokenizer: Any,
        test_prompts: list[str],
        max_new_tokens: int = 50,
    ) -> dict[str, Any]:
        """Compare outputs of original vs quantized model.

        Runs both models on the same prompts and compares:
        - Logit agreement (cosine similarity of output distributions)
        - Perplexity difference on the prompt text
        - Inference speed (tokens per second)

        Parameters
        ----------
        original : nn.Module
            The original FP32 model.
        quantized : nn.Module
            The quantized INT8 model.
        tokenizer : SimpleTokenizer
            Tokenizer for encoding prompts.
        test_prompts : list[str]
            List of test prompts to evaluate.
        max_new_tokens : int
            Maximum tokens for generation comparison.

        Returns
        -------
        dict[str, Any]
            Validation results with keys:

            - ``"mean_cosine_similarity"`` -- avg cosine sim of logits
            - ``"max_logit_diff"`` -- maximum absolute logit difference
            - ``"original_perplexity"`` -- avg perplexity on prompts (FP32)
            - ``"quantized_perplexity"`` -- avg perplexity on prompts (INT8)
            - ``"perplexity_delta"`` -- absolute perplexity increase
            - ``"original_tok_per_sec"`` -- FP32 inference speed
            - ``"quantized_tok_per_sec"`` -- INT8 inference speed
            - ``"speedup"`` -- INT8 / FP32 speed ratio
            - ``"per_prompt"`` -- list of per-prompt results
        """
        original.eval()
        quantized.eval()

        per_prompt_results: list[dict[str, Any]] = []
        all_cosine_sims: list[float] = []
        all_max_diffs: list[float] = []
        orig_perplexities: list[float] = []
        quant_perplexities: list[float] = []
        orig_times: list[float] = []
        quant_times: list[float] = []

        for prompt in test_prompts:
            ids = tokenizer.encode(prompt, add_special=True)
            input_tensor = torch.tensor([ids], device=torch.device("cpu"))

            # --- Forward pass: original ---
            t0 = time.perf_counter()
            with torch.no_grad():
                orig_logits, _ = original(input_tensor)
            orig_time = time.perf_counter() - t0

            # --- Forward pass: quantized ---
            t0 = time.perf_counter()
            with torch.no_grad():
                quant_logits, _ = quantized(input_tensor)
            quant_time = time.perf_counter() - t0

            # --- Cosine similarity of logit distributions ---
            orig_flat = orig_logits.reshape(-1)
            quant_flat = quant_logits.reshape(-1)
            cosine_sim = torch.nn.functional.cosine_similarity(
                orig_flat.unsqueeze(0),
                quant_flat.unsqueeze(0),
            ).item()

            max_diff = (orig_logits - quant_logits).abs().max().item()

            # --- Perplexity on prompt tokens ---
            orig_ppl = SLMQuantizer._compute_perplexity(
                orig_logits, input_tensor
            )
            quant_ppl = SLMQuantizer._compute_perplexity(
                quant_logits, input_tensor
            )

            n_tokens = len(ids)
            orig_tps = n_tokens / max(orig_time, 1e-10)
            quant_tps = n_tokens / max(quant_time, 1e-10)

            all_cosine_sims.append(cosine_sim)
            all_max_diffs.append(max_diff)
            orig_perplexities.append(orig_ppl)
            quant_perplexities.append(quant_ppl)
            orig_times.append(orig_tps)
            quant_times.append(quant_tps)

            per_prompt_results.append({
                "prompt": prompt[:80],
                "cosine_similarity": cosine_sim,
                "max_logit_diff": max_diff,
                "original_perplexity": orig_ppl,
                "quantized_perplexity": quant_ppl,
                "original_tok_per_sec": orig_tps,
                "quantized_tok_per_sec": quant_tps,
            })

        mean_cosine = sum(all_cosine_sims) / max(len(all_cosine_sims), 1)
        max_diff_overall = max(all_max_diffs) if all_max_diffs else 0.0
        mean_orig_ppl = sum(orig_perplexities) / max(len(orig_perplexities), 1)
        mean_quant_ppl = sum(quant_perplexities) / max(len(quant_perplexities), 1)
        mean_orig_tps = sum(orig_times) / max(len(orig_times), 1)
        mean_quant_tps = sum(quant_times) / max(len(quant_times), 1)

        result = {
            "mean_cosine_similarity": mean_cosine,
            "max_logit_diff": max_diff_overall,
            "original_perplexity": mean_orig_ppl,
            "quantized_perplexity": mean_quant_ppl,
            "perplexity_delta": mean_quant_ppl - mean_orig_ppl,
            "original_tok_per_sec": mean_orig_tps,
            "quantized_tok_per_sec": mean_quant_tps,
            "speedup": mean_quant_tps / max(mean_orig_tps, 1e-10),
            "per_prompt": per_prompt_results,
        }

        logger.info(
            "Quantization validation: cosine_sim=%.4f, ppl_delta=%.2f, "
            "speedup=%.2fx",
            mean_cosine,
            result["perplexity_delta"],
            result["speedup"],
        )

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_quantized(model: nn.Module, path: str) -> None:
        """Save a quantized model to disk.

        Uses ``torch.save`` with the full model state dict. The file
        includes the model's quantization configuration so it can be
        loaded correctly.

        Parameters
        ----------
        model : nn.Module
            The quantized model to save.
        path : str
            Filesystem path for the saved model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path)

        size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info("Quantized model saved: %s (%.2f MB)", save_path, size_mb)

    @staticmethod
    def load_quantized(
        path: str,
        model_class: type,
        **model_kwargs: Any,
    ) -> nn.Module:
        """Load a quantized model from disk.

        First instantiates a fresh FP32 model, applies dynamic
        quantization to create the correct module structure, then loads
        the saved quantized state dict.

        Parameters
        ----------
        path : str
            Path to the saved quantized state dict.
        model_class : type
            The model class (e.g., ``AdaptiveSLM``) to instantiate.
        **model_kwargs
            Keyword arguments forwarded to the model constructor.

        Returns
        -------
        nn.Module
            The loaded quantized model, ready for inference.
        """
        # 1. Create a fresh FP32 model with the same architecture
        fp32_model = model_class(**model_kwargs)

        # 2. Apply quantization to create the correct module structure
        quantized_model = SLMQuantizer.quantize_dynamic(fp32_model)

        # 3. Load the saved state dict (weights only -- quantized
        #    state_dicts contain tensors and pickled quantization
        #    metadata recognised by torch's safe-load allowlist).
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        quantized_model.load_state_dict(state_dict)

        quantized_model.eval()
        logger.info("Quantized model loaded from: %s", path)

        return quantized_model

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    @staticmethod
    def profile_inference(
        model: nn.Module,
        tokenizer: Any,
        prompt: str = "Hello, how are you today?",
        n_iterations: int = 100,
        max_seq_len: int = 64,
    ) -> dict[str, float]:
        """Profile inference latency and throughput.

        Runs the model forward pass multiple times on a fixed input and
        measures timing statistics.

        Parameters
        ----------
        model : nn.Module
            The model to profile (FP32 or quantized).
        tokenizer : SimpleTokenizer
            Tokenizer for encoding the prompt.
        prompt : str
            Test prompt for profiling.
        n_iterations : int
            Number of forward passes to average over.
        max_seq_len : int
            Sequence length for profiling (pads or truncates).

        Returns
        -------
        dict[str, float]
            Dictionary with keys:

            - ``"mean_latency_ms"`` -- average forward pass time
            - ``"std_latency_ms"`` -- standard deviation
            - ``"min_latency_ms"`` -- fastest forward pass
            - ``"max_latency_ms"`` -- slowest forward pass
            - ``"throughput_tok_per_sec"`` -- tokens per second
            - ``"memory_mb"`` -- estimated model memory footprint
        """
        model.eval()

        ids = tokenizer.encode(
            prompt, add_special=True, max_length=max_seq_len, padding=True
        )
        input_tensor = torch.tensor([ids], device=torch.device("cpu"))
        seq_len = len(ids)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(input_tensor)

        # Timed runs
        latencies: list[float] = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_tensor)
            elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            latencies.append(elapsed)

        mean_lat = sum(latencies) / len(latencies)
        std_lat = (
            sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)
        ) ** 0.5
        min_lat = min(latencies)
        max_lat = max(latencies)
        throughput = seq_len / (mean_lat / 1000.0)

        memory_mb = SLMQuantizer._measure_model_size(model)

        result = {
            "mean_latency_ms": mean_lat,
            "std_latency_ms": std_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat,
            "throughput_tok_per_sec": throughput,
            "memory_mb": memory_mb,
        }

        logger.info(
            "Inference profile: latency=%.2f +/- %.2f ms, "
            "throughput=%.0f tok/s, memory=%.2f MB",
            mean_lat,
            std_lat,
            throughput,
            memory_mb,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_model_size(model: nn.Module) -> float:
        """Save model to a temp file and return its size in MB.

        Parameters
        ----------
        model : nn.Module
            Model to measure.

        Returns
        -------
        float
            Model size in megabytes.
        """
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        try:
            torch.save(model.state_dict(), tmp_path)
            size_bytes = os.path.getsize(tmp_path)
        finally:
            os.unlink(tmp_path)

        return size_bytes / (1024 * 1024)

    @staticmethod
    def _compute_perplexity(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> float:
        """Compute perplexity of a sequence given model logits.

        Parameters
        ----------
        logits : torch.Tensor
            Model output logits of shape ``[1, seq_len, vocab_size]``.
        input_ids : torch.Tensor
            Token IDs of shape ``[1, seq_len]``.

        Returns
        -------
        float
            Perplexity value. Lower = more confident.
        """
        # Shift: logits[t] predicts input_ids[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        return torch.exp(loss).item()
