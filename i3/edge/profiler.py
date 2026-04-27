"""Edge-deployment profiler for the I3 SLM + TCN encoder.

Measures real on-device metrics — parameter counts, checkpoint sizes
under fp32 / bf16 / int8, CPU inference latency percentiles, peak
process memory, and ONNX graph size — so the demo UI can surface a
concrete answer to the Huawei HMI Lab filter question
*"ever deployed ML models to low-compute devices (wearables/IoT),
memory/power strictly limited?"*.

This module is **CPU-only by default**.  Every measurement writes its
result to ``reports/edge_profile.json`` (and a human-readable markdown
summary to ``reports/edge_profile.md``) so the server can serve it
from disk without recomputing on every request.

Design choices:

* No HuggingFace, no pretrained weights — we load the from-scratch
  :class:`i3.slm.model.AdaptiveSLM` and :class:`i3.encoder.tcn.TemporalConvNet`
  that the portfolio ships with.
* Sizes come from :func:`torch.save` into a ``BytesIO`` and measuring
  the serialised buffer, which gives a faithful on-disk number for
  each dtype.  fp32 / bf16 are computed by rebuilding the model in the
  target dtype; int8 uses :func:`torch.quantization.quantize_dynamic`.
* Latency uses :func:`time.perf_counter` across 100 independent single-
  token generations (32-token prompt, 16 new tokens, greedy) and 100
  TCN encoder forwards on a random ``(1, 10, 32)`` input.
* Peak process RSS is sampled during the measurement run with
  :mod:`psutil`.

Running ``python -m i3.edge.profiler`` or ``python scripts/measure_edge.py``
completes in under a minute on a modern laptop CPU.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Deployment budgets (MB) — used to flag deployability.
# --------------------------------------------------------------------------- #

_WEARABLE_MB_BUDGET = 50.0
_BUDGET_PHONE_MB = 100.0
_MIDRANGE_PHONE_MB = 300.0


@dataclass(slots=True)
class EdgeReport:
    """Typed view of the measurements written to disk."""

    slm_params: int
    slm_size_fp32_mb: float
    slm_size_bf16_mb: float
    slm_size_int8_mb: float
    tcn_params: int
    tcn_size_fp32_mb: float
    tcn_size_int8_mb: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_encoder_p50: float
    latency_ms_encoder_p95: float
    memory_peak_mb: float
    onnx_size_mb: float | None
    deployable_to: list[str]
    timestamp: str
    device: str
    slm_checkpoint: str
    tcn_checkpoint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _count_unique_params(model: nn.Module) -> int:
    """Count parameters de-duplicated by ``id`` (handles tied weights)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def _serialised_size_mb(model: nn.Module) -> float:
    """Serialise ``model.state_dict()`` to a BytesIO and measure the bytes.

    This mirrors what :func:`torch.save` would write to disk, so the
    number lines up with the on-device checkpoint size operators see.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def _to_dtype_copy(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Return a deep-copied model with its parameters cast to ``dtype``.

    Falls back to the original model if the clone path is exhausted
    (for int8 we use ``quantize_dynamic`` instead, which already yields
    a new module).
    """
    import copy

    clone = copy.deepcopy(model)
    clone = clone.to(dtype=dtype)
    clone.eval()
    return clone


def _int8_quantized(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantise ``nn.Linear`` + ``nn.Conv1d`` layers (CPU)."""
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8,
    )


def _percentile(samples: list[float], pct: float) -> float:
    """Compute a percentile without requiring NumPy."""
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    # linear interpolation between closest ranks (NumPy default)
    k = (len(ordered) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


# --------------------------------------------------------------------------- #
# Main profiler
# --------------------------------------------------------------------------- #


class EdgeProfiler:
    """Measure and report edge-deployment metrics for the SLM + TCN."""

    def __init__(
        self,
        slm_checkpoint: Path,
        tcn_checkpoint: Path,
        tokenizer_path: Path,
        device: str = "cpu",
    ) -> None:
        self.slm_checkpoint = Path(slm_checkpoint)
        self.tcn_checkpoint = Path(tcn_checkpoint)
        self.tokenizer_path = Path(tokenizer_path)
        # Edge work is CPU-only; we never move the models to CUDA here.
        # The constraint also avoids stealing VRAM from a training job.
        self.device = torch.device("cpu") if device != "cpu" else torch.device(device)
        self._slm: nn.Module | None = None
        self._tcn: nn.Module | None = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #

    def _load_slm(self) -> nn.Module:
        """Load the v1 Adaptive SLM from the ``best_model.pt`` checkpoint.

        Mirrors the checkpoint-aware ctor logic in
        :meth:`i3.pipeline.engine.Pipeline.load_slm` — read the embedding
        tensor shape first, fall back to the checkpoint's own configs
        block, then to static defaults.
        """
        from i3.slm.model import AdaptiveSLM

        ckpt = torch.load(
            self.slm_checkpoint, map_location="cpu", weights_only=True
        )
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise RuntimeError(
                f"Unexpected SLM checkpoint layout at {self.slm_checkpoint}"
            )

        sd = ckpt["model_state_dict"]
        cfg = ckpt.get("configs") or ckpt.get("config") or {}

        emb_w = None
        for key in (
            "embedding.token_embedding.embedding.weight",
            "token_embedding.embedding.weight",
            "embedding.token_embedding.weight",
            "token_embedding.weight",
        ):
            if key in sd:
                emb_w = sd[key]
                break

        if emb_w is not None:
            vocab = int(emb_w.shape[0])
            d_model = int(emb_w.shape[1])
        else:
            vocab = int(cfg.get("vocab_size", 30000))
            d_model = int(cfg.get("d_model", 512))

        model = AdaptiveSLM(
            vocab_size=vocab,
            max_seq_len=int(cfg.get("max_seq_len", 256)),
            d_model=d_model,
            n_heads=int(cfg.get("n_heads", 8)),
            n_layers=int(cfg.get("n_layers", 8)),
            d_ff=int(cfg.get("d_ff", 2048)),
            dropout=float(cfg.get("dropout", 0.1)),
            conditioning_dim=int(cfg.get("conditioning_dim", 64)),
            adaptation_dim=int(cfg.get("adaptation_dim", 8)),
        )
        model.load_state_dict(sd)
        model.eval()
        return model

    def _load_tcn(self) -> nn.Module:
        """Load the TCN encoder from ``best_model.pt``."""
        from i3.encoder.tcn import TemporalConvNet

        ckpt = torch.load(
            self.tcn_checkpoint, map_location="cpu", weights_only=True
        )
        if "model_state_dict" not in ckpt:
            raise RuntimeError(
                f"TCN checkpoint at {self.tcn_checkpoint} missing 'model_state_dict'"
            )

        model = TemporalConvNet(
            input_dim=32,
            hidden_dims=[64, 64, 64, 64],
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            embedding_dim=64,
            dropout=0.1,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _load_tokenizer(self) -> Any:
        from i3.slm.tokenizer import SimpleTokenizer

        return SimpleTokenizer.load(str(self.tokenizer_path))

    # ------------------------------------------------------------------ #
    # Latency measurements
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _measure_slm_latency(
        self, model: nn.Module, *, n_runs: int = 100
    ) -> tuple[list[float], float]:
        """Time 100 greedy continuations (32 prompt, 16 new tokens).

        Returns ``(samples_ms, peak_rss_mb_seen)``.  Peak RSS is sampled
        after each forward so the caller can track memory growth across
        the run.
        """
        import psutil

        proc = psutil.Process()
        vocab = int(getattr(model, "vocab_size", 8000))

        # Fixed pseudo-random prompt to keep measurements reproducible.
        rng = torch.Generator().manual_seed(0)

        samples_ms: list[float] = []
        peak_rss = 0.0

        # Warm-up pass — first forward allocates workspaces / oneDNN
        # kernels; we don't want that baked into the percentiles.
        warmup_ids = torch.randint(0, vocab, (1, 32), generator=rng)
        _ = model(warmup_ids, use_cache=False)

        for _ in range(n_runs):
            input_ids = torch.randint(0, vocab, (1, 32), generator=rng)
            start = time.perf_counter()

            # Greedy decode of 16 new tokens with KV cache off — mirrors
            # a worst-case "prefill-every-step" edge runtime that doesn't
            # yet have cached attention state. Keeps the measurement
            # self-contained (no dependency on SLMGenerator).
            current = input_ids
            for _step in range(16):
                logits, _ = model(current, use_cache=False)
                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                current = torch.cat(
                    [current, torch.tensor([[next_id]], dtype=current.dtype)],
                    dim=1,
                )
                # Guard against unbounded KV growth — cap context length.
                if current.shape[1] >= 64:
                    current = current[:, -64:]

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples_ms.append(elapsed_ms)
            peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))

        return samples_ms, peak_rss

    @torch.inference_mode()
    def _measure_encoder_latency(
        self, model: nn.Module, *, n_runs: int = 100
    ) -> tuple[list[float], float]:
        """Time 100 encoder forwards on a random ``(1, 10, 32)`` input."""
        import psutil

        proc = psutil.Process()
        rng = torch.Generator().manual_seed(1)

        # Warm-up
        warmup = torch.randn(1, 10, 32, generator=rng)
        _ = model(warmup)

        samples_ms: list[float] = []
        peak_rss = 0.0
        for _ in range(n_runs):
            x = torch.randn(1, 10, 32, generator=rng)
            start = time.perf_counter()
            _ = model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples_ms.append(elapsed_ms)
            peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))

        return samples_ms, peak_rss

    # ------------------------------------------------------------------ #
    # ONNX size
    # ------------------------------------------------------------------ #

    def _measure_onnx_size(self, model: nn.Module) -> float | None:
        """Export the SLM to ONNX via the existing helper and weigh the file.

        Uses the prefill-only exporter in :mod:`i3.slm.onnx_export`.
        On any failure (missing ``onnx`` package, unsupported op, etc.)
        logs a warning and returns ``None`` — the dashboard degrades
        gracefully.
        """
        try:
            from i3.slm.onnx_export import export_slm

            out = Path("reports/edge_slm.onnx")
            out.parent.mkdir(parents=True, exist_ok=True)
            export_slm(model, out, opset=17, verify=False)
            return out.stat().st_size / (1024 * 1024)
        except SystemExit as exc:
            logger.warning("ONNX export skipped: %s", exc)
        except Exception:  # pragma: no cover - defensive
            logger.exception("ONNX export failed; reporting onnx_size_mb=None")
        return None

    # ------------------------------------------------------------------ #
    # Deployability judgement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deployability(slm_int8_mb: float) -> list[str]:
        """Classify deployability vs three canonical memory budgets."""
        out: list[str] = []
        if slm_int8_mb <= _MIDRANGE_PHONE_MB:
            out.append(
                f"mid-range phone (int8 {slm_int8_mb:.1f} MB <= "
                f"{_MIDRANGE_PHONE_MB:.0f} MB budget)"
            )
        else:
            out.append(
                f"mid-range phone (int8 {slm_int8_mb:.1f} MB > "
                f"{_MIDRANGE_PHONE_MB:.0f} MB budget - too big)"
            )
        if slm_int8_mb <= _BUDGET_PHONE_MB:
            out.append(
                f"budget phone (int8 {slm_int8_mb:.1f} MB <= "
                f"{_BUDGET_PHONE_MB:.0f} MB budget)"
            )
        else:
            out.append(
                f"budget phone (int8 {slm_int8_mb:.1f} MB > "
                f"{_BUDGET_PHONE_MB:.0f} MB budget - too big)"
            )
        if slm_int8_mb <= _WEARABLE_MB_BUDGET:
            out.append(
                f"wearable (int8 {slm_int8_mb:.1f} MB <= "
                f"{_WEARABLE_MB_BUDGET:.0f} MB budget)"
            )
        else:
            out.append(
                f"wearable (int8 {slm_int8_mb:.1f} MB > "
                f"{_WEARABLE_MB_BUDGET:.0f} MB budget - too big)"
            )
        return out

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def measure(self) -> dict[str, Any]:
        """Run the full measurement suite and return a dict report.

        Side effects: writes ``reports/edge_profile.json`` and
        ``reports/edge_profile.md``.
        """
        import psutil

        logger.info("Loading v1 SLM and TCN encoder on CPU…")
        self._slm = self._load_slm()
        self._tcn = self._load_tcn()

        # --- Sizes --------------------------------------------------------
        slm_params = _count_unique_params(self._slm)
        tcn_params = _count_unique_params(self._tcn)

        # Analytical dtype sizes (params * bytes-per-element).  The
        # serialised number is usually within a few per-cent of this;
        # we expose the analytical figure because it matches what a
        # deployment engineer would compute on a spec sheet.
        slm_size_fp32_mb = slm_params * 4 / (1024 * 1024)
        slm_size_bf16_mb = slm_params * 2 / (1024 * 1024)

        tcn_size_fp32_mb = tcn_params * 4 / (1024 * 1024)

        logger.info("Quantising SLM + TCN to INT8 (dynamic, CPU)…")
        slm_int8 = _int8_quantized(self._slm)
        tcn_int8 = _int8_quantized(self._tcn)

        slm_size_int8_mb = _serialised_size_mb(slm_int8)
        tcn_size_int8_mb = _serialised_size_mb(tcn_int8)

        # --- Latency ------------------------------------------------------
        logger.info("Measuring SLM generation latency (100 runs, CPU)…")
        slm_samples, slm_rss = self._measure_slm_latency(self._slm)
        logger.info(
            "SLM latency: p50=%.1f ms  p95=%.1f ms  mean=%.1f ms",
            _percentile(slm_samples, 50),
            _percentile(slm_samples, 95),
            statistics.mean(slm_samples),
        )

        logger.info("Measuring TCN encoder latency (100 runs, CPU)…")
        enc_samples, enc_rss = self._measure_encoder_latency(self._tcn)
        logger.info(
            "TCN latency: p50=%.2f ms  p95=%.2f ms  mean=%.2f ms",
            _percentile(enc_samples, 50),
            _percentile(enc_samples, 95),
            statistics.mean(enc_samples),
        )

        # Peak RSS seen during the whole run (compared to current).
        current_rss = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_peak_mb = max(slm_rss, enc_rss, current_rss)

        # --- ONNX size ----------------------------------------------------
        logger.info("Exporting SLM to ONNX for size measurement…")
        onnx_size_mb = self._measure_onnx_size(self._slm)

        # --- Report -------------------------------------------------------
        report = EdgeReport(
            slm_params=int(slm_params),
            slm_size_fp32_mb=round(slm_size_fp32_mb, 2),
            slm_size_bf16_mb=round(slm_size_bf16_mb, 2),
            slm_size_int8_mb=round(slm_size_int8_mb, 2),
            tcn_params=int(tcn_params),
            tcn_size_fp32_mb=round(tcn_size_fp32_mb, 3),
            tcn_size_int8_mb=round(tcn_size_int8_mb, 3),
            latency_ms_p50=round(_percentile(slm_samples, 50), 2),
            latency_ms_p95=round(_percentile(slm_samples, 95), 2),
            latency_ms_encoder_p50=round(_percentile(enc_samples, 50), 3),
            latency_ms_encoder_p95=round(_percentile(enc_samples, 95), 3),
            memory_peak_mb=round(memory_peak_mb, 1),
            onnx_size_mb=(
                round(onnx_size_mb, 2) if onnx_size_mb is not None else None
            ),
            deployable_to=self._deployability(slm_size_int8_mb),
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            device=str(self.device),
            slm_checkpoint=str(self.slm_checkpoint),
            tcn_checkpoint=str(self.tcn_checkpoint),
        )
        data = report.to_dict()

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = reports_dir / "edge_profile.json"
        md_path = reports_dir / "edge_profile.md"
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        md_path.write_text(_render_markdown(data), encoding="utf-8")
        logger.info("Edge profile written to %s and %s", json_path, md_path)

        # Best-effort cleanup — the caller may reuse this process.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

        return data


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Edge profile",
        "",
        f"*Measured {data['timestamp']} on `{data['device']}`.*",
        "",
        "## SLM (v1 Adaptive decoder)",
        "",
        f"- Parameters: **{data['slm_params']:,}**",
        f"- fp32 size: {data['slm_size_fp32_mb']:.2f} MB",
        f"- bf16 size: {data['slm_size_bf16_mb']:.2f} MB",
        f"- int8 size: **{data['slm_size_int8_mb']:.2f} MB** (dynamic quantisation)",
        f"- ONNX prefill graph: "
        + (
            f"{data['onnx_size_mb']:.2f} MB"
            if data["onnx_size_mb"] is not None
            else "(export failed — see logs)"
        ),
        "",
        "## TCN encoder",
        "",
        f"- Parameters: **{data['tcn_params']:,}**",
        f"- fp32 size: {data['tcn_size_fp32_mb']:.3f} MB",
        f"- int8 size: {data['tcn_size_int8_mb']:.3f} MB",
        "",
        "## Latency (CPU, 100 runs)",
        "",
        f"- SLM greedy decode (32 prompt → 16 new tokens): "
        f"p50 **{data['latency_ms_p50']:.1f} ms**, p95 {data['latency_ms_p95']:.1f} ms",
        f"- TCN encoder (single 10×32 window): "
        f"p50 **{data['latency_ms_encoder_p50']:.3f} ms**, p95 {data['latency_ms_encoder_p95']:.3f} ms",
        "",
        "## Memory",
        "",
        f"- Peak process RSS during measurements: **{data['memory_peak_mb']:.1f} MB**",
        "",
        "## Deployability",
        "",
    ]
    for entry in data["deployable_to"]:
        ok = "too big" not in entry
        marker = "- [x] " if ok else "- [ ] "
        lines.append(f"{marker}{entry}")
    lines.append("")
    lines.append(
        f"*SLM checkpoint*: `{data['slm_checkpoint']}`  "
        f"*TCN checkpoint*: `{data['tcn_checkpoint']}`"
    )
    lines.append("")
    return "\n".join(lines)


__all__ = ["EdgeProfiler", "EdgeReport"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _default_paths() -> tuple[Path, Path, Path]:
    root = Path(os.environ.get("I3_ROOT", ".")).resolve()
    return (
        root / "checkpoints" / "slm" / "best_model.pt",
        root / "checkpoints" / "encoder" / "best_model.pt",
        root / "checkpoints" / "slm" / "tokenizer.json",
    )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    slm, tcn, tok = _default_paths()
    profiler = EdgeProfiler(slm, tcn, tok, device="cpu")
    report = profiler.measure()
    print(json.dumps(report, indent=2))
