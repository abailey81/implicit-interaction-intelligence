"""Benchmark speculative decoding against baseline greedy generation.

Runs a 100-iteration speed comparison between:

  (a) Baseline greedy autoregressive generation on the target model.
  (b) :class:`i3.slm.speculative_decoding.SpeculativeDecoder` with
      ``num_drafts`` in {2, 4, 8}.

Reports P50 / P95 latency, acceptance rate, and measured speed-up,
and writes a Markdown report under ``reports/``.

Usage::

    python scripts/benchmark_speculative.py \\
        --iterations 100 \\
        --max-new-tokens 32 \\
        --seed 42

The models used are random-initialised AdaptiveSLM instances. The
absolute numerical values from this benchmark are **not** a quality
signal (there is no trained checkpoint); the benchmark measures only
the decoding *throughput* delta between baseline and speculative paths.

References
----------
* Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers
  via Speculative Decoding*, ICML 2023.
* Chen et al. (2023), *Accelerating LLM Decoding with Speculative
  Sampling*, DeepMind tech report.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from i3.slm.model import AdaptiveSLM
from i3.slm.speculative_decoding import SpeculativeDecoder, SpeculativeStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Aggregated result of one benchmark variant (baseline or speculative).

    Attributes:
        label: Human-readable variant label (e.g. ``"baseline"`` or
            ``"speculative_k=4"``).
        iterations: Number of runs aggregated.
        latencies_ms: List of per-iteration latencies in milliseconds.
        acceptance_rates: List of per-iteration acceptance rates
            (empty for the baseline variant).
        speedups: List of per-iteration speed-up-vs-target estimates.
    """

    label: str
    iterations: int
    latencies_ms: list[float]
    acceptance_rates: list[float]
    speedups: list[float]

    # ---- derived statistics -----------------------------------------------

    @property
    def p50(self) -> float:
        """Median latency (milliseconds). Returns ``0.0`` if empty."""
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        """95th-percentile latency (milliseconds). ``0.0`` if empty."""
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = max(0, int(round(0.95 * (len(sorted_lat) - 1))))
        return sorted_lat[idx]

    @property
    def mean_speedup(self) -> float:
        """Mean empirical speed-up (``1.0`` for baseline runs)."""
        return statistics.fmean(self.speedups) if self.speedups else 1.0

    @property
    def mean_acceptance(self) -> float:
        """Mean per-iteration acceptance rate. ``0.0`` for baseline."""
        return (
            statistics.fmean(self.acceptance_rates)
            if self.acceptance_rates
            else 0.0
        )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_target_model(seed: int) -> AdaptiveSLM:
    """Construct a small random-initialised target :class:`AdaptiveSLM`.

    The configuration is intentionally modest (~5-6 M parameters) so the
    benchmark finishes in seconds on a laptop CPU.

    Args:
        seed: RNG seed forwarded to ``torch.manual_seed``.

    Returns:
        An :class:`AdaptiveSLM` in eval mode.
    """
    torch.manual_seed(seed)
    model = AdaptiveSLM(
        vocab_size=2000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        max_seq_len=128,
        conditioning_dim=64,
        adaptation_dim=8,
        n_cross_heads=2,
        n_conditioning_tokens=4,
        dropout=0.0,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------


@torch.no_grad()
def _baseline_generate(
    target: AdaptiveSLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    """Baseline greedy autoregressive generation on ``target``.

    Args:
        target: The target model.
        prompt_ids: ``[1, prompt_len]`` prompt token IDs.
        max_new_tokens: Number of new tokens to generate greedily.

    Returns:
        Tuple ``(generated_ids, latency_ms)``.
    """
    t0 = time.perf_counter()
    generated = prompt_ids.clone()
    target.clear_cache()
    for _ in range(max_new_tokens):
        logits, _ = target(generated, use_cache=False)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated = torch.cat(
            [generated, torch.tensor([[next_id]], device=prompt_ids.device)],
            dim=1,
        )
    return generated, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def run_benchmark(
    iterations: int,
    max_new_tokens: int,
    seed: int,
    num_drafts_list: list[int],
    acceptance_threshold: float,
) -> list[RunResult]:
    """Run the full benchmark suite.

    Args:
        iterations: Number of prompts per variant.
        max_new_tokens: Number of tokens to generate per prompt.
        seed: RNG seed.
        num_drafts_list: Speculative-decoding ``num_drafts`` values to
            benchmark.
        acceptance_threshold: Forwarded to the
            :class:`SpeculativeDecoder`.

    Returns:
        List of :class:`RunResult`, one per variant (baseline first).
    """
    target = _build_target_model(seed)
    prompt_len = 8

    # Random prompts — identical sequence of prompts across variants so
    # the latency comparison is apples-to-apples.
    torch.manual_seed(seed)
    prompts = [
        torch.randint(
            low=1,
            high=target.vocab_size,
            size=(1, prompt_len),
            dtype=torch.long,
        )
        for _ in range(iterations)
    ]

    results: list[RunResult] = []

    # ---- Baseline ---------------------------------------------------------
    baseline_latencies: list[float] = []
    for prompt in prompts:
        _, lat = _baseline_generate(target, prompt, max_new_tokens)
        baseline_latencies.append(lat)
    results.append(
        RunResult(
            label="baseline_greedy",
            iterations=iterations,
            latencies_ms=baseline_latencies,
            acceptance_rates=[],
            speedups=[1.0] * iterations,
        )
    )

    # ---- Speculative variants --------------------------------------------
    for k in num_drafts_list:
        # Fresh decoder per variant so fallback-mode detection is correct.
        decoder = SpeculativeDecoder(
            target_model=target,
            draft_model=None,
            num_drafts=k,
            acceptance_threshold=acceptance_threshold,
        )
        var_latencies: list[float] = []
        var_accept: list[float] = []
        var_speedups: list[float] = []
        for prompt in prompts:
            _, stats = decoder.generate(
                prompt_ids=prompt,
                conditioning_tokens=None,
                max_new_tokens=max_new_tokens,
            )
            assert isinstance(stats, SpeculativeStats)
            var_latencies.append(stats.total_latency_ms)
            var_accept.append(stats.acceptance_rate)
            var_speedups.append(stats.speedup_vs_target)

        results.append(
            RunResult(
                label=f"speculative_k={k}",
                iterations=iterations,
                latencies_ms=var_latencies,
                acceptance_rates=var_accept,
                speedups=var_speedups,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_markdown(
    results: list[RunResult],
    *,
    iterations: int,
    max_new_tokens: int,
    acceptance_threshold: float,
    baseline_p50: float,
) -> str:
    """Render the benchmark results as a Markdown document.

    Args:
        results: Output of :func:`run_benchmark`.
        iterations: Iterations per variant.
        max_new_tokens: Tokens generated per iteration.
        acceptance_threshold: Threshold used across all speculative
            variants.
        baseline_p50: Baseline P50 latency, used to compute wall-clock
            speed-up vs baseline.

    Returns:
        A Markdown string.
    """
    lines: list[str] = []
    lines.append("# Speculative Decoding Benchmark")
    lines.append("")
    lines.append(
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_"
    )
    lines.append("")
    lines.append(
        "Baseline: greedy autoregressive generation on the target model."
    )
    lines.append(
        "Speculative variants: draft-and-verify via "
        "`SpeculativeDecoder` (Leviathan et al. 2023)."
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Iterations per variant: **{iterations}**")
    lines.append(f"- Max new tokens per run: **{max_new_tokens}**")
    lines.append(
        f"- Acceptance threshold: **{acceptance_threshold:.2f}**"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Variant | P50 (ms) | P95 (ms) | Mean accept rate | "
        "Mean internal speed-up | Wall-clock speed-up vs baseline |"
    )
    lines.append(
        "|---------|---------:|---------:|-----------------:|"
        "-----------------------:|---------------------------------:|"
    )
    for r in results:
        wall_speedup = baseline_p50 / r.p50 if r.p50 > 0 else 1.0
        lines.append(
            f"| `{r.label}` | {r.p50:.2f} | {r.p95:.2f} | "
            f"{r.mean_acceptance:.3f} | {r.mean_speedup:.3f} | "
            f"{wall_speedup:.3f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The *internal speed-up* column is the Leviathan et al. 2023 "
        "per-pass estimate `(accepted + passes) / passes`; it is not "
        "affected by implementation overhead."
    )
    lines.append(
        "- The *wall-clock speed-up vs baseline* column divides baseline "
        "P50 by the variant P50 and therefore reflects actual end-to-end "
        "overhead on this machine."
    )
    lines.append(
        "- With random-initialised weights the draft is unlikely to "
        "match the target well, so acceptance rates will be low. After "
        "distillation (Huawei's Celia approach) the acceptance rate — "
        "and therefore the speed-up — would rise substantially."
    )
    return "\n".join(lines) + "\n"


def _write_report(
    results: list[RunResult],
    out_dir: Path,
    *,
    iterations: int,
    max_new_tokens: int,
    acceptance_threshold: float,
) -> Path:
    """Write the Markdown report to ``out_dir`` and return its path.

    Args:
        results: Output of :func:`run_benchmark`.
        out_dir: Directory to write into (created if missing).
        iterations: Iterations per variant.
        max_new_tokens: Tokens generated per iteration.
        acceptance_threshold: Threshold used across speculative variants.

    Returns:
        Filesystem path of the written report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"speculative_decoding_{ts}.md"
    baseline_p50 = next(
        (r.p50 for r in results if r.label == "baseline_greedy"),
        0.0,
    )
    md = _render_markdown(
        results,
        iterations=iterations,
        max_new_tokens=max_new_tokens,
        acceptance_threshold=acceptance_threshold,
        baseline_p50=baseline_p50,
    )
    path.write_text(md, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding vs baseline.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of prompts per variant (default: 100).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Tokens generated per prompt (default: 32).",
    )
    parser.add_argument(
        "--num-drafts",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Speculative num_drafts values to benchmark (default: 2 4 8).",
    )
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=0.7,
        help="Acceptance threshold forwarded to SpeculativeDecoder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write the Markdown report (default: reports/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary to stdout in addition to the Markdown report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)

    if args.iterations <= 0:
        logger.error("--iterations must be positive")
        return 2
    if args.max_new_tokens <= 0:
        logger.error("--max-new-tokens must be positive")
        return 2

    results = run_benchmark(
        iterations=args.iterations,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        num_drafts_list=list(args.num_drafts),
        acceptance_threshold=args.acceptance_threshold,
    )

    report_path = _write_report(
        results,
        out_dir=args.output_dir,
        iterations=args.iterations,
        max_new_tokens=args.max_new_tokens,
        acceptance_threshold=args.acceptance_threshold,
    )
    logger.info("Wrote report to %s", report_path)

    if args.json:
        summary = {
            "report_path": str(report_path),
            "results": [
                {
                    **{
                        k: v
                        for k, v in asdict(r).items()
                        if k not in {"latencies_ms", "acceptance_rates", "speedups"}
                    },
                    "p50_ms": r.p50,
                    "p95_ms": r.p95,
                    "mean_speedup": r.mean_speedup,
                    "mean_acceptance_rate": r.mean_acceptance,
                }
                for r in results
            ],
        }
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
