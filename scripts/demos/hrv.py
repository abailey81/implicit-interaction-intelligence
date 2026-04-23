"""PPG / HRV extraction demo (Batch F-2).

Generates a synthetic wrist-worn PPG signal, runs :class:`PPGHRVExtractor`,
and prints the resulting 8-dim feature vector with a one-line Markdown
interpretation referencing the Task Force 1996 HRV standard.

Usage::

    python -m scripts.run_hrv_demo
    python -m scripts.run_hrv_demo --stress-scenario
    python -m scripts.run_hrv_demo --duration 120 --sample-rate 50

The ``--stress-scenario`` flag injects an elevated-HR / reduced-HRV
response alongside the baseline and prints the delta, illustrating how
the I³ pipeline surfaces cognitive-load shifts.

References
----------
* Task Force of the European Society of Cardiology (1996).  *Heart rate
  variability: standards of measurement...*  Circulation 93(5).
* Shaffer & Ginsberg (2017).  *An overview of heart rate variability
  metrics and norms.*  Frontiers in Public Health 5.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

from i3.multimodal.ppg_hrv import PPGFeatureVector, PPGHRVExtractor


# ---------------------------------------------------------------------------
# Synthetic signal generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticPPGConfig:
    """Configuration for a synthetic wrist PPG signal.

    Attributes:
        duration_s: Total signal length in seconds.
        sample_rate: Samples per second (25 Hz matches Huawei Watch 5).
        mean_hr_bpm: Target mean heart rate.
        hrv_std_ms: Target standard deviation of IBIs in milliseconds
            (proxy for overall HRV).
        noise_std: Additive Gaussian noise amplitude.
        drift_amplitude: Amplitude of a slow 0.1-Hz baseline drift.
        rng_seed: Seed for :func:`numpy.random.default_rng`.
    """

    duration_s: float = 60.0
    sample_rate: float = 25.0
    mean_hr_bpm: float = 72.0
    hrv_std_ms: float = 45.0
    noise_std: float = 0.05
    drift_amplitude: float = 0.15
    rng_seed: int = 42


def _synthesise_ppg(cfg: SyntheticPPGConfig) -> np.ndarray:
    """Return a synthetic wrist-worn PPG waveform.

    The waveform is a sinusoidal pulse train whose inter-beat intervals
    are drawn from a log-normal distribution centred at
    ``60 / mean_hr_bpm`` seconds with dispersion controlled by
    ``hrv_std_ms``, plus additive Gaussian noise and a slow baseline drift.

    Args:
        cfg: Synthesis configuration.

    Returns:
        1-D ``float32`` numpy array of length
        ``int(cfg.duration_s * cfg.sample_rate)``.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    n_samples = int(cfg.duration_s * cfg.sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / cfg.sample_rate

    mean_ibi_s = 60.0 / cfg.mean_hr_bpm
    # Generate beat times drawn from a Gaussian around the mean IBI.
    beat_times: list[float] = []
    cur = 0.0
    while cur < cfg.duration_s:
        ibi = float(
            rng.normal(loc=mean_ibi_s, scale=cfg.hrv_std_ms / 1000.0)
        )
        ibi = max(0.33, min(2.0, ibi))  # physiological guard
        cur += ibi
        beat_times.append(cur)

    signal = np.zeros(n_samples, dtype=np.float64)
    # Model each beat as a narrow Gaussian pulse.
    pulse_sigma = 0.08  # seconds
    for bt in beat_times:
        if bt >= cfg.duration_s:
            continue
        signal += np.exp(-0.5 * ((t - bt) / pulse_sigma) ** 2)

    # Slow baseline drift (respiration + motion).
    signal += cfg.drift_amplitude * np.sin(2.0 * np.pi * 0.1 * t)
    # Additive noise.
    signal += rng.normal(loc=0.0, scale=cfg.noise_std, size=n_samples)
    return signal.astype(np.float32)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _markdown_interpretation(vec: PPGFeatureVector) -> str:
    """Return a one-line Markdown interpretation of the feature vector.

    Args:
        vec: The :class:`PPGFeatureVector` to interpret.

    Returns:
        A Markdown string referencing the Task Force 1996 norms.
    """
    # Very loose heuristics, intended as illustrative only.
    if vec.rmssd_ms < 15.0:
        tone = "low vagal tone (possible stress / fatigue)"
    elif vec.rmssd_ms < 35.0:
        tone = "moderate vagal tone"
    else:
        tone = "within resting-vagal-tone range"

    return (
        f"**HR {vec.hr_bpm:.1f} bpm, RMSSD {vec.rmssd_ms:.1f} ms, "
        f"SDNN {vec.sdnn_ms:.1f} ms, LF/HF {vec.lf_hf_ratio:.2f}** — "
        f"{tone} [Task Force 1996; Shaffer & Ginsberg 2017]."
    )


def _print_vector(label: str, vec: PPGFeatureVector) -> None:
    """Print a labelled feature vector as a neatly formatted block.

    Args:
        label: Short name identifying the scenario.
        vec: Feature vector to print.
    """
    payload = {
        "hr_bpm": vec.hr_bpm,
        "rmssd_ms": vec.rmssd_ms,
        "sdnn_ms": vec.sdnn_ms,
        "pnn50_percent": vec.pnn50_percent,
        "lf_power": vec.lf_power,
        "hf_power": vec.hf_power,
        "lf_hf_ratio": vec.lf_hf_ratio,
        "sample_entropy": vec.sample_entropy,
    }
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2))
    print(_markdown_interpretation(vec))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the demo CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "PPG / HRV extraction demo using a synthetic wrist waveform "
            "(Huawei Watch 5-style 25 Hz PPG)."
        )
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Signal duration in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=25.0,
        help="PPG sample rate in Hz (default: 25.0).",
    )
    parser.add_argument(
        "--mean-hr",
        type=float,
        default=72.0,
        help="Target mean heart rate in bpm (default: 72).",
    )
    parser.add_argument(
        "--hrv-std-ms",
        type=float,
        default=45.0,
        help="Target IBI std-dev in milliseconds (default: 45).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic synthesis (default: 42).",
    )
    parser.add_argument(
        "--stress-scenario",
        action="store_true",
        help="Also synthesise an elevated-HR / reduced-HRV stress response "
        "and print the delta.",
    )
    return parser


def _run_one(cfg: SyntheticPPGConfig, label: str) -> PPGFeatureVector:
    """Synthesise a PPG signal, run the extractor, and print the result.

    Args:
        cfg: Synthesis configuration.
        label: Scenario label for display.

    Returns:
        The extracted :class:`PPGFeatureVector`.
    """
    signal = _synthesise_ppg(cfg)
    extractor = PPGHRVExtractor()
    vec = extractor.extract(signal, sample_rate=cfg.sample_rate)
    _print_vector(label, vec)
    return vec


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Unix exit code (0 on success).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    base_cfg = SyntheticPPGConfig(
        duration_s=float(args.duration),
        sample_rate=float(args.sample_rate),
        mean_hr_bpm=float(args.mean_hr),
        hrv_std_ms=float(args.hrv_std_ms),
        rng_seed=int(args.seed),
    )

    baseline = _run_one(base_cfg, label="Baseline (resting)")

    if args.stress_scenario:
        stress_cfg = SyntheticPPGConfig(
            duration_s=base_cfg.duration_s,
            sample_rate=base_cfg.sample_rate,
            mean_hr_bpm=base_cfg.mean_hr_bpm + 25.0,
            hrv_std_ms=max(5.0, base_cfg.hrv_std_ms * 0.3),
            noise_std=base_cfg.noise_std,
            drift_amplitude=base_cfg.drift_amplitude,
            rng_seed=base_cfg.rng_seed + 1,
        )
        stress = _run_one(stress_cfg, label="Stress response")

        print("\n=== Delta (stress - baseline) ===")
        print(
            json.dumps(
                {
                    "d_hr_bpm": stress.hr_bpm - baseline.hr_bpm,
                    "d_rmssd_ms": stress.rmssd_ms - baseline.rmssd_ms,
                    "d_sdnn_ms": stress.sdnn_ms - baseline.sdnn_ms,
                    "d_lf_hf_ratio": stress.lf_hf_ratio - baseline.lf_hf_ratio,
                },
                indent=2,
            )
        )
        print(
            "Interpretation: elevated HR with compressed RMSSD/SDNN is the "
            "classical sympathetic-dominance signature [Shaffer & Ginsberg "
            "2017; Makivic 2013]."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
