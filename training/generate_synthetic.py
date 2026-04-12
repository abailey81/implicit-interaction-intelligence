"""Synthetic interaction data generator for training the User State Encoder.

Generates realistic sequences of 32-dim interaction feature vectors labelled
with one of 8 user-state archetypes.  State transitions within a session
follow a Markov chain that favours naturalistic patterns (e.g. starting
energetic and gradually fatiguing).

Usage::

    python -m training.generate_synthetic            # from project root
    python training/generate_synthetic.py             # direct execution

Outputs are saved to ``data/synthetic/`` as ``.pt`` files:

- ``train.pt``  (80 %)
- ``val.pt``    (10 %)
- ``test.pt``   (10 %)

Each file contains a dict with keys ``"sequences"`` (float32) and
``"labels"`` (int64), windowed and ready for the DataLoader.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8 User-state archetypes
# ---------------------------------------------------------------------------

USER_STATES: dict[str, dict[str, tuple[float, float]]] = {
    "energetic_engaged": {
        "mean_iki_ms": (80, 120),
        "typing_burst_length": (15, 30),
        "pause_duration_ms": (200, 500),
        "backspace_ratio": (0.02, 0.05),
        "message_length_words": (15, 40),
        "vocabulary_richness": (0.7, 0.9),
        "formality": (0.3, 0.6),
        "response_latency_ms": (500, 2000),
    },
    "tired_disengaging": {
        "mean_iki_ms": (200, 400),
        "typing_burst_length": (3, 8),
        "pause_duration_ms": (1000, 3000),
        "backspace_ratio": (0.05, 0.12),
        "message_length_words": (3, 10),
        "vocabulary_richness": (0.3, 0.5),
        "formality": (0.2, 0.4),
        "response_latency_ms": (5000, 15000),
    },
    "stressed_urgent": {
        "mean_iki_ms": (60, 100),
        "typing_burst_length": (5, 15),
        "pause_duration_ms": (100, 300),
        "backspace_ratio": (0.08, 0.15),
        "message_length_words": (5, 15),
        "vocabulary_richness": (0.4, 0.6),
        "formality": (0.1, 0.3),
        "response_latency_ms": (200, 1000),
    },
    "relaxed_conversational": {
        "mean_iki_ms": (120, 180),
        "typing_burst_length": (8, 20),
        "pause_duration_ms": (500, 1500),
        "backspace_ratio": (0.03, 0.06),
        "message_length_words": (10, 25),
        "vocabulary_richness": (0.5, 0.7),
        "formality": (0.3, 0.5),
        "response_latency_ms": (2000, 5000),
    },
    "focused_deep": {
        "mean_iki_ms": (100, 160),
        "typing_burst_length": (20, 50),
        "pause_duration_ms": (2000, 5000),
        "backspace_ratio": (0.01, 0.03),
        "message_length_words": (20, 60),
        "vocabulary_richness": (0.7, 0.95),
        "formality": (0.5, 0.8),
        "response_latency_ms": (3000, 8000),
    },
    "motor_difficulty": {
        "mean_iki_ms": (300, 800),
        "typing_burst_length": (1, 4),
        "pause_duration_ms": (2000, 8000),
        "backspace_ratio": (0.15, 0.35),
        "message_length_words": (2, 8),
        "vocabulary_richness": (0.3, 0.5),
        "formality": (0.2, 0.4),
        "response_latency_ms": (5000, 20000),
    },
    "distracted_multitasking": {
        "mean_iki_ms": (100, 300),
        "typing_burst_length": (3, 10),
        "pause_duration_ms": (3000, 15000),
        "backspace_ratio": (0.05, 0.10),
        "message_length_words": (5, 15),
        "vocabulary_richness": (0.4, 0.6),
        "formality": (0.2, 0.4),
        "response_latency_ms": (10000, 60000),
    },
    "formal_professional": {
        "mean_iki_ms": (120, 200),
        "typing_burst_length": (10, 25),
        "pause_duration_ms": (1000, 3000),
        "backspace_ratio": (0.03, 0.07),
        "message_length_words": (15, 40),
        "vocabulary_richness": (0.6, 0.85),
        "formality": (0.7, 0.95),
        "response_latency_ms": (2000, 6000),
    },
}

STATE_NAMES: list[str] = list(USER_STATES.keys())
STATE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(STATE_NAMES)}
NUM_STATES: int = len(STATE_NAMES)

# ---------------------------------------------------------------------------
# Markov transition matrix
# ---------------------------------------------------------------------------

# Row = current state, col = next state.  Sessions tend to drift from
# energetic toward tired/distracted, with abrupt jumps to stressed or
# back to engaged possible.
TRANSITION_MATRIX: np.ndarray = np.array(
    [
        #  enrg   tired  strss  relxd  focus  motor  distr  forml
        [0.40,  0.10,  0.08,  0.15,  0.12,  0.01,  0.09,  0.05],  # energetic_engaged
        [0.05,  0.45,  0.05,  0.10,  0.03,  0.10,  0.18,  0.04],  # tired_disengaging
        [0.10,  0.08,  0.35,  0.05,  0.10,  0.02,  0.20,  0.10],  # stressed_urgent
        [0.15,  0.12,  0.05,  0.38,  0.10,  0.02,  0.10,  0.08],  # relaxed_conversational
        [0.08,  0.10,  0.05,  0.10,  0.45,  0.02,  0.10,  0.10],  # focused_deep
        [0.03,  0.20,  0.05,  0.05,  0.02,  0.50,  0.10,  0.05],  # motor_difficulty
        [0.10,  0.15,  0.10,  0.10,  0.05,  0.05,  0.40,  0.05],  # distracted_multitasking
        [0.08,  0.05,  0.05,  0.10,  0.15,  0.02,  0.05,  0.50],  # formal_professional
    ],
    dtype=np.float64,
)

# Normalise rows to sum to 1 (safety net)
TRANSITION_MATRIX /= TRANSITION_MATRIX.sum(axis=1, keepdims=True)

# Starting-state distribution: sessions usually begin engaged or relaxed
START_PROBS: np.ndarray = np.array(
    [0.30, 0.05, 0.10, 0.20, 0.15, 0.02, 0.08, 0.10], dtype=np.float64
)
START_PROBS /= START_PROBS.sum()


# ---------------------------------------------------------------------------
# Normalisation ranges (used to map archetype params into [0, 1])
# ---------------------------------------------------------------------------

# These are "global" ranges covering the extremes of all archetypes.
FEATURE_GLOBAL_RANGES: dict[str, tuple[float, float]] = {
    "mean_iki_ms": (50, 900),
    "typing_burst_length": (1, 55),
    "pause_duration_ms": (50, 16000),
    "backspace_ratio": (0.0, 0.40),
    "message_length_words": (1, 65),
    "vocabulary_richness": (0.2, 1.0),
    "formality": (0.0, 1.0),
    "response_latency_ms": (100, 65000),
}


def _normalise(value: float, lo: float, hi: float) -> float:
    """Linearly scale *value* from [lo, hi] to [0, 1], clamped."""
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Feature vector generation
# ---------------------------------------------------------------------------


def _generate_feature_vector(
    state_name: str, rng: np.random.Generator
) -> np.ndarray:
    """Sample a single 32-dim feature vector for *state_name*.

    The 8 archetype parameters are sampled uniformly within their range,
    normalised to [0, 1], and then expanded with correlated noise to fill
    the remaining 24 dimensions.

    Returns:
        ``np.ndarray`` of shape ``(32,)`` with values in roughly [0, 1].
    """
    params = USER_STATES[state_name]

    # -- Sample the 8 primary parameters ------------------------------------
    raw: dict[str, float] = {}
    for feat_name, (lo, hi) in params.items():
        raw[feat_name] = rng.uniform(lo, hi)

    # -- Normalise to [0, 1] ------------------------------------------------
    normed: list[float] = []
    for feat_name in FEATURE_GLOBAL_RANGES:
        glo, ghi = FEATURE_GLOBAL_RANGES[feat_name]
        normed.append(_normalise(raw[feat_name], glo, ghi))

    primary = np.array(normed, dtype=np.float64)  # shape (8,)

    # -- Derive the remaining 24 dimensions ---------------------------------
    # Group 2: Message content (8 dims) -- derived from primary features
    msg_len_n = primary[4]      # message_length_words normalised
    vocab_n = primary[5]        # vocabulary_richness normalised
    formality_n = primary[6]    # formality normalised
    iki_n = primary[0]          # mean_iki normalised

    content = np.array([
        msg_len_n,                                          # message_length
        vocab_n,                                            # type_token_ratio
        np.clip(vocab_n * 0.8 + rng.normal(0, 0.05), 0, 1),  # mean_word_length
        np.clip(0.5 + (vocab_n - 0.5) * 0.6 + rng.normal(0, 0.05), 0, 1),  # flesch_kincaid
        np.clip(rng.uniform(0.05, 0.25), 0, 1),            # question_ratio
        formality_n,                                        # formality
        np.clip(1.0 - formality_n + rng.normal(0, 0.05), 0, 1),  # emoji_density
        np.clip(rng.normal(0.5, 0.15), -1, 1) * 0.5 + 0.5,  # sentiment_valence [0,1]
    ], dtype=np.float64)

    # Group 3: Session dynamics (8 dims)
    engagement = 1.0 - primary[7]  # inverse of response_latency
    dynamics = np.array([
        np.clip(rng.normal(0, 0.1), -0.5, 0.5) + 0.5,     # length_trend
        np.clip(rng.normal(0, 0.1), -0.5, 0.5) + 0.5,     # latency_trend
        np.clip(rng.normal(0, 0.1), -0.5, 0.5) + 0.5,     # vocab_trend
        np.clip(engagement + rng.normal(0, 0.05), 0, 1),   # engagement_velocity
        np.clip(rng.uniform(0.3, 0.9), 0, 1),              # topic_coherence
        rng.uniform(0.0, 1.0),                              # session_progress
        np.clip(rng.normal(0, 0.2), -1, 1) * 0.5 + 0.5,    # time_deviation
        np.clip(engagement * 0.7 + rng.normal(0, 0.1), 0, 1),  # response_depth
    ], dtype=np.float64)

    # Group 4: Deviation metrics (8 dims) -- z-score-like, centred at 0.5
    deviations = np.clip(
        rng.normal(0.5, 0.15, size=8), 0, 1
    ).astype(np.float64)

    vec = np.concatenate([primary, content, dynamics, deviations])  # (32,)

    # Add small global noise for realism
    noise = rng.normal(0, 0.02, size=32)
    vec = np.clip(vec + noise, 0.0, 1.0)

    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Session generation (Markov chain)
# ---------------------------------------------------------------------------


def generate_state_sequence(
    n_steps: int, rng: np.random.Generator
) -> list[str]:
    """Sample a Markov-chain state sequence of length *n_steps*.

    Args:
        n_steps: Number of timesteps (messages) in the session.
        rng:     NumPy random generator.

    Returns:
        List of state name strings.
    """
    seq: list[str] = []
    state_idx = rng.choice(NUM_STATES, p=START_PROBS)
    for _ in range(n_steps):
        seq.append(STATE_NAMES[state_idx])
        state_idx = rng.choice(NUM_STATES, p=TRANSITION_MATRIX[state_idx])
    return seq


def generate_synthetic_session(
    state_sequence: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate feature vectors and labels for one session.

    Args:
        state_sequence: List of state names, one per message.
        rng:            NumPy random generator.

    Returns:
        Tuple ``(features, labels)`` where features is
        ``[n_messages, 32]`` float32 and labels is ``[n_messages]`` int64.
    """
    features: list[np.ndarray] = []
    labels: list[int] = []
    for state_name in state_sequence:
        vec = _generate_feature_vector(state_name, rng)
        features.append(vec)
        labels.append(STATE_TO_ID[state_name])
    return np.stack(features), np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Windowed dataset creation
# ---------------------------------------------------------------------------


def _create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over one session to produce training samples.

    Each window of ``window_size`` consecutive vectors yields one sample.
    The label is that of the **last** vector in the window.

    If the session is shorter than ``window_size``, it is left-padded with
    zeros.

    Args:
        features: ``[n_messages, 32]`` feature array.
        labels:   ``[n_messages]`` label array.
        window_size: Window width.

    Returns:
        Tuple ``(X, y)`` where X is ``[n_windows, window_size, 32]`` and
        y is ``[n_windows]``.
    """
    n = len(features)

    if n < window_size:
        # Pad and return a single sample
        pad = np.zeros((window_size - n, 32), dtype=np.float32)
        padded = np.concatenate([pad, features], axis=0)
        return padded[np.newaxis], labels[-1:][np.newaxis].flatten()

    windows_x: list[np.ndarray] = []
    windows_y: list[int] = []
    for start in range(n - window_size + 1):
        end = start + window_size
        windows_x.append(features[start:end])
        windows_y.append(int(labels[end - 1]))  # label of last in window

    return np.stack(windows_x), np.array(windows_y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    n_sessions: int = 10_000,
    messages_per_session: int = 20,
    window_size: int = 10,
    seed: int = 42,
    output_dir: str | Path = "data/synthetic",
) -> dict[str, Path]:
    """Generate the full synthetic dataset and save train/val/test splits.

    Args:
        n_sessions:           Total number of sessions to generate.
        messages_per_session: Messages per session.
        window_size:          Sliding window width.
        seed:                 RNG seed for reproducibility.
        output_dir:           Directory for ``.pt`` output files.

    Returns:
        Dict mapping split name to saved file path.
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating %d sessions x %d messages (window=%d) ...",
        n_sessions,
        messages_per_session,
        window_size,
    )

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for i in range(n_sessions):
        states = generate_state_sequence(messages_per_session, rng)
        feats, labs = generate_synthetic_session(states, rng)
        wx, wy = _create_windows(feats, labs, window_size)
        all_x.append(wx)
        all_y.append(wy)

        if (i + 1) % 2000 == 0:
            logger.info("  ... %d / %d sessions", i + 1, n_sessions)

    X = np.concatenate(all_x, axis=0)  # [N, window_size, 32]
    Y = np.concatenate(all_y, axis=0)  # [N]

    # Shuffle
    perm = rng.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    # Split: 80/10/10
    n_total = len(X)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train": (X[:n_train], Y[:n_train]),
        "val": (X[n_train : n_train + n_val], Y[n_train : n_train + n_val]),
        "test": (X[n_train + n_val :], Y[n_train + n_val :]),
    }

    saved_paths: dict[str, Path] = {}
    for name, (sx, sy) in splits.items():
        path = out / f"{name}.pt"
        torch.save(
            {
                "sequences": torch.tensor(sx, dtype=torch.float32),
                "labels": torch.tensor(sy, dtype=torch.int64),
            },
            path,
        )
        logger.info(
            "  Saved %s: %d samples -> %s", name, len(sx), path
        )
        saved_paths[name] = path

    # Log label distribution for the training set
    train_labels = splits["train"][1]
    for sid, sname in enumerate(STATE_NAMES):
        count = int((train_labels == sid).sum())
        logger.info("    %s: %d (%.1f%%)", sname, count, 100 * count / len(train_labels))

    logger.info("Dataset generation complete. Total samples: %d", n_total)
    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic interaction data for TCN training."
    )
    parser.add_argument(
        "--sessions", type=int, default=10_000,
        help="Number of sessions to generate (default: 10000).",
    )
    parser.add_argument(
        "--messages", type=int, default=20,
        help="Messages per session (default: 20).",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Sliding window size (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/synthetic",
        help="Output directory (default: data/synthetic).",
    )
    args = parser.parse_args()

    # SEC: bound-check positive integers so a typo can't trigger a
    # zero- or negative-length loop. Note: window > messages is allowed
    # (the windowing helper left-pads short sessions with zeros).
    if args.sessions <= 0:
        parser.error("--sessions must be a positive integer")
    if args.messages <= 0:
        parser.error("--messages must be a positive integer")
    if args.window <= 0:
        parser.error("--window must be a positive integer")
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    args = _parse_args()
    generate_dataset(
        n_sessions=args.sessions,
        messages_per_session=args.messages,
        window_size=args.window,
        seed=args.seed,
        output_dir=args.output_dir,
    )
