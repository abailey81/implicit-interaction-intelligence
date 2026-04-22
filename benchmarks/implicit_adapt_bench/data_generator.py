"""Synthetic-data generator for ImplicitAdaptBench.

The generator deterministically samples ``n_records_per_archetype`` records
for every canonical archetype and every canonical prompt. All synthetic
splits are fully reproducible given a fixed seed.

Example::

    from pathlib import Path
    from benchmarks.implicit_adapt_bench.data_generator import (
        generate_synthetic_split, write_benchmark_jsonl,
    )

    records = generate_synthetic_split(
        n_records_per_archetype=4, split="dev", seed=42
    )
    write_benchmark_jsonl(records, Path("dev.jsonl"))
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

from benchmarks.implicit_adapt_bench.data_schema import (
    BehaviouralWindow,
    BenchmarkRecord,
    BenchmarkSplit,
    FormalityBucket,
    LengthBucket,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-imported canonical prompts / archetypes from the I³ Batch A codebase
# ---------------------------------------------------------------------------

try:
    from i3.eval.ablation_experiment import (
        _CANONICAL_PROMPTS as _I3_CANONICAL_PROMPTS,
        canonical_archetypes as _i3_canonical_archetypes,
    )
    _PROMPTS: tuple[str, ...] = tuple(_I3_CANONICAL_PROMPTS)
    _ARCHETYPE_FN = _i3_canonical_archetypes
except ImportError as exc:  # pragma: no cover - I³ not importable
    logger.info(
        "i3.eval.ablation_experiment not importable (%s); using the parallel "
        "in-module prompt list.",
        exc,
    )
    _PROMPTS = (
        "Tell me about your weekend.",
        "What is your favourite season, and why?",
        "How have you been feeling lately?",
        "Do you enjoy reading novels?",
        "If you could travel anywhere, where would you go?",
        "Explain how a transformer self-attention layer works.",
        "What is the difference between variance and standard deviation?",
        "Summarise the key idea behind reinforcement learning.",
        "Why is cross-validation important in machine learning?",
        "Explain gradient descent to a beginner.",
        "I feel overwhelmed by work lately.",
        "I cannot seem to focus today.",
        "I had an argument with a close friend.",
        "I am anxious about an upcoming interview.",
        "I am struggling to sleep this week.",
        "Please write a two-sentence apology email.",
        "Draft a polite request to reschedule a meeting.",
        "Give me three tips for writing cleaner code.",
        "List four healthy lunch ideas.",
        "Help me plan a 30-minute home workout.",
    )
    _ARCHETYPE_FN = None


# ---------------------------------------------------------------------------
# Archetype -> target style / length / formality
# ---------------------------------------------------------------------------


_ARCHETYPE_REFERENCE: dict[
    str, tuple[str, LengthBucket, FormalityBucket]
] = {
    "neutral": ("neutral_neutral_medium", "medium", "neutral"),
    "low_load_warm": ("warm_casual_short", "short", "casual"),
    "high_load_technical": ("objective_formal_long", "long", "formal"),
    "urgent_formal": ("objective_formal_short", "short", "formal"),
    "accessible_simple": ("warm_casual_short", "short", "casual"),
    "casual_verbose": ("warm_casual_long", "long", "casual"),
    "direct_terse": ("objective_neutral_short", "short", "neutral"),
    "reflective_neutral": ("reserved_neutral_medium", "medium", "neutral"),
}
"""Deterministic mapping from archetype name to (style_label, length, formality).

The label format is ``<tone>_<formality>_<length>`` matching the axes parsed
by :func:`benchmarks.implicit_adapt_bench.metrics._parse_style_label`.
"""


# Fallback archetype vectors (used only when the I³ tree is not importable).
_FALLBACK_ARCHETYPE_VECTORS: dict[str, list[float]] = {
    "neutral": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
    "low_load_warm": [0.1, 0.2, 0.3, 0.8, 0.3, 0.1, 0.6, 0.0],
    "high_load_technical": [0.9, 0.8, 0.8, 0.2, 0.8, 0.7, 0.1, 0.0],
    "urgent_formal": [0.6, 0.9, 0.2, 0.2, 0.95, 0.5, 0.2, 0.0],
    "accessible_simple": [0.15, 0.3, 0.3, 0.6, 0.5, 0.3, 0.95, 0.0],
    "casual_verbose": [0.4, 0.1, 0.9, 0.7, 0.4, 0.2, 0.3, 0.0],
    "direct_terse": [0.5, 0.5, 0.1, 0.2, 0.95, 0.6, 0.2, 0.0],
    "reflective_neutral": [0.5, 0.5, 0.7, 0.5, 0.3, 0.4, 0.4, 0.0],
}


def _resolve_archetype_vectors() -> dict[str, list[float]]:
    """Build the ``name -> 8-dim list`` mapping.

    Uses :func:`i3.eval.ablation_experiment.canonical_archetypes` when
    importable, otherwise falls back to :data:`_FALLBACK_ARCHETYPE_VECTORS`.
    """
    if _ARCHETYPE_FN is None:
        return {k: list(v) for k, v in _FALLBACK_ARCHETYPE_VECTORS.items()}
    out: dict[str, list[float]] = {}
    for name, av in _ARCHETYPE_FN().items():
        t = av.to_tensor()
        out[name] = [float(x) for x in t.tolist()]
    return out


# ---------------------------------------------------------------------------
# Behavioural-window synthesiser
# ---------------------------------------------------------------------------


def _synthesise_behavioural_window(
    adaptation_vector: Sequence[float],
    rng: random.Random,
) -> BehaviouralWindow:
    """Synthesise a plausible 32-dim feature vector from an adaptation vector.

    The generator is intentionally simple: it uses the archetype's first few
    adaptation dimensions to drive correlated movements in groups of the
    32-dim feature vector (keystroke dynamics, message content, session
    dynamics, deviation metrics), then adds per-channel Gaussian noise so
    that two records with the same archetype are never identical.

    Args:
        adaptation_vector: An 8-dim AdaptationVector target.
        rng: A seeded :class:`random.Random` instance.

    Returns:
        A fully populated :class:`BehaviouralWindow`.
    """
    cognitive_load = float(adaptation_vector[0])
    formality = float(adaptation_vector[1])
    verbosity = float(adaptation_vector[2])
    emotionality = float(adaptation_vector[3])
    directness = float(adaptation_vector[4])
    emotional_tone = float(adaptation_vector[5])
    accessibility = float(adaptation_vector[6])

    def _jitter(mu: float, sigma: float = 0.05) -> float:
        return max(0.0, min(1.0, rng.gauss(mu, sigma)))

    def _signed_jitter(mu: float, sigma: float = 0.05) -> float:
        return max(-1.0, min(1.0, rng.gauss(mu, sigma)))

    # --- 8 keystroke-dynamics features ----------------------------------
    # High cognitive load + high accessibility => slower, more edits.
    slow_typing = 0.5 + 0.3 * cognitive_load + 0.3 * accessibility
    keystroke = [
        _jitter(slow_typing),                  # mean_iki
        _jitter(0.4 + 0.3 * cognitive_load),   # std_iki
        _jitter(0.6 - 0.3 * accessibility),    # mean_burst_length
        _jitter(0.3 + 0.4 * cognitive_load),   # mean_pause_duration
        _jitter(0.2 + 0.5 * accessibility),    # backspace_ratio
        _jitter(1.0 - slow_typing),            # composition_speed
        _jitter(0.3 + 0.3 * cognitive_load),   # pause_before_send
        _jitter(0.2 + 0.4 * accessibility),    # editing_effort
    ]

    # --- 8 message-content features --------------------------------------
    content = [
        _jitter(0.2 + 0.6 * verbosity),        # message_length
        _jitter(0.3 + 0.5 * cognitive_load),   # type_token_ratio
        _jitter(0.4 + 0.3 * formality),        # mean_word_length
        _jitter(0.3 + 0.5 * cognitive_load),   # flesch_kincaid
        _jitter(0.2 + 0.2 * directness),       # question_ratio
        _jitter(formality),                    # formality
        _jitter(0.3 * emotionality),           # emoji_density
        _signed_jitter(0.3 * emotionality - 0.3 * emotional_tone),  # sentiment
    ]

    # --- 8 session-dynamics features -------------------------------------
    session = [
        _signed_jitter(0.1 * verbosity - 0.05),      # length_trend
        _signed_jitter(0.1 * (1.0 - slow_typing)),   # latency_trend
        _signed_jitter(0.1 * cognitive_load),        # vocab_trend
        _jitter(0.4 + 0.3 * emotionality),           # engagement_velocity
        _jitter(0.5 + 0.2 * formality),              # topic_coherence
        _jitter(rng.random()),                       # session_progress
        _signed_jitter(0.0, 0.1),                    # time_deviation
        _jitter(0.4 + 0.4 * emotionality),           # response_depth
    ]

    # --- 8 deviation features (z-scores, centred at 0) -------------------
    deviation = [_signed_jitter(0.0, 0.15) for _ in range(8)]

    feature_vector = keystroke + content + session + deviation

    # Raw keystroke intervals: 10–60 intervals whose mean scales with
    # ``slow_typing``. Values in ms.
    n_intervals = rng.randint(10, 60)
    target_mean_ms = 80.0 + 220.0 * slow_typing
    raw_iki = [
        max(10.0, rng.gauss(target_mean_ms, target_mean_ms * 0.4))
        for _ in range(n_intervals)
    ]

    feature_names = [
        "mean_iki",
        "std_iki",
        "mean_burst_length",
        "mean_pause_duration",
        "backspace_ratio",
        "composition_speed",
        "pause_before_send",
        "editing_effort",
        "message_length",
        "type_token_ratio",
        "mean_word_length",
        "flesch_kincaid",
        "question_ratio",
        "formality",
        "emoji_density",
        "sentiment_valence",
        "length_trend",
        "latency_trend",
        "vocab_trend",
        "engagement_velocity",
        "topic_coherence",
        "session_progress",
        "time_deviation",
        "response_depth",
        "iki_deviation",
        "length_deviation",
        "vocab_deviation",
        "formality_deviation",
        "speed_deviation",
        "engagement_deviation",
        "complexity_deviation",
        "pattern_deviation",
    ]
    deviation_dict = {
        feature_names[24 + i]: deviation[i] for i in range(8)
    }

    return BehaviouralWindow(
        feature_vector=feature_vector,
        raw_keystroke_intervals_ms=raw_iki,
        baseline_deviation_metrics=deviation_dict,
    )


# ---------------------------------------------------------------------------
# Split -> seed offset (so that train/dev/test don't collide)
# ---------------------------------------------------------------------------


_SPLIT_SEED_OFFSET: dict[BenchmarkSplit, int] = {
    "train": 0,
    "dev": 10_000,
    "test": 20_000,
    "held_out_human": 30_000,
}


def _maybe_human_preference(
    rng: random.Random, split: BenchmarkSplit
) -> float | None:
    """Populate a plausible human preference score for the ``held_out_human`` split.

    The mock preference score is uniformly drawn from ``[0.3, 0.9]`` — these
    are placeholder values that let scoring code exercise the
    :func:`benchmarks.implicit_adapt_bench.metrics.preference_rate` path
    end-to-end. Real preference scores are populated via the IRB-lite
    protocol described in ``docs/research/implicit_adapt_bench.md``.

    Args:
        rng: A seeded PRNG.
        split: The split this record belongs to.

    Returns:
        ``None`` for every non-``held_out_human`` split; else a float in
        ``[0.3, 0.9]``.
    """
    if split != "held_out_human":
        return None
    return float(rng.uniform(0.3, 0.9))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_split(
    n_records_per_archetype: int,
    split: BenchmarkSplit,
    seed: int,
) -> list[BenchmarkRecord]:
    """Generate a reproducible synthetic split.

    For every archetype in :func:`_resolve_archetype_vectors` the function
    emits ``n_records_per_archetype`` records whose prompts cycle through
    the canonical prompt list. Record IDs are ``"<split>-<index>"``.

    Args:
        n_records_per_archetype: Number of records per archetype (> 0).
        split: One of the four canonical :data:`BenchmarkSplit` values.
        seed: Integer seed for the PRNG. The actual seed is
            ``seed + _SPLIT_SEED_OFFSET[split]`` so that different splits
            produced with the same base seed are disjoint.

    Returns:
        A deterministic list of :class:`BenchmarkRecord` instances.

    Raises:
        ValueError: If ``n_records_per_archetype`` is not strictly positive
            or the split string is unrecognised.
    """
    if n_records_per_archetype <= 0:
        raise ValueError(
            f"n_records_per_archetype must be > 0, got {n_records_per_archetype}."
        )
    if split not in _SPLIT_SEED_OFFSET:
        raise ValueError(
            f"Unknown split {split!r}; must be one of {list(_SPLIT_SEED_OFFSET)}."
        )

    rng = random.Random(seed + _SPLIT_SEED_OFFSET[split])
    archetype_vectors = _resolve_archetype_vectors()

    records: list[BenchmarkRecord] = []
    index = 0
    for name, vec in archetype_vectors.items():
        style_label, length_bucket, formality_bucket = _ARCHETYPE_REFERENCE.get(
            name, ("neutral_neutral_medium", "medium", "neutral")
        )
        for k in range(n_records_per_archetype):
            prompt = _PROMPTS[(index + k) % len(_PROMPTS)]
            window = _synthesise_behavioural_window(vec, rng)
            record = BenchmarkRecord(
                record_id=f"{split}-{index:04d}",
                behavioural_window=window,
                prompt=prompt,
                target_archetype=name,
                target_adaptation_vector=list(vec),
                reference_style_label=style_label,
                reference_length_bucket=length_bucket,
                reference_formality_bucket=formality_bucket,
                human_preference_score=_maybe_human_preference(rng, split),
            )
            records.append(record)
            index += 1

    return records


def write_benchmark_jsonl(
    records: Sequence[BenchmarkRecord], path: Path
) -> None:
    """Write records as one JSON object per line.

    Args:
        records: Records to write.
        path: Output file path. Parent directories are created if missing.

    Raises:
        OSError: If the path cannot be created or written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # SEC: Write with explicit utf-8 + newline to avoid platform-specific
    # BOM / line-ending surprises when the file is consumed by downstream
    # tooling (e.g. a CI scorer on Linux reading a file written on Windows).
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(rec.model_dump_json())
            fh.write("\n")


def read_benchmark_jsonl(path: Path) -> list[BenchmarkRecord]:
    """Read records from a JSONL file written by :func:`write_benchmark_jsonl`.

    Args:
        path: JSONL file path.

    Returns:
        A list of :class:`BenchmarkRecord` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or does not conform to the
            :class:`BenchmarkRecord` schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    records: list[BenchmarkRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {i + 1} of {path}: {exc}"
                ) from exc
            records.append(BenchmarkRecord.model_validate(payload))
    return records
