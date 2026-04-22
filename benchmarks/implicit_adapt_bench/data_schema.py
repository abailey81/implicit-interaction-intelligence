"""Pydantic record schema for ImplicitAdaptBench.

The schema is deliberately narrow: every benchmark record pairs a compact
*behavioural window* (the implicit signal) with a prompt and a set of
reference style targets. A held-out ``human_preference_score`` is optional
and is only populated for records that have passed the preference-collection
protocol described in ``docs/research/implicit_adapt_bench.md``.

All models are frozen so that records hashable and immutable once built —
this prevents scoring code from accidentally mutating the gold reference.

Typical usage::

    from benchmarks.implicit_adapt_bench.data_schema import BenchmarkRecord

    record = BenchmarkRecord(
        record_id="dev-0001",
        behavioural_window=BehaviouralWindow(...),
        prompt="Tell me about your weekend.",
        target_archetype="low_load_warm",
        target_adaptation_vector=[0.1, 0.2, 0.3, 0.8, 0.3, 0.1, 0.6, 0.0],
        reference_style_label="warm_casual_short",
        reference_length_bucket="short",
        reference_formality_bucket="casual",
    )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Literal aliases
# ---------------------------------------------------------------------------

BenchmarkSplit = Literal["train", "dev", "test", "held_out_human"]
"""The four canonical benchmark splits.

``held_out_human`` is the preference-annotated micro-set; the other three are
synthetic.
"""

LengthBucket = Literal["short", "medium", "long"]
"""Discretised target response length.

* ``short``  — 1–2 sentences, <= 40 tokens.
* ``medium`` — 3–5 sentences, 40–120 tokens.
* ``long``   — 6+ sentences, > 120 tokens.
"""

FormalityBucket = Literal["casual", "neutral", "formal"]
"""Discretised target formality register."""


# ---------------------------------------------------------------------------
# BehaviouralWindow
# ---------------------------------------------------------------------------


class BehaviouralWindow(BaseModel):
    """The implicit-signal input for a single benchmark record.

    The window carries three complementary views of the same short sliding
    window of user behaviour:

    1. ``feature_vector`` — the canonical 32-dim
       :class:`~i3.interaction.types.InteractionFeatureVector` as a flat list
       (order matches ``i3.interaction.types.FEATURE_NAMES``).
    2. ``raw_keystroke_intervals_ms`` — raw inter-key intervals for the
       last typed message, in milliseconds. Provided so that submitters may
       build their own feature extractors.
    3. ``baseline_deviation_metrics`` — z-score deviations from the user's
       warm-up baseline, keyed by the feature name.

    Attributes:
        feature_vector: 32-dim feature vector as a list of floats.
        raw_keystroke_intervals_ms: Variable-length list of IKI in ms.
        baseline_deviation_metrics: Sparse dict of feature -> z-score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_vector: list[float] = Field(
        ...,
        description="Flat 32-dim InteractionFeatureVector, matching FEATURE_NAMES.",
    )
    raw_keystroke_intervals_ms: list[float] = Field(
        default_factory=list,
        description="Variable-length IKI sequence for the most recent message (ms).",
    )
    baseline_deviation_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Sparse z-score deviations from the user's baseline.",
    )

    @field_validator("feature_vector")
    @classmethod
    def _validate_feature_vector_length(cls, v: list[float]) -> list[float]:
        """Enforce the 32-dim feature-vector contract."""
        if len(v) != 32:
            raise ValueError(
                f"feature_vector must have exactly 32 elements, got {len(v)}."
            )
        for i, x in enumerate(v):
            if not isinstance(x, (int, float)):
                raise ValueError(
                    f"feature_vector[{i}] must be numeric, got {type(x).__name__}."
                )
        return [float(x) for x in v]

    @field_validator("raw_keystroke_intervals_ms")
    @classmethod
    def _validate_keystroke_intervals(cls, v: list[float]) -> list[float]:
        """All IKI values must be non-negative milliseconds."""
        for i, x in enumerate(v):
            if not isinstance(x, (int, float)):
                raise ValueError(
                    f"raw_keystroke_intervals_ms[{i}] must be numeric."
                )
            if x < 0.0:
                raise ValueError(
                    f"raw_keystroke_intervals_ms[{i}] must be >= 0 (got {x})."
                )
        return [float(x) for x in v]


# ---------------------------------------------------------------------------
# BenchmarkRecord
# ---------------------------------------------------------------------------


class BenchmarkRecord(BaseModel):
    """One gold benchmark record.

    A record is input (``behavioural_window`` + ``prompt``) plus
    archetype-derived style targets. ``human_preference_score`` is the only
    field that may be ``None`` — and is ``None`` for every record outside the
    ``held_out_human`` split.

    Attributes:
        record_id: Globally unique record id (split-prefixed).
        behavioural_window: The implicit-signal input.
        prompt: The user prompt to be responded to.
        target_archetype: Canonical archetype label (matches
            :func:`i3.eval.ablation_experiment.canonical_archetypes`).
        target_adaptation_vector: 8-dim target AdaptationVector as a list.
        reference_style_label: Human-readable style descriptor used by
            :func:`benchmarks.implicit_adapt_bench.metrics.style_match_score`.
        reference_length_bucket: Target length bucket.
        reference_formality_bucket: Target formality register.
        human_preference_score: Optional preference score in [0, 1].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., min_length=1)
    behavioural_window: BehaviouralWindow
    prompt: str = Field(..., min_length=1)
    target_archetype: str = Field(..., min_length=1)
    target_adaptation_vector: list[float] = Field(..., min_length=8, max_length=8)
    reference_style_label: str = Field(..., min_length=1)
    reference_length_bucket: LengthBucket
    reference_formality_bucket: FormalityBucket
    human_preference_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("target_adaptation_vector")
    @classmethod
    def _validate_adaptation_vector(cls, v: list[float]) -> list[float]:
        """AdaptationVector must be exactly 8 floats in a reasonable range."""
        if len(v) != 8:
            raise ValueError(
                f"target_adaptation_vector must have exactly 8 elements, got {len(v)}."
            )
        out: list[float] = []
        for i, x in enumerate(v):
            if not isinstance(x, (int, float)):
                raise ValueError(
                    f"target_adaptation_vector[{i}] must be numeric."
                )
            fx = float(x)
            # AdaptationVector values are nominally in [0, 1]; allow a small
            # slack for numerical jitter but reject obviously-bad values.
            if fx < -0.1 or fx > 1.1:
                raise ValueError(
                    f"target_adaptation_vector[{i}] must be in [0, 1] (got {fx})."
                )
            out.append(fx)
        return out


# ---------------------------------------------------------------------------
# BenchmarkSubmission
# ---------------------------------------------------------------------------


class BenchmarkSubmission(BaseModel):
    """One per-record submission row from a candidate system.

    Attributes:
        record_id: Must match a :class:`BenchmarkRecord.record_id` exactly.
        generated_text: The system's free-text response.
        method_name: Short identifier of the submitting method.
        runtime_ms_p50: Median forward latency for this record in ms.
        runtime_ms_p95: 95-percentile forward latency for this record in ms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., min_length=1)
    generated_text: str
    method_name: str = Field(..., min_length=1)
    runtime_ms_p50: float = Field(..., ge=0.0)
    runtime_ms_p95: float = Field(..., ge=0.0)

    @field_validator("runtime_ms_p95")
    @classmethod
    def _validate_p95_geq_p50(cls, v: float, info: object) -> float:
        """``runtime_ms_p95`` must be >= ``runtime_ms_p50`` (the usual invariant).

        The check uses ``info.data`` to access ``runtime_ms_p50`` once it has
        already passed its own validator.
        """
        # ``info`` is a ValidationInfo, but we avoid importing its type to keep
        # Pydantic 2 compatibility across minor versions.
        data = getattr(info, "data", {}) or {}
        p50 = data.get("runtime_ms_p50")
        if p50 is not None and v + 1e-9 < float(p50):
            raise ValueError(
                f"runtime_ms_p95 ({v}) must be >= runtime_ms_p50 ({p50})."
            )
        return v


# ---------------------------------------------------------------------------
# BenchmarkScore (output of scoring)
# ---------------------------------------------------------------------------


class BenchmarkScore(BaseModel):
    """The scoring result for a submission against a record set.

    Attributes:
        method_name: Identifier copied from the submissions.
        n_records: Number of records scored.
        per_metric: Mean metric values keyed by metric name.
        aggregate: Weighted aggregate scalar in [0, 1].
        notes: Freeform human-readable comments.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_name: str = Field(..., min_length=1)
    n_records: int = Field(..., ge=0)
    per_metric: dict[str, float] = Field(default_factory=dict)
    aggregate: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""
