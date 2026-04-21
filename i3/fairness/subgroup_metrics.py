"""Per-archetype fairness metrics over the I³ interaction diary.

For each of the 8 Epp/Vizer/Zimmermann user-state archetypes the brief
defines in ``training/generate_synthetic.py``, we compute:

1. The **mean AdaptationVector** over every diary exchange labelled with
   that archetype.
2. The **95% bootstrap CI** around each dimension of that mean.
3. The **disparity** across archetypes for each adaptation dimension
   — defined as ``max - min`` over subgroup means.

The resulting :class:`FairnessReport` flags dimensions whose disparity
exceeds a configurable threshold.  The canonical threshold (0.15 on the
[0, 1] adaptation scale) is taken from the I³ brief's conservative bias
tolerance (§11, §13 testing discipline); callers can override.

References
----------
* Epp, C., Lippold, M., Mandryk, R. L. (2011).  *Identifying emotional
  states using keystroke dynamics.*  CHI.
* Vizer, L. M., Zhou, L., Sears, A. (2009).  *Automated stress detection
  using keystroke and linguistic features.*  International Journal of
  Human-Computer Studies 67(10).
* Zimmermann, P., Guttormsen, S., Danuser, B., Gomez, P. (2014).
  *Affective computing — a rationale for measuring mood with mouse and
  keyboard.*  International Journal of Occupational Safety and Ergonomics.
* Barocas, S., Hardt, M., Narayanan, A. (2019). *Fairness and Machine
  Learning.*  fairmlbook.org.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from i3.fairness.confidence_intervals import bootstrap_mean_ci

logger = logging.getLogger(__name__)


# Canonical adaptation dimension order — mirrors
# :meth:`AdaptationVector.to_tensor` layout.
_ADAPTATION_DIMS: list[str] = [
    "cognitive_load",
    "style_formality",
    "style_verbosity",
    "style_emotionality",
    "style_directness",
    "emotional_tone",
    "accessibility",
    "reserved",
]

# The 8 canonical archetypes from `training/generate_synthetic.py`.  Kept as a
# constant so callers do not need to depend on the training module.
EPP_VIZER_ZIMMERMANN_ARCHETYPES: list[str] = [
    "energetic_engaged",
    "tired_disengaging",
    "stressed_urgent",
    "relaxed_conversational",
    "focused_deep",
    "motor_difficulty",
    "distracted_multitasking",
    "formal_professional",
]


# ---------------------------------------------------------------------------
# Duck-typed diary protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AsyncDiary(Protocol):
    """Minimal duck-typed interface we need from the I³ diary.

    The canonical :class:`~i3.diary.store.DiaryStore` satisfies this
    automatically — it exposes ``get_user_sessions`` and
    ``get_session_exchanges``.
    """

    async def get_user_sessions(self, user_id: str, limit: int = ...) -> list[dict]: ...

    async def get_session_exchanges(self, session_id: str) -> list[dict]: ...


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

class ArchetypeMetrics(BaseModel):
    """Fairness metrics for a single archetype.

    Attributes:
        archetype: Human-readable label.
        n_exchanges: How many diary exchanges contributed to the summary.
        mean_adaptation: Eight-element array of per-dimension means (keyed
            by adaptation dimension name).
        ci_lower: Eight-element dict of lower CI bounds.
        ci_upper: Eight-element dict of upper CI bounds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    archetype: str
    n_exchanges: int
    mean_adaptation: dict[str, float] = Field(default_factory=dict)
    ci_lower: dict[str, float] = Field(default_factory=dict)
    ci_upper: dict[str, float] = Field(default_factory=dict)


class FairnessReport(BaseModel):
    """Aggregate fairness report across all archetypes.

    Attributes:
        per_archetype: Metrics for each archetype in
            :data:`EPP_VIZER_ZIMMERMANN_ARCHETYPES` order.
        disparity: For each adaptation dimension, the ``max - min`` of the
            archetype means.
        flagged_dimensions: Dimensions whose disparity exceeds the
            configured threshold.
        threshold: The disparity threshold used.
        total_exchanges: Total diary exchanges that contributed to the
            report.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    per_archetype: list[ArchetypeMetrics]
    disparity: dict[str, float]
    flagged_dimensions: list[str]
    threshold: float
    total_exchanges: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _extract_adaptation_vector(exchange: dict) -> np.ndarray | None:
    """Pull an 8-dim ``float32`` array from a diary exchange row.

    The diary stores ``adaptation_vector`` as JSON text.  The exchange dict
    is expected to carry either the parsed dict (the store returns parsed
    JSON for ``topics`` but not for the adaptation vector in all paths) or
    the original string; we handle both.
    """
    raw = exchange.get("adaptation_vector")
    if raw is None:
        return None
    import json as _json

    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except _json.JSONDecodeError:
            logger.warning("Unparseable adaptation_vector in diary row; skipped")
            return None
    if not isinstance(raw, dict):
        return None
    style = raw.get("style_mirror") or {}
    try:
        return np.array(
            [
                float(raw.get("cognitive_load", 0.5)),
                float(style.get("formality", 0.5)),
                float(style.get("verbosity", 0.5)),
                float(style.get("emotionality", 0.5)),
                float(style.get("directness", 0.5)),
                float(raw.get("emotional_tone", 0.5)),
                float(raw.get("accessibility", 0.0)),
                0.0,
            ],
            dtype=np.float32,
        )
    except (TypeError, ValueError):
        return None


async def _collect_adaptations_by_archetype(
    diary: AsyncDiary,
    archetypes: "Sequence[str]",
    archetype_resolver: Optional[Callable[[dict, dict], Optional[str]]] = None,
    user_id: str | None = None,
    max_sessions: int = 500,
) -> dict[str, list[np.ndarray]]:
    """Walk the diary once; bucket adaptation vectors by archetype label.

    The archetype for an exchange is derived from ``exchange["archetype"]``
    if the field exists; otherwise falls back to ``session["dominant_emotion"]``.
    A caller-supplied ``archetype_resolver`` takes precedence.
    """
    out: dict[str, list[np.ndarray]] = {a: [] for a in archetypes}
    if user_id is None:
        logger.warning("No user_id supplied; fairness report will be empty")
        return out

    sessions = await diary.get_user_sessions(user_id=user_id, limit=max_sessions)
    for sess in sessions:
        sess_id = sess.get("session_id")
        if not sess_id:
            continue
        exchanges = await diary.get_session_exchanges(sess_id)
        for ex in exchanges:
            vec = _extract_adaptation_vector(ex)
            if vec is None:
                continue
            if archetype_resolver is not None:
                label = archetype_resolver(ex, sess)
            else:
                label = ex.get("archetype") or sess.get("dominant_emotion")
            if label in out:
                out[label].append(vec)
    return out


async def compute_per_archetype_adaptation_bias(
    diary: AsyncDiary,
    archetypes: list[str] | None = None,
    user_id: str | None = None,
    *,
    disparity_threshold: float = 0.15,
    ci_level: float = 0.95,
    num_resamples: int = 2000,
    seed: int | None = None,
    archetype_resolver: Optional[Callable[[dict, dict], Optional[str]]] = None,
    max_sessions: int = 500,
) -> FairnessReport:
    """Compute the per-archetype adaptation fairness report.

    For each archetype in ``archetypes``, compute the mean adaptation
    vector and a 95% bootstrap CI on each dimension.  Aggregate into a
    :class:`FairnessReport`.

    Args:
        diary: An :class:`AsyncDiary`-compatible object.  The canonical
            :class:`~i3.diary.store.DiaryStore` works out of the box.
        archetypes: List of archetype labels.  Defaults to the 8
            Epp/Vizer/Zimmermann names.
        user_id: User identifier to filter the diary by.
        disparity_threshold: Dimensions whose ``max - min`` across
            archetypes exceeds this are flagged as biased.
        ci_level: Confidence level for bootstrap intervals.
        num_resamples: Number of bootstrap resamples per dimension.
        seed: Optional RNG seed.
        archetype_resolver: Optional callable
            ``(exchange, session) -> archetype_label``.  Overrides the
            default extraction from ``exchange["archetype"]`` /
            ``session["dominant_emotion"]``.
        max_sessions: Cap on sessions pulled.  Default 500.

    Returns:
        A :class:`FairnessReport` with per-archetype metrics plus the
        cross-archetype disparity summary.
    """
    if archetypes is None:
        archetypes = list(EPP_VIZER_ZIMMERMANN_ARCHETYPES)

    buckets = await _collect_adaptations_by_archetype(
        diary=diary,
        archetypes=archetypes,
        archetype_resolver=archetype_resolver,
        user_id=user_id,
        max_sessions=max_sessions,
    )

    per_archetype_list: list[ArchetypeMetrics] = []
    means_by_dim: dict[str, list[float]] = {d: [] for d in _ADAPTATION_DIMS}
    total_exchanges = 0

    for archetype in archetypes:
        vectors = buckets.get(archetype, [])
        n = len(vectors)
        total_exchanges += n
        if n == 0:
            per_archetype_list.append(
                ArchetypeMetrics(archetype=archetype, n_exchanges=0)
            )
            continue

        stacked = np.stack(vectors, axis=0)
        means: dict[str, float] = {}
        ci_lower: dict[str, float] = {}
        ci_upper: dict[str, float] = {}
        for i, dim in enumerate(_ADAPTATION_DIMS):
            col = stacked[:, i]
            result = bootstrap_mean_ci(
                col.astype(np.float64),
                num_resamples=num_resamples,
                level=ci_level,
                seed=seed,
            )
            means[dim] = result.point_estimate
            ci_lower[dim] = result.lower
            ci_upper[dim] = result.upper
            means_by_dim[dim].append(result.point_estimate)

        per_archetype_list.append(
            ArchetypeMetrics(
                archetype=archetype,
                n_exchanges=n,
                mean_adaptation=means,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    # -- Disparity --------------------------------------------------------
    disparity: dict[str, float] = {}
    flagged: list[str] = []
    for dim, vals in means_by_dim.items():
        if len(vals) < 2:
            disparity[dim] = 0.0
            continue
        d = float(max(vals) - min(vals))
        disparity[dim] = d
        if d > disparity_threshold:
            flagged.append(dim)

    return FairnessReport(
        per_archetype=per_archetype_list,
        disparity=disparity,
        flagged_dimensions=flagged,
        threshold=disparity_threshold,
        total_exchanges=total_exchanges,
    )
