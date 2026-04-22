"""Closed-loop evaluator for persona-driven simulation harness.

Given a fully initialised :class:`~i3.pipeline.engine.Pipeline` and a list
of :class:`~i3.eval.simulation.personas.HCIPersona`, this module drives a
multi-persona, multi-session simulation through the pipeline and scores:

* **Persona recovery rate** -- 1-NN cluster recovery in the inferred
  adaptation-vector space. For each session's final message, we check
  whether the persona whose ground-truth ``expected_adaptation`` is
  closest (L2) to the inferred :class:`AdaptationVector` matches the
  session's true persona.
* **Per-message L2 adaptation error** -- :math:`\\lVert
  \\hat{a}_t - a^{*}_t \\rVert_2` for every message, aggregated per
  persona.
* **Convergence speed** -- index of the first message in a session at
  which the L2 error drops below ``adapt_converged_threshold``. Sessions
  that never converge contribute ``None``.
* **Persona-conditional router bias** -- fraction of ``local_slm``
  routing decisions per persona. The accessibility personas
  (``motor_impaired_user`` and ``low_vision_user``) are compared
  against the baseline fresh-user local-decision rate as a sanity check
  that the accessibility prior is translating to more privacy-preserving
  (local) routing.

All metrics are reported with 95 % bootstrap confidence intervals via
:func:`i3.eval.ablation_statistics.bootstrap_ci`.

The evaluator is fully deterministic given the ``(pipeline, personas,
seed)`` triple.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector
from i3.eval.ablation_statistics import bootstrap_ci
from i3.eval.simulation.personas import ALL_PERSONAS, HCIPersona
from i3.eval.simulation.user_simulator import SimulatedMessage, UserSimulator
from i3.pipeline.types import PipelineInput, PipelineOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: L2 distance between two AdaptationVectors
# ---------------------------------------------------------------------------


def _adaptation_vector_l2(a: AdaptationVector, b: AdaptationVector) -> float:
    """Compute Euclidean distance between two :class:`AdaptationVector`s.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Non-negative float distance between the 7 meaningful dimensions
        (the reserved 8th dimension is always 0 and is skipped).
    """
    va = a.to_tensor().numpy().astype(np.float64)[:7]
    vb = b.to_tensor().numpy().astype(np.float64)[:7]
    return float(np.linalg.norm(va - vb))


def _adaptation_from_pipeline_output(
    output: PipelineOutput,
) -> AdaptationVector:
    """Reconstruct an :class:`AdaptationVector` from a pipeline output.

    The pipeline returns ``adaptation`` as a nested dict; this helper
    hydrates it back into a typed vector so the evaluator can run
    vector-space arithmetic against persona ground truth.

    Args:
        output: A :class:`PipelineOutput` instance.

    Returns:
        The reconstructed :class:`AdaptationVector`. Any missing or
        malformed fields fall back to :meth:`AdaptationVector.default`
        so the evaluator never crashes mid-run.
    """
    payload = output.adaptation
    if not isinstance(payload, dict):
        return AdaptationVector.default()
    try:
        return AdaptationVector.from_dict(payload)
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "closed-loop evaluator: could not parse adaptation dict; "
            "falling back to default vector."
        )
        return AdaptationVector.default()


# ---------------------------------------------------------------------------
# Per-message record
# ---------------------------------------------------------------------------


class MessageRecord(BaseModel):
    """One row of the closed-loop evaluation dataframe.

    Attributes:
        persona_name: Ground-truth persona name.
        session_index: Zero-based session index within this persona.
        message_index: Zero-based message index within the session.
        l2_error: L2 distance between inferred and ground-truth
            adaptation vectors.
        route_chosen: Router decision (``"local_slm"`` or ``"cloud_llm"``
            or ``"error_fallback"``).
        inferred: Inferred 7-dimensional adaptation vector (reserved
            dimension dropped). Stored as a list for JSON serialisation.
        ground_truth: Ground-truth 7-dimensional adaptation vector.
        embedding_2d: 2-D projection of the inferred user-state
            embedding emitted by the pipeline.
    """

    model_config = ConfigDict(frozen=True)

    persona_name: str
    session_index: int
    message_index: int
    l2_error: float
    route_chosen: str
    inferred: list[float]
    ground_truth: list[float]
    embedding_2d: tuple[float, float]


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


class ClosedLoopResult(BaseModel):
    """Aggregated closed-loop evaluation result.

    All fields are JSON-serialisable. ``persona_confusion_matrix`` is
    represented as a list of lists (row = true persona, column =
    inferred persona at the session-final message) to preserve ordering
    semantics across languages.

    Attributes:
        persona_order: The persona names in row/column order of the
            confusion matrix.
        per_persona_recovery_rate: 1-NN final-message recovery rate, one
            entry per persona.
        per_persona_recovery_ci: 95 % bootstrap CIs for the recovery
            rates.
        per_persona_adaptation_error: Mean L2 adaptation error per
            persona.
        per_persona_adaptation_error_ci: 95 % bootstrap CIs for the
            adaptation errors.
        per_persona_error_by_message: For each persona, the mean L2
            error at each message index (length ==
            ``n_messages_per_session``).
        convergence_speeds: For each persona, mean message index at
            which error first drops below
            ``adapt_converged_threshold``. ``None`` if no session
            converged.
        persona_confusion_matrix: Row = true persona, column =
            nearest-neighbour persona at the final message. Rows sum to
            the number of sessions per persona.
        aggregate_recovery_rate: Mean recovery rate across all personas.
        aggregate_recovery_rate_ci: 95 % bootstrap CI of the aggregate.
        aggregate_adaptation_error: Mean L2 error across all messages.
        aggregate_adaptation_error_ci: 95 % bootstrap CI of the mean.
        router_bias: Dict of router-bias diagnostics (see
            :meth:`ClosedLoopEvaluator._compute_router_bias`).
        per_message_records: Full flat list of all
            :class:`MessageRecord` entries. Useful for offline analysis.
        n_sessions_per_persona: Echo of the constructor argument.
        n_messages_per_session: Echo of the constructor argument.
        adapt_converged_threshold: Echo of the constructor argument.
        wall_clock_seconds: Total evaluation wall-clock time.
    """

    model_config = ConfigDict(frozen=True)

    persona_order: list[str] = Field(default_factory=list)

    per_persona_recovery_rate: dict[str, float] = Field(default_factory=dict)
    per_persona_recovery_ci: dict[str, tuple[float, float]] = Field(
        default_factory=dict
    )

    per_persona_adaptation_error: dict[str, float] = Field(default_factory=dict)
    per_persona_adaptation_error_ci: dict[str, tuple[float, float]] = Field(
        default_factory=dict
    )

    per_persona_error_by_message: dict[str, list[float]] = Field(
        default_factory=dict
    )

    convergence_speeds: dict[str, Optional[float]] = Field(default_factory=dict)

    persona_confusion_matrix: list[list[int]] = Field(default_factory=list)

    aggregate_recovery_rate: float = 0.0
    aggregate_recovery_rate_ci: tuple[float, float] = (0.0, 0.0)

    aggregate_adaptation_error: float = 0.0
    aggregate_adaptation_error_ci: tuple[float, float] = (0.0, 0.0)

    router_bias: dict[str, float] = Field(default_factory=dict)

    per_message_records: list[MessageRecord] = Field(default_factory=list)

    n_sessions_per_persona: int = 0
    n_messages_per_session: int = 0
    adapt_converged_threshold: float = 0.0

    wall_clock_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ClosedLoopEvaluator:
    """Drive a :class:`Pipeline` through a persona-indexed simulation.

    The evaluator is intentionally agnostic about what the pipeline
    actually contains -- any object that exposes an
    ``async process_message(PipelineInput) -> PipelineOutput`` method and
    (optionally) ``async start_session(user_id) -> str`` satisfies the
    duck-typed contract. This enables unit tests to inject a minimal
    mock pipeline without loading the real SLM stack.

    Args:
        pipeline: A :class:`~i3.pipeline.engine.Pipeline` (or mock) with
            an initialised state.
        personas: List of :class:`HCIPersona` to evaluate. Defaults to
            :data:`ALL_PERSONAS`.
        n_sessions_per_persona: Number of sessions to simulate per
            persona. Must be >= 1.
        n_messages_per_session: Messages per session. Must be >= 1.
        adapt_converged_threshold: L2-error threshold below which the
            inferred adaptation vector is considered "converged".
        seed: Deterministic seed used to drive every
            :class:`UserSimulator` instance.
        bootstrap_rng_seed: Deterministic seed for the bootstrap RNG so
            confidence intervals are reproducible.

    Example::

        evaluator = ClosedLoopEvaluator(
            pipeline=pipeline,
            n_sessions_per_persona=5,
            n_messages_per_session=15,
        )
        result = await evaluator.run()
        print(result.aggregate_recovery_rate)
    """

    def __init__(
        self,
        pipeline: Any,
        personas: Iterable[HCIPersona] = ALL_PERSONAS,
        n_sessions_per_persona: int = 5,
        n_messages_per_session: int = 15,
        adapt_converged_threshold: float = 0.3,
        seed: int = 42,
        bootstrap_rng_seed: int = 4242,
    ) -> None:
        if n_sessions_per_persona < 1:
            raise ValueError(
                f"n_sessions_per_persona must be >= 1, "
                f"got {n_sessions_per_persona}"
            )
        if n_messages_per_session < 1:
            raise ValueError(
                f"n_messages_per_session must be >= 1, "
                f"got {n_messages_per_session}"
            )
        if not (0.0 < adapt_converged_threshold):
            raise ValueError(
                f"adapt_converged_threshold must be > 0, "
                f"got {adapt_converged_threshold}"
            )

        self.pipeline = pipeline
        self.personas: list[HCIPersona] = list(personas)
        if not self.personas:
            raise ValueError("personas list must be non-empty")
        self.n_sessions_per_persona = int(n_sessions_per_persona)
        self.n_messages_per_session = int(n_messages_per_session)
        self.adapt_converged_threshold = float(adapt_converged_threshold)
        self.seed = int(seed)
        self._bootstrap_rng = np.random.default_rng(int(bootstrap_rng_seed))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> ClosedLoopResult:
        """Execute the full closed-loop evaluation.

        Returns:
            A populated :class:`ClosedLoopResult` with bootstrapped 95 %
            confidence intervals for every headline metric.
        """
        t0 = time.perf_counter()

        persona_order = [p.name for p in self.personas]
        persona_index = {name: i for i, name in enumerate(persona_order)}

        # -- data structures -------------------------------------------------
        records: list[MessageRecord] = []
        confusion = [
            [0 for _ in persona_order] for _ in persona_order
        ]
        # For bootstrap aggregation, per-persona lists of per-session
        # metrics (recovery is 0/1 per session, error is mean per message).
        per_persona_sessions_recovery: dict[str, list[float]] = {
            name: [] for name in persona_order
        }
        per_persona_session_errors: dict[str, list[float]] = {
            name: [] for name in persona_order
        }
        per_persona_message_errors: dict[str, list[float]] = {
            name: [] for name in persona_order
        }
        per_persona_convergence: dict[str, list[Optional[int]]] = {
            name: [] for name in persona_order
        }
        # For router bias: list of 1/0 per message (1 => local_slm).
        per_persona_route_local: dict[str, list[int]] = {
            name: [] for name in persona_order
        }
        # For per-message-index mean error traces.
        per_persona_error_by_msg_sum: dict[str, list[float]] = {
            name: [0.0] * self.n_messages_per_session for name in persona_order
        }
        per_persona_error_by_msg_count: dict[str, list[int]] = {
            name: [0] * self.n_messages_per_session for name in persona_order
        }

        # -- main simulation loop -------------------------------------------
        for persona in self.personas:
            logger.info(
                "Running persona=%s (%d sessions × %d messages)",
                persona.name,
                self.n_sessions_per_persona,
                self.n_messages_per_session,
            )
            for session_idx in range(self.n_sessions_per_persona):
                session_seed = self.seed + 1000 * session_idx
                simulator = UserSimulator(persona, seed=session_seed)
                messages = simulator.run_session(
                    n_messages=self.n_messages_per_session
                )

                final_inferred: AdaptationVector | None = None
                session_errors: list[float] = []
                converged_at: Optional[int] = None
                user_id = f"persona:{persona.name}:session:{session_idx}"
                session_id = await self._start_session(user_id)

                for msg in messages:
                    pipeline_output = await self._process_message(
                        user_id=user_id,
                        session_id=session_id,
                        message=msg,
                    )
                    inferred = _adaptation_from_pipeline_output(pipeline_output)
                    final_inferred = inferred
                    err = _adaptation_vector_l2(
                        inferred, persona.expected_adaptation
                    )
                    session_errors.append(err)
                    per_persona_message_errors[persona.name].append(err)
                    per_persona_error_by_msg_sum[persona.name][
                        msg.message_index
                    ] += err
                    per_persona_error_by_msg_count[persona.name][
                        msg.message_index
                    ] += 1
                    per_persona_route_local[persona.name].append(
                        1 if pipeline_output.route_chosen == "local_slm" else 0
                    )
                    if converged_at is None and err < self.adapt_converged_threshold:
                        converged_at = msg.message_index

                    inferred_vec = inferred.to_tensor().numpy().astype(
                        float
                    )[:7].tolist()
                    ground_vec = persona.expected_adaptation.to_tensor().numpy().astype(
                        float
                    )[:7].tolist()
                    records.append(
                        MessageRecord(
                            persona_name=persona.name,
                            session_index=session_idx,
                            message_index=msg.message_index,
                            l2_error=err,
                            route_chosen=pipeline_output.route_chosen,
                            inferred=inferred_vec,
                            ground_truth=ground_vec,
                            embedding_2d=(
                                float(pipeline_output.user_state_embedding_2d[0]),
                                float(pipeline_output.user_state_embedding_2d[1]),
                            ),
                        )
                    )

                # -- session summary --
                per_persona_session_errors[persona.name].append(
                    float(np.mean(session_errors)) if session_errors else 0.0
                )
                per_persona_convergence[persona.name].append(converged_at)

                # 1-NN persona recovery: which persona's expected_adaptation
                # is closest to final inferred?
                if final_inferred is None:
                    recovered_name = persona.name  # degenerate edge case
                else:
                    recovered_name = self._nearest_persona(final_inferred)
                recovered_idx = persona_index[recovered_name]
                true_idx = persona_index[persona.name]
                confusion[true_idx][recovered_idx] += 1
                per_persona_sessions_recovery[persona.name].append(
                    1.0 if recovered_name == persona.name else 0.0
                )

        # -- aggregate metrics ----------------------------------------------
        (
            per_persona_recovery_rate,
            per_persona_recovery_ci,
        ) = self._summarise_per_persona(
            per_persona_sessions_recovery
        )
        (
            per_persona_adaptation_error,
            per_persona_adaptation_error_ci,
        ) = self._summarise_per_persona(
            per_persona_message_errors
        )

        per_persona_error_by_message: dict[str, list[float]] = {}
        for name in persona_order:
            trace: list[float] = []
            sums = per_persona_error_by_msg_sum[name]
            counts = per_persona_error_by_msg_count[name]
            for i in range(self.n_messages_per_session):
                trace.append(
                    float(sums[i] / counts[i]) if counts[i] > 0 else 0.0
                )
            per_persona_error_by_message[name] = trace

        convergence_speeds: dict[str, Optional[float]] = {}
        for name in persona_order:
            converged_indices = [
                float(v) for v in per_persona_convergence[name] if v is not None
            ]
            convergence_speeds[name] = (
                float(np.mean(converged_indices))
                if converged_indices
                else None
            )

        # aggregate recovery & error
        all_recovery = [
            v
            for name in persona_order
            for v in per_persona_sessions_recovery[name]
        ]
        all_errors = [
            v
            for name in persona_order
            for v in per_persona_message_errors[name]
        ]
        aggregate_recovery_rate = (
            float(np.mean(all_recovery)) if all_recovery else 0.0
        )
        aggregate_recovery_rate_ci = (
            bootstrap_ci(np.asarray(all_recovery), rng=self._bootstrap_rng)
            if all_recovery
            else (0.0, 0.0)
        )
        aggregate_adaptation_error = (
            float(np.mean(all_errors)) if all_errors else 0.0
        )
        aggregate_adaptation_error_ci = (
            bootstrap_ci(np.asarray(all_errors), rng=self._bootstrap_rng)
            if all_errors
            else (0.0, 0.0)
        )

        router_bias = self._compute_router_bias(per_persona_route_local)

        wall = time.perf_counter() - t0
        return ClosedLoopResult(
            persona_order=persona_order,
            per_persona_recovery_rate=per_persona_recovery_rate,
            per_persona_recovery_ci=per_persona_recovery_ci,
            per_persona_adaptation_error=per_persona_adaptation_error,
            per_persona_adaptation_error_ci=per_persona_adaptation_error_ci,
            per_persona_error_by_message=per_persona_error_by_message,
            convergence_speeds=convergence_speeds,
            persona_confusion_matrix=confusion,
            aggregate_recovery_rate=aggregate_recovery_rate,
            aggregate_recovery_rate_ci=aggregate_recovery_rate_ci,
            aggregate_adaptation_error=aggregate_adaptation_error,
            aggregate_adaptation_error_ci=aggregate_adaptation_error_ci,
            router_bias=router_bias,
            per_message_records=records,
            n_sessions_per_persona=self.n_sessions_per_persona,
            n_messages_per_session=self.n_messages_per_session,
            adapt_converged_threshold=self.adapt_converged_threshold,
            wall_clock_seconds=float(wall),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _start_session(self, user_id: str) -> str:
        """Open a pipeline session when the pipeline supports it.

        Falls back to a synthetic session id if the pipeline does not
        expose ``start_session`` (e.g. a mock pipeline in tests).
        """
        starter = getattr(self.pipeline, "start_session", None)
        if starter is None:
            return f"{user_id}:synthetic"
        try:
            return await starter(user_id)
        except (AttributeError, RuntimeError, TypeError):
            logger.warning(
                "closed-loop evaluator: pipeline.start_session failed for %s; "
                "using synthetic session id.",
                user_id,
            )
            return f"{user_id}:synthetic"

    async def _process_message(
        self,
        user_id: str,
        session_id: str,
        message: SimulatedMessage,
    ) -> PipelineOutput:
        """Dispatch one :class:`SimulatedMessage` through the pipeline."""
        ks = message.keystroke_intervals_ms
        composition_ms = float(sum(ks))
        pause_before_send_ms = float(ks[-1]) if ks else 0.0
        pipeline_input = PipelineInput(
            user_id=user_id,
            session_id=session_id,
            message_text=message.text,
            timestamp=float(message.timestamp),
            composition_time_ms=composition_ms,
            edit_count=message.edit_count,
            pause_before_send_ms=pause_before_send_ms,
            keystroke_timings=list(ks),
        )
        output = await self.pipeline.process_message(pipeline_input)
        return output

    def _nearest_persona(self, inferred: AdaptationVector) -> str:
        """Return the persona whose ``expected_adaptation`` is closest.

        Ties are broken by the persona's position in ``self.personas``
        (earliest wins) so the comparator is strictly deterministic.
        """
        best_name = self.personas[0].name
        best_d = float("inf")
        for persona in self.personas:
            d = _adaptation_vector_l2(inferred, persona.expected_adaptation)
            if d < best_d:
                best_d = d
                best_name = persona.name
        return best_name

    def _summarise_per_persona(
        self, values: dict[str, list[float]]
    ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
        """Return ``(mean, 95% bootstrap CI)`` per persona."""
        means: dict[str, float] = {}
        cis: dict[str, tuple[float, float]] = {}
        for name, xs in values.items():
            if not xs:
                means[name] = 0.0
                cis[name] = (0.0, 0.0)
                continue
            arr = np.asarray(xs, dtype=np.float64)
            means[name] = float(arr.mean())
            cis[name] = bootstrap_ci(arr, rng=self._bootstrap_rng)
        return means, cis

    def _compute_router_bias(
        self, per_persona_route_local: dict[str, list[int]]
    ) -> dict[str, float]:
        """Summarise the persona-conditional local-routing bias.

        Reports the local-routing fraction per persona and the
        accessibility-vs-baseline delta used as a sanity check that the
        accessibility prior is translating into privacy-preserving
        routing.

        Args:
            per_persona_route_local: Map persona_name ->
                per-message list of 0/1 where 1 == local_slm.

        Returns:
            A dict with keys:

            * ``local_rate_<persona>``: Local-routing rate per persona.
            * ``accessibility_local_rate``: Mean local rate across
              motor_impaired_user and low_vision_user (if present).
            * ``baseline_local_rate``: Local rate for fresh_user (if
              present) else mean across the non-accessibility personas.
            * ``accessibility_vs_baseline_delta``: Accessibility rate
              minus baseline rate (positive => accessibility personas
              are more often routed locally).
        """
        bias: dict[str, float] = {}
        for name, bits in per_persona_route_local.items():
            bias[f"local_rate_{name}"] = (
                float(np.mean(bits)) if bits else 0.0
            )

        accessibility_names = {"motor_impaired_user", "low_vision_user"}
        baseline_name = "fresh_user"

        acc_values = [
            bias[f"local_rate_{n}"]
            for n in accessibility_names
            if f"local_rate_{n}" in bias
        ]
        if acc_values:
            bias["accessibility_local_rate"] = float(np.mean(acc_values))

        if f"local_rate_{baseline_name}" in bias:
            baseline_rate = bias[f"local_rate_{baseline_name}"]
        else:
            non_acc = [
                v
                for key, v in bias.items()
                if key.startswith("local_rate_")
                and not any(
                    key == f"local_rate_{n}" for n in accessibility_names
                )
            ]
            baseline_rate = float(np.mean(non_acc)) if non_acc else 0.0
        bias["baseline_local_rate"] = float(baseline_rate)

        if "accessibility_local_rate" in bias:
            bias["accessibility_vs_baseline_delta"] = float(
                bias["accessibility_local_rate"] - baseline_rate
            )
        return bias


__all__ = [
    "ClosedLoopEvaluator",
    "ClosedLoopResult",
    "MessageRecord",
]
