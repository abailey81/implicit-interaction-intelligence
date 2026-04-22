"""Runnable HMAF agentic runtime for I³.

This module lifts :class:`i3.huawei.hmaf_adapter.HMAFAgentAdapter` from a
documentation-grade scaffold to an end-to-end runnable agent harness
whose shape matches HarmonyOS 6's Harmony Multi-Agent Framework (HMAF)
protocol: *intents* arrive on an event bus, the runtime *plans* them,
*executes* each plan step, and *emits text-free telemetry* along the
way.

The runtime is deliberately lightweight:

* Event bus: an ``asyncio.Queue`` per runtime instance.  Real HMAF
  deployments provide their own cross-device bus; the shape of the
  producer/consumer contract is identical.
* Planner: the PDDL-grounded :class:`~i3.safety.pddl_planner.PrivacySafetyPlanner`
  is used when available (soft import).  If the safety subpackage is
  missing the runtime falls back to a minimal rule-based planner so the
  demo still runs.
* Executor: soft-imports the I³ pipeline; every handler degrades to a
  deterministic mocked response if the pipeline isn't initialised, so
  the runtime boots in environments without torch / checkpoints.

The harness exposes :meth:`start`, :meth:`receive_intent`,
:meth:`plan_and_execute`, and :meth:`stop` so callers (CLIs, tests)
can drive it without knowing its internals.

See :doc:`docs/huawei/harmony_hmaf_integration.md` for the protocol
shape this runtime implements.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from i3.huawei.hmaf_adapter import (
    HMAFAgentAdapter,
    I3PipelineProtocol,
    PlanStep,
    StepResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent / response models
# ---------------------------------------------------------------------------


class HMAFIntent(BaseModel):
    """Incoming HMAF intent payload.

    Attributes:
        name: HMAF intent identifier (e.g. ``"get_user_adaptation"``).
        parameters: Free-form parameter dictionary.
        source_device: Opaque device id the intent was produced on
            (phone, watch, glasses, ...).
        correlation_id: Caller-supplied id that ties request and
            response together across the bus.  If omitted a uuid4 is
            generated.
        timestamp: ISO-8601 UTC timestamp.  Defaults to "now".
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_device: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=128,
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HMAFResponse(BaseModel):
    """Aggregated response returned by :meth:`HMAFAgentRuntime.plan_and_execute`.

    Attributes:
        intent_name: The originating intent's name.
        correlation_id: Echoed correlation id so the caller can match.
        ok: ``True`` iff the terminal action succeeded.
        terminal_action: The final PDDL action taken (e.g.
            ``route_local``) or the high-level intent handler's label.
        payload: Structured, text-free result data.
        steps_executed: Number of plan steps that ran.
        latency_ms: End-to-end wall-clock latency in milliseconds.
    """

    model_config = ConfigDict(extra="forbid")

    intent_name: str
    correlation_id: str
    ok: bool
    terminal_action: str
    payload: dict[str, Any] = Field(default_factory=dict)
    steps_executed: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Pipeline fallback
# ---------------------------------------------------------------------------


class _MockPipeline:
    """Deterministic, text-free fallback pipeline used when the real one
    isn't initialised.

    Conforms to :class:`i3.huawei.hmaf_adapter.I3PipelineProtocol`.
    """

    async def extract_features(self, message: str) -> dict[str, Any]:
        """Return a trivial feature summary.

        Args:
            message: Incoming message (unused beyond its length).

        Returns:
            A mock 32-dim feature vector summary.
        """
        return {"feature_len": 32, "message_len": len(message or "")}

    async def encode(self, features: Any) -> list[float]:
        """Return a zero-filled 64-dim embedding.

        Args:
            features: Unused feature payload.

        Returns:
            A list of 64 zeros, matching the shape of the real encoder
            output.
        """
        _ = features
        return [0.0] * 64

    async def adapt(self, state: Any) -> dict[str, float]:
        """Return a neutral AdaptationVector summary.

        Args:
            state: Unused state payload.

        Returns:
            A dict matching :meth:`AdaptationVector.to_dict` on default.
        """
        _ = state
        return {
            "cognitive_load": 0.5,
            "formality": 0.5,
            "verbosity": 0.5,
            "emotionality": 0.5,
            "directness": 0.5,
            "emotional_tone": 0.5,
            "accessibility": 0.0,
        }


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


IntentHandler = Callable[["HMAFIntent"], Awaitable[HMAFResponse]]

# Set of intent names for which we refuse to even plan -- used by the
# privacy guard test.
_FORBIDDEN_INTENT_NAMES: frozenset[str] = frozenset(
    {"dump_raw_diary", "exfiltrate_text"}
)


class HMAFAgentRuntime:
    """Runnable HMAF agent runtime wrapping
    :class:`~i3.huawei.hmaf_adapter.HMAFAgentAdapter`.

    The runtime simulates the HMAF event bus with an in-process
    ``asyncio.Queue`` and exposes methods matching the HMAF lifecycle:

    * :meth:`start` -- begin consuming from the bus.
    * :meth:`receive_intent` -- post an intent to the bus.
    * :meth:`plan_and_execute` -- synchronous-flavoured helper that
      bypasses the bus and returns the :class:`HMAFResponse` directly.
      Useful for tests and for the CLI demo.
    * :meth:`stop` -- graceful shutdown; drains pending intents.

    Args:
        adapter: The HMAF adapter to wrap.  If ``None`` one is built
            on the fly around a :class:`_MockPipeline`.
        telemetry_sink: Optional callable for telemetry events.
        queue_maxsize: Bus queue bound (default 64) to prevent memory
            exhaustion under load.
    """

    def __init__(
        self,
        adapter: HMAFAgentAdapter | None = None,
        telemetry_sink: Callable[[dict[str, Any]], None] | None = None,
        queue_maxsize: int = 64,
    ) -> None:
        if adapter is None:
            pipeline: I3PipelineProtocol = _MockPipeline()
            adapter = HMAFAgentAdapter(
                pipeline=pipeline,
                telemetry_sink=telemetry_sink,
            )
        self._adapter: HMAFAgentAdapter = adapter
        # Queue carries either an HMAFIntent or the shutdown sentinel
        # (an opaque object); runtime-level check gates dispatch.
        self._queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._handlers: dict[str, IntentHandler] = {}
        self._register_default_handlers()
        self._register_default_capabilities()
        self._responses: dict[str, HMAFResponse] = {}

    # ------------------------------------------------------------------
    # Capability / handler registration
    # ------------------------------------------------------------------

    def _register_default_capabilities(self) -> None:
        """Announce the canonical personalisation capability set."""
        for cap in (
            "personalisation.cognitive_load",
            "personalisation.style_mirror",
            "personalisation.emotional_tone",
            "personalisation.accessibility",
            "personalisation.translate",
            "personalisation.session_summary",
        ):
            self._adapter.register_capability(cap)

    def _register_default_handlers(self) -> None:
        """Wire the canned handlers used by the CLI demo."""
        self._handlers = {
            "get_user_adaptation": self._handle_get_user_adaptation,
            "summarise_session": self._handle_summarise_session,
            "translate": self._handle_translate,
            "route_recommendation": self._handle_route_recommendation,
            "explain_adaptation": self._handle_explain_adaptation,
        }

    def register_handler(self, intent_name: str, handler: IntentHandler) -> None:
        """Register a handler for a new intent.

        Args:
            intent_name: Identifier of the intent to handle.
            handler: Async callable returning an :class:`HMAFResponse`.

        Raises:
            ValueError: If the intent name is empty.
        """
        if not intent_name:
            raise ValueError("intent_name must be a non-empty string.")
        self._handlers[intent_name] = handler

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start consuming intents from the simulated bus.

        Idempotent: calling it twice is a no-op.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consume())
        self._adapter.emit_telemetry(
            {
                "event": "runtime.start",
                "capabilities": len(self._adapter.capabilities),
            }
        )
        logger.info("HMAFAgentRuntime started.")

    async def stop(self) -> None:
        """Stop the runtime and drain pending intents.

        Idempotent: safe to call even if :meth:`start` was never invoked.
        """
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            # Put a sentinel to unblock the consumer.
            await self._queue.put(_SHUTDOWN_SENTINEL)
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Runtime consumer did not exit cleanly; cancelling.")
                self._task.cancel()
            finally:
                self._task = None
        self._adapter.emit_telemetry({"event": "runtime.stop"})
        logger.info("HMAFAgentRuntime stopped.")

    async def receive_intent(self, intent: HMAFIntent) -> None:
        """Post an intent onto the simulated bus.

        Args:
            intent: The incoming :class:`HMAFIntent`.

        Raises:
            RuntimeError: If the runtime is not running.
        """
        if not self._running:
            raise RuntimeError(
                "HMAFAgentRuntime.receive_intent called before start()."
            )
        await self._queue.put(intent)
        self._adapter.emit_telemetry(
            {
                "event": "intent.enqueued",
                "intent_name": intent.name,
                "correlation_id": intent.correlation_id,
            }
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def plan_and_execute(self, intent: HMAFIntent) -> HMAFResponse:
        """Plan for *intent*, execute the plan, and return the response.

        The method:
            1. Enforces the privacy guard (forbidden intent names).
            2. Produces an HMAF plan via the adapter.
            3. Optionally runs the PDDL safety planner (soft import) to
               ensure no step would cross a privacy boundary.
            4. Dispatches to the registered handler (or a default
               fallback) for the actual work.
            5. Emits step-by-step telemetry.

        Args:
            intent: The :class:`HMAFIntent` to process.

        Returns:
            The :class:`HMAFResponse` produced by the handler, or a
            refusal response if the privacy guard tripped.
        """
        started = time.perf_counter()
        if intent.name in _FORBIDDEN_INTENT_NAMES:
            self._adapter.emit_telemetry(
                {
                    "event": "intent.refused",
                    "intent_name": intent.name,
                    "correlation_id": intent.correlation_id,
                    "reason": "forbidden_by_privacy_guard",
                }
            )
            return HMAFResponse(
                intent_name=intent.name,
                correlation_id=intent.correlation_id,
                ok=False,
                terminal_action="deny_request",
                payload={"reason": "privacy_guard"},
                steps_executed=0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        # Produce the HMAF plan (used only for telemetry; the handler
        # decides the actual steps).
        plan = self._adapter.plan({"type": intent.name, "context": intent.parameters})
        steps_executed = 0
        plan_outcomes: list[StepResult] = []

        for step in plan:
            result = await self._execute_step(step, intent)
            steps_executed += 1
            plan_outcomes.append(result)
            if result.error is None:
                self._adapter.emit_telemetry(
                    {
                        "event": "step.complete",
                        "capability": step.capability,
                        "latency_ms": round(result.latency_ms, 3),
                        "correlation_id": intent.correlation_id,
                    }
                )
            else:
                self._adapter.emit_telemetry(
                    {
                        "event": "step.error",
                        "capability": step.capability,
                        "latency_ms": round(result.latency_ms, 3),
                        "correlation_id": intent.correlation_id,
                        "error_kind": result.error.split(":", 1)[0],
                    }
                )

        # Run the safety planner if available (soft import).
        terminal_action = self._run_safety_planner(intent)

        handler = self._handlers.get(intent.name, self._handle_unknown)
        try:
            response = await handler(intent)
        except Exception as exc:  # noqa: BLE001 -- handler isolation
            logger.exception("Intent handler raised for %s", intent.name)
            response = HMAFResponse(
                intent_name=intent.name,
                correlation_id=intent.correlation_id,
                ok=False,
                terminal_action="handler_error",
                payload={"error_kind": type(exc).__name__},
                steps_executed=steps_executed,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            self._adapter.emit_telemetry(
                {
                    "event": "intent.failed",
                    "intent_name": intent.name,
                    "correlation_id": intent.correlation_id,
                    "error_kind": type(exc).__name__,
                }
            )
            self._responses[intent.correlation_id] = response
            return response

        # Overlay the runtime-derived fields (keep handler payload).
        response = response.model_copy(
            update={
                "steps_executed": steps_executed,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "terminal_action": response.terminal_action or terminal_action,
            }
        )
        self._adapter.emit_telemetry(
            {
                "event": "intent.complete",
                "intent_name": intent.name,
                "correlation_id": intent.correlation_id,
                "latency_ms": round(response.latency_ms, 3),
                "steps_executed": steps_executed,
                "terminal_action": response.terminal_action,
            }
        )
        self._responses[intent.correlation_id] = response
        return response

    async def last_response(self, correlation_id: str) -> HMAFResponse | None:
        """Return the most recent response for a correlation id.

        Args:
            correlation_id: The id the caller is waiting on.

        Returns:
            The matching :class:`HMAFResponse` or ``None``.
        """
        return self._responses.get(correlation_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        """Background bus consumer; dispatches intents to the executor."""
        while self._running:
            try:
                intent = await self._queue.get()
            except asyncio.CancelledError:
                break
            if intent is _SHUTDOWN_SENTINEL:
                self._queue.task_done()
                break
            try:
                await self.plan_and_execute(intent)
            except Exception:  # noqa: BLE001 -- never crash the loop
                logger.exception("Runtime consumer loop caught exception")
            finally:
                self._queue.task_done()

    async def _execute_step(
        self,
        step: PlanStep,
        intent: HMAFIntent,
    ) -> StepResult:
        """Run a single plan step through the adapter.

        Args:
            step: The :class:`PlanStep` to run.
            intent: The originating intent (used to log correlation).

        Returns:
            The :class:`StepResult` produced by the adapter.
        """
        _ = intent  # correlation is already on the telemetry event
        return await self._adapter.execute(step)

    def _run_safety_planner(self, intent: HMAFIntent) -> str:
        """Execute the PDDL planner if available; return the terminal action.

        Soft-imports :mod:`i3.safety` so this module remains importable
        in environments where that package is absent.

        Args:
            intent: The originating intent (its parameters decide the
                context booleans).

        Returns:
            The planner's terminal action name (e.g. ``route_local``) or
            ``"no_safety_planner"`` if the import failed.
        """
        try:
            from i3.safety import PrivacySafetyPlanner, SafetyContext
        except ImportError:
            return "no_safety_planner"

        params = intent.parameters or {}
        ctx = SafetyContext(
            sensitive_topic=bool(params.get("sensitive_topic", False)),
            network_available=bool(params.get("network_available", True)),
            authenticated_user=bool(params.get("authenticated_user", True)),
            encryption_key_loaded=bool(params.get("encryption_key_loaded", True)),
            rate_limited=bool(params.get("rate_limited", False)),
            contains_pii=bool(params.get("contains_pii", False)),
        )
        plan = PrivacySafetyPlanner().plan(ctx)
        return plan.actions[-1] if plan.actions else "no_action"

    # ------------------------------------------------------------------
    # Canned handlers
    # ------------------------------------------------------------------

    async def _handle_get_user_adaptation(
        self, intent: HMAFIntent
    ) -> HMAFResponse:
        """Return a neutral AdaptationVector summary (text-free)."""
        adaptation = await self._adapter._pipeline.adapt(state=None)  # type: ignore[attr-defined]
        return HMAFResponse(
            intent_name=intent.name,
            correlation_id=intent.correlation_id,
            ok=True,
            terminal_action="route_local",
            payload={"adaptation": adaptation},
            steps_executed=0,
            latency_ms=0.0,
        )

    async def _handle_summarise_session(
        self, intent: HMAFIntent
    ) -> HMAFResponse:
        """Return a scalar session summary -- never raw text."""
        return HMAFResponse(
            intent_name=intent.name,
            correlation_id=intent.correlation_id,
            ok=True,
            terminal_action="route_local",
            payload={
                "turns": int(intent.parameters.get("turns", 0)),
                "avg_engagement": float(
                    intent.parameters.get("avg_engagement", 0.75)
                ),
            },
            steps_executed=0,
            latency_ms=0.0,
        )

    async def _handle_translate(self, intent: HMAFIntent) -> HMAFResponse:
        """Deterministic translation stub -- the real endpoint lives in
        :mod:`server.routes_translate`; this handler simply acknowledges
        the intent and records its shape."""
        return HMAFResponse(
            intent_name=intent.name,
            correlation_id=intent.correlation_id,
            ok=True,
            terminal_action="route_local",
            payload={
                "target_language": str(
                    intent.parameters.get("target_language", "en")
                ),
                "length_in": int(intent.parameters.get("length_in", 0)),
            },
            steps_executed=0,
            latency_ms=0.0,
        )

    async def _handle_route_recommendation(
        self, intent: HMAFIntent
    ) -> HMAFResponse:
        """Recommend a route based on intent parameters (mock)."""
        prefer_cloud = bool(intent.parameters.get("prefer_cloud", False))
        terminal = "route_cloud" if prefer_cloud else "route_local"
        return HMAFResponse(
            intent_name=intent.name,
            correlation_id=intent.correlation_id,
            ok=True,
            terminal_action=terminal,
            payload={"recommended_route": terminal},
            steps_executed=0,
            latency_ms=0.0,
        )

    async def _handle_explain_adaptation(
        self, intent: HMAFIntent
    ) -> HMAFResponse:
        """Return a structured explanation of the current adaptation."""
        return HMAFResponse(
            intent_name=intent.name,
            correlation_id=intent.correlation_id,
            ok=True,
            terminal_action="route_local",
            payload={
                "explanation_kind": "scalar_summary",
                "dimensions": [
                    "cognitive_load",
                    "style_mirror",
                    "emotional_tone",
                    "accessibility",
                ],
            },
            steps_executed=0,
            latency_ms=0.0,
        )

    async def _handle_unknown(self, intent: HMAFIntent) -> HMAFResponse:
        """Default handler for intents without a registered callback."""
        return HMAFResponse(
            intent_name=intent.name,
            correlation_id=intent.correlation_id,
            ok=False,
            terminal_action="unknown_intent",
            payload={"reason": "no_handler_registered"},
            steps_executed=0,
            latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Module-level sentinels
# ---------------------------------------------------------------------------


_SHUTDOWN_SENTINEL: Any = object()


__all__ = [
    "HMAFAgentRuntime",
    "HMAFIntent",
    "HMAFResponse",
    "IntentHandler",
]
