"""HMAF agent-protocol adapter for Implicit Interaction Intelligence (I³).

This module provides a **reference implementation** of an HMAF
(Harmony Multi-Agent Framework) agent adapter that wraps I³'s
adaptation pipeline. It exposes I³'s adaptation capabilities through
HMAF's four primitives: capability registration, planning, execution,
and telemetry emission.

The adapter is deliberately framework-agnostic: the HMAF message
shapes are declared as :class:`typing.Protocol` types, so the same
code can run against a real HMAF runtime, a test harness, or a
no-op stub.

See :doc:`docs/huawei/harmony_hmaf_integration.md` for the full
integration story.

Example:
    Minimal usage::

        from i3.huawei.hmaf_adapter import HMAFAgentAdapter

        adapter = HMAFAgentAdapter(pipeline=my_pipeline)
        adapter.register_capability("personalisation.cognitive_load")
        adapter.register_capability("personalisation.style_mirror")

        plan = adapter.plan({"type": "personalise_response_style"})
        for step in plan:
            result = await adapter.execute(step)
            adapter.emit_telemetry({
                "event": "step.complete",
                "capability": step.capability,
                "latency_ms": result.latency_ms,
            })
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HMAF message shape protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class HMAFIntent(Protocol):
    """HMAF intent payload shape.

    Attributes:
        type: Intent identifier (e.g. ``"personalise_response_style"``).
        context: Optional free-form context map produced by the HMAF planner.
    """

    type: str
    context: dict[str, Any]


@runtime_checkable
class HMAFPlanStep(Protocol):
    """One step in an HMAF execution plan.

    Attributes:
        capability: Namespaced capability id (e.g. ``"personalisation.cognitive_load"``).
        inputs: Input payload for this step.
        produces: Logical name of the output artefact.
    """

    capability: str
    inputs: dict[str, Any]
    produces: str


@runtime_checkable
class HMAFTelemetryEvent(Protocol):
    """HMAF telemetry event shape.

    Telemetry events must be **text-free** — only scalars and identifiers.
    I³'s adapter enforces this: any field whose name implies raw content
    (``text``, ``prompt``, ``response``, ``body``) is rejected before
    emission.
    """

    event: str


# ---------------------------------------------------------------------------
# Concrete data classes (matching the protocols above)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanStep:
    """Immutable concrete :class:`HMAFPlanStep` implementation.

    Attributes:
        capability: The HMAF capability id this step invokes.
        inputs: Inputs passed to the capability.
        produces: Logical name of the produced artefact.
    """

    capability: str
    inputs: dict[str, Any] = field(default_factory=dict)
    produces: str = "unspecified"


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of executing a single :class:`PlanStep`.

    Attributes:
        capability: Capability id that was executed.
        produced: The artefact produced by the step.
        latency_ms: Wall-clock execution latency in milliseconds.
        error: Optional error message if the step failed.
    """

    capability: str
    produced: Any
    latency_ms: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class I3PipelineProtocol(Protocol):
    """Minimal shape required of an I³ pipeline by the HMAF adapter.

    Any object exposing these async methods can back the adapter. The
    full :class:`i3.pipeline.engine.PipelineEngine` satisfies this
    protocol.
    """

    async def extract_features(self, message: str) -> Any:
        """Extract an :class:`InteractionFeatureVector` from raw input."""
        ...

    async def encode(self, features: Any) -> Any:
        """Run the TCN encoder to produce a 64-dim user-state embedding."""
        ...

    async def adapt(self, state: Any) -> Any:
        """Produce an 8-dim :class:`AdaptationVector`."""
        ...


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


# Text-sounding keys we refuse to emit in telemetry, for pillar-4 compliance.
_FORBIDDEN_TELEMETRY_KEYS: frozenset[str] = frozenset(
    {"text", "prompt", "response", "body", "content", "raw"}
)


class HMAFAgentAdapter:
    """HMAF-native agent wrapper around an I³ pipeline.

    The adapter exposes I³'s adaptation pipeline through the four
    HMAF primitives:

    * :meth:`register_capability` — announce a capability to the HMAF
      runtime's discovery layer.
    * :meth:`plan` — turn an intent into an executable plan whose steps
      are idempotent I³ pipeline stages.
    * :meth:`execute` — run a single plan step and return its result.
    * :meth:`emit_telemetry` — publish a text-free telemetry event.

    Args:
        pipeline: Any object satisfying :class:`I3PipelineProtocol`.
            Typically :class:`i3.pipeline.engine.PipelineEngine`.
        telemetry_sink: Optional callable invoked with every emitted
            telemetry event. Defaults to a module-level logger call.

    Attributes:
        capabilities: The set of registered HMAF capability ids.
    """

    def __init__(
        self,
        pipeline: I3PipelineProtocol,
        telemetry_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._pipeline: I3PipelineProtocol = pipeline
        self._telemetry_sink: Callable[[dict[str, Any]], None] = (
            telemetry_sink if telemetry_sink is not None else self._default_sink
        )
        self.capabilities: set[str] = set()

    # ------------------------------------------------------------------ #
    # Capability registration
    # ------------------------------------------------------------------ #

    def register_capability(self, name: str) -> None:
        """Announce a capability to the HMAF discovery layer.

        Args:
            name: Namespaced HMAF capability id (e.g.
                ``"personalisation.cognitive_load"``).

        Raises:
            ValueError: If the capability id is empty or does not contain
                a namespace separator.
        """
        if not name or "." not in name:
            raise ValueError(
                f"HMAF capability id must be dot-namespaced; got {name!r}."
            )
        self.capabilities.add(name)
        logger.info("Registered HMAF capability: %s", name)

    def list_capabilities(self) -> list[str]:
        """Return the sorted list of currently-registered capability ids.

        Returns:
            Sorted capability ids (deterministic ordering for HMAF
            discovery caches).
        """
        return sorted(self.capabilities)

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #

    def plan(self, intent: dict[str, Any] | HMAFIntent) -> list[PlanStep]:
        """Translate an HMAF intent into an executable plan.

        The current plan is the five-step I³ flow. HMAF's planner may
        prune steps whose outputs are already cached, parallelise
        independent steps, or refuse the plan entirely.

        Args:
            intent: HMAF intent, either as a dict or a protocol-conforming
                object.

        Returns:
            An ordered list of :class:`PlanStep` ready for
            :meth:`execute`.
        """
        intent_type = self._intent_type(intent)
        logger.debug("Planning for intent type: %s", intent_type)

        plan: list[PlanStep] = [
            PlanStep(
                capability="interaction.feature_extract",
                inputs={"intent_type": intent_type},
                produces="InteractionFeatureVector",
            ),
            PlanStep(
                capability="encoder.tcn",
                inputs={"from": "InteractionFeatureVector"},
                produces="UserStateEmbedding",
            ),
            PlanStep(
                capability="adaptation.controller",
                inputs={"from": "UserStateEmbedding"},
                produces="AdaptationVector",
            ),
            PlanStep(
                capability="router.thompson",
                inputs={"from": "AdaptationVector"},
                produces="RoutingDecision",
            ),
            PlanStep(
                capability="generation.route",
                inputs={"from": "RoutingDecision"},
                produces="Response",
            ),
        ]
        return plan

    @staticmethod
    def _intent_type(intent: dict[str, Any] | HMAFIntent) -> str:
        """Coerce an intent (dict or protocol) into a string id.

        Args:
            intent: Incoming HMAF intent.

        Returns:
            Intent type string, or ``"unknown"`` if not determinable.
        """
        if isinstance(intent, dict):
            return str(intent.get("type", "unknown"))
        return getattr(intent, "type", "unknown")

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def execute(self, step: PlanStep | HMAFPlanStep) -> StepResult:
        """Execute a single plan step against the wrapped pipeline.

        Each step is pure-function-shaped: same inputs, same outputs,
        no hidden state (pipeline state is threaded explicitly via the
        pipeline object).

        Args:
            step: The plan step to execute.

        Returns:
            A :class:`StepResult` carrying the produced artefact and
            wall-clock latency.
        """
        capability = step.capability
        inputs = step.inputs
        started = time.perf_counter()

        try:
            produced = await self._dispatch(capability, inputs)
            error: str | None = None
        except NotImplementedError as exc:
            produced = None
            error = f"not_implemented:{exc}"
            logger.warning("Step %s not implemented: %s", capability, exc)
        except Exception as exc:  # noqa: BLE001 — report all failures
            produced = None
            error = f"{type(exc).__name__}:{exc}"
            logger.error("Step %s failed: %s", capability, exc)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return StepResult(
            capability=capability,
            produced=produced,
            latency_ms=elapsed_ms,
            error=error,
        )

    async def _dispatch(
        self, capability: str, inputs: dict[str, Any]
    ) -> Any:
        """Dispatch a capability id to the appropriate pipeline method.

        Args:
            capability: HMAF capability id.
            inputs: Payload for the capability.

        Returns:
            The capability's output artefact.

        Raises:
            NotImplementedError: If the capability is unknown. HMAF
                runtimes treat this as a soft failure (the plan skips
                the step and proceeds).
        """
        if capability == "interaction.feature_extract":
            message = str(inputs.get("message", ""))
            return await self._pipeline.extract_features(message)
        if capability == "encoder.tcn":
            features = inputs.get("features")
            return await self._pipeline.encode(features)
        if capability == "adaptation.controller":
            state = inputs.get("state")
            return await self._pipeline.adapt(state)
        # router.thompson and generation.route are intentionally stubbed:
        # the reference pipeline protocol covers only the always-on path.
        raise NotImplementedError(
            f"Capability {capability!r} is not wired in this reference "
            "adapter; extend I3PipelineProtocol to support it."
        )

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #

    def emit_telemetry(self, event: dict[str, Any] | HMAFTelemetryEvent) -> None:
        """Publish a telemetry event, enforcing the no-raw-text invariant.

        Any field whose name suggests raw user content
        (``text``, ``prompt``, ``response``, ``body``, ``content``,
        ``raw``) is rejected before emission. This is the programmatic
        guard that corresponds to HMAF Pillar 4's "secure and
        trustworthy agents" contract.

        Args:
            event: Telemetry event payload.

        Raises:
            ValueError: If the event contains a forbidden key.
        """
        payload = self._event_to_dict(event)
        self._validate_no_raw_content(payload)
        self._telemetry_sink(payload)

    @staticmethod
    def _event_to_dict(
        event: dict[str, Any] | HMAFTelemetryEvent,
    ) -> dict[str, Any]:
        """Coerce an incoming event to a plain dict.

        Args:
            event: Either a dict or a protocol-conforming object.

        Returns:
            A plain dict payload.
        """
        if isinstance(event, dict):
            return dict(event)
        return {
            k: getattr(event, k)
            for k in dir(event)
            if not k.startswith("_") and not callable(getattr(event, k))
        }

    @staticmethod
    def _validate_no_raw_content(payload: dict[str, Any]) -> None:
        """Raise if the payload contains any raw-content-looking keys.

        Args:
            payload: The telemetry event to check.

        Raises:
            ValueError: If any forbidden key is present.
        """
        forbidden = _FORBIDDEN_TELEMETRY_KEYS & {k.lower() for k in payload}
        if forbidden:
            raise ValueError(
                f"Refusing to emit telemetry event with raw-content keys: "
                f"{sorted(forbidden)}. Telemetry must be text-free (HMAF "
                "Pillar 4 compliance)."
            )

    @staticmethod
    def _default_sink(payload: dict[str, Any]) -> None:
        """Default telemetry sink: log at INFO level.

        Args:
            payload: Validated telemetry event payload.
        """
        logger.info("HMAF telemetry: %s", payload)


__all__ = [
    "HMAFAgentAdapter",
    "HMAFIntent",
    "HMAFPlanStep",
    "HMAFTelemetryEvent",
    "I3PipelineProtocol",
    "PlanStep",
    "StepResult",
]
