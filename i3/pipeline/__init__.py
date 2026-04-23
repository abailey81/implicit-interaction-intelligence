"""Pipeline Orchestration Engine for Implicit Interaction Intelligence (I3).

The pipeline module provides the central orchestrator that connects every
subsystem into a single end-to-end message-processing flow.  It is the
primary entry point for the application server and the demo interface.

Key exports:

- :class:`Pipeline` -- The main orchestrator class.  Construct with a
  :class:`~src.config.Config`, call ``await pipeline.initialize()``, then
  process messages via ``await pipeline.process_message(input)``.

- :class:`PipelineInput` -- Dataclass carrying a user message and its
  associated keystroke/timing metadata.

- :class:`PipelineOutput` -- Dataclass returned after processing, containing
  the AI response, routing decision, adaptation vector, and all user-state
  metrics needed by the frontend dashboard.

- :class:`EngagementEstimator` -- Stateless utility that computes a composite
  engagement score from raw interaction metrics.

Usage::

    from i3.config import load_config
    from i3.pipeline import Pipeline, PipelineInput

    config = load_config("configs/default.yaml")
    pipeline = Pipeline(config)
    await pipeline.initialize()

    session_id = await pipeline.start_session("user_42")
    output = await pipeline.process_message(PipelineInput(
        user_id="user_42",
        session_id=session_id,
        message_text="Hello!",
        timestamp=time.time(),
        composition_time_ms=1500.0,
        edit_count=0,
        pause_before_send_ms=200.0,
    ))
"""

from i3.pipeline.engine import Pipeline
from i3.pipeline.types import (
    EngagementEstimator,
    EngagementSignal,
    PipelineInput,
    PipelineOutput,
)

__all__ = [
    "EngagementEstimator",
    "EngagementSignal",
    "Pipeline",
    "PipelineInput",
    "PipelineOutput",
]
