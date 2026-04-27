"""Per-turn pipeline trace collector for the live "Flow" dashboard.

Powers the third flagship surface: the system architecture flow
dashboard (``web/js/flow_dashboard.js``).  Every stage of the
:class:`i3.pipeline.engine.Pipeline._process_message_inner` method is
wrapped in a :meth:`PipelineTraceCollector.stage` context manager, which
records:

  * stage_id (encoder, adaptation, biometric, ...)
  * start / end / latency in ms (relative to the turn start)
  * a short input / output / notes summary (≤ 4 KB total per turn)
  * the ``fired`` flag (False for conditional stages that didn't run)
  * the ``is_tool`` flag (True for tool routes — math/refuse/entity)
  * arrow flows (``encoder → adaptation``, ``adaptation → router``, ...)

The completed trace dict is shipped on the WebSocket response /
response_done frame as ``pipeline_trace``.  The browser side animates
each stage box pulsing in the order they fired.

Design constraints
~~~~~~~~~~~~~~~~~~
* Pure Python, no torch, no numpy.
* Total trace dict ≤ 4 KB (we cap inputs/outputs/notes by length).
* Bounded in-memory ring buffer (default 200 traces) for retrospective
  replays via the ``GET /api/flow/recent`` endpoint.
* Thread-safe enough for the asyncio single-threaded event loop +
  background task pool — we don't expect concurrent writers to the
  *same* TurnHandle, but the recent-traces deque is guarded by a
  threading.Lock so HTTP routes and the engine cannot race.
* Never raises out of an instrumented section.  Failure to record a
  stage is *always* a soft failure — we log at WARN and move on.

Public surface (used by ``i3.pipeline.engine`` and ``server.routes_flow``):

    StageRecord, PipelineTrace, TurnHandle, PipelineTraceCollector

Privacy
~~~~~~~
The collector NEVER stores raw message text or full embeddings.  The
inputs / outputs / notes maps carry small JSON-safe summaries
(dimensions, scores, durations, route names).  When in doubt, the
``_truncate`` helper trims any string value to 80 chars.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Maximum number of stages ever recorded for a single turn.  Defends
#: against a runaway loop that calls ``stage()`` thousands of times.
_MAX_STAGES_PER_TURN: int = 32

#: Maximum number of arrow flows ever recorded for a single turn.
_MAX_ARROWS_PER_TURN: int = 64

#: Maximum length (chars) of any single input / output / notes string.
_MAX_VALUE_CHARS: int = 60

#: Maximum number of keys in any single inputs / outputs map.
_MAX_DICT_KEYS: int = 6

#: Default size of the in-memory recent-traces deque.
_DEFAULT_MAX_TRACES: int = 200


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """One pipeline stage's execution record.

    Attributes
    ----------
    stage_id:
        Stable machine identifier — the front-end keys its diagram
        layout off this.  One of: ``encoder``, ``adaptation``,
        ``biometric``, ``state_classifier``, ``affect_shift``,
        ``entity_tracker``, ``accessibility``, ``router``,
        ``retrieval``, ``generation``, ``critique``, ``postprocess``,
        ``multimodal_fusion``, ``personalisation``.
    label:
        Human-readable short string for the stage tile (e.g. ``TCN
        encoder``, ``Adaptation controller``).
    fired:
        ``True`` if the stage actually ran on this turn.  Some stages
        are conditional (``critique`` only fires on the SLM path).
    started_at_ms / ended_at_ms / latency_ms:
        All in milliseconds, ``started_at_ms`` and ``ended_at_ms``
        relative to turn start.  ``latency_ms = ended - started``.
    inputs / outputs:
        Short JSON-safe summary maps.  String values are clipped to
        ``_MAX_VALUE_CHARS`` chars and the map is clipped to
        ``_MAX_DICT_KEYS`` keys to keep the trace under 4 KB.
    notes:
        One-line description of what happened.  Clipped to 160 chars.
    is_tool:
        ``True`` for tool-route stages so the front end paints them
        yellow (math / refuse / entity / compare).
    """

    stage_id: str
    label: str
    fired: bool = False
    started_at_ms: float = 0.0
    ended_at_ms: float = 0.0
    latency_ms: float = 0.0
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    notes: str = ""
    is_tool: bool = False


@dataclass
class PipelineTrace:
    """Full per-turn trace.  Serialised onto the WS response frame."""

    turn_id: str
    user_id: str
    session_id: str
    started_at_ms: float
    ended_at_ms: float
    total_latency_ms: float
    stages: list[StageRecord] = field(default_factory=list)
    arrow_flows: list[dict] = field(default_factory=list)


@dataclass
class TurnHandle:
    """Opaque handle returned by :meth:`PipelineTraceCollector.start_turn`.

    The engine carries it through ``_process_message_inner`` and passes
    it back to the collector on every ``stage()`` / ``note()`` /
    ``arrow()`` call so concurrent users do not cross-pollute traces.
    """

    turn_id: str
    user_id: str
    session_id: str
    monotonic_start: float          # time.monotonic() reference
    stages: list[StageRecord] = field(default_factory=list)
    arrow_flows: list[dict] = field(default_factory=list)
    # Map stage_id -> StageRecord for fast lookup in note() / arrow().
    _stage_index: dict[str, StageRecord] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(value: Any) -> Any:
    """Clip a value down to a JSON-safe summary form.

    Strings longer than ``_MAX_VALUE_CHARS`` are truncated with an
    ellipsis suffix.  Numbers pass through (rounded to 4 d.p. for
    floats).  Anything else gets ``str()``'d and truncated.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Reject NaN / inf to keep the JSON valid.
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 4)
    if isinstance(value, str):
        if len(value) > _MAX_VALUE_CHARS:
            return value[: _MAX_VALUE_CHARS - 1] + "…"
        return value
    text = str(value)
    if len(text) > _MAX_VALUE_CHARS:
        text = text[: _MAX_VALUE_CHARS - 1] + "…"
    return text


def _truncate_dict(d: dict) -> dict:
    """Bound a summary dict to ``_MAX_DICT_KEYS`` keys + truncated values."""
    out: dict = {}
    if not isinstance(d, dict):
        return out
    for i, (k, v) in enumerate(d.items()):
        if i >= _MAX_DICT_KEYS:
            break
        try:
            key = str(k)[:32]
        except Exception:
            continue
        out[key] = _truncate(v)
    return out


def _now_ms(monotonic_start: float) -> float:
    """Return current time in ms relative to the turn's monotonic_start."""
    return (time.monotonic() - monotonic_start) * 1000.0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class PipelineTraceCollector:
    """Per-turn collector with a bounded in-memory recent-traces deque.

    The engine calls :meth:`start_turn` at the top of
    ``_process_message_inner``, wraps each major stage in a ``with
    self._trace_collector.stage(handle, ...):`` block, and finally calls
    :meth:`finalise` to get back a JSON-safe dict that ships on the WS
    response frame.

    Recent traces are kept in a thread-safe deque (default 200) so the
    ``GET /api/flow/recent`` route can replay any past turn.

    Example
    -------
    >>> collector = PipelineTraceCollector()
    >>> handle = collector.start_turn("alice", "session-1")
    >>> with collector.stage(handle, "encoder", "TCN encoder"):
    ...     collector.note(handle, "encoder", out_dim=64)
    >>> trace = collector.finalise(handle)
    >>> trace["stages"][0]["stage_id"]
    'encoder'
    """

    def __init__(self, max_traces_in_memory: int = _DEFAULT_MAX_TRACES) -> None:
        if max_traces_in_memory <= 0:
            max_traces_in_memory = _DEFAULT_MAX_TRACES
        self._recent: deque[dict] = deque(maxlen=int(max_traces_in_memory))
        # Index turn_id -> trace dict for O(1) get_turn().  We rebuild
        # this lazily so we don't keep two copies in memory; the deque
        # is the source of truth.
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_turn(self, user_id: str, session_id: str) -> TurnHandle:
        """Begin a new trace.  Returns the handle the engine carries."""
        return TurnHandle(
            turn_id=str(uuid.uuid4()),
            user_id=str(user_id)[:64],
            session_id=str(session_id)[:128],
            monotonic_start=time.monotonic(),
        )

    # ------------------------------------------------------------------
    # Stage recording
    # ------------------------------------------------------------------

    @contextmanager
    def stage(
        self,
        handle: TurnHandle,
        stage_id: str,
        label: str,
        *,
        is_tool: bool = False,
    ) -> Iterator[StageRecord]:
        """Enter a stage; auto-time start and end.

        The yielded :class:`StageRecord` can have ``inputs`` /
        ``outputs`` / ``notes`` set directly inside the ``with`` block,
        or via :meth:`note`.  The record is appended to
        ``handle.stages`` only after the ``with`` block completes — so
        an exception inside the stage still records the timing and
        marks ``fired=True`` (the engine considers a started-but-failed
        stage to have fired for visualisation purposes).
        """
        if not isinstance(handle, TurnHandle):  # pragma: no cover - defensive
            # Yield a throw-away record so the caller's ``with`` body
            # doesn't crash; we just don't record anything.
            yield StageRecord(stage_id=stage_id, label=label)
            return

        if len(handle.stages) >= _MAX_STAGES_PER_TURN:
            yield StageRecord(stage_id=stage_id, label=label)
            return

        record = StageRecord(
            stage_id=str(stage_id)[:32],
            label=str(label)[:48],
            fired=True,
            started_at_ms=round(_now_ms(handle.monotonic_start), 3),
            is_tool=bool(is_tool),
        )
        # Index BEFORE yielding so ``note()`` calls inside the with
        # body can find the record by stage_id.  Without this the
        # index is only populated in the ``finally`` clause and every
        # in-body note() call silently no-ops.
        handle._stage_index[record.stage_id] = record
        try:
            yield record
        finally:
            record.ended_at_ms = round(_now_ms(handle.monotonic_start), 3)
            record.latency_ms = round(
                max(0.0, record.ended_at_ms - record.started_at_ms), 3
            )
            # Bound the inputs / outputs map shapes.
            try:
                record.inputs = _truncate_dict(record.inputs)
                record.outputs = _truncate_dict(record.outputs)
                if isinstance(record.notes, str) and len(record.notes) > 80:
                    record.notes = record.notes[:79] + "…"
            except Exception:  # pragma: no cover - defensive
                logger.warning("trace truncation failed for stage=%s", stage_id)
            handle.stages.append(record)

    # ------------------------------------------------------------------
    # Manual stage entry (no timing context — for stages that didn't fire)
    # ------------------------------------------------------------------

    def record_skipped(
        self,
        handle: TurnHandle,
        stage_id: str,
        label: str,
        *,
        reason: str = "",
        is_tool: bool = False,
    ) -> None:
        """Record a stage that was conditionally skipped this turn.

        Used by stages like ``critique`` (only fires on SLM path) or
        ``retrieval`` (only when route_chosen != "tool").  The front
        end paints these dimmed grey.
        """
        if not isinstance(handle, TurnHandle):
            return
        if len(handle.stages) >= _MAX_STAGES_PER_TURN:
            return
        now_ms = round(_now_ms(handle.monotonic_start), 3)
        record = StageRecord(
            stage_id=str(stage_id)[:32],
            label=str(label)[:48],
            fired=False,
            started_at_ms=now_ms,
            ended_at_ms=now_ms,
            latency_ms=0.0,
            notes=str(reason)[:159] if reason else "",
            is_tool=bool(is_tool),
        )
        handle.stages.append(record)
        handle._stage_index[record.stage_id] = record

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def note(
        self,
        handle: TurnHandle,
        stage_id: str,
        **kv: Any,
    ) -> None:
        """Attach summary fields to the most recent record for *stage_id*.

        Recognised special keys:

        * ``_input``      — merged into ``record.inputs``
        * ``_output``     — merged into ``record.outputs``
        * ``_notes``      — overrides ``record.notes`` (string)
        * ``_is_tool``    — sets ``record.is_tool`` (bool)

        Anything else is dropped into ``record.outputs`` for
        convenience (most stages care about output dimensions /
        scores far more than inputs).  ``_input``/``_output`` may be
        passed dicts to merge in bulk.
        """
        if not isinstance(handle, TurnHandle):
            return
        record = handle._stage_index.get(str(stage_id)[:32])
        if record is None:
            return
        try:
            for key, value in kv.items():
                if key == "_input" and isinstance(value, dict):
                    record.inputs.update(value)
                elif key == "_output" and isinstance(value, dict):
                    record.outputs.update(value)
                elif key == "_notes":
                    record.notes = str(value)[:80]
                elif key == "_is_tool":
                    record.is_tool = bool(value)
                else:
                    record.outputs[key] = value
        except Exception:  # pragma: no cover - defensive
            logger.warning("trace.note failed for stage=%s", stage_id)

    # ------------------------------------------------------------------
    # Arrow flow recording
    # ------------------------------------------------------------------

    def arrow(
        self,
        handle: TurnHandle,
        from_id: str,
        to_id: str,
        *,
        payload_summary: str = "",
        size_bytes: int = 0,
    ) -> None:
        """Record a directed data-flow arrow between two stages.

        Used by the front end to draw the SVG paths between stage
        boxes.  ``payload_summary`` is a short string like
        ``"64-d embedding"`` or ``"cosine=0.82"``.
        """
        if not isinstance(handle, TurnHandle):
            return
        if len(handle.arrow_flows) >= _MAX_ARROWS_PER_TURN:
            return
        try:
            handle.arrow_flows.append(
                {
                    "from": str(from_id)[:32],
                    "to": str(to_id)[:32],
                    "payload_summary": str(payload_summary)[:_MAX_VALUE_CHARS],
                    "size_bytes": int(max(0, min(1_000_000, int(size_bytes or 0)))),
                }
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("trace.arrow failed for %s->%s", from_id, to_id)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalise(self, handle: TurnHandle) -> dict:
        """Close the trace and return the JSON-safe dict.

        Also stashes the dict in the bounded recent-traces deque for
        retrospective replay via ``GET /api/flow/recent``.
        """
        if not isinstance(handle, TurnHandle):
            return {}
        ended_ms = round(_now_ms(handle.monotonic_start), 3)
        # Total latency: prefer the difference between the latest
        # stage_ended and the turn start; fall back to ended_ms.
        if handle.stages:
            latest_end = max(s.ended_at_ms for s in handle.stages)
            total = max(latest_end, ended_ms)
        else:
            total = ended_ms
        trace = PipelineTrace(
            turn_id=handle.turn_id,
            user_id=handle.user_id,
            session_id=handle.session_id,
            started_at_ms=0.0,
            ended_at_ms=ended_ms,
            total_latency_ms=round(total, 3),
            stages=list(handle.stages),
            arrow_flows=list(handle.arrow_flows),
        )
        try:
            payload = asdict(trace)
        except Exception:  # pragma: no cover - defensive
            logger.exception("trace finalise failed")
            return {}

        # Stash for /api/flow/recent.
        try:
            with self._lock:
                self._recent.append(payload)
        except Exception:  # pragma: no cover - defensive
            logger.warning("failed to stash trace in recent-deque")
        return payload

    # ------------------------------------------------------------------
    # Retrospective access
    # ------------------------------------------------------------------

    def recent(self, *, user_id: str | None = None, n: int = 10) -> list[dict]:
        """Return the most-recent ``n`` traces, newest first.

        When *user_id* is given, traces for other users are filtered
        out.  ``n`` is clamped to the deque's maxlen.
        """
        n = max(1, min(int(n or 10), self._recent.maxlen or _DEFAULT_MAX_TRACES))
        with self._lock:
            traces = list(self._recent)
        traces.reverse()  # newest first
        if user_id:
            traces = [t for t in traces if t.get("user_id") == user_id]
        return traces[:n]

    def get_turn(self, turn_id: str) -> dict | None:
        """Return one specific trace by ``turn_id``, or ``None`` if missing."""
        if not turn_id:
            return None
        with self._lock:
            for t in reversed(self._recent):
                if t.get("turn_id") == turn_id:
                    return t
        return None

    def __len__(self) -> int:
        return len(self._recent)


__all__ = [
    "PipelineTrace",
    "PipelineTraceCollector",
    "StageRecord",
    "TurnHandle",
]
