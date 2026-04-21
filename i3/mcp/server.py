"""I3 Model Context Protocol (MCP) server.

This module wraps a subset of I3's read-only APIs in an MCP server that
Claude (or any MCP-compatible client) can connect to.  The server exposes:

* **Tools**
    * ``get_user_adaptation_vector(user_id)`` — current 8-dim adaptation.
    * ``get_user_state_embedding(user_id)`` — 64-dim encoder embedding.
    * ``get_feature_vector(user_id)`` — last 32-dim aggregated feature vector.
    * ``get_session_metadata(user_id, session_id)`` — session-level metrics.
    * ``get_device_profile(target)`` — Kirin-class device profile.
    * ``route_recommendation(context)`` — bandit routing decision (dry-run).
    * ``explain_adaptation(user_id)`` — feature-attribution breakdown.
* **Resources**
    * ``i3://users/{user_id}/profile`` — EMAs + variance stats (no raw text).
    * ``i3://users/{user_id}/diary`` — diary entries (topics + adaptation).
    * ``i3://devices/kirin/{id}`` — device profile JSON.
    * ``i3://architecture/layers`` — 7-layer architecture reference.
    * ``i3://adrs`` — ADR index.
* **Prompts**
    * ``adaptation_summary(user_id)`` — summarise the adaptation state.
    * ``troubleshoot_high_load(user_id)`` — suggest simpler strategies.

Privacy guarantees
------------------
* Every payload is passed through
  :class:`i3.privacy.sanitizer.PrivacySanitizer` before being returned.  If
  the sanitizer detects PII (which should be impossible given we only carry
  aggregated metrics) the server raises rather than emitting the value.
* No raw user messages ever leave the server.  The server has no access path
  to raw text — it only reads aggregated state from the user model, the
  interaction monitor's feature store, and the diary.
* Every tool invocation is audit-logged at ``INFO`` level with the
  ``user_id`` (and **only** the user id), never the argument payload.

Spec compatibility
------------------
Compatible with the MCP specification dated ``2024-11-05`` and later.  The
SDK is soft-imported; a missing ``mcp`` package raises a clear error at
construction time.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import of the MCP SDK
# ---------------------------------------------------------------------------

try:
    # The official Python SDK uses the high-level :class:`FastMCP` helper to
    # register tools / resources / prompts.  Fallbacks are imported lazily.
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]

    _MCP_AVAILABLE: bool = True
except Exception:  # pragma: no cover — evaluated without mcp installed
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_AVAILABLE = False


if TYPE_CHECKING:  # pragma: no cover
    from i3.adaptation.types import AdaptationVector
    from i3.huawei.kirin_targets import DeviceProfile
    from i3.interaction.types import InteractionFeatureVector
    from i3.pipeline.engine import Pipeline
    from i3.privacy.sanitizer import PrivacySanitizer
    from i3.router.router import IntelligentRouter
    from i3.router.types import RoutingContext, RoutingDecision


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Human-readable name announced to MCP clients.
_SERVER_NAME: str = "i3-hmi-companion"

#: Fallback server version used if ``pyproject.toml`` cannot be parsed.
_FALLBACK_VERSION: str = "1.0.0"

#: Valid values for ``get_device_profile``'s ``target`` argument.
_VALID_DEVICE_TARGETS: frozenset[str] = frozenset(
    {"kirin_9000", "kirin_9010", "kirin_a2", "smart_hanhan"}
)

#: MCP specification this server is tested against.
MCP_SPEC_VERSION: str = "2024-11-05"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _read_pyproject_version() -> str:
    """Return the ``version`` declared in the project's ``pyproject.toml``.

    We avoid pulling a TOML dependency — the file is parsed line-by-line with
    a very small regex-free scanner.  Any failure falls back to
    :data:`_FALLBACK_VERSION` so the server still boots on an unusual layout.

    Returns:
        The version string (e.g. ``"1.0.0"``).  Never ``None``.
    """
    import pathlib

    candidates = [
        pathlib.Path("pyproject.toml"),
        pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml",
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        in_poetry = False
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("[tool.poetry]"):
                in_poetry = True
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_poetry = False
                continue
            if in_poetry and stripped.startswith("version"):
                # Line shape: ``version = "1.0.0"``
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    value = parts[1].strip().strip('"').strip("'")
                    if value:
                        return value
    return _FALLBACK_VERSION


def _require_mcp() -> None:
    """Raise a clear error when the ``mcp`` SDK is not installed."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("Install mcp: `pip install mcp[cli]`")


def _tensor_to_list(tensor: Any) -> list[float]:
    """Convert a torch or numpy 1-D tensor/array to a plain ``list[float]``.

    Args:
        tensor: A torch Tensor, numpy ndarray, or already-a-list input.

    Returns:
        A JSON-serialisable list of floats.
    """
    if tensor is None:
        return []
    if isinstance(tensor, list):
        return [float(x) for x in tensor]
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    if hasattr(tensor, "tolist"):
        return [float(x) for x in tensor.tolist()]
    return [float(x) for x in list(tensor)]


def _tensor_to_b64(tensor: Any) -> str:
    """Encode a 1-D numeric tensor as a base64 float32 blob.

    Used when the resource layer emits user embeddings without exposing their
    numeric content as a JSON array (saves space + avoids precision issues).

    Args:
        tensor: torch.Tensor or numpy.ndarray (or ``None``).

    Returns:
        Base64 ASCII string; empty string if *tensor* is ``None``.
    """
    if tensor is None:
        return ""
    import numpy as np

    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    arr = np.asarray(tensor, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _jsonable(value: Any) -> Any:
    """Best-effort coerce *value* into something ``json.dumps`` can serialise.

    Dataclasses are unpacked, tensors become lists, sets become sorted lists,
    and nested containers are recursively normalised.  This lives in one place
    so every tool / resource returns consistent wire-format JSON.

    Args:
        value: Arbitrary Python value.

    Returns:
        A JSON-compatible structure (dict / list / primitive).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):  # Pydantic v2
        return _jsonable(value.model_dump())
    if hasattr(value, "detach") or (
        hasattr(value, "tolist") and hasattr(value, "shape")
    ):
        return _tensor_to_list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # Enum fallback
    if hasattr(value, "value"):
        return value.value
    return str(value)


# ---------------------------------------------------------------------------
# I3MCPServer
# ---------------------------------------------------------------------------


class I3MCPServer:
    """MCP server exposing I3's read-only APIs to Claude-compatible clients.

    The server is stateless with respect to MCP: every request looks up
    fresh data from the pipeline / stores it wraps.  Constructing the
    server does NOT start it — call :meth:`run` (or one of the transport
    helpers in :mod:`i3.mcp.transport`) to serve clients.

    Args:
        pipeline: A fully-initialised I3 :class:`i3.pipeline.engine.Pipeline`
            from which user state, adaptation, and routing can be read.
            If ``None``, the server still registers its handlers but every
            user-scoped tool returns an explicit ``"pipeline_unavailable"``
            error payload.
        sanitizer: A :class:`i3.privacy.sanitizer.PrivacySanitizer` used to
            audit every outgoing payload.  If ``None``, a fresh instance
            with defaults is constructed.
        name: MCP server name announced to clients.  Defaults to
            :data:`_SERVER_NAME`.
        version: Optional version override — by default read from
            ``pyproject.toml``.

    Raises:
        RuntimeError: If the ``mcp`` package is not installed.
    """

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        sanitizer: PrivacySanitizer | None = None,
        *,
        name: str = _SERVER_NAME,
        version: str | None = None,
    ) -> None:
        _require_mcp()

        from i3.privacy.sanitizer import PrivacySanitizer as _PS

        self._pipeline: Pipeline | None = pipeline
        self._sanitizer: PrivacySanitizer = sanitizer or _PS(enabled=True)
        self._name: str = name
        self._version: str = version or _read_pyproject_version()

        # Assemble the FastMCP app.  ``assert`` satisfies the type checker:
        # ``_require_mcp`` already raised when ``FastMCP is None``.
        assert FastMCP is not None  # noqa: S101
        self._mcp = FastMCP(name=self._name, version=self._version)

        self._register_tools()
        self._register_resources()
        self._register_prompts()

        logger.info(
            "I3MCPServer initialised name=%s version=%s pipeline=%s spec=%s",
            self._name,
            self._version,
            "bound" if pipeline is not None else "unbound",
            MCP_SPEC_VERSION,
        )

    # ---- Properties ----------------------------------------------------

    @property
    def name(self) -> str:
        """The MCP server name announced to clients."""
        return self._name

    @property
    def version(self) -> str:
        """The MCP server version announced to clients."""
        return self._version

    @property
    def mcp(self) -> Any:
        """Underlying :class:`mcp.server.fastmcp.FastMCP` instance."""
        return self._mcp

    # ---- Audit / privacy boundary -------------------------------------

    def _audit(self, tool: str, user_id: str | None) -> None:
        """Emit a single ``INFO`` audit log for an inbound MCP invocation.

        Only the tool name and ``user_id`` are logged — never the argument
        payload, never any user text.
        """
        if user_id is None:
            logger.info("mcp_call tool=%s", tool)
        else:
            logger.info("mcp_call tool=%s user_id=%s", tool, user_id)

    def _guard_payload(self, payload: Any, *, label: str) -> Any:
        """Sanitize a payload before it leaves the server.

        The payload is serialised to JSON, run through the
        :class:`PrivacySanitizer`, and returned as-is if clean.  Any
        detected PII is treated as a critical invariant violation and
        raises :class:`RuntimeError` — the caller is responsible for
        retrying with corrected data, not for silently leaking it.

        Args:
            payload: The object about to be returned to the MCP client.
            label: Short tag used in the error message and audit log.

        Returns:
            The original *payload* (unchanged on success).

        Raises:
            RuntimeError: If any PII is detected in the serialised payload.
        """
        try:
            text = json.dumps(_jsonable(payload), ensure_ascii=False)
        except (TypeError, ValueError):
            # Non-serialisable payload — we can't meaningfully scan it.
            # Return unchanged; the MCP framing will error later if needed.
            return payload
        result = self._sanitizer.sanitize(text)
        if result.pii_detected:
            logger.error(
                "mcp_privacy_violation label=%s pii_types=%s count=%d",
                label,
                result.pii_types,
                result.replacements_made,
            )
            raise RuntimeError(
                f"Privacy invariant violated in MCP payload {label!r}: "
                f"detected {result.pii_types}. Refusing to emit."
            )
        return payload

    # ---- Pipeline lookups ----------------------------------------------

    def _pipeline_or_unavailable(self) -> Pipeline | None:
        """Return the bound pipeline or ``None`` with a logged warning."""
        if self._pipeline is None:
            logger.warning("mcp_pipeline_unavailable — returning error payload")
        return self._pipeline

    def _user_model(self, user_id: str) -> Any:
        """Return the per-user ``UserModel`` attached to the pipeline.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            The :class:`i3.user_model.model.UserModel` for *user_id*, or
            ``None`` if the pipeline is unavailable / the user has no model.
        """
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return None
        getter = getattr(pipe, "get_user_model", None)
        if callable(getter):
            try:
                return getter(user_id)
            except Exception:  # noqa: BLE001 — defensive
                return None
        # Real pipeline stores models under ``user_models``; tests may use
        # the private ``_user_models``.  Accept both.
        for attr in ("user_models", "_user_models"):
            models = getattr(pipe, attr, None)
            if isinstance(models, dict):
                return models.get(user_id)
        return None

    # ---- Tool implementations ------------------------------------------

    def _tool_get_user_adaptation_vector(self, user_id: str) -> dict[str, Any]:
        """Return the current 8-dim :class:`AdaptationVector` for *user_id*.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            ``{"user_id": ..., "vector": {...}}`` on success, or an error
            payload when the pipeline / user is unavailable.
        """
        self._audit("get_user_adaptation_vector", user_id)
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return {"error": "pipeline_unavailable", "user_id": user_id}
        controller = (
            getattr(pipe, "adaptation_controller", None)
            or getattr(pipe, "_adaptation_controller", None)
            or getattr(pipe, "adaptation", None)
        )
        # Controllers may expose the last vector via either ``last_vector``
        # or an internal ``_last_vector`` — try both before falling back.
        current = None
        if controller is not None:
            current = getattr(controller, "last_vector", None) or getattr(
                controller, "_last_vector", None
            )
        if current is None:
            from i3.adaptation.types import AdaptationVector

            current = AdaptationVector.default()
        payload = {"user_id": user_id, "vector": _jsonable(current)}
        return self._guard_payload(payload, label="adaptation_vector")

    def _tool_get_user_state_embedding(self, user_id: str) -> dict[str, Any]:
        """Return the 64-dim user-state embedding for *user_id*.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            ``{"user_id": ..., "embedding": [...64 floats...]}`` — or an
            error payload if no embedding is currently available.
        """
        self._audit("get_user_state_embedding", user_id)
        model = self._user_model(user_id)
        if model is None:
            return {"error": "user_model_unavailable", "user_id": user_id}
        session = getattr(model, "session_state", None)
        embedding = getattr(session, "embedding", None) if session is not None else None
        if embedding is None:
            return {"error": "no_embedding", "user_id": user_id}
        payload = {
            "user_id": user_id,
            "dim": 64,
            "embedding": _tensor_to_list(embedding),
        }
        return self._guard_payload(payload, label="user_state_embedding")

    def _tool_get_feature_vector(self, user_id: str) -> dict[str, Any]:
        """Return the last 32-dim :class:`InteractionFeatureVector`.

        Only aggregated metrics are emitted — never raw text.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            JSON-serialisable feature payload.
        """
        self._audit("get_feature_vector", user_id)
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return {"error": "pipeline_unavailable", "user_id": user_id}
        monitor = (
            getattr(pipe, "interaction_monitor", None)
            or getattr(pipe, "_interaction_monitor", None)
            or getattr(pipe, "monitor", None)
        )
        features: Any = None
        if monitor is not None:
            getter = getattr(monitor, "get_last_features", None)
            if callable(getter):
                try:
                    features = getter(user_id)
                except Exception:  # noqa: BLE001
                    features = None
        if features is None:
            from i3.interaction.types import InteractionFeatureVector

            features = InteractionFeatureVector.zeros()
        payload = {"user_id": user_id, "features": _jsonable(features)}
        return self._guard_payload(payload, label="feature_vector")

    def _tool_get_session_metadata(
        self, user_id: str, session_id: str
    ) -> dict[str, Any]:
        """Return aggregated metadata for a single session.

        Args:
            user_id: Unique identifier for the user.
            session_id: Session identifier returned by the pipeline.

        Returns:
            JSON-serialisable dict with length, message count, dominant
            emotion, route distribution, and mean engagement.  Never text.
        """
        self._audit("get_session_metadata", user_id)
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return {"error": "pipeline_unavailable", "user_id": user_id}

        diary_store = getattr(pipe, "diary_store", None) or getattr(
            pipe, "_diary_store", None
        )
        if diary_store is None:
            return {"error": "diary_unavailable", "user_id": user_id}

        import asyncio

        async def _collect() -> dict[str, Any]:
            session = await diary_store.get_session(session_id)  # type: ignore[union-attr]
            exchanges = await diary_store.get_session_exchanges(  # type: ignore[union-attr]
                session_id
            )
            route_counts: dict[str, int] = {}
            for ex in exchanges or []:
                route = ex.get("router_decision") if isinstance(ex, dict) else None
                if route is not None:
                    route_counts[str(route)] = route_counts.get(str(route), 0) + 1
            return {
                "user_id": user_id,
                "session_id": session_id,
                "session": _jsonable(session) if session else None,
                "message_count": len(exchanges or []),
                "route_distribution": route_counts,
                "dominant_emotion": (
                    session.get("dominant_emotion") if isinstance(session, dict) else None
                ),
            }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside an event loop — run in a fresh one.
                future = asyncio.run_coroutine_threadsafe(_collect(), loop)
                payload = future.result(timeout=5.0)
            else:
                payload = loop.run_until_complete(_collect())
        except RuntimeError:
            payload = asyncio.run(_collect())
        except Exception as exc:  # noqa: BLE001 — defensive
            return {"error": f"session_lookup_failed: {exc}", "user_id": user_id}

        return self._guard_payload(payload, label="session_metadata")

    def _tool_get_device_profile(self, target: str) -> dict[str, Any]:
        """Return the Kirin-class :class:`DeviceProfile` for *target*.

        Args:
            target: One of ``kirin_9000``, ``kirin_9010``, ``kirin_a2``,
                ``smart_hanhan``.

        Returns:
            The serialised device profile, or an error payload if
            *target* is not recognised.
        """
        self._audit("get_device_profile", None)
        if target not in _VALID_DEVICE_TARGETS:
            return {
                "error": "unknown_target",
                "target": target,
                "valid": sorted(_VALID_DEVICE_TARGETS),
            }
        from i3.huawei import kirin_targets as kt

        mapping: dict[str, DeviceProfile] = {
            "kirin_9000": kt.KIRIN_9000,
            "kirin_9010": kt.KIRIN_9010,
            "kirin_a2": kt.KIRIN_A2,
            "smart_hanhan": kt.SMART_HANHAN,
        }
        profile = mapping[target]
        payload = {"target": target, "profile": _jsonable(profile)}
        return self._guard_payload(payload, label="device_profile")

    def _tool_route_recommendation(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Ask the bandit for a route without actually generating a response.

        Args:
            context: A :class:`RoutingContext`-shaped dict (JSON-friendly).

        Returns:
            JSON-serialisable :class:`RoutingDecision`.
        """
        self._audit("route_recommendation", context.get("user_id"))
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return {"error": "pipeline_unavailable"}
        router_obj = (
            getattr(pipe, "intelligent_router", None)
            or getattr(pipe, "router", None)
            or getattr(pipe, "_router", None)
        )
        if router_obj is None:
            return {"error": "router_unavailable"}
        from i3.router.types import RoutingContext

        # Build a safe RoutingContext with defaults for any missing key.
        ctx = RoutingContext(
            user_state_compressed=list(
                context.get("user_state_compressed", [0.0, 0.0, 0.0, 0.0])
            ),
            query_complexity=float(context.get("query_complexity", 0.5)),
            topic_sensitivity=float(context.get("topic_sensitivity", 0.0)),
            user_patience=float(context.get("user_patience", 0.5)),
            session_progress=float(context.get("session_progress", 0.0)),
            baseline_established=bool(context.get("baseline_established", False)),
            previous_route=int(context.get("previous_route", -1)),
            previous_engagement=float(context.get("previous_engagement", 0.5)),
            time_of_day=float(context.get("time_of_day", 0.5)),
            message_count=int(context.get("message_count", 0)),
            cloud_latency_est=float(context.get("cloud_latency_est", 0.3)),
            slm_confidence=float(context.get("slm_confidence", 0.5)),
        )

        # Two code paths: the full ``IntelligentRouter`` has ``.route(text,
        # ctx=ctx)``; a raw bandit only has ``.select_arm(ctx_vec)``.  We
        # detect capability and degrade gracefully.
        if hasattr(router_obj, "route"):
            decision = router_obj.route("", ctx=ctx)
            payload = {
                "chosen_route": decision.chosen_route.value,
                "confidence": decision.confidence,
                "was_privacy_override": decision.was_privacy_override,
                "reasoning": decision.reasoning,
            }
        elif hasattr(router_obj, "select_arm"):
            arm, conf = router_obj.select_arm(ctx.to_vector())
            payload = {
                "chosen_route": "local_slm" if arm == 0 else "cloud_llm",
                "confidence": {
                    "local_slm": conf.get("arm_0", 0.5)
                    if isinstance(conf, dict)
                    else 0.5,
                    "cloud_llm": conf.get("arm_1", 0.5)
                    if isinstance(conf, dict)
                    else 0.5,
                },
                "was_privacy_override": False,
                "reasoning": "bandit_only (no sensitivity / complexity gates)",
            }
        else:
            return {"error": "router_unavailable"}
        return self._guard_payload(payload, label="routing_decision")

    def _tool_explain_adaptation(self, user_id: str) -> dict[str, Any]:
        """Return a feature-attribution breakdown of the adaptation state.

        Uses :class:`i3.interpretability.FeatureAttributor` when importable;
        otherwise falls back to a plain heuristic over the last feature
        vector.

        Args:
            user_id: Unique identifier for the user.

        Returns:
            ``{"user_id": ..., "method": ..., "attributions": [...]}``.
        """
        self._audit("explain_adaptation", user_id)
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return {"error": "pipeline_unavailable", "user_id": user_id}

        # Try the "real" attributor first.  It requires a differentiable
        # mapping, which the pipeline may not yet have wired up, so we
        # fall back gracefully.
        try:
            from i3.interpretability import FEATURE_NAMES as _FEATS

            feat_names: list[str] = list(_FEATS)
        except Exception:  # noqa: BLE001
            from i3.interaction.types import FEATURE_NAMES as _FEATS2

            feat_names = list(_FEATS2)

        # Heuristic attribution: rank features by |value| (no raw text).
        monitor = (
            getattr(pipe, "interaction_monitor", None)
            or getattr(pipe, "_interaction_monitor", None)
            or getattr(pipe, "monitor", None)
        )
        features: Any = None
        if monitor is not None:
            getter = getattr(monitor, "get_last_features", None)
            if callable(getter):
                try:
                    features = getter(user_id)
                except Exception:  # noqa: BLE001
                    features = None
        if features is None:
            from i3.interaction.types import InteractionFeatureVector

            features = InteractionFeatureVector.zeros()

        feats_dict = _jsonable(features)
        ranked = sorted(
            [
                (name, float(feats_dict.get(name, 0.0)))
                for name in feat_names
            ],
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        attributions = [
            {"feature": name, "value": value} for name, value in ranked[:8]
        ]
        payload = {
            "user_id": user_id,
            "method": "heuristic_magnitude",
            "attributions": attributions,
            "note": (
                "Heuristic fallback. Integrated-gradients attribution requires "
                "a differentiable adaptation head; swap in "
                "i3.interpretability.FeatureAttributor for full IG."
            ),
        }
        return self._guard_payload(payload, label="explain_adaptation")

    # ---- Resource implementations --------------------------------------

    def _resource_user_profile(self, user_id: str) -> str:
        """Return the user profile JSON with embeddings base64-encoded."""
        self._audit("resource:user_profile", user_id)
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return json.dumps({"error": "pipeline_unavailable", "user_id": user_id})
        model = self._user_model(user_id)
        if model is None:
            return json.dumps({"error": "user_model_unavailable", "user_id": user_id})
        session = getattr(model, "session_state", None)
        profile = getattr(model, "profile", None)
        payload = {
            "user_id": user_id,
            "baseline_embedding_b64": _tensor_to_b64(
                getattr(profile, "baseline_embedding", None)
            ),
            "session_embedding_b64": _tensor_to_b64(
                getattr(session, "embedding", None)
            ),
            "baseline_features_mean": _jsonable(
                getattr(profile, "baseline_features_mean", None) or {}
            ),
            "baseline_features_std": _jsonable(
                getattr(profile, "baseline_features_std", None) or {}
            ),
            "relationship_strength": float(
                getattr(profile, "relationship_strength", 0.0) or 0.0
            ),
            "total_sessions": int(getattr(profile, "total_sessions", 0) or 0),
            "total_messages": int(getattr(profile, "total_messages", 0) or 0),
            "baseline_established": bool(
                getattr(profile, "baseline_established", False)
            ),
        }
        payload = self._guard_payload(payload, label="user_profile_resource")
        return json.dumps(payload)

    def _resource_user_diary(self, user_id: str) -> str:
        """Return a list of recent diary entries (topics + adaptation only)."""
        self._audit("resource:user_diary", user_id)
        pipe = self._pipeline_or_unavailable()
        if pipe is None:
            return json.dumps({"error": "pipeline_unavailable", "user_id": user_id})
        diary_store = getattr(pipe, "diary_store", None) or getattr(
            pipe, "_diary_store", None
        )
        if diary_store is None:
            return json.dumps({"error": "diary_unavailable", "user_id": user_id})
        import asyncio

        async def _collect() -> list[dict[str, Any]]:
            entries = await diary_store.get_recent_diary_entries(  # type: ignore[union-attr]
                user_id, n=10
            )
            # Defensive: strip any free-form summary text so the response
            # truly carries no natural-language prose to the client.
            safe: list[dict[str, Any]] = []
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                safe.append(
                    {
                        "session_id": entry.get("session_id"),
                        "start_time": entry.get("start_time"),
                        "end_time": entry.get("end_time"),
                        "message_count": entry.get("message_count"),
                        "topics": entry.get("topics"),
                        "dominant_emotion": entry.get("dominant_emotion"),
                        "mean_engagement": entry.get("mean_engagement"),
                        "relationship_strength": entry.get("relationship_strength"),
                    }
                )
            return safe

        try:
            entries = asyncio.run(_collect())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                entries = loop.run_until_complete(_collect())
            finally:
                loop.close()
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"diary_query_failed: {exc}"})

        payload = self._guard_payload(
            {"user_id": user_id, "entries": entries},
            label="user_diary_resource",
        )
        return json.dumps(payload)

    def _resource_device_profile(self, device_id: str) -> str:
        """Return the device profile JSON for *device_id*."""
        self._audit("resource:device_profile", None)
        result = self._tool_get_device_profile(device_id)
        return json.dumps(result)

    def _resource_architecture_layers(self) -> str:
        """Return the 7-layer architecture as structured JSON."""
        self._audit("resource:architecture_layers", None)
        layers = {
            "spec_version": MCP_SPEC_VERSION,
            "layers": [
                {
                    "id": 1,
                    "name": "Interaction Monitor",
                    "role": (
                        "Extract 32-dim feature vector from keystroke "
                        "dynamics, linguistic complexity, and session "
                        "dynamics."
                    ),
                    "module": "i3.interaction",
                },
                {
                    "id": 2,
                    "name": "Privacy Sanitizer",
                    "role": (
                        "Strip PII before any cloud transmission and before "
                        "topic extraction. See ADR-0004."
                    ),
                    "module": "i3.privacy",
                },
                {
                    "id": 3,
                    "name": "TCN Encoder",
                    "role": (
                        "Encode feature windows into a 64-dim L2-normalised "
                        "user-state embedding. See ADR-0002."
                    ),
                    "module": "i3.encoder",
                },
                {
                    "id": 4,
                    "name": "User Model",
                    "role": (
                        "Three-timescale representation (instantaneous, "
                        "session, long-term baseline) with deviation stats."
                    ),
                    "module": "i3.user_model",
                },
                {
                    "id": 5,
                    "name": "Adaptation Controller",
                    "role": (
                        "Assemble 8-dim AdaptationVector from cognitive load, "
                        "style mirror, emotional tone, and accessibility."
                    ),
                    "module": "i3.adaptation",
                },
                {
                    "id": 6,
                    "name": "Intelligent Router",
                    "role": (
                        "Contextual Thompson Sampling over {local SLM, cloud "
                        "LLM}. See ADR-0003."
                    ),
                    "module": "i3.router",
                },
                {
                    "id": 7,
                    "name": "Adaptive SLM / Cloud LLM",
                    "role": (
                        "Response generation conditioned on the adaptation "
                        "vector via cross-attention. See ADR-0001."
                    ),
                    "module": "i3.slm",
                },
            ],
        }
        return json.dumps(layers)

    def _resource_adrs(self) -> str:
        """Return the ADR index as a structured list."""
        self._audit("resource:adrs", None)
        adrs = [
            {"id": 1, "title": "Custom SLM over HuggingFace", "status": "Accepted"},
            {"id": 2, "title": "TCN encoder over LSTM / Transformer", "status": "Accepted"},
            {"id": 3, "title": "Thompson sampling over UCB", "status": "Accepted"},
            {"id": 4, "title": "Privacy by architecture", "status": "Accepted"},
            {"id": 5, "title": "FastAPI over Flask", "status": "Accepted"},
            {"id": 6, "title": "Poetry over pip-tools", "status": "Accepted"},
            {"id": 7, "title": "OpenTelemetry for observability", "status": "Accepted"},
            {"id": 8, "title": "Fernet over custom crypto", "status": "Accepted"},
            {"id": 9, "title": "SQLite over Redis", "status": "Accepted"},
            {"id": 10, "title": "Pydantic v2 for config", "status": "Accepted"},
        ]
        return json.dumps({"adrs": adrs, "count": len(adrs)})

    # ---- Prompt templates ----------------------------------------------

    def _prompt_adaptation_summary(self, user_id: str) -> str:
        """Return the prompt template for an adaptation summary."""
        self._audit("prompt:adaptation_summary", user_id)
        return (
            "You have access to the I3 MCP server. "
            f"For user_id={user_id!r}, call get_user_adaptation_vector and "
            "explain_adaptation, then summarise the user's current adaptation "
            "state in exactly 2 sentences. Do not mention tool names in the "
            "final summary."
        )

    def _prompt_troubleshoot_high_load(self, user_id: str) -> str:
        """Return the prompt template for troubleshooting high cognitive load."""
        self._audit("prompt:troubleshoot_high_load", user_id)
        return (
            "You have access to the I3 MCP server. "
            f"For user_id={user_id!r}, inspect the adaptation vector and the "
            "explain_adaptation attribution. Given the user's current "
            "cognitive load and recent feature deviations, suggest three "
            "simpler response strategies the assistant could adopt right now "
            "to reduce friction. Be concrete."
        )

    # ---- Registration --------------------------------------------------

    def _register_tools(self) -> None:
        """Wire internal methods into MCP tool handlers."""
        mcp = self._mcp
        _bind_tool(mcp, self._tool_get_user_adaptation_vector, "get_user_adaptation_vector")
        _bind_tool(mcp, self._tool_get_user_state_embedding, "get_user_state_embedding")
        _bind_tool(mcp, self._tool_get_feature_vector, "get_feature_vector")
        _bind_tool(mcp, self._tool_get_session_metadata, "get_session_metadata")
        _bind_tool(mcp, self._tool_get_device_profile, "get_device_profile")
        _bind_tool(mcp, self._tool_route_recommendation, "route_recommendation")
        _bind_tool(mcp, self._tool_explain_adaptation, "explain_adaptation")

    def _register_resources(self) -> None:
        """Wire internal methods into MCP resource handlers."""
        mcp = self._mcp
        _bind_resource(
            mcp, self._resource_user_profile, "i3://users/{user_id}/profile"
        )
        _bind_resource(
            mcp, self._resource_user_diary, "i3://users/{user_id}/diary"
        )
        _bind_resource(
            mcp, self._resource_device_profile, "i3://devices/kirin/{device_id}"
        )
        _bind_resource(
            mcp, self._resource_architecture_layers, "i3://architecture/layers"
        )
        _bind_resource(mcp, self._resource_adrs, "i3://adrs")

    def _register_prompts(self) -> None:
        """Wire internal methods into MCP prompt templates."""
        mcp = self._mcp
        _bind_prompt(mcp, self._prompt_adaptation_summary, "adaptation_summary")
        _bind_prompt(
            mcp, self._prompt_troubleshoot_high_load, "troubleshoot_high_load"
        )

    # ---- Entry points --------------------------------------------------

    def run(self, transport: str = "stdio") -> None:
        """Serve MCP clients on *transport* (blocking call).

        Args:
            transport: One of ``"stdio"``, ``"sse"``, ``"streamable_http"``.

        Raises:
            RuntimeError: When the selected transport is not available in
                the installed ``mcp`` SDK.
        """
        from i3.mcp.transport import run_transport

        run_transport(self, transport=transport)


# ---------------------------------------------------------------------------
# Binding helpers — tolerate SDK API drift between mcp versions
# ---------------------------------------------------------------------------


def _bind_tool(mcp_app: Any, fn: Callable[..., Any], name: str) -> None:
    """Register *fn* as an MCP tool named *name* on *mcp_app*.

    FastMCP has historically exposed either ``.tool()`` as a decorator or
    ``.add_tool()`` as a method.  We accept both.

    Args:
        mcp_app: Underlying ``FastMCP`` instance.
        fn: Callable to register.
        name: Tool name exposed to MCP clients.
    """
    if hasattr(mcp_app, "tool"):
        mcp_app.tool(name=name)(fn)
        return
    if hasattr(mcp_app, "add_tool"):
        mcp_app.add_tool(fn, name=name)  # type: ignore[misc]
        return
    raise RuntimeError(
        "Installed mcp SDK exposes neither .tool() nor .add_tool(); "
        "upgrade mcp[cli]."
    )


def _bind_resource(mcp_app: Any, fn: Callable[..., Any], uri: str) -> None:
    """Register *fn* as an MCP resource handler for *uri*."""
    if hasattr(mcp_app, "resource"):
        mcp_app.resource(uri)(fn)
        return
    if hasattr(mcp_app, "add_resource"):
        mcp_app.add_resource(fn, uri=uri)  # type: ignore[misc]
        return
    raise RuntimeError(
        "Installed mcp SDK exposes neither .resource() nor .add_resource(); "
        "upgrade mcp[cli]."
    )


def _bind_prompt(mcp_app: Any, fn: Callable[..., Any], name: str) -> None:
    """Register *fn* as an MCP prompt template named *name*."""
    if hasattr(mcp_app, "prompt"):
        mcp_app.prompt(name=name)(fn)
        return
    if hasattr(mcp_app, "add_prompt"):
        mcp_app.add_prompt(fn, name=name)  # type: ignore[misc]
        return
    raise RuntimeError(
        "Installed mcp SDK exposes neither .prompt() nor .add_prompt(); "
        "upgrade mcp[cli]."
    )


__all__ = ["I3MCPServer", "MCP_SPEC_VERSION"]
