"""Unit tests for the I³ MCP server.

These tests exercise the :class:`i3.mcp.server.I3MCPServer` handlers in
isolation.  The MCP SDK is required — tests are skipped automatically if
``mcp`` is not importable.  The I³ pipeline is replaced with a tiny
hand-rolled mock so the tests do not depend on any I/O or training
artefacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from i3.adaptation.types import AdaptationVector, StyleVector
from i3.interaction.types import InteractionFeatureVector

# Skip every test in this module when the mcp SDK is unavailable.
mcp = pytest.importorskip(
    "mcp", reason="MCP SDK not installed — `pip install mcp[cli]` to enable."
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _StubDiaryStore:
    """Minimal async DiaryStore stand-in.

    Provides the two methods the MCP server touches:
    :meth:`get_session` and :meth:`get_session_exchanges`, plus
    :meth:`get_recent_diary_entries`.
    """

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return {
            "session_id": session_id,
            "start_time": "2026-04-22T10:00:00+00:00",
            "end_time": "2026-04-22T10:15:00+00:00",
            "message_count": 3,
            "dominant_emotion": "neutral",
            "mean_engagement": 0.62,
            "topics": ["learning", "python"],
        }

    async def get_session_exchanges(self, session_id: str) -> list[dict[str, Any]]:
        return [
            {"router_decision": "local_slm", "latency_ms": 120.0},
            {"router_decision": "cloud_llm", "latency_ms": 340.0},
            {"router_decision": "local_slm", "latency_ms": 110.0},
        ]

    async def get_recent_diary_entries(
        self, user_id: str, n: int = 5
    ) -> list[dict[str, Any]]:
        return [
            {
                "session_id": "sess_1",
                "start_time": "2026-04-22T10:00:00+00:00",
                "end_time": "2026-04-22T10:15:00+00:00",
                "message_count": 3,
                "topics": ["python"],
                "dominant_emotion": "neutral",
                "mean_engagement": 0.62,
                "relationship_strength": 0.18,
            }
        ]


@dataclass
class _StubProfile:
    """Stand-in for :class:`UserProfile` — scalar + dict fields only."""

    user_id: str = "u_test"
    baseline_embedding: Any = None
    baseline_features_mean: dict[str, float] = field(default_factory=dict)
    baseline_features_std: dict[str, float] = field(default_factory=dict)
    total_sessions: int = 5
    total_messages: int = 42
    relationship_strength: float = 0.33
    long_term_style: dict[str, float] = field(default_factory=dict)
    baseline_established: bool = True


@dataclass
class _StubSession:
    """Stand-in for :class:`SessionState`."""

    embedding: Any = None
    message_count: int = 3
    dominant_emotion: str = "neutral"


class _StubUserModel:
    def __init__(self) -> None:
        import torch

        self.profile = _StubProfile(
            baseline_embedding=torch.zeros(64),
        )
        self.session_state = _StubSession(embedding=torch.ones(64) * 0.1)


class _StubMonitor:
    def get_last_features(self, user_id: str) -> InteractionFeatureVector:
        v = InteractionFeatureVector.zeros()
        v.backspace_ratio = 0.42
        v.iki_deviation = 1.1
        v.length_trend = -0.3
        return v


class _StubAdaptationController:
    def __init__(self) -> None:
        self.last_vector = AdaptationVector(
            cognitive_load=0.3,
            style_mirror=StyleVector(
                formality=0.5, verbosity=0.7, emotionality=0.4, directness=0.4
            ),
            emotional_tone=0.2,
            accessibility=0.0,
        )


class _StubRouter:
    """Stub router exposing just the one method MCP calls."""

    def route(self, text: str, ctx: Any) -> Any:
        from i3.router.types import RouteChoice, RoutingDecision

        return RoutingDecision(
            chosen_route=RouteChoice.LOCAL_SLM,
            confidence={"local_slm": 0.8, "cloud_llm": 0.2},
            context=ctx,
            was_privacy_override=False,
            reasoning="stub",
        )


class _StubPipeline:
    def __init__(self) -> None:
        self._user_models: dict[str, _StubUserModel] = {"u_test": _StubUserModel()}
        self.interaction_monitor = _StubMonitor()
        self.adaptation_controller = _StubAdaptationController()
        self.router = _StubRouter()
        self.diary_store = _StubDiaryStore()

    def get_user_model(self, user_id: str) -> _StubUserModel | None:
        return self._user_models.get(user_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server():
    """Construct an I3MCPServer with the stub pipeline attached."""
    from i3.mcp.server import I3MCPServer

    return I3MCPServer(pipeline=_StubPipeline())


@pytest.fixture
def unbound_server():
    """Construct an I3MCPServer without any pipeline attached."""
    from i3.mcp.server import I3MCPServer

    return I3MCPServer(pipeline=None)


# ---------------------------------------------------------------------------
# Tests: basic construction and metadata
# ---------------------------------------------------------------------------


def test_server_name_and_version(server) -> None:
    """Server advertises the canonical name and a non-empty version."""
    assert server.name == "i3-hmi-companion"
    assert isinstance(server.version, str) and len(server.version) > 0


def test_server_requires_mcp_sdk(monkeypatch) -> None:
    """Constructing the server raises a clear RuntimeError if mcp is absent."""
    import i3.mcp.server as srv

    monkeypatch.setattr(srv, "_MCP_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="pip install mcp"):
        srv.I3MCPServer(pipeline=None)


# ---------------------------------------------------------------------------
# Tests: tool handlers
# ---------------------------------------------------------------------------


def test_tool_get_device_profile_happy_path(server) -> None:
    """Kirin 9000 profile is returned with the canonical fields."""
    payload = server._tool_get_device_profile("kirin_9000")
    assert payload["target"] == "kirin_9000"
    profile = payload["profile"]
    for field_name in ("name", "device_class", "ram_mb", "int8_tops"):
        assert field_name in profile
    assert profile["device_class"] == "phone"


def test_tool_get_device_profile_rejects_unknown(server) -> None:
    """Unknown device targets return a structured error, not a crash."""
    payload = server._tool_get_device_profile("pentium_iii")
    assert payload["error"] == "unknown_target"
    assert "kirin_9000" in payload["valid"]


def test_tool_get_user_adaptation_vector_unbound(unbound_server) -> None:
    """No pipeline => explicit pipeline_unavailable error."""
    payload = unbound_server._tool_get_user_adaptation_vector("u_test")
    assert payload == {"error": "pipeline_unavailable", "user_id": "u_test"}


def test_tool_get_user_adaptation_vector_happy_path(server) -> None:
    """Adaptation vector includes every expected axis."""
    payload = server._tool_get_user_adaptation_vector("u_test")
    assert payload["user_id"] == "u_test"
    vec = payload["vector"]
    for key in ("cognitive_load", "style_mirror", "emotional_tone", "accessibility"):
        assert key in vec
    assert set(vec["style_mirror"]) == {
        "formality",
        "verbosity",
        "emotionality",
        "directness",
    }


def test_tool_get_user_state_embedding_shape(server) -> None:
    """Embedding has length 64 and the ``dim`` field advertises it."""
    payload = server._tool_get_user_state_embedding("u_test")
    assert payload["dim"] == 64
    assert len(payload["embedding"]) == 64


def test_tool_get_feature_vector_has_all_32_fields(server) -> None:
    """Feature vector payload carries the full 32-field aggregate."""
    payload = server._tool_get_feature_vector("u_test")
    assert payload["user_id"] == "u_test"
    assert "backspace_ratio" in payload["features"]
    assert payload["features"]["backspace_ratio"] == pytest.approx(0.42)
    assert len(payload["features"]) == 32


def test_tool_route_recommendation(server) -> None:
    """Router returns a structured decision without needing raw text."""
    payload = server._tool_route_recommendation(
        {
            "user_state_compressed": [0.0, 0.1, -0.1, 0.0],
            "query_complexity": 0.4,
            "topic_sensitivity": 0.0,
        }
    )
    assert payload["chosen_route"] in {"local_slm", "cloud_llm"}
    assert set(payload["confidence"]) == {"local_slm", "cloud_llm"}


def test_tool_explain_adaptation_ranks_features(server) -> None:
    """Explanation returns ranked attributions (magnitude fallback)."""
    payload = server._tool_explain_adaptation("u_test")
    assert payload["method"] == "heuristic_magnitude"
    attrs = payload["attributions"]
    assert len(attrs) == 8
    # Sorted by absolute magnitude descending.
    mags = [abs(a["value"]) for a in attrs]
    assert mags == sorted(mags, reverse=True)


# ---------------------------------------------------------------------------
# Tests: resource handlers
# ---------------------------------------------------------------------------


def test_resource_architecture_layers(server) -> None:
    """Architecture resource enumerates all 7 layers."""
    payload = json.loads(server._resource_architecture_layers())
    assert len(payload["layers"]) == 7
    ids = [layer["id"] for layer in payload["layers"]]
    assert ids == sorted(ids)


def test_resource_adrs(server) -> None:
    """ADR resource carries a non-empty, numbered index."""
    payload = json.loads(server._resource_adrs())
    assert payload["count"] >= 10
    assert all(isinstance(item["id"], int) for item in payload["adrs"])


def test_resource_user_profile_base64_embedding(server) -> None:
    """User profile resource emits a base64-encoded embedding, not raw floats."""
    payload = json.loads(server._resource_user_profile("u_test"))
    assert "baseline_embedding_b64" in payload
    assert isinstance(payload["baseline_embedding_b64"], str)
    # The embeddings themselves never leak as plain arrays.
    assert "baseline_embedding" not in payload


# ---------------------------------------------------------------------------
# Tests: privacy guard
# ---------------------------------------------------------------------------


def test_guard_payload_refuses_pii(server) -> None:
    """If a payload somehow contains PII, the guard raises rather than emitting it."""
    with pytest.raises(RuntimeError, match="Privacy invariant violated"):
        server._guard_payload(
            {"note": "reach me at alice@example.com"}, label="test"
        )


def test_guard_payload_passes_clean(server) -> None:
    """Clean payloads pass through unchanged."""
    payload = {"cognitive_load": 0.3, "accessibility": 0.0}
    assert server._guard_payload(payload, label="test") is payload


# ---------------------------------------------------------------------------
# Tests: audit logging
# ---------------------------------------------------------------------------


def test_audit_logs_user_id_only(server, caplog) -> None:
    """Audit log line contains tool name + user_id, not payload fields."""
    import logging

    with caplog.at_level(logging.INFO, logger="i3.mcp.server"):
        server._tool_get_user_adaptation_vector("u_test")
    relevant = [
        rec for rec in caplog.records if "mcp_call" in rec.getMessage()
    ]
    assert relevant, "no mcp_call audit log line emitted"
    msg = relevant[0].getMessage()
    assert "user_id=u_test" in msg
    assert "get_user_adaptation_vector" in msg
