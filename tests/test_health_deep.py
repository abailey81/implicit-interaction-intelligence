"""Iter 56 — /api/health/deep endpoint contract tests.

Exercises the deep-health aggregator without booting FastAPI by
constructing a minimal Request stand-in.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.responses import JSONResponse


def _fake_request(pipeline=None) -> SimpleNamespace:
    """Build the smallest Request shape the helper inspects."""
    app_state = SimpleNamespace(pipeline=pipeline)
    app = SimpleNamespace(state=app_state, version="1.0.0-test")
    return SimpleNamespace(app=app)


def test_health_deep_no_pipeline():
    from server.routes_health import health_deep
    req = _fake_request(pipeline=None)
    resp = asyncio.get_event_loop().run_until_complete(health_deep(req))
    assert isinstance(resp, JSONResponse)
    body = resp.body.decode("utf-8")
    import json
    data = json.loads(body)
    # Sections always present
    for k in ("status", "version", "uptime_s", "slm_v2", "encoder",
              "intent", "cloud", "privacy", "cascade", "checkpoints"):
        assert k in data, f"missing {k!r} in /api/health/deep"
    assert data["status"] == "ok"
    # Subsystem booleans must be JSON-serialisable
    assert isinstance(data["intent"]["qwen_adapter_present"], bool)
    assert isinstance(data["intent"]["gemini_api_key_set"], bool)
    assert isinstance(data["privacy"]["encryption_key_set"], bool)


def test_health_deep_intent_section_shape():
    from server.routes_health import health_deep
    req = _fake_request(pipeline=None)
    resp = asyncio.get_event_loop().run_until_complete(health_deep(req))
    import json
    data = json.loads(resp.body.decode("utf-8"))
    intent = data["intent"]
    for k in ("qwen_adapter_present", "qwen_metrics_present",
              "gemini_plan_present", "gemini_api_key_set"):
        assert k in intent, f"missing intent.{k}"


def test_health_deep_cloud_providers_listed():
    """Iter 51 wires anthropic/google/mistral/etc. into the registry."""
    from server.routes_health import health_deep
    req = _fake_request(pipeline=None)
    resp = asyncio.get_event_loop().run_until_complete(health_deep(req))
    import json
    data = json.loads(resp.body.decode("utf-8"))
    providers = data["cloud"]["registered_providers"]
    assert isinstance(providers, list)
    # Must include at least the core 4
    for p in ("anthropic", "openai", "google"):
        assert p in providers, f"provider {p!r} not in {providers}"


def test_health_deep_with_minimal_pipeline_stub():
    """Verify the helper degrades gracefully when only a partial
    pipeline shape is present (no profiling, no cascade tracker)."""
    from server.routes_health import health_deep
    pipeline = SimpleNamespace()  # nothing implemented
    req = _fake_request(pipeline=pipeline)
    resp = asyncio.get_event_loop().run_until_complete(health_deep(req))
    import json
    data = json.loads(resp.body.decode("utf-8"))
    # Should still return; cascade is empty dict, profiling absent
    assert isinstance(data["cascade"], dict)
