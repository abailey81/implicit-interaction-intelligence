"""Tests for :mod:`i3.cloud.pydantic_ai_adapter`.

Reference: Pydantic AI documentation (2024-12 release).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")


def test_module_imports_with_soft_import_path() -> None:
    """The adapter module exposes the typed schema and client class."""
    from i3.cloud import pydantic_ai_adapter

    assert pydantic_ai_adapter.is_available() is True
    assert hasattr(pydantic_ai_adapter, "AdaptiveResponse")
    assert hasattr(pydantic_ai_adapter, "PydanticAICloudClient")
    # Schema must define the agreed fields
    fields = set(pydantic_ai_adapter.AdaptiveResponse.model_fields.keys())
    assert fields == {"text", "tone", "estimated_complexity", "used_simplification"}


def test_adapter_class_constructs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Client constructs with patched model/agent factories."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000")

    with patch("i3.cloud.pydantic_ai_adapter.AnthropicModel") as mock_model_cls, patch(
        "i3.cloud.pydantic_ai_adapter.Agent"
    ) as mock_agent_cls:
        mock_model_cls.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        from i3.cloud.pydantic_ai_adapter import PydanticAICloudClient

        client = PydanticAICloudClient(model_name="claude-sonnet-4-5")
        assert client is not None
        assert client.model_name == "claude-sonnet-4-5"
        mock_model_cls.assert_called_once()
        mock_agent_cls.assert_called_once()


@pytest.mark.asyncio
async def test_happy_path_call_with_mocked_llm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``generate()`` returns a validated :class:`AdaptiveResponse`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000")

    from i3.cloud.pydantic_ai_adapter import AdaptiveResponse

    expected = AdaptiveResponse(
        text="Hi there.",
        tone="warm",
        estimated_complexity=0.2,
        used_simplification=True,
    )

    class _Result:
        data = expected

    async def _fake_run(_prompt: str) -> _Result:  # noqa: ANN001
        return _Result()

    with patch("i3.cloud.pydantic_ai_adapter.AnthropicModel") as mock_model_cls, patch(
        "i3.cloud.pydantic_ai_adapter.Agent"
    ) as mock_agent_cls:
        mock_model_cls.return_value = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run = _fake_run
        mock_agent_cls.return_value = mock_agent

        from i3.cloud.pydantic_ai_adapter import PydanticAICloudClient

        client = PydanticAICloudClient()
        out = await client.generate("Say hi")
        assert out is expected
        assert out.tone == "warm"
        assert out.used_simplification is True
