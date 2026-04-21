"""Tests for :mod:`i3.cloud.instructor_adapter`.

Reference: Jason Liu, *Instructor: Structured Outputs for LLMs* (2024).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

instructor = pytest.importorskip("instructor")
pytest.importorskip("anthropic")


class _Reply(BaseModel):
    text: str
    score: float


def test_module_imports_with_soft_import_path() -> None:
    """The adapter module exposes the public class and availability flag."""
    from i3.cloud import instructor_adapter

    assert instructor_adapter.is_available() is True
    assert hasattr(instructor_adapter, "InstructorAdapter")


def test_adapter_class_constructs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Constructor patches the Anthropic client through Instructor."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000")

    with patch("i3.cloud.instructor_adapter._anthropic") as mock_anthropic_mod, patch(
        "i3.cloud.instructor_adapter.instructor"
    ) as mock_instructor_mod:
        mock_raw = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_raw
        mock_instructor_mod.from_anthropic.return_value = MagicMock()

        from i3.cloud.instructor_adapter import InstructorAdapter

        adapter = InstructorAdapter(model_name="claude-sonnet-4-5")
        assert adapter is not None
        mock_anthropic_mod.Anthropic.assert_called_once()
        mock_instructor_mod.from_anthropic.assert_called_once_with(mock_raw)


def test_happy_path_call_with_mocked_llm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``structured_generate`` returns the Pydantic instance created by the stub."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000")

    expected = _Reply(text="done", score=0.97)

    with patch("i3.cloud.instructor_adapter._anthropic") as mock_anthropic_mod, patch(
        "i3.cloud.instructor_adapter.instructor"
    ) as mock_instructor_mod:
        mock_anthropic_mod.Anthropic.return_value = MagicMock()
        wrapped = MagicMock()
        wrapped.messages.create.return_value = expected
        mock_instructor_mod.from_anthropic.return_value = wrapped

        from i3.cloud.instructor_adapter import InstructorAdapter

        adapter = InstructorAdapter(max_retries=1)
        out = adapter.structured_generate(
            "Give me a score.", _Reply, system="Return the score."
        )
        assert out is expected
        wrapped.messages.create.assert_called_once()
        kwargs = wrapped.messages.create.call_args.kwargs
        assert kwargs["response_model"] is _Reply
        assert kwargs["system"] == "Return the score."
        assert kwargs["max_retries"] == 1
