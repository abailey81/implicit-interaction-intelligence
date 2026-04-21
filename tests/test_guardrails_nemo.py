"""Tests for :mod:`i3.cloud.guardrails_nemo`.

Reference: Rebedea et al. 2023, "NeMo Guardrails."  EMNLP 2023 sys-demo.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

nemoguardrails = pytest.importorskip("nemoguardrails")


def test_module_imports_with_soft_import_path() -> None:
    """The wrapper module exposes its public symbols."""
    from i3.cloud import guardrails_nemo

    assert guardrails_nemo.is_available() is True
    assert hasattr(guardrails_nemo, "GuardrailedCloudClient")
    assert hasattr(guardrails_nemo, "GuardedResponse")


def test_adapter_class_constructs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The wrapper constructs when handed a rails dir (even an empty stub)."""
    from i3.cloud.guardrails_nemo import GuardrailedCloudClient

    # The rails aren't loaded eagerly; construction should succeed as
    # long as the inner client and the path are supplied.
    inner = MagicMock()
    guarded = GuardrailedCloudClient(
        inner, rails_path=str(tmp_path), fail_closed=False
    )
    assert guarded is not None
    assert guarded._inner is inner
    assert guarded._fail_closed is False


@pytest.mark.asyncio
async def test_happy_path_with_mocked_llm() -> None:
    """A benign prompt passes through both rails and returns the inner text."""
    from i3.cloud.guardrails_nemo import GuardrailedCloudClient

    inner = MagicMock()
    inner.generate = AsyncMock(
        return_value={"text": "Photosynthesis is the process...", "input_tokens": 10, "output_tokens": 12, "latency_ms": 42.0}
    )

    guarded = GuardrailedCloudClient(inner, fail_closed=False)

    # Stub out the rails dispatcher so the test does not require a real
    # Colang engine / API key.
    guarded._run_input_rails = AsyncMock(return_value={"allow": True, "triggered": []})  # type: ignore[method-assign]
    guarded._run_output_rails = AsyncMock(
        return_value={
            "allow": True,
            "text": "Photosynthesis is the process...",
            "triggered": [],
        }
    )  # type: ignore[method-assign]

    result = await guarded.generate("What is photosynthesis?")
    assert result.blocked is False
    assert "Photosynthesis" in result.text
    assert result.rails_triggered == []
    inner.generate.assert_awaited_once()
