"""Tests for :mod:`i3.cloud.dspy_adapter`.

The module is soft-imported; these tests are skipped in environments
without ``dspy-ai``.  Three tests cover:

    (a) module import with soft-import path,
    (b) adapter class construction without error,
    (c) a minimal happy-path call with a mocked underlying LLM.

Reference:
    Khattab et al. (2023) "DSPy: Compiling Declarative LM Calls."
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

dspy = pytest.importorskip("dspy")


def test_module_imports_with_soft_import_path() -> None:
    """The adapter module imports and exposes the expected symbols."""
    from i3.cloud import dspy_adapter

    assert dspy_adapter.is_available() is True
    assert hasattr(dspy_adapter, "I3AdaptivePromptProgram")
    assert hasattr(dspy_adapter, "optimize_program")
    assert dspy_adapter.AdaptiveResponseSignature is not None


def test_adapter_class_constructs() -> None:
    """The DSPy program constructs with a stub LM."""
    from i3.cloud.dspy_adapter import I3AdaptivePromptProgram

    lm = MagicMock()
    lm.kwargs = {}
    program = I3AdaptivePromptProgram(lm=lm)
    assert program is not None
    assert program._module is not None
    assert hasattr(program._module, "think")


def test_happy_path_call_with_mocked_llm() -> None:
    """``forward()`` delegates to the inner module with a string payload."""
    from i3.cloud.dspy_adapter import I3AdaptivePromptProgram

    program = I3AdaptivePromptProgram(lm=MagicMock())

    # Replace the inner chain-of-thought with a stub that records args
    stub_pred = dspy.Prediction(response="mocked reply")
    program._module = MagicMock(return_value=stub_pred)

    av = {"formality": 0.2, "verbosity": 0.5, "empathy": 0.9}
    result = program.forward(
        user_state="calm, focused",
        adaptation_vector=av,
        message="Hello",
    )
    assert result is stub_pred
    program._module.assert_called_once()
    call_kwargs = program._module.call_args.kwargs
    assert call_kwargs["user_state"] == "calm, focused"
    assert call_kwargs["message"] == "Hello"
    # Adaptation vector must be serialised to a deterministic string
    assert isinstance(call_kwargs["adaptation_vector"], str)
    assert "formality" in call_kwargs["adaptation_vector"]
