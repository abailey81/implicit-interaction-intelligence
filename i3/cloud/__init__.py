"""Cloud LLM integration layer for Implicit Interaction Intelligence (I3).

This package provides the components needed to route queries to the
Anthropic Claude API when the intelligent router selects the cloud path:

- :class:`CloudLLMClient` -- Async client for the Anthropic Messages API
  with retry logic, token tracking, and graceful fallback.
- :class:`PromptBuilder` -- Dynamically constructs system prompts from
  the current :class:`~src.adaptation.types.AdaptationVector`.
- :class:`ResponsePostProcessor` -- Ensures cloud responses conform to
  adaptation constraints (length, vocabulary, accessibility).
"""

from i3.cloud.client import CloudLLMClient
from i3.cloud.postprocess import ResponsePostProcessor
from i3.cloud.prompt_builder import PromptBuilder

__all__ = [
    "CloudLLMClient",
    "PromptBuilder",
    "ResponsePostProcessor",
]
