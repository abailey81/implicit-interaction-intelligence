"""Pydantic AI typed client for the I3 cloud layer.

`Pydantic AI <https://ai.pydantic.dev>`_ is the Pydantic team's
opinionated, type-first LLM client.  It converts Pydantic model schemas
into the provider-native structured-output contracts (Anthropic tool
use, OpenAI ``response_format``, etc.) and validates model output on
return, raising a :class:`pydantic.ValidationError` on schema drift.

For I3 we constrain Claude's response to the :class:`AdaptiveResponse`
schema below so downstream code can rely on ``tone``,
``estimated_complexity``, and ``used_simplification`` being present and
well-typed.

References:
    * Pydantic AI documentation (2024-12 release):
      https://ai.pydantic.dev/
    * Anthropic tool-use structured-output spec:
      https://docs.anthropic.com/en/docs/build-with-claude/tool-use

Install hint::

    pip install "pydantic-ai>=0.0.13"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import of pydantic-ai
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    from pydantic_ai import Agent  # type: ignore[import-not-found]
    from pydantic_ai.models.anthropic import (  # type: ignore[import-not-found]
        AnthropicModel,
    )

    _PYDANTIC_AI_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    Agent = None  # type: ignore[assignment, misc]
    AnthropicModel = None  # type: ignore[assignment, misc]
    _PYDANTIC_AI_AVAILABLE = False


_INSTALL_HINT: str = (
    "pydantic-ai is not installed. Install with: "
    '`pip install "pydantic-ai>=0.0.13"`.'
)


def is_available() -> bool:
    """Return ``True`` iff the ``pydantic_ai`` package is importable."""
    return _PYDANTIC_AI_AVAILABLE


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


Tone = Literal["warm", "neutral", "direct"]


class AdaptiveResponse(BaseModel):
    """Schema-enforced shape of an I3 adaptive reply.

    The field names and semantics mirror those used by the on-device
    SLM so both code paths can be compared directly in evaluation.

    Attributes:
        text: The natural-language reply.
        tone: One of ``"warm"``, ``"neutral"``, ``"direct"``.
        estimated_complexity: Model's self-reported complexity of its
            own answer, on ``[0, 1]``.  Higher = denser vocabulary /
            longer sentences.
        used_simplification: ``True`` when the model reduced complexity
            in response to an :class:`AdaptationVector` with high
            ``simplification`` score.
    """

    text: str = Field(..., description="Assistant reply.")
    tone: Tone = Field(..., description="Dominant tone class.")
    estimated_complexity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-estimated response complexity on [0,1].",
    )
    used_simplification: bool = Field(
        ...,
        description="Whether the model applied a simplification step.",
    )


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PydanticAICloudClient:
    """Typed Anthropic client driven by Pydantic AI.

    Args:
        model_name: Anthropic model id (default ``claude-sonnet-4-5``).
        api_key: Optional explicit API key; falls back to
            ``ANTHROPIC_API_KEY``.
        system_prompt: Static system prompt prepended to every call.
        result_type: The Pydantic model to enforce on the output.
            Defaults to :class:`AdaptiveResponse`.

    Raises:
        ImportError: If ``pydantic-ai`` is not installed.
    """

    def __init__(
        self,
        *,
        model_name: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        system_prompt: str = (
            "You are a warm, adaptive AI companion. Respond with the "
            "exact schema specified; do not include any other text."
        ),
        result_type: Type[BaseModel] = AdaptiveResponse,
    ) -> None:
        if not _PYDANTIC_AI_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        assert Agent is not None and AnthropicModel is not None

        resolved_key = api_key if api_key is not None else os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        if not resolved_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set; PydanticAICloudClient calls "
                "will fail at request time."
            )

        self.model_name = model_name
        self._system_prompt = system_prompt
        self._result_type: Type[BaseModel] = result_type
        self._model = AnthropicModel(
            model_name=model_name,
            api_key=resolved_key or None,
        )
        self._agent = Agent(
            self._model,
            result_type=self._result_type,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, user_message: str) -> BaseModel:
        """Return a schema-validated response instance.

        Args:
            user_message: The current user message.

        Returns:
            An instance of the configured ``result_type`` (default
            :class:`AdaptiveResponse`).

        Raises:
            pydantic.ValidationError: If the model returns output that
                fails schema validation even after Pydantic AI's
                self-repair retries.
        """
        result = await self._agent.run(user_message)
        # pydantic-ai >=0.0.13 exposes the validated data as `.data`
        validated = getattr(result, "data", None) or getattr(result, "output", result)
        return validated  # type: ignore[return-value]

    def generate_sync(self, user_message: str) -> BaseModel:
        """Synchronous variant of :meth:`generate`."""
        result = self._agent.run_sync(user_message)
        validated = getattr(result, "data", None) or getattr(result, "output", result)
        return validated  # type: ignore[return-value]

    @property
    def agent(self) -> Any:
        """Return the underlying :class:`pydantic_ai.Agent`."""
        return self._agent


__all__ = [
    "AdaptiveResponse",
    "PydanticAICloudClient",
    "Tone",
    "is_available",
]
