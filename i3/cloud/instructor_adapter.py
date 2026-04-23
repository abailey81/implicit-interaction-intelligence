"""Instructor-based structured output for the I3 cloud layer.

`Instructor <https://python.useinstructor.com>`_ is a lightweight
structured-output wrapper that patches the provider SDK (Anthropic in
our case) so every call returns an instance of a user-supplied
Pydantic model.  It relies on the provider's native tool-use /
JSON-mode hook rather than regex-parsing free-form text, and performs
up to ``max_retries`` automatic self-repair cycles on validation
failures.

For I3 we treat Instructor as an alternative to Pydantic AI that is
useful when the caller already has a raw ``anthropic`` client they
want to keep: ``instructor.from_anthropic`` is a single-line upgrade.

References:
    * Jason Liu, "Instructor: Structured Outputs for LLMs."
      https://python.useinstructor.com/ (v1.x documentation, 2024).
    * Anthropic tool-use spec (used by Instructor's Anthropic backend):
      https://docs.anthropic.com/en/docs/build-with-claude/tool-use

Install hint::

    pip install "instructor>=1.6" anthropic
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    import instructor  # type: ignore[import-not-found]

    _INSTRUCTOR_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    instructor = None  # type: ignore[assignment]
    _INSTRUCTOR_AVAILABLE = False

try:  # pragma: no cover - environment-dependent
    import anthropic as _anthropic  # type: ignore[import-not-found]

    _ANTHROPIC_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    _anthropic = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False


_INSTALL_HINT: str = (
    "instructor (and anthropic) are not installed. Install with: "
    '`pip install "instructor>=1.6" anthropic`.'
)


def is_available() -> bool:
    """Return ``True`` iff both ``instructor`` and ``anthropic`` are importable."""
    return _INSTRUCTOR_AVAILABLE and _ANTHROPIC_AVAILABLE


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class InstructorAdapter:
    """Thin wrapper around ``instructor.from_anthropic``.

    Args:
        model_name: Anthropic model id (default ``claude-sonnet-4-5``).
        api_key: Optional API key; falls back to ``ANTHROPIC_API_KEY``.
        max_tokens: Max completion tokens per call.
        max_retries: Self-repair retries on validation failure.

    Raises:
        ImportError: If ``instructor`` or ``anthropic`` is not installed.
    """

    def __init__(
        self,
        *,
        model_name: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_tokens: int = 1024,
        max_retries: int = 2,
    ) -> None:
        if not is_available():
            raise ImportError(_INSTALL_HINT)
        assert instructor is not None and _anthropic is not None

        resolved_key = api_key if api_key is not None else os.environ.get(
            "ANTHROPIC_API_KEY", ""
        )
        if not resolved_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set; InstructorAdapter calls will "
                "fail at request time."
            )

        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self._raw_client = _anthropic.Anthropic(api_key=resolved_key or None)
        # ``from_anthropic`` wraps the raw client and returns a new object
        # whose ``.messages.create`` method accepts ``response_model=...``.
        self._client = instructor.from_anthropic(self._raw_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def structured_generate(
        self,
        prompt: str,
        response_model: type[T],
        *,
        system: str | None = None,
    ) -> T:
        """Return a validated instance of *response_model*.

        Args:
            prompt: The user prompt.
            response_model: Pydantic model class to validate the reply.
            system: Optional system prompt; if omitted the caller is
                expected to encode context inside *prompt*.

        Returns:
            A validated instance of *response_model*.
        """
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "response_model": response_model,
            "max_retries": self.max_retries,
        }
        if system:
            kwargs["system"] = system
        return self._client.messages.create(**kwargs)  # type: ignore[no-any-return]

    @property
    def client(self) -> Any:
        """Return the underlying instructor-wrapped Anthropic client."""
        return self._client


__all__ = [
    "InstructorAdapter",
    "is_available",
]
