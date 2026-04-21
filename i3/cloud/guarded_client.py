"""Guardrail-wrapped cloud LLM client.

Composes an :class:`i3.cloud.client.CloudLLMClient` with the input
and output guardrails defined in :mod:`i3.cloud.guardrails`. This
preserves the original client unchanged — no monkey-patching, no
subclassing of the Anthropic HTTP surface.

Usage::

    from i3.cloud.client import CloudLLMClient
    from i3.cloud.guarded_client import GuardedCloudClient
    from i3.cloud.guardrails import InputGuardrail, OutputGuardrail

    base = CloudLLMClient(config)
    guarded = GuardedCloudClient(
        base,
        input_guardrail=InputGuardrail(max_tokens=2048),
        output_guardrail=OutputGuardrail(),
    )
    result = await guarded.generate(
        user_message="hello",
        system_prompt="...",
        user_id="alice",
    )

On input-guardrail failure the guarded client returns a safe
fallback-shaped dict (identical schema to the underlying client) so
that callers do not need to branch on exceptions — the fallback is
annotated with ``blocked=True`` and a ``block_reason`` string.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from i3.cloud.guardrails import (
    GuardrailViolation,
    InputGuardrail,
    OutputGuardrail,
)

logger = logging.getLogger(__name__)


class GuardedCloudClient:
    """Wrap a :class:`CloudLLMClient` with input/output guardrails.

    The class is duck-typed on the underlying client — any object
    that exposes an awaitable ``generate(user_message, system_prompt,
    conversation_history=None) -> dict`` is acceptable. This makes it
    trivial to unit-test the wrapper with a stub.

    Attributes:
        client: The underlying cloud client.
        input_guardrail: The :class:`InputGuardrail` instance.
        output_guardrail: The :class:`OutputGuardrail` instance.
    """

    def __init__(
        self,
        client: Any,
        *,
        input_guardrail: Optional[InputGuardrail] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
    ) -> None:
        """Initialise the guarded client.

        Args:
            client: An instance of :class:`CloudLLMClient` or any
                object with a compatible async ``generate`` method.
            input_guardrail: Custom input guardrail; a default one is
                created if omitted.
            output_guardrail: Custom output guardrail; a default one
                is created if omitted.
        """
        self.client: Any = client
        self.input_guardrail: InputGuardrail = (
            input_guardrail or InputGuardrail()
        )
        self.output_guardrail: OutputGuardrail = (
            output_guardrail or OutputGuardrail()
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blocked_response(reason: str, category: str) -> dict[str, Any]:
        """Return a dict mirroring :class:`CloudLLMClient.generate`'s shape."""
        return {
            "text": (
                "Your request could not be processed due to a safety "
                "policy. Please rephrase and try again."
            ),
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
            "blocked": True,
            "block_reason": reason,
            "block_category": category,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        user_message: str,
        system_prompt: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
        *,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run input guardrails, call the inner client, sanitise output.

        Args:
            user_message: The user's prompt.
            system_prompt: System-level instructions forwarded to the
                inner client.
            conversation_history: Optional prior turns.
            user_id: If supplied, any reflection of this identifier in
                the model output is redacted by the output guardrail.

        Returns:
            A dict with the same schema as
            :meth:`CloudLLMClient.generate`, with additional
            ``blocked``, ``block_reason``, and ``violations`` keys when
            guardrails intervene.

        Raises:
            RuntimeError: Only if the underlying client raises and the
                guardrail chose not to intercept.
        """
        # 1. Input guardrail — may return a blocked response.
        try:
            self.input_guardrail.enforce(user_message)
        except GuardrailViolation as gv:
            logger.warning(
                "Input guardrail blocked prompt (category=%s): %s",
                gv.category,
                gv.reason,
            )
            return self._blocked_response(gv.reason, gv.category)

        # 2. Forward to the underlying client.
        result = await self.client.generate(
            user_message=user_message,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
        )

        # 3. Output guardrail.
        text = result.get("text", "") if isinstance(result, dict) else ""
        sensitive: list[str] = []
        if user_id:
            sensitive.append(user_id)
        sanitised = self.output_guardrail.sanitize(
            text, sensitive_tokens=tuple(sensitive)
        )
        new_result = dict(result) if isinstance(result, dict) else {}
        new_result["text"] = sanitised.text
        new_result["output_modified"] = sanitised.modified
        new_result["violations"] = sanitised.violations
        new_result.setdefault("blocked", False)
        return new_result

    async def generate_session_summary(
        self, session_metadata: dict[str, Any]
    ) -> str:
        """Forward metadata-only summary calls without guardrails.

        The underlying session-summary path already enforces a strict
        metadata allow-list and never sees raw user text, so the
        guardrail layer is redundant. We still pass the output through
        the :class:`OutputGuardrail` to strip any accidental key
        reflections.

        Args:
            session_metadata: Aggregated per-session metrics.

        Returns:
            A short natural-language summary string.
        """
        summary_fn = getattr(self.client, "generate_session_summary", None)
        if not callable(summary_fn):
            raise RuntimeError(
                "Underlying client has no generate_session_summary() method."
            )
        text = await summary_fn(session_metadata)
        sanitised = self.output_guardrail.sanitize(text)
        return sanitised.text

    # ------------------------------------------------------------------
    # Pass-through properties / lifecycle
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Mirror :attr:`CloudLLMClient.is_available`."""
        return bool(getattr(self.client, "is_available", False))

    @property
    def usage_stats(self) -> dict[str, int]:
        """Mirror :attr:`CloudLLMClient.usage_stats`."""
        return dict(getattr(self.client, "usage_stats", {}))

    async def close(self) -> None:
        """Close the underlying client, if it supports it."""
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            await close_fn()

    async def __aenter__(self) -> "GuardedCloudClient":
        enter_fn = getattr(self.client, "__aenter__", None)
        if callable(enter_fn):
            await enter_fn()
        return self

    async def __aexit__(self, *exc: object) -> None:
        exit_fn = getattr(self.client, "__aexit__", None)
        if callable(exit_fn):
            await exit_fn(*exc)


__all__ = ["GuardedCloudClient"]
