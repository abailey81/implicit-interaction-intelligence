"""NeMo Guardrails integration for the I3 cloud client.

Wraps the existing :class:`~i3.cloud.client.CloudLLMClient` with
programmable input / output rails declared in Colang 2.  The rails
block prompt-injection attempts, redact PII, clamp response length to
the current :class:`~i3.adaptation.types.AdaptationVector.verbosity`
target, and refuse jailbreak-style content.

Design:
    * **Non-invasive.** The underlying client is never monkey-patched;
      this wrapper only calls its public ``generate`` method.
    * **Soft dependency.** ``nemoguardrails`` is imported inside a
      ``try``/``except ImportError`` block so the core pipeline boots
      without it.
    * **Config-driven.** Rails live in ``configs/guardrails/*.co`` and
      are loaded through ``RailsConfig.from_path``; no Python changes
      are required to adjust rail behaviour.

References:
    Rebedea, T. *et al.* (2023). **NeMo Guardrails: A Toolkit for
    Controllable and Safe LLM Applications with Programmable Rails.**
    Proceedings of EMNLP 2023 (System Demonstrations), arXiv:2310.10501.

Install hint::

    pip install "nemoguardrails>=0.11"
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    from nemoguardrails import (  # type: ignore[import-not-found]
        LLMRails,
        RailsConfig,
    )

    _NEMO_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    LLMRails = None  # type: ignore[assignment, misc]
    RailsConfig = None  # type: ignore[assignment, misc]
    _NEMO_AVAILABLE = False


_INSTALL_HINT: str = (
    "NeMo Guardrails is not installed. Install with: "
    '`pip install "nemoguardrails>=0.11"`.'
)


# Default location of the Colang rails within the repo
_DEFAULT_RAILS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "configs" / "guardrails"
)


def is_available() -> bool:
    """Return ``True`` iff ``nemoguardrails`` is importable."""
    return _NEMO_AVAILABLE


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GuardedResponse:
    """Response envelope after rails have been applied.

    Attributes:
        text: The final (possibly rail-modified) assistant text.
        blocked: ``True`` if an input or output rail blocked the call.
        block_reason: Free-text explanation when *blocked* is True.
        rails_triggered: Names of rails that fired (for observability).
        raw: The underlying client's raw response dict, or ``None`` if
            the call was blocked before reaching the LLM.
    """

    text: str
    blocked: bool
    block_reason: str | None
    rails_triggered: list[str]
    raw: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class GuardrailedCloudClient:
    """Rail-protected facade around :class:`CloudLLMClient`.

    Args:
        inner_client: An instance of
            :class:`i3.cloud.client.CloudLLMClient` (or a mock with
            the same ``generate`` signature).
        rails_path: Directory containing ``config.yml`` and the
            ``*.co`` Colang files.  Defaults to ``configs/guardrails``.
        fail_closed: When ``True`` (default) any rail evaluation error
            yields a blocked response; when ``False`` the underlying
            client's reply is returned unchanged.

    Raises:
        ImportError: If ``nemoguardrails`` is not installed.
    """

    def __init__(
        self,
        inner_client: Any,
        *,
        rails_path: str | os.PathLike[str] | None = None,
        fail_closed: bool = True,
    ) -> None:
        if not _NEMO_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        self._inner = inner_client
        self._fail_closed = bool(fail_closed)
        self._rails_dir = Path(rails_path) if rails_path else _DEFAULT_RAILS_PATH
        self._rails: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _ensure_rails(self) -> Any:
        """Lazily load the :class:`LLMRails` instance."""
        if self._rails is not None:
            return self._rails
        if not self._rails_dir.exists():
            raise FileNotFoundError(
                f"Rails directory not found: {self._rails_dir}"
            )
        assert RailsConfig is not None and LLMRails is not None
        config = RailsConfig.from_path(str(self._rails_dir))
        self._rails = LLMRails(config)
        logger.info("NeMo rails loaded from %s", self._rails_dir)
        return self._rails

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        context: Mapping[str, Any] | None = None,
    ) -> GuardedResponse:
        """Generate a response with input and output rails applied.

        Args:
            prompt: The user prompt (already privacy-scrubbed upstream,
                but the input rails re-check as defence-in-depth).
            context: Optional extra context forwarded to the rails.

        Returns:
            A :class:`GuardedResponse`.
        """
        rails_triggered: list[str] = []
        ctx = dict(context or {})

        # -------- Input rails --------
        try:
            input_verdict = await self._run_input_rails(prompt, ctx)
        except Exception as exc:
            logger.exception("Input rail evaluation failed: %s", exc)
            if self._fail_closed:
                return GuardedResponse(
                    text="",
                    blocked=True,
                    block_reason=f"input-rail-error:{type(exc).__name__}",
                    rails_triggered=["input_rail_error"],
                    raw=None,
                )
            input_verdict = {"allow": True, "triggered": []}

        rails_triggered.extend(input_verdict.get("triggered", []))
        if not input_verdict.get("allow", True):
            return GuardedResponse(
                text=str(input_verdict.get("response", "")),
                blocked=True,
                block_reason=input_verdict.get("reason"),
                rails_triggered=rails_triggered,
                raw=None,
            )

        # -------- Underlying LLM call --------
        system_prompt: str = ctx.get("system_prompt") or ""
        history = ctx.get("conversation_history")
        raw = await self._inner.generate(
            user_message=prompt,
            system_prompt=system_prompt,
            conversation_history=history,
        )
        inner_text = str(raw.get("text", "")) if isinstance(raw, dict) else ""

        # -------- Output rails --------
        try:
            output_verdict = await self._run_output_rails(inner_text, ctx)
        except Exception as exc:
            logger.exception("Output rail evaluation failed: %s", exc)
            if self._fail_closed:
                return GuardedResponse(
                    text="",
                    blocked=True,
                    block_reason=f"output-rail-error:{type(exc).__name__}",
                    rails_triggered=rails_triggered + ["output_rail_error"],
                    raw=raw,
                )
            output_verdict = {"allow": True, "text": inner_text, "triggered": []}

        rails_triggered.extend(output_verdict.get("triggered", []))
        final_text = str(output_verdict.get("text", inner_text))
        allowed = bool(output_verdict.get("allow", True))

        return GuardedResponse(
            text=final_text if allowed else "",
            blocked=not allowed,
            block_reason=output_verdict.get("reason") if not allowed else None,
            rails_triggered=rails_triggered,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Rail dispatch
    # ------------------------------------------------------------------

    async def _run_input_rails(
        self, prompt: str, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Send *prompt* through NeMo's ``generate_async`` with input-only rails.

        We use NeMo's high-level :meth:`LLMRails.generate_async` API and
        inspect the result.  When a rail triggers, NeMo returns a short
        canned response; we detect this via the ``messages`` field.
        """
        rails = self._ensure_rails()
        result = await rails.generate_async(
            messages=[{"role": "user", "content": prompt}],
            options={"rails": ["input"]},
        )
        return self._normalise_rail_result(result, side="input")

    async def _run_output_rails(
        self, text: str, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Pass the LLM reply through output rails only."""
        rails = self._ensure_rails()
        result = await rails.generate_async(
            messages=[
                {"role": "user", "content": context.get("prompt", "")},
                {"role": "assistant", "content": text},
            ],
            options={"rails": ["output"]},
        )
        return self._normalise_rail_result(result, side="output", fallback_text=text)

    @staticmethod
    def _normalise_rail_result(
        result: Any,
        *,
        side: str,
        fallback_text: str = "",
    ) -> dict[str, Any]:
        """Convert NeMo's polymorphic return value into our verdict dict."""
        # NeMo may return dict, str, or a pydantic-like object across versions
        if isinstance(result, dict):
            triggered: list[str] = list(result.get("triggered_rails", []))
            content: str = str(result.get("content", "") or result.get("text", ""))
            status: str = str(result.get("status", "success")).lower()
        elif isinstance(result, str):
            triggered = []
            content = result
            status = "success"
        else:
            triggered = list(getattr(result, "triggered_rails", []) or [])
            content = str(
                getattr(result, "content", None)
                or getattr(result, "text", None)
                or ""
            )
            status = str(getattr(result, "status", "success")).lower()

        blocked_markers = (
            "i'm sorry",
            "i cannot",
            "i can't help",
            "that request was blocked",
        )
        low = content.lower()
        blocked = status in ("blocked", "rejected") or any(
            m in low for m in blocked_markers
        )
        if blocked:
            return {
                "allow": False,
                "text": content or fallback_text,
                "reason": f"{side}-rail-blocked",
                "triggered": triggered or [f"{side}_rail"],
                "response": content,
            }
        return {
            "allow": True,
            "text": content or fallback_text,
            "triggered": triggered,
        }


__all__ = [
    "GuardedResponse",
    "GuardrailedCloudClient",
    "is_available",
]
