"""Input and output guardrails for the cloud LLM client.

Implements a lightweight, stdlib-only guardrail layer that sits on
either side of the Anthropic call:

* :class:`InputGuardrail` — rejects prompts that exceed a token
  budget, contain prompt-injection keywords, or repeat recent prompts
  (simple loop detection).
* :class:`OutputGuardrail` — strips reflected sensitive tokens
  (user_id, api_key patterns), trims over-long responses, and logs
  detected policy-violation mentions.

The guardrails are composable and have no external dependencies — the
whole validation path can run on-device before a single byte leaves
the network stack.

Usage is demonstrated in :mod:`i3.cloud.guarded_client`, which wraps
an existing :class:`i3.cloud.client.CloudLLMClient` without modifying
it.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Final, Optional, Pattern

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GuardrailViolation(Exception):
    """Raised when an input prompt fails the input guardrail."""

    def __init__(self, reason: str, *, category: str) -> None:
        """Initialise the violation.

        Args:
            reason: Human-readable explanation.
            category: Machine-readable category
                (``"too_long"``, ``"prompt_injection"``, ``"loop"``).
        """
        super().__init__(reason)
        self.reason: str = reason
        self.category: str = category


# ---------------------------------------------------------------------------
# Result wrappers
# ---------------------------------------------------------------------------


@dataclass
class InputCheckResult:
    """Outcome of an input guardrail check."""

    ok: bool
    reason: Optional[str] = None
    category: Optional[str] = None


@dataclass
class OutputCheckResult:
    """Outcome of an output guardrail check.

    Attributes:
        text: The (possibly modified) response text.
        modified: ``True`` if the text was altered by sanitisation.
        violations: Policy-violation categories detected (may be empty
            even if ``modified`` is ``True`` — e.g. pure trimming).
    """

    text: str
    modified: bool = False
    violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------


class InputGuardrail:
    """Validate prompts before they leave the device.

    Checks, in order:

    1. Approximate token count (``len(prompt) / 4`` heuristic) against
       :attr:`max_tokens`.
    2. Substring match against a small allow-list of well-known
       prompt-injection patterns.
    3. Loop detection: the *same* prompt appearing three times inside
       the configured history window.

    The guardrail is stateful — it keeps a bounded deque of recent
    prompts so that loop detection works across calls. Thread safety
    is the caller's responsibility (the class is intended to run
    inside a single asyncio event loop).

    Attributes:
        max_tokens: Upper bound on approximate prompt tokens.
        history_window: Number of recent prompts kept in memory.
        injection_patterns: Compiled regexes used for detection.
    """

    DEFAULT_MAX_TOKENS: Final[int] = 4096
    DEFAULT_HISTORY_WINDOW: Final[int] = 8
    LOOP_THRESHOLD: Final[int] = 3
    _CHARS_PER_TOKEN_HEURISTIC: Final[float] = 4.0

    # Commonly-seen prompt-injection trigger substrings. The list is
    # intentionally conservative — aggressive filters produce high
    # false-positive rates on legitimate technical questions.
    DEFAULT_INJECTION_PATTERNS: Final[tuple[str, ...]] = (
        r"ignore\s+previous\s+(instructions|prompt|rules)",
        r"ignore\s+all\s+prior",
        r"disregard\s+(the\s+)?above",
        r"system\s*:",
        r"\[INST\]",
        r"<\s*/?\s*system\s*>",
        r"\bjailbreak\b",
        r"do\s+anything\s+now",
        r"\bDAN\b",
        r"forget\s+(the\s+)?(above|previous)",
        r"override\s+(your\s+)?instructions",
    )

    def __init__(
        self,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        injection_patterns: Optional[tuple[str, ...]] = None,
    ) -> None:
        """Initialise the guardrail.

        Args:
            max_tokens: Approximate token budget per prompt.
            history_window: Number of recent prompts kept for loop
                detection.
            injection_patterns: Custom patterns. Defaults to
                :attr:`DEFAULT_INJECTION_PATTERNS`.
        """
        self.max_tokens: int = int(max_tokens)
        self.history_window: int = int(history_window)
        patterns = injection_patterns or self.DEFAULT_INJECTION_PATTERNS
        self.injection_patterns: list[Pattern[str]] = [
            re.compile(p, flags=re.IGNORECASE) for p in patterns
        ]
        self._history: deque[str] = deque(maxlen=self.history_window)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Return the heuristic token count for *text*."""
        return max(
            1, int(len(text) / InputGuardrail._CHARS_PER_TOKEN_HEURISTIC)
        )

    def _detect_injection(self, prompt: str) -> Optional[str]:
        """Return the matching injection pattern, if any."""
        for pat in self.injection_patterns:
            if pat.search(prompt):
                return pat.pattern
        return None

    def _detect_loop(self, prompt: str) -> bool:
        """Return ``True`` if *prompt* equals the last LOOP_THRESHOLD entries."""
        if len(self._history) < self.LOOP_THRESHOLD - 1:
            return False
        tail = list(self._history)[-(self.LOOP_THRESHOLD - 1) :]
        return all(entry == prompt for entry in tail)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, prompt: str) -> InputCheckResult:
        """Validate *prompt* and update the loop history on success.

        Args:
            prompt: The user-supplied text to validate.

        Returns:
            An :class:`InputCheckResult`. On success ``ok`` is ``True``.
            On failure ``reason`` and ``category`` describe the problem.
        """
        if not isinstance(prompt, str):
            return InputCheckResult(
                ok=False,
                reason="prompt must be a string",
                category="bad_type",
            )

        if self._approx_tokens(prompt) > self.max_tokens:
            return InputCheckResult(
                ok=False,
                reason=(
                    f"prompt exceeds token budget "
                    f"(~{self._approx_tokens(prompt)} > {self.max_tokens})"
                ),
                category="too_long",
            )

        inj = self._detect_injection(prompt)
        if inj is not None:
            return InputCheckResult(
                ok=False,
                reason=f"prompt matched injection pattern: {inj}",
                category="prompt_injection",
            )

        if self._detect_loop(prompt):
            return InputCheckResult(
                ok=False,
                reason=(
                    f"prompt repeated {self.LOOP_THRESHOLD} times in a row"
                ),
                category="loop",
            )

        self._history.append(prompt)
        return InputCheckResult(ok=True)

    def enforce(self, prompt: str) -> str:
        """Like :meth:`check` but raises on failure.

        Returns:
            The validated prompt (unchanged).

        Raises:
            GuardrailViolation: If the prompt fails any check.
        """
        result = self.check(prompt)
        if not result.ok:
            assert result.reason is not None
            assert result.category is not None
            raise GuardrailViolation(
                result.reason, category=result.category
            )
        return prompt

    def reset(self) -> None:
        """Clear the loop-detection history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------


class OutputGuardrail:
    """Sanitise LLM responses before surfacing them to the user.

    Responsibilities:

    * Redact reflected sensitive tokens — user identifiers supplied by
      the caller, and anything matching the common API-key shapes
      (``sk-ant-...``, ``sk-...``, AWS-style keys, generic 40+ char
      alphanumeric blobs).
    * Trim responses that exceed :attr:`max_response_chars`, preserving
      a terminal ellipsis so downstream UI knows the text was cut.
    * Detect known policy-violation mentions (``"i cannot comply"``,
      ``"illegal content"``, …) and emit structured log events so they
      can be fed into observability dashboards.

    The class is intentionally stateless so instances can be shared
    across concurrent tasks.

    Attributes:
        max_response_chars: Upper bound on returned character count.
        policy_markers: Substrings that flag a policy-violation
            mention.
    """

    DEFAULT_MAX_RESPONSE_CHARS: Final[int] = 16_000

    _API_KEY_PATTERNS: Final[tuple[str, ...]] = (
        r"sk-ant-[A-Za-z0-9_\-]{20,}",  # Anthropic
        r"sk-[A-Za-z0-9]{20,}",  # OpenAI-style
        r"AKIA[0-9A-Z]{16}",  # AWS access key id
        r"ghp_[A-Za-z0-9]{20,}",  # GitHub PAT
        r"xox[baprs]-[A-Za-z0-9-]{10,}",  # Slack tokens
        r"AIza[0-9A-Za-z\-_]{35}",  # Google API key
    )

    DEFAULT_POLICY_MARKERS: Final[tuple[str, ...]] = (
        "i cannot comply",
        "i won't help with",
        "i will not help with",
        "illegal content",
        "harmful content",
        "unsafe request",
        "violates my policies",
    )

    def __init__(
        self,
        *,
        max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
        policy_markers: Optional[tuple[str, ...]] = None,
    ) -> None:
        """Initialise the guardrail.

        Args:
            max_response_chars: Maximum characters retained in the
                response.
            policy_markers: Substrings that flag a policy violation.
        """
        self.max_response_chars: int = int(max_response_chars)
        self.policy_markers: tuple[str, ...] = (
            policy_markers or self.DEFAULT_POLICY_MARKERS
        )
        self._compiled_key_patterns: list[Pattern[str]] = [
            re.compile(p) for p in self._API_KEY_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _redact_sensitive(
        self, text: str, sensitive: tuple[str, ...]
    ) -> tuple[str, bool]:
        """Strip reflected sensitive tokens from *text*.

        Args:
            text: The response text.
            sensitive: Literal tokens to redact (user_id, session_id,
                etc.). Matched case-insensitively as whole words where
                reasonable.

        Returns:
            ``(sanitised_text, modified_flag)``.
        """
        modified = False
        out = text
        for literal in sensitive:
            if not literal:
                continue
            # Use a word-boundary regex only when the literal is
            # alphanumeric; otherwise fall back to a plain replace.
            if re.fullmatch(r"[A-Za-z0-9_\-]+", literal):
                pat = re.compile(
                    rf"\b{re.escape(literal)}\b", flags=re.IGNORECASE
                )
                new, n = pat.subn("[REDACTED]", out)
                if n > 0:
                    modified = True
                    out = new
            else:
                if literal in out:
                    out = out.replace(literal, "[REDACTED]")
                    modified = True
        for pat in self._compiled_key_patterns:
            new, n = pat.subn("[REDACTED_KEY]", out)
            if n > 0:
                modified = True
                out = new
        return out, modified

    def _detect_violations(self, text: str) -> list[str]:
        """Return policy markers found in *text* (lower-cased)."""
        lowered = text.lower()
        return [m for m in self.policy_markers if m in lowered]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(
        self,
        text: str,
        *,
        sensitive_tokens: Optional[tuple[str, ...]] = None,
    ) -> OutputCheckResult:
        """Apply output guardrails to *text*.

        Args:
            text: The assistant's response text.
            sensitive_tokens: Literal tokens to redact (e.g. the
                caller's ``user_id``, session identifiers). May be
                empty or ``None``.

        Returns:
            An :class:`OutputCheckResult` with the sanitised text,
            a ``modified`` flag, and the list of detected policy
            violations.
        """
        if not isinstance(text, str):
            return OutputCheckResult(text="", modified=True, violations=[])

        modified = False
        out, redacted = self._redact_sensitive(
            text, sensitive_tokens or ()
        )
        modified = modified or redacted

        if len(out) > self.max_response_chars:
            out = out[: self.max_response_chars - 1].rstrip() + "…"
            modified = True
            logger.info(
                "OutputGuardrail trimmed response to %d chars",
                self.max_response_chars,
            )

        violations = self._detect_violations(out)
        if violations:
            # Log only the categories, not the full response.
            logger.info(
                "OutputGuardrail detected policy markers: %s",
                ", ".join(violations),
            )

        return OutputCheckResult(
            text=out, modified=modified, violations=violations
        )


__all__ = [
    "GuardrailViolation",
    "InputCheckResult",
    "InputGuardrail",
    "OutputCheckResult",
    "OutputGuardrail",
]
