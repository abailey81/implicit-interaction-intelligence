"""Dynamic system-prompt construction from AdaptationVector parameters.

The :class:`PromptBuilder` translates the numerical dimensions of an
:class:`~src.adaptation.types.AdaptationVector` into natural-language
instructions that are injected as the ``system`` prompt for the Claude
Messages API.  This lets the cloud LLM mirror the same adaptation
behaviour that the on-device SLM achieves through cross-attention
conditioning -- but via in-context instruction following.

Design rationale
~~~~~~~~~~~~~~~~
Rather than sending raw floats to the LLM we convert each dimension
into explicit behavioural instructions.  This produces more reliable
adherence than simply embedding numbers, because instruction-tuned
models respond well to natural-language directives but poorly to
arbitrary numerical scales they were not trained on.
"""

from __future__ import annotations

import logging
from typing import Any

from i3.adaptation.types import AdaptationVector, StyleVector

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds dynamic system prompts that encode adaptation parameters.

    The system prompt tells Claude *how* to respond based on the current
    adaptation vector, translating numerical dimensions into natural-
    language instructions that the model can reliably follow.

    Usage::

        builder = PromptBuilder()
        system = builder.build_system_prompt(adaptation_vector, user_summary)
        # Pass *system* to CloudLLMClient.generate()
    """

    BASE_PROMPT: str = (
        "You are a warm, adaptive AI companion. You adjust your "
        "communication style based on the user's current state and needs.\n"
        "\n"
        "Current adaptation parameters:\n"
        "{adaptation_instructions}\n"
        "\n"
        "User context:\n"
        "{user_context}\n"
        "\n"
        "Guidelines:\n"
        "- Respond naturally and conversationally\n"
        "- Follow the adaptation parameters closely\n"
        "- Keep responses concise (1-3 sentences unless the user needs more)\n"
        "- Never mention that you're adapting or monitoring the user\n"
        "- Be genuine, not performative"
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        adaptation_vector: AdaptationVector,
        user_summary: dict[str, Any] | None = None,
    ) -> str:
        """Build a complete system prompt from the current adaptation vector.

        Args:
            adaptation_vector: The 8-dimensional adaptation specification
                produced by the adaptation controller.
            user_summary: Optional aggregated profile data (session count,
                relationship strength, preferred topics, etc.).  No raw
                user text should appear here -- only metadata.

        Returns:
            A fully interpolated system prompt string ready for the
            Claude Messages API ``system`` parameter.
        """
        instructions = self._adaptation_to_instructions(adaptation_vector)
        context = self._build_user_context(user_summary)
        prompt = self.BASE_PROMPT.format(
            adaptation_instructions=instructions,
            user_context=context,
        )
        logger.debug(
            "Built system prompt  (cognitive_load=%.2f, formality=%.2f, "
            "accessibility=%.2f, prompt_len=%d)",
            adaptation_vector.cognitive_load,
            adaptation_vector.style_mirror.formality,
            adaptation_vector.accessibility,
            len(prompt),
        )
        return prompt

    # ------------------------------------------------------------------
    # Adaptation -> natural language
    # ------------------------------------------------------------------

    def _adaptation_to_instructions(self, av: AdaptationVector) -> str:
        """Convert the numerical adaptation vector to natural-language directives.

        Each dimension is mapped to one or more behavioural instructions
        via threshold-based rules.  Accessibility is treated with the
        highest priority because it directly affects usability for users
        with motor or cognitive difficulty.
        """
        lines: list[str] = []

        # -- Cognitive load -> response complexity ----------------------
        # Iter 53 — aligned with the post-processor's (and iter-38
        # rhythm-driven adapter's) semantics: HIGH cognitive_load means
        # the user is stressed / rushed / mentally taxed, NOT that they
        # have spare capacity for richer answers.  The prompt-builder
        # used to treat high cl as "richest/most complex response" per
        # the now-stale CognitiveLoadAdapter docstring; this flipped
        # the LLM into producing detailed prose for a stressed user
        # while the post-processor then trimmed it to one sentence —
        # wasted tokens and inverted semantics.  The thresholds below
        # mirror ``ResponsePostProcessor._enforce_length`` tiers so
        # the LLM produces the right shape from the start.
        if av.cognitive_load >= 0.8:
            lines.append(
                "- The user appears stressed or overloaded. Reply in a "
                "single concise sentence. No jargon, no follow-ups."
            )
        elif av.cognitive_load >= 0.6:
            lines.append(
                "- Keep the reply tight (<= 2 sentences). Skip preamble; "
                "lead with the answer."
            )
        elif av.cognitive_load >= 0.4:
            lines.append(
                "- Use moderate language complexity. Clear and accessible."
            )
        else:
            lines.append(
                "- The user has spare cognitive bandwidth. You may use "
                "richer vocabulary and 4-6 sentence explanations when "
                "the question warrants depth."
            )

        # -- Formality --------------------------------------------------
        self._add_formality_instructions(av.style_mirror, lines)

        # -- Verbosity --------------------------------------------------
        self._add_verbosity_instructions(av.style_mirror, lines)

        # -- Emotionality -----------------------------------------------
        self._add_emotionality_instructions(av.style_mirror, lines)

        # -- Directness -------------------------------------------------
        self._add_directness_instructions(av.style_mirror, lines)

        # -- Emotional tone (warmth) ------------------------------------
        if av.emotional_tone < 0.3:
            lines.append(
                "- Be extra supportive and encouraging. "
                "The user may need comfort."
            )
        elif av.emotional_tone > 0.7:
            lines.append("- Maintain a neutral, balanced tone.")
        # Mid-range: no special instruction needed.

        # -- Accessibility (highest priority override) ------------------
        if av.accessibility > 0.5:
            lines.append(
                "- IMPORTANT: The user may have motor or cognitive "
                "difficulty."
            )
            lines.append(
                "- Use very simple language. Ask yes/no questions "
                "when possible."
            )
            lines.append(
                "- Keep responses extremely short (under 15 words)."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Style sub-dimension helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_formality_instructions(
        style: StyleVector, lines: list[str]
    ) -> None:
        """Append formality instructions based on the style vector."""
        if style.formality < 0.3:
            lines.append(
                "- Be casual and conversational. Contractions are fine."
            )
        elif style.formality > 0.7:
            lines.append("- Maintain a professional, formal tone.")

    @staticmethod
    def _add_verbosity_instructions(
        style: StyleVector, lines: list[str]
    ) -> None:
        """Append verbosity instructions based on the style vector."""
        if style.verbosity < 0.3:
            lines.append("- Be concise. One sentence if possible.")
        elif style.verbosity > 0.7:
            lines.append("- Provide detailed, thorough responses.")

    @staticmethod
    def _add_emotionality_instructions(
        style: StyleVector, lines: list[str]
    ) -> None:
        """Append emotionality instructions based on the style vector."""
        if style.emotionality > 0.7:
            lines.append("- Be expressive and warm. Show empathy.")
        elif style.emotionality < 0.3:
            lines.append(
                "- Keep emotional expression restrained and measured."
            )

    @staticmethod
    def _add_directness_instructions(
        style: StyleVector, lines: list[str]
    ) -> None:
        """Append directness instructions based on the style vector."""
        if style.directness > 0.7:
            lines.append(
                "- Be direct and to the point. Avoid hedging."
            )
        elif style.directness < 0.3:
            lines.append(
                "- Use softer, indirect phrasing. Hedge suggestions."
            )

    # ------------------------------------------------------------------
    # User context
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_context(
        user_summary: dict[str, Any] | None = None,
    ) -> str:
        """Build a user-context string from an aggregated profile summary.

        The summary must not contain raw user text -- only metadata such
        as session counts, topic keywords, and engagement scores.

        Args:
            user_summary: Optional dict with keys like ``session_count``,
                ``relationship_strength``, ``preferred_topics``,
                ``avg_engagement``, ``last_session_days_ago``.

        Returns:
            A multi-line string describing the user context, or a default
            message for new / unknown users.
        """
        if not user_summary:
            return "New user, no history available."

        context_lines: list[str] = []

        session_count = user_summary.get("session_count")
        if session_count is not None:
            if session_count < 3:
                context_lines.append(
                    f"- Early-stage relationship ({session_count} prior sessions)."
                )
            elif session_count < 20:
                context_lines.append(
                    f"- Developing relationship ({session_count} prior sessions)."
                )
            else:
                context_lines.append(
                    f"- Established relationship ({session_count} prior sessions)."
                )

        relationship_strength = user_summary.get("relationship_strength")
        if relationship_strength is not None:
            if relationship_strength > 0.7:
                context_lines.append("- Strong rapport with user.")
            elif relationship_strength < 0.3:
                context_lines.append("- Still building rapport with user.")

        topics = user_summary.get("preferred_topics")
        if topics and isinstance(topics, list):
            topic_str = ", ".join(str(t) for t in topics[:5])
            context_lines.append(f"- Preferred topics: {topic_str}.")

        avg_engagement = user_summary.get("avg_engagement")
        if avg_engagement is not None:
            if avg_engagement > 0.7:
                context_lines.append("- User typically highly engaged.")
            elif avg_engagement < 0.3:
                context_lines.append(
                    "- User engagement has been low -- keep things light."
                )

        last_session = user_summary.get("last_session_days_ago")
        if last_session is not None and last_session > 7:
            context_lines.append(
                f"- User hasn't interacted in {last_session} days. "
                f"Welcome them back warmly."
            )

        return "\n".join(context_lines) if context_lines else "No detailed user context available."
