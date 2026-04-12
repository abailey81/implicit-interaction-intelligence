"""Session summariser for the Interaction Diary.

Generates privacy-safe, diary-style session summaries from aggregated
metadata.  Summaries are produced in one of two modes:

1. **Cloud mode** -- When a :class:`~src.cloud.client.CloudLLMClient` is
   available, a prompt containing *only* aggregated metadata (topic
   keywords, engagement scores, emotion labels, message count) is sent to
   the cloud LLM.  **No raw user text is ever transmitted.**

2. **Template mode** -- When the cloud client is unavailable or
   explicitly disabled, a deterministic template fills in the summary
   from the same metadata fields.  This is the default fallback.

PRIVACY GUARANTEE
~~~~~~~~~~~~~~~~~
Summaries are generated exclusively from aggregated session metadata --
topic keywords, scalar metrics, and emotion labels.  No raw user text or
AI response text is ever included in the prompt or the resulting summary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from i3.cloud.client import CloudLLMClient

logger = logging.getLogger(__name__)


class SessionSummarizer:
    """Generates privacy-safe session summaries.

    Summaries are derived from aggregated metadata only.  No raw user
    text is ever sent to the cloud -- this is a key privacy guarantee of
    the I3 system.

    Parameters
    ----------
    cloud_client:
        Optional :class:`~src.cloud.client.CloudLLMClient` instance.
        When provided *and* available, summaries are generated via the
        cloud LLM.  Otherwise the template fallback is used.
    """

    def __init__(self, cloud_client: Optional[CloudLLMClient] = None) -> None:
        self.cloud_client = cloud_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def summarize(self, session_data: dict[str, Any]) -> str:
        """Generate a privacy-safe session summary.

        Attempts cloud-based generation first; falls back to a
        deterministic template if the cloud is unavailable or errors.

        Parameters
        ----------
        session_data:
            Aggregated session metadata as produced by
            :meth:`~src.diary.logger.DiaryLogger.get_session_summary_data`.
            Expected keys: ``message_count``, ``topics``,
            ``mean_engagement``, ``mean_cognitive_load``,
            ``mean_accessibility``, ``dominant_emotion``,
            ``mean_energy``, ``duration_min`` (optional),
            ``start_emotion`` / ``end_emotion`` (optional).

        Returns
        -------
        str
            A warm, diary-style one-sentence summary suitable for
            display in the interaction diary UI.
        """
        if self.cloud_client and self._cloud_is_available():
            try:
                summary = await self._cloud_summary(session_data)
                logger.info("Session summary generated via cloud LLM")
                return summary
            except Exception:
                logger.warning(
                    "Cloud summary generation failed; falling back to template",
                    exc_info=True,
                )
        return self._template_summary(session_data)

    # ------------------------------------------------------------------
    # Cloud-based summary
    # ------------------------------------------------------------------

    async def _cloud_summary(self, data: dict[str, Any]) -> str:
        """Generate a summary by prompting the cloud LLM with metadata.

        The prompt contains **only** aggregated facts (topic keywords,
        engagement percentage, emotion labels, message count).  No raw
        user text is included.

        Parameters
        ----------
        data:
            Session metadata dictionary.

        Returns
        -------
        str
            Cloud-generated summary sentence.
        """
        topics = data.get("topics", ["general"])
        mean_engagement = data.get("mean_engagement", 0.5)
        mean_energy = data.get("mean_energy", 0.5)
        start_emotion = data.get("start_emotion", data.get("dominant_emotion", "neutral"))
        end_emotion = data.get("end_emotion", data.get("dominant_emotion", "neutral"))

        # Describe energy level from scalar
        if mean_energy > 0.6:
            energy_desc = "energetic"
        elif mean_energy > 0.3:
            energy_desc = "calm"
        else:
            energy_desc = "low-energy"

        prompt = (
            "Summarize this interaction session in one warm, diary-style sentence.\n"
            "\n"
            "Session facts (no raw text available):\n"
            f"- Duration: {data.get('duration_min', 0):.0f} minutes\n"
            f"- Messages exchanged: {data.get('message_count', 0)}\n"
            f"- Topics discussed: {', '.join(topics[:5])}\n"
            f"- Emotional arc: started {start_emotion}, ended {end_emotion}\n"
            f"- Engagement: {mean_engagement:.0%}\n"
            f"- User seemed: {energy_desc}\n"
            "\n"
            "Write a warm, diary-style one-sentence summary. Do not fabricate details."
        )

        assert self.cloud_client is not None  # guarded by caller
        response = await self.cloud_client.generate_session_summary(
            {"prompt": prompt}
        )

        # Ensure we return a string regardless of response shape
        if isinstance(response, str):
            return response.strip()
        logger.warning(
            "Unexpected cloud response type %s; falling back to template",
            type(response).__name__,
        )
        return self._template_summary(data)

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _template_summary(self, data: dict[str, Any]) -> str:
        """Deterministic template-based summary when cloud is unavailable.

        Produces a readable one-sentence summary from engagement level,
        message count, and top topic keywords.

        Parameters
        ----------
        data:
            Session metadata dictionary.

        Returns
        -------
        str
            Template-generated summary sentence.
        """
        topics = data.get("topics", ["general conversation"])
        engagement = data.get("mean_engagement", 0.5)
        msg_count = data.get("message_count", 0)
        emotion = data.get("dominant_emotion", "neutral")

        # Engagement descriptor
        if engagement > 0.7:
            eng_desc = "a highly engaged"
        elif engagement > 0.4:
            eng_desc = "an engaged"
        else:
            eng_desc = "a quiet"

        # Topic string (max 3)
        topic_str = ", ".join(topics[:3]) if topics else "general topics"

        # Emotion colouring
        emotion_suffix = ""
        if emotion and emotion != "neutral":
            emotion_suffix = f", with a {emotion} tone throughout"

        return (
            f"{eng_desc} session with {msg_count} messages about "
            f"{topic_str}{emotion_suffix}."
        ).capitalize()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cloud_is_available(self) -> bool:
        """Check whether the cloud client is configured and reachable.

        Inspects the ``is_available`` attribute/property on the cloud
        client if it exists; otherwise assumes available.
        """
        if self.cloud_client is None:
            return False
        if hasattr(self.cloud_client, "is_available"):
            return bool(self.cloud_client.is_available)
        # If the client exists but lacks is_available, assume usable
        return True
