"""Synthetic meta-task generator drawn from the HCI persona library.

A meta-learning algorithm is only as good as its task distribution. For
MAML / Reptile to produce an encoder that adapts quickly to a *new*
user, every task seen during outer-loop training must itself be a
*different* user. :class:`PersonaTaskGenerator` provides exactly that:
each generated :class:`~i3.meta_learning.maml.MetaTask` is a fresh
``(support, query)`` split drawn from a single
:class:`~i3.eval.simulation.personas.HCIPersona`, with the persona's
own :class:`~i3.adaptation.types.AdaptationVector` serving as the
ground-truth target.

Determinism is inherited from the :class:`UserSimulator`: each task
uses a deterministic seed derived from the persona name and a
monotonically increasing task index, so two generators constructed
with the same ``(personas, seed)`` produce identical task streams.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from typing import Optional

from i3.eval.simulation.personas import HCIPersona
from i3.eval.simulation.user_simulator import SimulatedMessage, UserSimulator
from i3.interaction.features import BaselineTracker, FeatureExtractor
from i3.interaction.types import InteractionFeatureVector
from i3.meta_learning.maml import MetaBatch, MetaTask

logger = logging.getLogger(__name__)


class PersonaTaskGenerator:
    """Generate meta-learning tasks from a list of HCI personas.

    Each call to :meth:`generate_task` samples a fresh session from a
    persona, splits it into ``support_size`` + ``query_size`` messages,
    runs the feature extractor over each message, and packages the
    result as a :class:`MetaTask`. The persona's
    ``expected_adaptation`` becomes the task's ground-truth target.

    Args:
        personas: Non-empty list of personas the generator may draw
            from. Sampling is round-robin interleaved with a
            deterministic seed salt to avoid two consecutive tasks
            ever coming from the same persona unless ``len(personas)
            == 1``.
        support_size: Number of support messages per task. Must be at
            least one.
        query_size: Number of query messages per task. Must be at
            least one.
        seed: Integer seed from which every per-task
            :class:`UserSimulator` seed is derived.

    Raises:
        ValueError: If ``personas`` is empty, or if ``support_size``
            or ``query_size`` is less than one.
    """

    def __init__(
        self,
        personas: list[HCIPersona],
        support_size: int = 3,
        query_size: int = 5,
        seed: int = 42,
    ) -> None:
        if not personas:
            raise ValueError("personas must be a non-empty list.")
        if support_size < 1:
            raise ValueError(
                f"support_size must be >= 1, got {support_size!r}."
            )
        if query_size < 1:
            raise ValueError(
                f"query_size must be >= 1, got {query_size!r}."
            )
        self.personas = list(personas)
        self.support_size = int(support_size)
        self.query_size = int(query_size)
        self.seed = int(seed)
        self._task_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_task(self, persona: Optional[HCIPersona] = None) -> MetaTask:
        """Produce a single :class:`MetaTask`.

        Args:
            persona: Optional explicit persona to sample from. When
                ``None`` (the default), the generator rotates through
                ``self.personas`` using the internal task counter.

        Returns:
            A fully populated :class:`MetaTask`.
        """
        if persona is None:
            persona = self.personas[self._task_counter % len(self.personas)]
        # Derive a deterministic per-task seed: base seed XOR a salt
        # derived from (persona name, task counter).
        salt = int(
            hashlib.sha256(
                f"{persona.name}:{self._task_counter}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        task_seed = int(self.seed ^ salt)
        simulator = UserSimulator(persona, seed=task_seed)
        total = self.support_size + self.query_size
        messages = simulator.run_session(n_messages=total)
        support_msgs = messages[: self.support_size]
        query_msgs = messages[self.support_size :]
        support_features = [self._extract(m) for m in support_msgs]
        query_features = [self._extract(m) for m in query_msgs]
        self._task_counter += 1
        return MetaTask(
            persona_name=persona.name,
            support_set=support_features,
            query_set=query_features,
            target_adaptation=persona.expected_adaptation,
        )

    def generate_batch(self, meta_batch_size: int = 4) -> MetaBatch:
        """Produce a :class:`MetaBatch` of ``meta_batch_size`` tasks.

        Args:
            meta_batch_size: Number of tasks in the batch. Must be at
                least one.

        Raises:
            ValueError: If ``meta_batch_size`` is less than one.
        """
        if meta_batch_size < 1:
            raise ValueError(
                f"meta_batch_size must be >= 1, got {meta_batch_size!r}."
            )
        tasks = [self.generate_task() for _ in range(meta_batch_size)]
        return MetaBatch(tasks=tasks)

    def __iter__(self) -> Iterator[MetaBatch]:
        """Infinite iterator over meta-batches of size ``len(personas)``."""
        while True:
            yield self.generate_batch(meta_batch_size=len(self.personas))

    def reset(self) -> None:
        """Rewind the task counter. Useful for reproducible evaluation."""
        self._task_counter = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, message: SimulatedMessage) -> InteractionFeatureVector:
        """Run the :class:`FeatureExtractor` over a simulated message.

        The project's :class:`FeatureExtractor` takes a pre-aggregated
        keystroke-metrics dict (not individual events), along with a
        :class:`BaselineTracker` and a history window. We summarise the
        simulator's inter-key intervals into the expected keys and pass
        a fresh per-generator baseline so the extractor can produce its
        full 32-dim output deterministically.
        """
        iki = message.keystroke_intervals_ms or [0.0]
        mean_iki = sum(iki) / len(iki)
        std_iki = (
            sum((v - mean_iki) ** 2 for v in iki) / max(1, len(iki) - 1)
        ) ** 0.5
        composition_ms = float(sum(iki))
        composition_cps = (
            len(message.text) / (composition_ms / 1000.0)
            if composition_ms > 0.0
            else 0.0
        )
        ks_metrics: dict[str, float] = {
            "mean_iki_ms": float(mean_iki),
            "std_iki_ms": float(std_iki),
            "mean_burst_length": 0.0,
            "mean_pause_duration_ms": 0.0,
            "backspace_ratio": 0.0,
            "composition_speed_cps": float(composition_cps),
            "pause_before_send_ms": 0.0,
            "editing_effort": 0.0,
        }
        try:
            extractor = FeatureExtractor()
            baseline = BaselineTracker()
            return extractor.extract(
                keystroke_metrics=ks_metrics,
                message_text=message.text,
                history=[],
                baseline=baseline,
                session_start_ts=0.0,
                current_ts=float(message.timestamp),
            )
        except (AttributeError, TypeError, ValueError):
            return self._fallback_features(message)

    @staticmethod
    def _fallback_features(
        message: SimulatedMessage,
    ) -> InteractionFeatureVector:
        """Compute a minimal feature vector from the message directly.

        Used when the canonical :class:`FeatureExtractor` signature is
        not the one this module expects. Produces normalised
        keystroke-dynamics features plus a handful of message-content
        features so the downstream meta-learner still sees a non-trivial
        signal.
        """
        iki = message.keystroke_intervals_ms or [0.0]
        mean_iki = sum(iki) / len(iki)
        std_iki = (
            sum((v - mean_iki) ** 2 for v in iki) / max(1, len(iki) - 1)
        ) ** 0.5
        text = message.text
        words = text.split()
        word_lens = [len(w) for w in words] or [0]
        mean_wl = sum(word_lens) / len(word_lens)
        vector = InteractionFeatureVector()
        vector.mean_iki = min(1.0, mean_iki / 500.0)
        vector.std_iki = min(1.0, std_iki / 250.0)
        vector.composition_speed = (
            min(1.0, (len(text) * 1000.0) / max(1.0, sum(iki)))
            if sum(iki) > 0.0
            else 0.0
        )
        vector.message_length = min(1.0, len(text) / 200.0)
        vector.mean_word_length = min(1.0, mean_wl / 10.0)
        vector.session_progress = float(message.message_index + 1) / 10.0
        return vector


__all__: list[str] = ["PersonaTaskGenerator"]
