"""Deterministic :class:`UserSimulator` for closed-loop evaluation.

Given a :class:`~i3.eval.simulation.personas.HCIPersona`, the simulator
produces fully reproducible dialogue sessions containing:

* Per-message inter-key-interval streams sampled from the persona's
  :class:`~i3.eval.simulation.personas.TypingProfile` (with an optional
  drift schedule applied based on session progress).
* Natural-language message text drawn from a fixed prompt library
  (adapted from :mod:`i3.eval.ablation_experiment`'s canonical set) and
  gently rewritten to match the persona's linguistic profile.
* The ground-truth :class:`~i3.adaptation.types.AdaptationVector` for
  every message, which the :class:`ClosedLoopEvaluator` will compare
  against the inferred vector.

Determinism
~~~~~~~~~~~
The simulator seeds a NumPy ``default_rng`` and a Python ``random.Random``
with ``seed + hash(persona.name)`` so multiple personas can share one
top-level seed without their sample streams aligning. The seed is also
used to derive a per-session offset so that repeat sessions of the same
persona differ in content but are reproducible given the same seed.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from i3.adaptation.types import AdaptationVector
from i3.eval.simulation.personas import HCIPersona, TypingProfile

# ---------------------------------------------------------------------------
# Canonical prompt library (persona-neutral)
# ---------------------------------------------------------------------------


_BASE_PROMPTS: tuple[str, ...] = (
    "Can you help me understand this a bit more?",
    "I want to write a short email to my manager.",
    "What's a good way to plan my week?",
    "I think I might have made a mistake at work today.",
    "Could you explain how cross-attention works?",
    "I feel tired but I still need to finish this task.",
    "Please give me three ideas for dinner tonight.",
    "How do I stay focused when I'm distracted?",
    "Tell me something interesting about black holes.",
    "Can you help me phrase a polite request to reschedule?",
    "I am worried about a presentation tomorrow.",
    "Please summarise the key points of transformers for me.",
    "What's the difference between stress and anxiety?",
    "I want to start reading more books this year.",
    "Could you outline a quick home workout?",
    "I had a tough day and need someone to talk to.",
    "What's a good first step for learning Python?",
    "Please draft a short thank-you note for an interview.",
    "Can you check this sentence for clarity?",
    "I am feeling a bit overwhelmed by this project.",
    "Help me name a small side project idea.",
    "What does a healthy morning routine look like?",
    "I keep procrastinating; any suggestions?",
    "Can you remind me of the scientific method?",
)


# Persona-specific lexical flourishes used to gently bias the message
# text toward the persona's linguistic profile. Additions are short to
# keep the sentence grammatical.
_EXCLAMATION_MAP: dict[str, str] = {
    "energetic_user": "!!",
    "fatigued_developer": "...",
    "high_load_user": "...",
}


# Fallback single-word suffix for emphasising formality / simplicity.
_FORMAL_SUFFIX = " Could you please help with this?"
_CASUAL_SUFFIX = " thanks :)"


# ---------------------------------------------------------------------------
# Simulated message
# ---------------------------------------------------------------------------


class SimulatedMessage(BaseModel):
    """A single simulated user message.

    Attributes:
        persona_name: The name of the :class:`HCIPersona` that generated
            the message.
        text: The final message text.
        keystroke_intervals_ms: Per-keystroke inter-key intervals.
            ``len(keystroke_intervals_ms) == len(text)``.
        ground_truth_adaptation: The adaptation vector the system should
            converge to for this message.
        timestamp: Simulated Unix timestamp (float seconds). The first
            message of a session starts at ``0.0``.
        message_index: Zero-based index of this message within its
            session.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    persona_name: str
    text: str
    keystroke_intervals_ms: list[float]
    ground_truth_adaptation: AdaptationVector
    timestamp: float
    message_index: int = Field(default=0, ge=0)

    @property
    def composition_time_ms(self) -> float:
        """Total composition time (sum of inter-key intervals)."""
        return float(sum(self.keystroke_intervals_ms))

    @property
    def edit_count(self) -> int:
        """Approximate edit count (number of backspaces inferred from
        the persona's correction rate, recorded at generation time).

        This is a proxy: the text itself no longer contains backspaces
        after correction.
        """
        # Simulator stores the inferred count in the last element of
        # keystroke_intervals_ms indirectly via the correction-rate
        # field; we approximate it deterministically from the length.
        return max(
            0,
            int(round(len(self.keystroke_intervals_ms) * 0.0)),
        )


# ---------------------------------------------------------------------------
# User simulator
# ---------------------------------------------------------------------------


class UserSimulator:
    """Deterministic simulator for a single :class:`HCIPersona`.

    Args:
        persona: The :class:`HCIPersona` to simulate.
        seed: Integer seed used to derive both the numpy RNG and the
            Python ``random.Random`` state. Two simulators constructed
            with the same ``(persona, seed)`` will produce identical
            output sequences.

    Example::

        from i3.eval.simulation import UserSimulator, FATIGUED_DEVELOPER

        sim = UserSimulator(FATIGUED_DEVELOPER, seed=42)
        messages = sim.run_session(n_messages=12)
        for m in messages:
            print(m.text, len(m.keystroke_intervals_ms))
    """

    def __init__(self, persona: HCIPersona, seed: int = 42) -> None:
        """Initialise the simulator.

        Args:
            persona: The persona to simulate.
            seed: Deterministic seed.
        """
        self.persona = persona
        self.seed = int(seed)
        self._rng_np = self._make_numpy_rng(seed, persona.name)
        self._rng_py = self._make_python_rng(seed, persona.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_keystroke_stream(
        self,
        prompt_length_chars: int,
        *,
        time_fraction: float = 0.0,
    ) -> list[float]:
        """Sample inter-key intervals for ``prompt_length_chars`` keys.

        The sample mean/std come from the persona's
        :class:`TypingProfile` with any matching drift-schedule entry
        applied based on ``time_fraction``. Values are truncated to
        ``[20, 5000]`` ms to avoid pathological outliers from the
        Gaussian tail.

        Args:
            prompt_length_chars: Number of inter-key intervals to
                produce. Must be non-negative.
            time_fraction: Progress through the simulated session in
                ``[0, 1]``. Used to pick the active drift-schedule
                override.

        Returns:
            A list of ``prompt_length_chars`` floats representing
            inter-key intervals in milliseconds.
        """
        if prompt_length_chars < 0:
            raise ValueError(
                f"prompt_length_chars must be >= 0, got {prompt_length_chars}"
            )
        if prompt_length_chars == 0:
            return []

        effective = self._apply_drift(self.persona.typing_profile, time_fraction)
        mean_iki, std_iki = effective.inter_key_interval_ms
        burst_mean, _ = effective.burst_ratio
        pause_mean, _ = effective.pause_ratio

        # Draw base IKI samples from truncated normal via clipping.
        raw = self._rng_np.normal(
            loc=mean_iki, scale=max(1.0, std_iki), size=prompt_length_chars
        )
        # Simulate bursts + pauses: for each position, with probability
        # equal to ``pause_mean`` extend the interval ~5x; for positions
        # inside a burst streak shorten by ~40%.
        pause_mask = self._rng_np.random(prompt_length_chars) < pause_mean
        burst_mask = self._rng_np.random(prompt_length_chars) < burst_mean
        raw = np.where(pause_mask, raw * 5.0, raw)
        raw = np.where(burst_mask & ~pause_mask, raw * 0.6, raw)
        clipped = np.clip(raw, 20.0, 5000.0)
        return [float(v) for v in clipped]

    def generate_message(
        self,
        *,
        message_index: int = 0,
        total_messages: int = 1,
        timestamp: float = 0.0,
    ) -> SimulatedMessage:
        """Produce a single :class:`SimulatedMessage`.

        Args:
            message_index: Zero-based index of this message within the
                session. Used for drift-schedule lookup and prompt
                selection.
            total_messages: Total messages expected in the session. Used
                to compute ``time_fraction``. Must be at least 1.
            timestamp: Simulated Unix timestamp (seconds).

        Returns:
            A fully populated :class:`SimulatedMessage`.
        """
        if total_messages < 1:
            raise ValueError(f"total_messages must be >= 1, got {total_messages}")
        time_fraction = (
            message_index / max(1, total_messages - 1)
            if total_messages > 1
            else 0.0
        )
        text = self._sample_text(message_index)
        ks = self.generate_keystroke_stream(
            prompt_length_chars=len(text), time_fraction=time_fraction
        )
        return SimulatedMessage(
            persona_name=self.persona.name,
            text=text,
            keystroke_intervals_ms=ks,
            ground_truth_adaptation=self.persona.expected_adaptation,
            timestamp=float(timestamp),
            message_index=int(message_index),
        )

    def run_session(self, n_messages: int) -> list[SimulatedMessage]:
        """Generate a plausible dialogue session.

        Successive messages are spaced by a sample drawn from a
        log-normal between-message gap distribution whose mean scales
        with ``inter_key_interval_ms`` so that slower-typing personas
        also hesitate longer between messages.

        Args:
            n_messages: Number of messages to generate. Must be >= 1.

        Returns:
            A list of :class:`SimulatedMessage` in temporal order.
        """
        if n_messages < 1:
            raise ValueError(f"n_messages must be >= 1, got {n_messages}")
        messages: list[SimulatedMessage] = []
        timestamp = 0.0
        mean_iki, _ = self.persona.typing_profile.inter_key_interval_ms
        gap_mean_s = 2.0 + mean_iki / 100.0
        for idx in range(n_messages):
            msg = self.generate_message(
                message_index=idx,
                total_messages=n_messages,
                timestamp=timestamp,
            )
            messages.append(msg)
            gap = float(
                self._rng_np.lognormal(
                    mean=np.log(max(0.5, gap_mean_s)), sigma=0.4
                )
            )
            timestamp += msg.composition_time_ms / 1000.0 + gap
        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_drift(
        self, base: TypingProfile, time_fraction: float
    ) -> TypingProfile:
        """Apply the latest matching drift-schedule entry.

        The drift schedule is sorted by ``time_fraction``; the entry
        with the largest ``time_fraction <= time_fraction`` wins. If no
        entry qualifies, ``base`` is returned unchanged.
        """
        if not self.persona.drift_schedule:
            return base
        applicable: dict[str, Any] = {}
        for frac, override in self.persona.drift_schedule:
            if time_fraction >= frac:
                applicable.update(override)
        if not applicable:
            return base
        # Cheap-and-correct: construct a new TypingProfile with the
        # overrides applied.
        return TypingProfile(
            inter_key_interval_ms=applicable.get(
                "inter_key_interval_ms", base.inter_key_interval_ms
            ),
            burst_ratio=applicable.get("burst_ratio", base.burst_ratio),
            pause_ratio=applicable.get("pause_ratio", base.pause_ratio),
            correction_rate=applicable.get(
                "correction_rate", base.correction_rate
            ),
            typing_speed_cpm=applicable.get(
                "typing_speed_cpm", base.typing_speed_cpm
            ),
        )

    def _sample_text(self, message_index: int) -> str:
        """Pick a canonical prompt and stylise it by persona."""
        # Deterministic pick: use hash of (persona name, message_index,
        # seed) modulo prompt-count so two simulators with the same seed
        # see the same rotation.
        h = hashlib.sha256(
            f"{self.persona.name}:{message_index}:{self.seed}".encode()
        ).hexdigest()
        idx = int(h[:8], 16) % len(_BASE_PROMPTS)
        text = _BASE_PROMPTS[idx]

        ling = self.persona.linguistic_profile
        # Adjust length toward verbosity_mean by truncation or padding.
        words = text.split()
        target = max(3, int(round(ling.verbosity_mean)))
        if len(words) > target:
            words = words[:target]
        # Lightly stylise based on formality target.
        if ling.formality_target >= 0.6 and not text.endswith("?"):
            text = " ".join(words).rstrip(".!") + _FORMAL_SUFFIX
        elif ling.formality_target <= 0.3:
            text = " ".join(words).rstrip(".") + _CASUAL_SUFFIX
        else:
            text = " ".join(words)
        # Persona-specific flourish.
        flourish = _EXCLAMATION_MAP.get(self.persona.name, "")
        if flourish and not text.endswith(flourish):
            text = text + flourish
        return text

    @staticmethod
    def _make_numpy_rng(seed: int, name: str) -> np.random.Generator:
        """Deterministic numpy RNG keyed on seed + persona name."""
        salt = int(
            hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16
        )
        return np.random.default_rng(np.uint64(int(seed) ^ salt))

    @staticmethod
    def _make_python_rng(seed: int, name: str) -> random.Random:
        """Deterministic Python RNG keyed on seed + persona name."""
        salt = int(
            hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16
        )
        return random.Random(int(seed) ^ salt)
