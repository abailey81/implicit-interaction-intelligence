"""Common types for the intent-parsing layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Canonical action vocabulary — must match training/build_intent_dataset.py.
SUPPORTED_ACTIONS: tuple[str, ...] = (
    "set_timer", "set_alarm", "send_message", "play_music",
    "navigate", "weather_query", "call", "set_reminder",
    "control_device", "cancel", "unsupported",
)

# Per-action slot schemas (loose — model output is rejected if slots
# include keys not in this set, to catch hallucinated structure).
ACTION_SLOTS: dict[str, set[str]] = {
    "set_timer":      {"duration_seconds"},
    "set_alarm":      {"time"},
    "send_message":   {"recipient", "message"},
    "play_music":     {"genre", "artist"},
    "navigate":       {"location"},
    "weather_query":  {"location"},
    "call":           {"recipient", "video"},
    "set_reminder":   {"task", "time", "when"},
    "control_device": {"device", "state", "value"},
    "cancel":         set(),
    "unsupported":    set(),
}


@dataclass
class IntentResult:
    """Parsed intent for a single utterance."""

    raw_input: str
    raw_output: str  # full model generation (for debugging)
    parsed: dict[str, Any] | None  # the JSON object, or None if invalid
    valid_json: bool
    valid_action: bool
    valid_slots: bool
    action: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    backend: str = ""  # "qwen-lora" / "gemini-vertex"
    error: str | None = None

    @property
    def confidence(self) -> float:
        """A simple heuristic confidence: 1.0 only when fully valid."""
        score = 0.0
        if self.valid_json:
            score += 0.4
        if self.valid_action:
            score += 0.3
        if self.valid_slots:
            score += 0.3
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "action": self.action,
            "params": self.params,
            "valid_json": self.valid_json,
            "valid_action": self.valid_action,
            "valid_slots": self.valid_slots,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 2),
            "backend": self.backend,
            "error": self.error,
        }
