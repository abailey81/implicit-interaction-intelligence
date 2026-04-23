"""Interaction monitoring and feature engineering layer for I3.

This package provides the core infrastructure for capturing, processing, and
analysing *how* users interact -- keystroke dynamics, typing patterns, linguistic
complexity -- rather than merely *what* they say.

Exported classes
----------------
InteractionEvent        Low-level interaction event envelope.
InteractionFeatureVector 32-dimensional feature vector extracted from interaction patterns.
KeystrokeEvent          Single keystroke timing record.
InteractionMonitor      Per-user keystroke buffer and session tracker.
FeatureExtractor        Computes the 32-dim feature vector from raw data.
LinguisticAnalyzer      From-scratch NLP utilities (no external libraries).
"""

from i3.interaction.features import FeatureExtractor
from i3.interaction.linguistic import LinguisticAnalyzer
from i3.interaction.monitor import InteractionMonitor
from i3.interaction.types import (
    FEATURE_NAMES,
    InteractionEvent,
    InteractionFeatureVector,
    KeystrokeEvent,
)

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractor",
    "InteractionEvent",
    "InteractionFeatureVector",
    "InteractionMonitor",
    "KeystrokeEvent",
    "LinguisticAnalyzer",
]
