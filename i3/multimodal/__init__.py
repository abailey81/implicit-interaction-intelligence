"""Multi-modal perception port for Implicit Interaction Intelligence (I³).

The I³ TCN encoder is modality-agnostic: it consumes any sequence of
8-dimensional feature groups (keystroke dynamics, message content, session
dynamics, deviation metrics) and produces a 64-dim user-state embedding.  The
insight motivating this package is that **the keystroke-dynamics group is only
one of many possible input modalities**.  Voice prosody, touchscreen pressure
patterns, gaze dwell, and wearable accelerometry all fit the same 8-dim
per-modality contract and can be fused into the encoder by concatenation plus
a learned modality embedding.

Reference: THE_COMPLETE_BRIEF §11 ("Multi-modal port: TCN is modality-agnostic
— keystroke → touch pressure → gaze duration → accelerometer").

All modality extractors **soft-import** their heavy dependencies (``librosa``,
``mediapipe``, ...).  When a dependency is unavailable, the extractor returns a
zero vector and logs a ``logger.warning`` so the pipeline degrades gracefully
rather than crashing.
"""

from __future__ import annotations

from i3.multimodal.accelerometer import (
    AccelerometerFeatureExtractor,
    AccelerometerFeatureVector,
)
from i3.multimodal.fusion import MODALITY_INDEX, ModalityEmbedding, ModalityFusion
from i3.multimodal.gaze import GazeFeatureExtractor, GazeFeatureVector
from i3.multimodal.gaze_classifier import (
    GAZE_LABELS,
    GazeClassifier,
    GazeFeatures,
    GazeFineTuneHead,
    load_pretrained_backbone,
)
from i3.multimodal.prosody import (
    GAZE_FEATURE_KEYS,
    PROSODY_FEATURE_KEYS,
    GazeEncoder,
    MultimodalFusion,
    ProsodyEncoder,
    ProsodyFeatures,
    gaze_payload_to_tensor,
    prosody_payload_to_tensor,
    validate_gaze_payload,
    validate_prosody_payload,
)
from i3.multimodal.touch import TouchFeatureExtractor, TouchFeatureVector
from i3.multimodal.voice import VoiceFeatureExtractor, VoiceFeatureVector

__all__ = [
    "GAZE_FEATURE_KEYS",
    "GAZE_LABELS",
    "MODALITY_INDEX",
    "PROSODY_FEATURE_KEYS",
    "AccelerometerFeatureExtractor",
    "AccelerometerFeatureVector",
    "GazeClassifier",
    "GazeEncoder",
    "GazeFeatureExtractor",
    "GazeFeatureVector",
    "GazeFeatures",
    "GazeFineTuneHead",
    "ModalityEmbedding",
    "ModalityFusion",
    "MultimodalFusion",
    "ProsodyEncoder",
    "ProsodyFeatures",
    "TouchFeatureExtractor",
    "TouchFeatureVector",
    "VoiceFeatureExtractor",
    "VoiceFeatureVector",
    "gaze_payload_to_tensor",
    "load_pretrained_backbone",
    "prosody_payload_to_tensor",
    "validate_gaze_payload",
    "validate_prosody_payload",
]
