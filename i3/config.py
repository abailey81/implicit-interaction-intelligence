"""Configuration loader for Implicit Interaction Intelligence (I3).

Provides a fully-typed, immutable configuration system built on Pydantic v2
BaseModel with YAML file loading, multi-file overlay merging, environment
variable overrides, and deterministic seed management.

Usage:
    from i3.config import load_config

    # Load default config
    cfg = load_config("configs/default.yaml")

    # Load with demo overlay
    cfg = load_config("configs/default.yaml", overlays=["configs/demo.yaml"])

    # Access typed fields
    print(cfg.encoder.embedding_dim)  # 64
    print(cfg.slm.training.learning_rate)  # 0.0003
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


# ---------------------------------------------------------------------------
# Helper: deep-merge two dicts (overlay wins)
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into a copy of *base*.

    For nested dicts the merge is recursive; for all other types the overlay
    value replaces the base value entirely.
    """
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Pydantic models -- one per YAML section
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Top-level project metadata."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    seed: int = 42


class InteractionConfig(BaseModel):
    """Interaction capture settings."""

    model_config = ConfigDict(frozen=True)

    feature_window: int = 10
    keystroke_features: bool = True
    linguistic_features: bool = True
    feature_dim: int = 32


class EncoderConfig(BaseModel):
    """Temporal Convolutional Encoder settings."""

    model_config = ConfigDict(frozen=True)

    architecture: str = "tcn"
    input_dim: int = 32
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 64, 64, 64])
    kernel_size: int = 3
    dilations: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.1
    embedding_dim: int = 64
    use_layer_norm: bool = True
    use_residual: bool = True

    @field_validator("dropout")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v


class UserModelConfig(BaseModel):
    """User baseline and deviation tracking settings."""

    model_config = ConfigDict(frozen=True)

    session_ema_alpha: float = 0.3
    longterm_ema_alpha: float = 0.1
    baseline_warmup: int = 5
    deviation_threshold: float = 1.5
    max_history_sessions: int = 50


class CognitiveLoadConfig(BaseModel):
    """Cognitive load adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    min_response_length: int = 10
    max_response_length: int = 150
    vocabulary_levels: int = 3


class StyleMirrorConfig(BaseModel):
    """Style mirroring adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    dimensions: int = 4
    adaptation_rate: float = 0.2


class EmotionalToneConfig(BaseModel):
    """Emotional tone adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    warmth_range: tuple[float, float] = (0.0, 1.0)
    default: float = 0.5

    @field_validator("warmth_range", mode="before")
    @classmethod
    def _coerce_warmth_range(cls, v: Any) -> tuple[float, float]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (float(v[0]), float(v[1]))
        raise ValueError(f"warmth_range must be a 2-element sequence, got {v}")


class AccessibilityConfig(BaseModel):
    """Accessibility adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    detection_threshold: float = 0.7
    simplification_levels: int = 3


class AdaptationConfig(BaseModel):
    """Aggregated adaptation engine settings."""

    model_config = ConfigDict(frozen=True)

    cognitive_load: CognitiveLoadConfig = Field(default_factory=CognitiveLoadConfig)
    style_mirror: StyleMirrorConfig = Field(default_factory=StyleMirrorConfig)
    emotional_tone: EmotionalToneConfig = Field(default_factory=EmotionalToneConfig)
    accessibility: AccessibilityConfig = Field(default_factory=AccessibilityConfig)


class RouterConfig(BaseModel):
    """Contextual Thompson Sampling router settings."""

    model_config = ConfigDict(frozen=True)

    arms: list[str] = Field(default_factory=lambda: ["local_slm", "cloud_llm"])
    bandit_type: str = "contextual_thompson"
    context_dim: int = 12
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    exploration_bonus: float = 0.1
    min_cloud_complexity: float = 0.6
    privacy_override: bool = True


class SLMTrainingConfig(BaseModel):
    """SLM training hyper-parameters."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = 32
    learning_rate: float = 3.0e-4
    warmup_steps: int = 500
    max_steps: int = 50000
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    checkpoint_every: int = 5000


class SLMGenerationConfig(BaseModel):
    """SLM text generation parameters."""

    model_config = ConfigDict(frozen=True)

    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 100
    repetition_penalty: float = 1.2


class SLMQuantizationConfig(BaseModel):
    """SLM quantization settings for edge deployment."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    dtype: str = "int8"
    method: str = "dynamic"


class SLMConfig(BaseModel):
    """Custom Small Language Model settings."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = 8000
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    conditioning_dim: int = 64
    adaptation_dim: int = 8
    cross_attention_heads: int = 2
    use_pre_ln: bool = True
    tie_weights: bool = True
    training: SLMTrainingConfig = Field(default_factory=SLMTrainingConfig)
    generation: SLMGenerationConfig = Field(default_factory=SLMGenerationConfig)
    quantization: SLMQuantizationConfig = Field(default_factory=SLMQuantizationConfig)


class CloudConfig(BaseModel):
    """Cloud LLM integration settings."""

    model_config = ConfigDict(frozen=True)

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 200
    timeout: float = 10.0
    fallback_on_error: bool = True


class DiaryConfig(BaseModel):
    """Interaction diary storage settings."""

    model_config = ConfigDict(frozen=True)

    db_path: str = "data/diary.db"
    max_entries: int = 1000
    session_summary_model: str = "cloud"
    encrypt_at_rest: bool = True


class PrivacyConfig(BaseModel):
    """Privacy and data protection settings."""

    model_config = ConfigDict(frozen=True)

    strip_pii: bool = True
    never_store_raw_text: bool = True
    encrypt_embeddings: bool = True
    encryption_key_env: str = "I3_ENCRYPTION_KEY"

    def get_encryption_key(self) -> str | None:
        """Retrieve the encryption key from the environment variable.

        Returns:
            The key string, or ``None`` if the variable is not set.
        """
        return os.environ.get(self.encryption_key_env)


class TargetDeviceConfig(BaseModel):
    """Hardware target for edge-device profiling."""

    model_config = ConfigDict(frozen=True)

    name: str
    memory_mb: int
    tops: float


class ProfilingConfig(BaseModel):
    """Profiling and benchmarking settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    benchmark_iterations: int = 100
    target_devices: list[TargetDeviceConfig] = Field(default_factory=list)


class ServerConfig(BaseModel):
    """FastAPI / WebSocket server settings."""

    model_config = ConfigDict(frozen=True)

    # Default to loopback; operators must explicitly set 0.0.0.0 to
    # accept external connections.  See SECURITY.md for the rationale.
    host: str = "127.0.0.1"
    port: int = 8000
    # Explicit allow-list; wildcard "*" is rejected at server start
    # unless I3_ALLOW_CORS_WILDCARD=1.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    )
    websocket_ping_interval: int = 30


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    """Root configuration aggregating all sections.

    All sub-models are frozen (immutable) once constructed so that
    configuration cannot be accidentally mutated at runtime.
    """

    model_config = ConfigDict(frozen=True)

    project: ProjectConfig = Field(default_factory=lambda: ProjectConfig(name="I3", version="0.0.0"))
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    user_model: UserModelConfig = Field(default_factory=UserModelConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    slm: SLMConfig = Field(default_factory=SLMConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    diary: DiaryConfig = Field(default_factory=DiaryConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    """Set deterministic seeds for reproducibility across frameworks.

    Sets seeds for the Python ``random`` module, NumPy, and PyTorch (CPU and,
    if available, CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is optional at import time (e.g., during tests without GPU)
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    path: str | Path,
    overlays: list[str | Path] | None = None,
    *,
    set_seeds: bool = True,
) -> Config:
    """Load and validate the I3 configuration from one or more YAML files.

    Args:
        path: Path to the base YAML configuration file.
        overlays: Optional list of additional YAML files whose values are
            deep-merged on top of the base configuration.  Later files in
            the list take precedence over earlier ones.
        set_seeds: If ``True`` (default), set deterministic seeds for
            ``random``, ``numpy``, and ``torch`` using ``project.seed``.

    Returns:
        A fully validated, immutable :class:`Config` instance.

    Raises:
        ConfigError: If a file cannot be read, parsed, or fails validation.
    """
    base_path = Path(path)
    if not base_path.is_file():
        raise ConfigError(f"Configuration file not found: {base_path}")

    try:
        with open(base_path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML from {base_path}: {exc}") from exc

    # Apply overlays in order
    for overlay_path in overlays or []:
        overlay_file = Path(overlay_path)
        if not overlay_file.is_file():
            raise ConfigError(f"Overlay file not found: {overlay_file}")
        try:
            with open(overlay_file) as fh:
                overlay_data: dict[str, Any] = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML from {overlay_file}: {exc}") from exc
        data = _deep_merge(data, overlay_data)

    # Apply environment variable overrides for sensitive values
    encryption_key_env = (
        data.get("privacy", {}).get("encryption_key_env", "I3_ENCRYPTION_KEY")
    )
    if os.environ.get(encryption_key_env):
        # Key exists in environment -- no action needed at config level,
        # but validate it is set when encryption is enabled.
        pass
    elif data.get("privacy", {}).get("encrypt_embeddings", True):
        import warnings
        warnings.warn(
            f"Environment variable '{encryption_key_env}' is not set but "
            f"encrypt_embeddings is enabled.  Encryption operations will fail "
            f"at runtime unless the key is provided.",
            stacklevel=2,
        )

    # Validate and construct the immutable Config
    try:
        config = Config(**data)
    except Exception as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc

    # Seed management
    if set_seeds:
        _set_seeds(config.project.seed)

    return config
