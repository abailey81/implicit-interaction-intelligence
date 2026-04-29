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
from urllib.parse import urlparse

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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

    name: str = Field(..., min_length=1, description="Human-readable project name")
    version: str = Field(..., min_length=1, description="Semantic version string")
    # SEC: bound seed to a non-negative 32-bit integer; numpy's seed API
    # rejects values outside [0, 2**32-1] at runtime, so failing fast at
    # config load is preferable to a stack trace deep inside training.
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    # PERF: device selection knob consumed by training scripts and the
    # runtime bootstrap.  Accepted values: 'auto' (CUDA > MPS > CPU),
    # 'cpu', 'cuda', 'cuda:N', 'mps'.  Resolution lives in
    # :func:`i3.runtime.device.pick_device`.
    device: str = Field(default="auto", min_length=1)
    # PERF: enable torch.amp mixed-precision training when CUDA/MPS are
    # available.  Honoured by the training loops; always a no-op on CPU.
    mixed_precision: bool = True


class InteractionConfig(BaseModel):
    """Interaction capture settings."""

    model_config = ConfigDict(frozen=True)

    # SEC: positive bounds on every dimensional field — a zero or negative
    # value would silently disable the feature pipeline at runtime.
    feature_window: int = Field(default=10, gt=0)
    keystroke_features: bool = True
    linguistic_features: bool = True
    feature_dim: int = Field(default=32, gt=0)


class EncoderConfig(BaseModel):
    """Temporal Convolutional Encoder settings."""

    model_config = ConfigDict(frozen=True)

    architecture: str = Field(default="tcn", min_length=1)
    input_dim: int = Field(default=32, gt=0)
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 64, 64, 64])
    kernel_size: int = Field(default=3, gt=0)
    dilations: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    # SEC: dropout is a probability — Pydantic enforces [0, 1) and the
    # custom validator below rejects exactly 1.0 (which would zero every
    # activation and silently break training).
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    embedding_dim: int = Field(default=64, gt=0)
    use_layer_norm: bool = True
    use_residual: bool = True

    @field_validator("dropout")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v

    @field_validator("hidden_dims", "dilations")
    @classmethod
    def _positive_list(cls, v: list[int]) -> list[int]:
        # SEC: every entry in a topology list must be strictly positive —
        # a zero hidden dim collapses the layer to a no-op.
        if not v:
            raise ValueError("list must contain at least one element")
        if any(x <= 0 for x in v):
            raise ValueError(f"all entries must be > 0, got {v}")
        return v


class UserModelConfig(BaseModel):
    """User baseline and deviation tracking settings."""

    model_config = ConfigDict(frozen=True)

    # SEC: EMA smoothing constants must lie in (0, 1] — alpha=0 freezes
    # the baseline forever, alpha>1 produces an unstable filter.
    session_ema_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    longterm_ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    # SEC: baseline_warmup must be > 0 — a value of 0 would short-circuit
    # the warm-up phase and emit deviation scores against an empty mean.
    baseline_warmup: int = Field(default=5, gt=0)
    deviation_threshold: float = Field(default=1.5, gt=0.0)
    max_history_sessions: int = Field(default=50, gt=0)


class CognitiveLoadConfig(BaseModel):
    """Cognitive load adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    min_response_length: int = Field(default=10, gt=0)
    max_response_length: int = Field(default=150, gt=0)
    vocabulary_levels: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _check_min_le_max(self) -> CognitiveLoadConfig:
        # SEC: a min > max produces an empty range and an unreachable
        # generation constraint at runtime.
        if self.min_response_length > self.max_response_length:
            raise ValueError(
                f"min_response_length ({self.min_response_length}) must be "
                f"<= max_response_length ({self.max_response_length})"
            )
        return self


class StyleMirrorConfig(BaseModel):
    """Style mirroring adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    dimensions: int = Field(default=4, gt=0)
    # SEC: adaptation_rate is a probability — bound to [0, 1].
    # Iter 40: raised from 0.2 → 0.35 so consistent declarative
    # messages cross the directness > 0.7 threshold within 2 turns
    # instead of 4.  The previous rate left the StyleMirror lagging
    # so far behind real user style that the post-processor's
    # directness/verbosity hooks rarely fired in short demo
    # sessions.  0.35 is the empirical sweet spot — fast enough
    # for single-message responsiveness, slow enough that one
    # atypical message doesn't whiplash the style.
    adaptation_rate: float = Field(default=0.35, ge=0.0, le=1.0)


class EmotionalToneConfig(BaseModel):
    """Emotional tone adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    warmth_range: tuple[float, float] = (0.0, 1.0)
    default: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("warmth_range", mode="before")
    @classmethod
    def _coerce_warmth_range(cls, v: Any) -> tuple[float, float]:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            # SEC: warmth_range must be a non-degenerate interval inside [0, 1].
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"warmth_range must satisfy 0 <= lo <= hi <= 1, got {v}"
                )
            return (lo, hi)
        raise ValueError(f"warmth_range must be a 2-element sequence, got {v}")


class AccessibilityConfig(BaseModel):
    """Accessibility adaptation parameters."""

    model_config = ConfigDict(frozen=True)

    # Iter 47 — lowered from 0.7 to 0.5.  The score is the mean of
    # four difficulty signals (iki slowdown, speed drop, backspace
    # ratio, editing effort).  Each signal in practice tops out
    # around 0.5–0.8 even for clearly motor-impaired users, so
    # ``mean ≥ 0.7`` required all four signals to be near-maximum
    # simultaneously and the path almost never fired.  At 0.5 the
    # path activates for users whose typing actually shows
    # multiple signs of motor difficulty without being so loose
    # it trips on momentarily-distracted typing.
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    simplification_levels: int = Field(default=3, gt=0)


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
    bandit_type: str = Field(default="contextual_thompson", min_length=1)
    context_dim: int = Field(default=12, gt=0)
    # SEC: Beta-distribution priors for the cold-start Beta-Bernoulli
    # path inside the bandit — must be strictly positive.
    prior_alpha: float = Field(default=1.0, gt=0.0)
    prior_beta: float = Field(default=1.0, gt=0.0)
    # SEC (H-9, 2026-04-23 audit): Gaussian **precision** prior on the
    # Laplace-approximated logistic regression weights.  This is a
    # different quantity from ``prior_alpha`` (the Beta prior); mixing
    # them produced coupled mistuning.  Default ``1.0`` matches the
    # bandit's prior behaviour before the audit.
    prior_precision: float = Field(default=1.0, gt=0.0)
    exploration_bonus: float = Field(default=0.1, ge=0.0)
    min_cloud_complexity: float = Field(default=0.6, ge=0.0, le=1.0)
    privacy_override: bool = True

    @field_validator("arms")
    @classmethod
    def _arms_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("router.arms must contain at least one arm")
        if len(set(v)) != len(v):
            raise ValueError(f"router.arms must be unique, got {v}")
        return v


class SLMTrainingConfig(BaseModel):
    """SLM training hyper-parameters."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(default=32, gt=0)
    # SEC: learning_rate strictly > 0 — a zero LR makes training a no-op
    # but the loss curve still decreases (numerical noise), masking the bug.
    learning_rate: float = Field(default=3.0e-4, gt=0.0)
    warmup_steps: int = Field(default=500, ge=0)
    max_steps: int = Field(default=50_000, gt=0)
    gradient_clip: float = Field(default=1.0, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    checkpoint_every: int = Field(default=5000, gt=0)

    @model_validator(mode="after")
    def _warmup_le_max(self) -> SLMTrainingConfig:
        if self.warmup_steps > self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be "
                f"<= max_steps ({self.max_steps})"
            )
        return self


class SLMGenerationConfig(BaseModel):
    """SLM text generation parameters."""

    model_config = ConfigDict(frozen=True)

    # SEC: temperature must be > 0 (=0 collapses to greedy and is handled
    # via top_k=1 elsewhere); top_p in (0, 1]; top_k > 0.
    temperature: float = Field(default=0.8, gt=0.0, le=10.0)
    top_k: int = Field(default=50, gt=0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=100, gt=0)
    repetition_penalty: float = Field(default=1.2, gt=0.0)


class SLMQuantizationConfig(BaseModel):
    """SLM quantization settings for edge deployment."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    dtype: str = Field(default="int8")
    method: str = Field(default="dynamic")

    @field_validator("dtype")
    @classmethod
    def _dtype_supported(cls, v: str) -> str:
        # SEC: only allow dtypes the runtime quantizer actually supports.
        allowed = {"int8", "int4", "fp16", "bf16"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("method")
    @classmethod
    def _method_supported(cls, v: str) -> str:
        allowed = {"dynamic", "static", "qat"}
        if v not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}, got {v!r}")
        return v


class SLMConfig(BaseModel):
    """Custom Small Language Model settings."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = Field(default=8000, gt=0)
    max_seq_len: int = Field(default=256, gt=0)
    d_model: int = Field(default=256, gt=0)
    n_heads: int = Field(default=4, gt=0)
    n_layers: int = Field(default=4, gt=0)
    d_ff: int = Field(default=512, gt=0)
    # SEC: dropout matches the encoder constraint — [0, 1) is the only
    # mathematically valid range.
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    conditioning_dim: int = Field(default=64, gt=0)
    adaptation_dim: int = Field(default=8, gt=0)
    cross_attention_heads: int = Field(default=2, gt=0)
    use_pre_ln: bool = True
    tie_weights: bool = True
    training: SLMTrainingConfig = Field(default_factory=SLMTrainingConfig)
    generation: SLMGenerationConfig = Field(default_factory=SLMGenerationConfig)
    quantization: SLMQuantizationConfig = Field(default_factory=SLMQuantizationConfig)

    @model_validator(mode="after")
    def _check_head_dim(self) -> SLMConfig:
        # SEC: multi-head attention requires d_model to be divisible by
        # n_heads (and by cross_attention_heads).  Otherwise PyTorch raises
        # a confusing reshape error inside the model.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.d_model % self.cross_attention_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"cross_attention_heads ({self.cross_attention_heads})"
            )
        # SEC: tied input/output embeddings require vocab_size and d_model
        # to align with the embedding table — both must be > 0 (already
        # enforced) and the generation horizon must fit the context.
        if self.generation.max_new_tokens >= self.max_seq_len:
            raise ValueError(
                f"generation.max_new_tokens ({self.generation.max_new_tokens}) "
                f"must be < max_seq_len ({self.max_seq_len})"
            )
        return self


class CloudConfig(BaseModel):
    """Cloud LLM integration settings."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(default="anthropic", min_length=1)
    # SEC (M-3, 2026-04-23 audit): default matches the brief-§8-locked
    # id used throughout ``configs/default.yaml`` and the CHANGELOG.
    # Prior default (``claude-sonnet-4-20250514``) drifted from the YAML.
    model: str = Field(default="claude-sonnet-4-5", min_length=1)
    max_tokens: int = Field(default=200, gt=0, le=200_000)
    # SEC: positive timeout — a 0 or negative value would block forever
    # or raise inside httpx with an unhelpful message.
    timeout: float = Field(default=10.0, gt=0.0)
    fallback_on_error: bool = True


class DiaryConfig(BaseModel):
    """Interaction diary storage settings."""

    model_config = ConfigDict(frozen=True)

    db_path: str = Field(default="data/diary.db", min_length=1)
    max_entries: int = Field(default=1000, gt=0)
    session_summary_model: str = Field(default="cloud", min_length=1)
    encrypt_at_rest: bool = True

    @field_validator("db_path")
    @classmethod
    def _no_path_traversal(cls, v: str) -> str:
        # SEC: reject parent-directory components.  An attacker who can
        # influence the YAML (e.g. via a crafted overlay) could otherwise
        # cause the diary to be written outside the project tree.
        if ".." in Path(v).parts:
            raise ValueError(f"db_path must not contain '..' components: {v!r}")
        return v


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

    name: str = Field(..., min_length=1)
    # SEC: device memory and compute throughput must be strictly positive
    # (memory) / non-negative (tops) — zero or negative values would
    # short-circuit the profiler's feasibility checks.
    memory_mb: int = Field(..., gt=0)
    tops: float = Field(..., ge=0.0)


class ProfilingConfig(BaseModel):
    """Profiling and benchmarking settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    benchmark_iterations: int = Field(default=100, gt=0)
    target_devices: list[TargetDeviceConfig] = Field(default_factory=list)


class ServerConfig(BaseModel):
    """FastAPI / WebSocket server settings."""

    model_config = ConfigDict(frozen=True)

    # Default to loopback; operators must explicitly set 0.0.0.0 to
    # accept external connections.  See SECURITY.md for the rationale.
    host: str = Field(default="127.0.0.1", min_length=1)
    # SEC: bind port must be a valid TCP port.  Ports < 1024 require
    # root on POSIX and are flagged as a footgun, but we allow them with
    # a documented warning rather than refusing outright.
    port: int = Field(default=8000, ge=1, le=65535)
    # Explicit allow-list; wildcard "*" is rejected at server start
    # unless I3_ALLOW_CORS_WILDCARD=1.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    )
    websocket_ping_interval: int = Field(default=30, gt=0)

    @field_validator("cors_origins")
    @classmethod
    def _validate_cors_origins(cls, v: list[str]) -> list[str]:
        """Each entry must be either '*' or a parseable http(s) URL.

        SEC: catches typos like ``localhost:8000`` (missing scheme) or
        ``http:/localhost`` (single slash) at config-load time rather
        than letting Starlette silently treat them as opaque strings.
        """
        cleaned: list[str] = []
        for origin in v:
            if not isinstance(origin, str) or not origin.strip():
                raise ValueError(f"cors_origins entry must be a non-empty str, got {origin!r}")
            origin = origin.strip()
            if origin == "*":
                cleaned.append(origin)
                continue
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(
                    f"cors_origins entry must be 'http(s)://host[:port]', got {origin!r}"
                )
            cleaned.append(origin)
        return cleaned

    @field_validator("host")
    @classmethod
    def _validate_host(cls, v: str) -> str:
        # SEC: 0.0.0.0 is allowed but flagged in audit logs by the
        # caller (server/app.py).  Reject empty / whitespace-only hosts
        # which would silently bind to all interfaces under uvicorn.
        if not v or not v.strip():
            raise ValueError("server.host must be a non-empty string")
        return v.strip()


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    """Root configuration aggregating all sections.

    All sub-models are frozen (immutable) once constructed so that
    configuration cannot be accidentally mutated at runtime.  The root
    enforces ``extra="forbid"`` so a typoed top-level section in
    ``configs/default.yaml`` fails loudly at load time rather than
    silently dropping the real section's settings (M-2, 2026-04-23
    audit).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    # V2 sections consumed by ``i3/slm/train_v2.py`` only.  Kept as free-
    # form dicts here (not Pydantic sub-models) because the trainer owns
    # their schema and the runtime pipeline doesn't look at them; adding
    # them keeps ``extra="forbid"`` happy without duplicating the
    # TrainingV2/ModelV2 validators across two files.
    slm_v2: dict | None = None
    training_v2: dict | None = None


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    """Set deterministic seeds for reproducibility across frameworks.

    Sets seeds for the Python ``random`` module, NumPy, and PyTorch (CPU and,
    if available, CUDA).  Also enables cuDNN deterministic mode so that
    convolution kernels select reproducible algorithms — note this can
    cost up to ~30% throughput on GPU but is required for bit-exact
    replays of training runs.
    """
    # SEC: PYTHONHASHSEED is read by CPython at startup; setting it here
    # affects only child processes spawned later, not the current one.
    # We still write it so that downstream subprocess.Popen() runs use
    # the same hash seed.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # SEC: deterministic cuDNN — required for reproducible runs.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is optional at import time (e.g., during tests without GPU)
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Read and parse a single YAML file with strict, safe semantics.

    SEC notes:
        * Uses ``yaml.safe_load`` only — never ``yaml.load`` — to refuse
          arbitrary Python object instantiation tags.
        * Refuses non-regular files and dangling symlinks.  A *resolved*
          symlink that still points to a regular file is permitted (the
          common Linux deployment pattern), but the resolved target is
          re-validated.
        * Caps file size at 1 MiB to limit memory exhaustion via a
          maliciously huge YAML document.
    """
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"  Hint: pass an existing path to load_config()."
        )
    if not path.is_file():
        raise ConfigError(
            f"Configuration path is not a regular file: {path}"
        )

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ConfigError(f"Cannot stat configuration file {path}: {exc}") from exc

    # SEC: 1 MiB is far larger than any sane I3 config (default is ~3 KB).
    max_bytes = 1 * 1024 * 1024
    if size > max_bytes:
        raise ConfigError(
            f"Configuration file {path} is {size} bytes; refusing to load "
            f"anything larger than {max_bytes} bytes."
        )

    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        # PyYAML attaches a problem_mark with line/column information.
        line = getattr(getattr(exc, "problem_mark", None), "line", None)
        col = getattr(getattr(exc, "problem_mark", None), "column", None)
        loc = f" (line {line + 1}, column {col + 1})" if line is not None else ""
        raise ConfigError(
            f"Failed to parse YAML from {path}{loc}: {exc}\n"
            f"  Hint: validate the file with `python -c 'import yaml; "
            f"yaml.safe_load(open({str(path)!r}))'`"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read configuration file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}"
        )
    return data


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
    data = _load_yaml_file(base_path)

    # Apply overlays in order
    for overlay_path in overlays or []:
        overlay_data = _load_yaml_file(Path(overlay_path))
        data = _deep_merge(data, overlay_data)

    # SEC: Surface a clear warning when encryption is enabled but the key
    # env var is missing.  We deliberately do NOT inject the key into the
    # Config itself — secrets must never reach the immutable model so they
    # cannot be accidentally serialized via .model_dump_json().
    encryption_key_env = (
        data.get("privacy", {}).get("encryption_key_env", "I3_ENCRYPTION_KEY")
    )
    if (
        not os.environ.get(encryption_key_env)
        and data.get("privacy", {}).get("encrypt_embeddings", True)
    ):
        import warnings
        warnings.warn(
            f"Environment variable '{encryption_key_env}' is not set but "
            f"encrypt_embeddings is enabled.  Encryption operations will fail "
            f"at runtime unless the key is provided.  See .env.example.",
            stacklevel=2,
        )

    # Validate and construct the immutable Config.  We catch
    # ValidationError specifically so the error message is structured;
    # everything else is wrapped with the file path for context.
    try:
        config = Config(**data)
    except ValidationError as exc:
        # SEC: pydantic's str(exc) already enumerates each failing field
        # with its location, so we surface it verbatim under ConfigError.
        raise ConfigError(
            f"Configuration validation failed for {base_path}:\n{exc}"
        ) from exc
    except TypeError as exc:
        raise ConfigError(
            f"Configuration structure invalid for {base_path}: {exc}\n"
            f"  Hint: an unknown top-level section was passed to Config()."
        ) from exc

    # Seed management
    if set_seeds:
        _set_seeds(config.project.seed)

    return config
