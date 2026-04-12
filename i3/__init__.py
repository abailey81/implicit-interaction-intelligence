"""
╭─────────────────────────────────────────────────────────────────────╮
│                                                                     │
│   Implicit Interaction Intelligence  (I³)                           │
│                                                                     │
│   Adaptive AI companion systems that learn from *how* you           │
│   interact — keystroke dynamics, linguistic complexity, temporal    │
│   patterns — and continuously adapt their responses across          │
│   cognitive load, communication style, emotional tone, and          │
│   accessibility needs.                                              │
│                                                                     │
│   ─────────────────────────────────────────────────────────────     │
│   Architecture                                                      │
│   ─────────────────────────────────────────────────────────────     │
│       Layer 1 — Perception       (i3.interaction)                   │
│       Layer 2 — Encoding         (i3.encoder)                       │
│       Layer 3 — User Model       (i3.user_model)                    │
│       Layer 4 — Adaptation       (i3.adaptation)                    │
│       Layer 5 — Routing          (i3.router)                        │
│       Layer 6 — Generation       (i3.slm / i3.cloud)                │
│       Layer 7 — Diary            (i3.diary)                         │
│                                                                     │
│       Cross-cutting: Privacy     (i3.privacy)                       │
│       Cross-cutting: Profiling   (i3.profiling)                     │
│       Orchestration:             (i3.pipeline)                      │
│                                                                     │
│   ─────────────────────────────────────────────────────────────     │
│   Quick start                                                       │
│   ─────────────────────────────────────────────────────────────     │
│       >>> from i3 import Pipeline, load_config                      │
│       >>> cfg = load_config("config/default.yaml")                  │
│       >>> pipe = Pipeline(cfg)                                      │
│       >>> out  = pipe.run(interaction_stream)                       │
│                                                                     │
╰─────────────────────────────────────────────────────────────────────╯
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# ─── Package metadata ────────────────────────────────────────────────
__version__ = "1.0.0"
__author__ = "Tamer Atesyakar"
__license__ = "MIT"

# ─── Default logger (library consumers should override) ─────────────
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ─── Eager public API (lightweight — no torch) ──────────────────────
from i3.config import Config, load_config
from i3.interaction.types import InteractionFeatureVector, KeystrokeEvent
from i3.adaptation.types import AdaptationVector, StyleVector
from i3.pipeline.types import PipelineInput, PipelineOutput

# ─── Lazy public API (heavy — torch / transformers / etc.) ──────────
# These are resolved on first attribute access via PEP 562 __getattr__,
# so `import i3` stays fast and side-effect free.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Pipeline":                ("i3.pipeline.engine",      "Pipeline"),
    "TemporalConvNet":         ("i3.encoder.tcn",          "TemporalConvNet"),
    "AdaptiveSLM":             ("i3.slm.model",            "AdaptiveSLM"),
    "UserModel":               ("i3.user_model.model",     "UserModel"),
    "AdaptationController":    ("i3.adaptation.controller","AdaptationController"),
    "ContextualThompsonBandit":("i3.router.bandit",        "ContextualThompsonBandit"),
    "EdgeProfiler":            ("i3.profiling.report",     "EdgeProfiler"),
}


def __getattr__(name: str):
    """PEP 562 lazy loader for heavy components."""
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache on the package module
        return value
    raise AttributeError(f"module 'i3' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy names to `dir(i3)` and IDE auto-completion."""
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


# ─── Type-checker hints (never executed at runtime) ─────────────────
if TYPE_CHECKING:  # pragma: no cover
    from i3.adaptation.controller import AdaptationController
    from i3.encoder.tcn import TemporalConvNet
    from i3.pipeline.engine import Pipeline
    from i3.profiling.report import EdgeProfiler
    from i3.router.bandit import ContextualThompsonBandit
    from i3.slm.model import AdaptiveSLM
    from i3.user_model.model import UserModel


__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    # Eagerly loaded
    "Config",
    "load_config",
    "InteractionFeatureVector",
    "KeystrokeEvent",
    "AdaptationVector",
    "StyleVector",
    "PipelineInput",
    "PipelineOutput",
    # Lazy (resolved via __getattr__)
    "Pipeline",
    "TemporalConvNet",
    "AdaptiveSLM",
    "UserModel",
    "AdaptationController",
    "ContextualThompsonBandit",
    "EdgeProfiler",
]
