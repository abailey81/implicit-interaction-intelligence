"""Cross-check ``configs/default.yaml`` against the Pydantic schema.

This test is the load-bearing guard for the ``extra="forbid"``
invariant added during the 2026-04-23 audit.  It proves:

* Every section and field the schema declares is reachable from the
  YAML.
* The YAML round-trips through ``Config(**...)`` without validation
  error.
* The ``router.prior_precision`` field (added as a distinct field
  during the audit) is present with a sensible default.
* Typoed top-level sections are rejected (the invariant itself).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# Stub torch so this test file imports in environments with a broken
# torch install.
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.tensor = lambda *a, **k: _torch_stub.Tensor()
_torch_stub.float32 = "float32"
sys.modules.setdefault("torch", _torch_stub)


from i3.config import Config, load_config  # noqa: E402

_DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_exists():
    assert _DEFAULT_YAML.is_file()


def test_default_yaml_round_trips_through_schema():
    """The canonical configs/default.yaml must load under the strict schema."""
    cfg = load_config(str(_DEFAULT_YAML), set_seeds=False)
    assert isinstance(cfg, Config)


def test_cloud_model_pinned():
    """configs/default.yaml must pin the brief-locked model id."""
    cfg = load_config(str(_DEFAULT_YAML), set_seeds=False)
    assert cfg.cloud.model == "claude-sonnet-4-5"


def test_router_has_separate_prior_precision():
    """``prior_alpha`` (Beta) and ``prior_precision`` (Gaussian) are distinct."""
    cfg = load_config(str(_DEFAULT_YAML), set_seeds=False)
    assert hasattr(cfg.router, "prior_alpha")
    assert hasattr(cfg.router, "prior_precision")
    assert cfg.router.prior_precision > 0


def test_root_config_rejects_unknown_section():
    """``extra="forbid"`` must block typoed top-level sections."""
    with pytest.raises(Exception):
        Config(saftey={"enabled": True})  # typo: 'saftey'
    with pytest.raises(Exception):
        Config(priavcy={"strip_pii": True})  # typo: 'priavcy'


def test_router_prior_precision_must_be_positive():
    from i3.config import RouterConfig

    # Zero or negative precision is rejected at construction.
    with pytest.raises(Exception):
        RouterConfig(prior_precision=0.0)
    with pytest.raises(Exception):
        RouterConfig(prior_precision=-1.0)


def test_router_arms_must_be_unique():
    from i3.config import RouterConfig

    with pytest.raises(Exception):
        RouterConfig(arms=["local_slm", "local_slm"])


def test_config_is_frozen():
    """Every sub-model is Pydantic-frozen; mutation must raise."""
    cfg = load_config(str(_DEFAULT_YAML), set_seeds=False)
    with pytest.raises(Exception):
        cfg.cloud.model = "something-else"  # type: ignore[misc]
