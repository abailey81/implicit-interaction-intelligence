"""MLflow experiment-tracking wrapper for I3.

This module wraps the optional :mod:`mlflow` dependency behind an
``ExperimentTracker`` class.  If MLflow is not installed (or fails to
import for any reason), every tracker call is a no-op and a single
warning is logged at construction time.  The training pipeline therefore
never breaks because of missing MLOps tooling.

Typical usage::

    from i3.mlops.tracking import ExperimentTracker

    tracker = ExperimentTracker()
    with tracker.run("encoder-train-v1", tags={"arch": "tcn"}):
        tracker.log_params({"lr": 1e-3, "batch_size": 64})
        for step, loss in enumerate(losses):
            tracker.log_metrics({"loss": loss}, step=step)
        tracker.log_artifact("checkpoints/best.pt")

Environment variables read at construction time:

``MLFLOW_TRACKING_URI``
    Target tracking server URI.  Defaults to ``file:./mlruns``.
``MLFLOW_EXPERIMENT_NAME``
    Experiment name.  Defaults to ``i3-experiments``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Soft imports
# --------------------------------------------------------------------------- #

_mlflow: ModuleType | None
try:  # pragma: no cover - trivial import guard
    import mlflow as _mlflow  # type: ignore[import-not-found]
except Exception:
    _mlflow = None


_DEFAULT_TRACKING_URI = "file:./mlruns"
_DEFAULT_EXPERIMENT_NAME = "i3-experiments"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _git_commit_sha() -> str:
    """Return the short git SHA of ``HEAD`` or ``"unknown"`` on failure.

    Returns:
        Seven-character git SHA if the command succeeds, else ``"unknown"``.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _torch_version() -> str:
    """Return the installed torch version or ``"unavailable"``."""
    try:
        import torch  # type: ignore[import-not-found]

        return str(torch.__version__)
    except Exception:
        return "unavailable"


def _config_hash(config: Mapping[str, Any] | None) -> str:
    """Stable SHA-256 of a config mapping, for tagging runs.

    Args:
        config: Arbitrary JSON-serialisable configuration mapping.

    Returns:
        Hex digest (first 16 chars) or ``"none"`` if ``config`` is ``None``.
    """
    if config is None:
        return "none"
    try:
        payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        return "unhashable"
    return hashlib.sha256(payload).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# ExperimentTracker
# --------------------------------------------------------------------------- #


class ExperimentTracker:
    """Thin, defensively-coded MLflow wrapper.

    All methods are safe to call even when MLflow is not installed: the
    tracker enters a silent no-op mode, logging a single warning on
    construction so the caller is aware.

    Args:
        tracking_uri: Optional MLflow tracking URI override.  If ``None``,
            reads from ``MLFLOW_TRACKING_URI``.  Default
            ``file:./mlruns``.
        experiment_name: Optional experiment name override.  If ``None``,
            reads from ``MLFLOW_EXPERIMENT_NAME``.  Default
            ``i3-experiments``.
        config: Optional JSON-serialisable configuration object whose
            hash will be attached to every run as a tag.

    Attributes:
        enabled: ``True`` when MLflow is available and initialised.
        tracking_uri: Effective tracking URI.
        experiment_name: Effective experiment name.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.tracking_uri: str = (
            tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
            or _DEFAULT_TRACKING_URI
        )
        self.experiment_name: str = (
            experiment_name
            or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or _DEFAULT_EXPERIMENT_NAME
        )
        self._config_hash: str = _config_hash(config)
        self._active_run_id: str | None = None

        if _mlflow is None:
            logger.warning(
                "mlflow is not installed; ExperimentTracker is running in "
                "no-op mode. Install with `pip install mlflow` to enable."
            )
            self.enabled = False
            return

        try:
            _mlflow.set_tracking_uri(self.tracking_uri)
            _mlflow.set_experiment(self.experiment_name)
            self.enabled = True
            logger.info(
                "ExperimentTracker initialised: uri=%s experiment=%s",
                self.tracking_uri,
                self.experiment_name,
            )
        except Exception as exc:
            logger.warning(
                "ExperimentTracker failed to initialise MLflow (%s); "
                "falling back to no-op mode.",
                exc,
            )
            self.enabled = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start_run(
        self,
        run_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> str | None:
        """Begin a new MLflow run.

        Args:
            run_name: Human-readable run name.
            tags: Optional additional tags to attach to the run.

        Returns:
            Run ID string, or ``None`` when tracking is disabled.
        """
        if not self.enabled or _mlflow is None:
            return None
        merged_tags: dict[str, str] = {
            "i3.git_sha": _git_commit_sha(),
            "i3.python": platform.python_version(),
            "i3.torch": _torch_version(),
            "i3.config_hash": self._config_hash,
        }
        if tags:
            merged_tags.update({str(k): str(v) for k, v in tags.items()})
        try:
            run = _mlflow.start_run(run_name=run_name, tags=merged_tags)
            self._active_run_id = run.info.run_id
            return self._active_run_id
        except Exception as exc:
            logger.warning("start_run failed: %s", exc)
            return None

    def end_run(self, status: str = "FINISHED") -> None:
        """End the currently-active MLflow run.

        Args:
            status: MLflow run status (``"FINISHED"``, ``"FAILED"`` ...).
        """
        if not self.enabled or _mlflow is None or self._active_run_id is None:
            return
        try:
            _mlflow.end_run(status=status)
        except Exception as exc:
            logger.warning("end_run failed: %s", exc)
        finally:
            self._active_run_id = None

    @contextmanager
    def run(
        self,
        run_name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[str | None]:
        """Context manager wrapper around :meth:`start_run` / :meth:`end_run`.

        Usage::

            with tracker.run("train-v1", tags={"arch": "tcn"}) as run_id:
                tracker.log_params({"lr": 1e-3})

        Yields:
            The MLflow run ID, or ``None`` when tracking is disabled.
        """
        rid = self.start_run(run_name, tags=tags)
        try:
            yield rid
        except BaseException:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")

    # ------------------------------------------------------------------ #
    # Logging primitives
    # ------------------------------------------------------------------ #

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log a batch of hyperparameters.

        Args:
            params: Mapping of parameter name to value.  Values are
                stringified by MLflow, so complex objects should be
                pre-flattened by the caller.
        """
        if not self.enabled or _mlflow is None:
            return
        try:
            _mlflow.log_params({str(k): v for k, v in params.items()})
        except Exception as exc:
            logger.warning("log_params failed: %s", exc)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log a batch of scalar metrics at a given step.

        Args:
            metrics: Mapping of metric name to float value.
            step: Optional integer step index (e.g. epoch or batch).
        """
        if not self.enabled or _mlflow is None:
            return
        try:
            for name, value in metrics.items():
                _mlflow.log_metric(str(name), float(value), step=step)
        except Exception as exc:
            logger.warning("log_metrics failed: %s", exc)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Upload a local file or directory to the active run.

        Args:
            path: Path to file or directory on the local filesystem.
            artifact_path: Destination subdirectory within the run.
        """
        if not self.enabled or _mlflow is None:
            return
        try:
            _mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:
            logger.warning("log_artifact failed: %s", exc)

    def log_model(self, model: Any, name: str) -> None:
        """Log a PyTorch model using MLflow's model registry conventions.

        Args:
            model: The ``torch.nn.Module`` to serialise.
            name: Artifact path / model name within the run.
        """
        if not self.enabled or _mlflow is None:
            return
        try:
            # Prefer the torch flavour if available.
            try:
                from mlflow import pytorch as mlflow_pytorch  # type: ignore
            except Exception:
                mlflow_pytorch = None

            if mlflow_pytorch is not None:
                mlflow_pytorch.log_model(model, name)
            else:  # pragma: no cover - fallback only when pytorch flavour missing
                logger.warning(
                    "mlflow.pytorch unavailable; skipping log_model(%s).", name
                )
        except Exception as exc:
            logger.warning("log_model failed: %s", exc)

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on the active run.

        Args:
            key: Tag name.
            value: Tag value (stringified).
        """
        if not self.enabled or _mlflow is None:
            return
        try:
            _mlflow.set_tag(str(key), str(value))
        except Exception as exc:
            logger.warning("set_tag failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def active_run_id(self) -> str | None:
        """Return the currently active run ID, if any."""
        return self._active_run_id

    def __repr__(self) -> str:
        return (
            f"ExperimentTracker(enabled={self.enabled}, "
            f"uri={self.tracking_uri!r}, experiment={self.experiment_name!r})"
        )


__all__ = ["ExperimentTracker"]


# Re-export Python version helper for diagnostics / tests.
_PYTHON_VERSION = sys.version
