"""Model registry helpers for I3.

The registry is a three-tier abstraction:

1. **Local filesystem** -- always available.  Models are stored under
   ``<root>/<name>/v<version>/`` with an ``index.json`` manifest per
   version and a top-level ``latest.json`` alias.
2. **MLflow model registry** -- optional; engaged only when MLflow is
   installed and a ``tracking_uri`` is configured.
3. **Weights & Biases artefacts** -- optional; engaged only when the
   ``wandb`` package is installed and ``WANDB_API_KEY`` is set.

All three layers are written to best-effort; a failure to mirror to
MLflow or W&B never prevents a successful local registration.

The registry never executes pickled code: every lookup returns paths,
and integrity is delegated to :func:`i3.mlops.checkpoint.load_verified`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from i3.mlops.checkpoint import compute_sha256

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Soft imports
# --------------------------------------------------------------------------- #

_mlflow: ModuleType | None
try:  # pragma: no cover
    import mlflow as _mlflow  # type: ignore[import-not-found]
except Exception:
    _mlflow = None

_wandb: ModuleType | None
try:  # pragma: no cover
    import wandb as _wandb  # type: ignore[import-not-found]
except Exception:
    _wandb = None


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class RegistryEntry:
    """Single registered model version.

    Attributes:
        name: Model name (e.g. ``"encoder"`` or ``"slm"``).
        version: Monotonically increasing integer version.
        path: Absolute path to the checkpoint on the local filesystem.
        sha256: Hex digest of the checkpoint.
        created_at: ISO-8601 UTC timestamp of registration.
        tags: Free-form string tags (e.g. ``{"stage": "staging"}``).
        metrics: Scalar metrics captured at registration time.
        metadata: Arbitrary extra metadata (run ids, git sha, ...).
    """

    name: str
    version: int
    path: str
    sha256: str
    created_at: str
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Return the entry as a sorted, indented JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegistryEntry:
        """Reconstruct an entry from a parsed JSON manifest.

        Args:
            data: Parsed JSON mapping.

        Returns:
            A fully-populated :class:`RegistryEntry`.
        """
        return cls(
            name=str(data["name"]),
            version=int(data["version"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            created_at=str(data["created_at"]),
            tags={str(k): str(v) for k, v in dict(data.get("tags") or {}).items()},
            metrics={
                str(k): float(v)
                for k, v in dict(data.get("metrics") or {}).items()
            },
            metadata=dict(data.get("metadata") or {}),
        )


# --------------------------------------------------------------------------- #
# ModelRegistry
# --------------------------------------------------------------------------- #


class ModelRegistry:
    """Local filesystem model registry with optional MLflow / W&B mirrors.

    Layout on disk::

        <root>/
            <name>/
                v1/
                    model.pt
                    model.pt.sha256
                    index.json
                v2/
                    ...
                latest.json              # -> {"version": 2}

    Args:
        root: Directory that stores all registered models.  Created
            lazily on the first ``register`` call.
        use_mlflow: If ``True`` and MLflow is installed, mirror
            registrations to the MLflow model registry.
        use_wandb: If ``True`` and ``wandb`` is installed, mirror
            registrations as W&B artefacts.
        wandb_project: W&B project name.  Defaults to the
            ``WANDB_PROJECT`` env var or ``"i3"``.

    Attributes:
        root: Registry root directory.
        mlflow_enabled: Whether MLflow mirroring is active.
        wandb_enabled: Whether W&B mirroring is active.
    """

    def __init__(
        self,
        root: str | Path = "registry",
        use_mlflow: bool = True,
        use_wandb: bool = False,
        wandb_project: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.mlflow_enabled = bool(use_mlflow and _mlflow is not None)
        self.wandb_enabled = bool(use_wandb and _wandb is not None)
        self.wandb_project = (
            wandb_project or os.environ.get("WANDB_PROJECT") or "i3"
        )

        if use_mlflow and not self.mlflow_enabled:
            logger.warning(
                "use_mlflow=True but mlflow is not installed; "
                "MLflow mirroring disabled."
            )
        if use_wandb and not self.wandb_enabled:
            logger.warning(
                "use_wandb=True but wandb is not installed; "
                "W&B mirroring disabled."
            )

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def _model_dir(self, name: str) -> Path:
        return self.root / name

    def _version_dir(self, name: str, version: int) -> Path:
        return self._model_dir(name) / f"v{version}"

    def _next_version(self, name: str) -> int:
        model_dir = self._model_dir(name)
        if not model_dir.exists():
            return 1
        versions = [
            int(p.name[1:])
            for p in model_dir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        return (max(versions) + 1) if versions else 1

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        source_path: str | Path,
        metrics: Mapping[str, float] | None = None,
        tags: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        mlflow_run_id: str | None = None,
    ) -> RegistryEntry:
        """Copy a checkpoint into the registry and create a new version.

        Args:
            name: Model name.
            source_path: Path to an already-saved checkpoint file.
            metrics: Optional scalar metrics captured at registration.
            tags: Optional string tags (e.g. ``{"stage": "staging"}``).
            metadata: Optional free-form metadata mapping.
            mlflow_run_id: Optional MLflow run ID for cross-referencing.

        Returns:
            The created :class:`RegistryEntry`.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"source checkpoint not found: {src}")

        version = self._next_version(name)
        vdir = self._version_dir(name, version)
        vdir.mkdir(parents=True, exist_ok=True)

        dest = vdir / src.name
        shutil.copy2(src, dest)
        # Mirror the sha256 sidecar if present; otherwise compute fresh.
        src_sidecar = src.with_suffix(src.suffix + ".sha256")
        if src_sidecar.exists():
            shutil.copy2(src_sidecar, dest.with_suffix(dest.suffix + ".sha256"))
            digest = src_sidecar.read_text(encoding="utf-8").split()[0].lower()
        else:
            digest = compute_sha256(dest)
            dest.with_suffix(dest.suffix + ".sha256").write_text(
                f"{digest}  {dest.name}\n", encoding="utf-8"
            )

        entry = RegistryEntry(
            name=name,
            version=version,
            path=str(dest.resolve()),
            sha256=digest,
            created_at=datetime.now(timezone.utc).isoformat(),
            tags={str(k): str(v) for k, v in dict(tags or {}).items()},
            metrics={str(k): float(v) for k, v in dict(metrics or {}).items()},
            metadata=dict(metadata or {}),
        )
        if mlflow_run_id is not None:
            entry.metadata["mlflow_run_id"] = str(mlflow_run_id)

        (vdir / "index.json").write_text(entry.to_json(), encoding="utf-8")
        (self._model_dir(name) / "latest.json").write_text(
            json.dumps({"version": version}, indent=2), encoding="utf-8"
        )

        # Best-effort mirrors.
        self._mirror_mlflow(entry, mlflow_run_id)
        self._mirror_wandb(entry)

        logger.info("Registered %s v%d -> %s", name, version, dest)
        return entry

    def get(self, name: str, version: int | None = None) -> RegistryEntry:
        """Fetch a registered model entry.

        Args:
            name: Model name.
            version: Specific version; if ``None`` the latest is returned.

        Returns:
            The matching :class:`RegistryEntry`.
        """
        if version is None:
            latest_path = self._model_dir(name) / "latest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"no versions registered for {name}")
            version = int(json.loads(latest_path.read_text(encoding="utf-8"))["version"])
        manifest = self._version_dir(name, version) / "index.json"
        if not manifest.exists():
            raise FileNotFoundError(f"{name} v{version} not found")
        return RegistryEntry.from_dict(json.loads(manifest.read_text(encoding="utf-8")))

    def list_versions(self, name: str) -> list[RegistryEntry]:
        """Return every registered version of ``name`` in ascending order.

        Args:
            name: Model name.

        Returns:
            List of entries, oldest first.
        """
        model_dir = self._model_dir(name)
        if not model_dir.exists():
            return []
        entries: list[RegistryEntry] = []
        for p in sorted(model_dir.iterdir()):
            if not (p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()):
                continue
            manifest = p / "index.json"
            if manifest.exists():
                entries.append(
                    RegistryEntry.from_dict(
                        json.loads(manifest.read_text(encoding="utf-8"))
                    )
                )
        entries.sort(key=lambda e: e.version)
        return entries

    def list_models(self) -> list[str]:
        """Return all registered model names.

        Returns:
            Sorted list of model names with at least one version.
        """
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def promote(self, name: str, version: int, stage: str) -> RegistryEntry:
        """Set the ``stage`` tag for a given version (``staging`` / ``prod``).

        Args:
            name: Model name.
            version: Version to promote.
            stage: Target stage label.

        Returns:
            The updated entry.
        """
        entry = self.get(name, version)
        entry.tags["stage"] = stage
        (self._version_dir(name, version) / "index.json").write_text(
            entry.to_json(), encoding="utf-8"
        )
        logger.info("Promoted %s v%d -> %s", name, version, stage)
        return entry

    # ------------------------------------------------------------------ #
    # Mirrors (best effort)
    # ------------------------------------------------------------------ #

    def _mirror_mlflow(
        self,
        entry: RegistryEntry,
        run_id: str | None,
    ) -> None:
        """Mirror a registered entry into the MLflow model registry.

        Args:
            entry: The entry to mirror.
            run_id: Optional MLflow run ID that produced the model.
        """
        if not self.mlflow_enabled or _mlflow is None:
            return
        try:
            client = _mlflow.tracking.MlflowClient()
            try:
                client.create_registered_model(entry.name)
            except Exception:
                pass  # Already exists.
            source = str(Path(entry.path).resolve().as_uri())
            client.create_model_version(
                name=entry.name,
                source=source,
                run_id=run_id,
                tags={**entry.tags, "sha256": entry.sha256},
            )
        except Exception as exc:
            logger.warning("MLflow mirror failed for %s: %s", entry.name, exc)

    def _mirror_wandb(self, entry: RegistryEntry) -> None:
        """Mirror a registered entry as a W&B artefact.

        Args:
            entry: The entry to mirror.
        """
        if not self.wandb_enabled or _wandb is None:
            return
        try:
            run = _wandb.init(
                project=self.wandb_project,
                job_type="model-register",
                reinit=True,
                name=f"register-{entry.name}-v{entry.version}",
            )
            try:
                artifact = _wandb.Artifact(
                    entry.name,
                    type="model",
                    metadata={
                        "sha256": entry.sha256,
                        "version": entry.version,
                        **entry.metadata,
                    },
                )
                artifact.add_file(entry.path)
                run.log_artifact(artifact, aliases=list(set(["latest", *entry.tags.values()])))
            finally:
                run.finish()
        except Exception as exc:
            logger.warning("W&B mirror failed for %s: %s", entry.name, exc)


__all__ = ["ModelRegistry", "RegistryEntry"]


def iter_stages(entries: Iterable[RegistryEntry], stage: str) -> list[RegistryEntry]:
    """Filter entries by ``stage`` tag.

    Args:
        entries: Iterable of entries (e.g. from :meth:`ModelRegistry.list_versions`).
        stage: Stage label to match against ``entry.tags.get("stage")``.

    Returns:
        Filtered list of entries.
    """
    return [e for e in entries if e.tags.get("stage") == stage]
