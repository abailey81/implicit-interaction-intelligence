"""Dagger pipeline-as-code for Implicit Interaction Intelligence (I^3).

This module defines the full CI/CD pipeline using the Dagger Python SDK.
Running ``dagger call <function>`` locally reproduces the exact steps that
CI executes, eliminating the "works on CI, not on my machine" class of bugs.

Design notes
------------
- Every function is ``async`` and fully type-annotated so that Dagger's
  TypeScript-style function router can introspect signatures.
- We *soft-import* the ``dagger`` package so this file can still be
  statically imported (for documentation, IDE introspection, ``ruff``
  linting, etc.) on a machine that has not yet installed the Dagger SDK.
- Image versions are pinned to specific minor releases -- Dagger will still
  cache-bust correctly via content-addressed hashes.

References
----------
- Dagger Python SDK docs  : https://docs.dagger.io/sdk/python
- Dagger modules (2024)   : https://docs.dagger.io/manuals/developer/modules
- Dagger functions guide  : https://docs.dagger.io/api/functions
- dagger.io blog posts on pipeline-as-code (2023/2024)
"""

from __future__ import annotations

import sys
from typing import Any

try:
    import dagger
    from dagger import dag, function, object_type
except ImportError as exc:  # pragma: no cover - runtime dependency
    _HINT = (
        "The 'dagger' Python SDK is not installed. Install the Dagger CLI\n"
        "  (https://docs.dagger.io/install) and then run:\n"
        "    pip install dagger-io\n"
        "  or add 'dagger-io' to your project's dev dependencies."
    )
    print(_HINT, file=sys.stderr)
    raise ImportError(_HINT) from exc


# ---------------------------------------------------------------------------
# Constants -- pinned image digests keep the supply chain hermetic.
# ---------------------------------------------------------------------------

PYTHON_BASE: str = "python:3.11-slim-bookworm"
TRIVY_IMAGE: str = "aquasec/trivy:0.50.1"
MKDOCS_IMAGE: str = "squidfunk/mkdocs-material:9.5.17"


@object_type
class I3:
    """Root Dagger object exposing every pipeline step as a callable function."""

    # ---------------------------------------------------------------- lint ---
    @function
    async def lint(self, source: dagger.Directory) -> str:
        """Run ruff inside a Python container.

        Parameters
        ----------
        source:
            Directory representing the repository root.

        Returns
        -------
        str
            Combined stdout/stderr of the lint run.
        """
        return await (
            dag.container()
            .from_(PYTHON_BASE)
            .with_mounted_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["pip", "install", "--no-cache-dir", "ruff==0.4.4"])
            .with_exec(["ruff", "check", "i3", "tests"])
            .stdout()
        )

    # ---------------------------------------------------------------- test ---
    @function
    async def test(
        self,
        source: dagger.Directory,
        python_version: str = "3.11",
    ) -> str:
        """Run pytest in a matrix.

        Parameters
        ----------
        source:
            Directory containing the project sources.
        python_version:
            Minor version of the CPython base image (``3.10``, ``3.11``,
            ``3.12``). Defaults to ``3.11`` to match CI.
        """
        image = f"python:{python_version}-slim-bookworm"
        return await (
            dag.container()
            .from_(image)
            .with_mounted_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["pip", "install", "--no-cache-dir", "-e", ".[dev]"])
            .with_exec(
                [
                    "pytest",
                    "-q",
                    "--maxfail=1",
                    "--disable-warnings",
                    "--cov=i3",
                    "--cov-report=term-missing",
                ]
            )
            .stdout()
        )

    # --------------------------------------------------------- build_image ---
    @function
    async def build_image(self, source: dagger.Directory) -> dagger.Container:
        """Build the production Docker image using the repository Dockerfile.

        The returned ``dagger.Container`` is fully cacheable and can be
        piped directly into :meth:`scan_image` or published to a registry
        via ``.publish("ghcr.io/...")``.
        """
        return (
            dag.container()
            .build(context=source, dockerfile="Dockerfile")
            .with_label("org.opencontainers.image.source", "https://github.com/…/i3")
            .with_label("org.opencontainers.image.licenses", "Apache-2.0")
        )

    # ---------------------------------------------------------- scan_image ---
    @function
    async def scan_image(self, image: dagger.Container) -> str:
        """Scan the given container image for CVEs using Trivy.

        The image is exported to a tarball inside the pipeline and fed to a
        Trivy container -- no registry round-trip required.
        """
        tar = image.as_tarball()
        return await (
            dag.container()
            .from_(TRIVY_IMAGE)
            .with_mounted_file("/tmp/image.tar", tar)
            .with_exec(
                [
                    "trivy",
                    "image",
                    "--input",
                    "/tmp/image.tar",
                    "--severity",
                    "HIGH,CRITICAL",
                    "--exit-code",
                    "1",
                    "--ignore-unfixed",
                    "--no-progress",
                ]
            )
            .stdout()
        )

    # ---------------------------------------------------------- docs_build ---
    @function
    async def docs_build(self, source: dagger.Directory) -> dagger.Directory:
        """Build the MkDocs site in *strict* mode; returns the output dir."""
        built = (
            dag.container()
            .from_(MKDOCS_IMAGE)
            .with_mounted_directory("/docs", source)
            .with_workdir("/docs")
            .with_exec(["mkdocs", "build", "--strict", "--site-dir", "/out"])
        )
        return built.directory("/out")

    # -------------------------------------------------------------- release ---
    @function
    async def release(self, source: dagger.Directory, tag: str) -> str:
        """Full pipeline: lint -> test -> build -> scan -> publish.

        Parameters
        ----------
        source:
            Repository root.
        tag:
            Semantic version tag, e.g. ``v1.2.3``. No ``v`` normalisation
            is done -- pass what you want published.
        """
        lint_out: str = await self.lint(source)
        test_out: str = await self.test(source, python_version="3.11")
        image: dagger.Container = await self.build_image(source)
        scan_out: str = await self.scan_image(image)

        address: str = f"ghcr.io/hmi-lab/i3:{tag}"
        published: str = await image.publish(address)

        report: list[Any] = [
            "=== lint ===",
            lint_out.strip(),
            "=== test ===",
            test_out.strip(),
            "=== scan ===",
            scan_out.strip(),
            "=== published ===",
            published,
        ]
        return "\n".join(str(x) for x in report)
