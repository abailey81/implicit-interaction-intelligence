"""Dagger pipeline module for Implicit Interaction Intelligence (I^3).

This package exposes a Dagger Python-SDK module that encodes the entire
CI/CD pipeline as code. See ``dagger/main.py`` for function definitions.

Usage (once the Dagger CLI is installed, https://dagger.io):

    dagger call lint --source=. stdout
    dagger call test --source=. --python-version=3.11 stdout
    dagger call build-image --source=. publish --address=...
    dagger call scan-image --image=...
    dagger call docs-build --source=. export --path=./site
    dagger call release --source=. --tag=v1.2.3 stdout

References:
    - Dagger Python SDK: https://docs.dagger.io/sdk/python
    - Dagger modules: https://docs.dagger.io/manuals/developer/modules
"""

from __future__ import annotations

__all__ = ["main"]
