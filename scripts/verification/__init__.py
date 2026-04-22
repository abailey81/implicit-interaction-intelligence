"""I3 verification harness package.

This package hosts the ``scripts/verify_all.py`` master CLI and its
supporting check modules.  Checks are auto-discovered on import: each
``checks_*.py`` module registers its own :class:`Check` instances with
the global :class:`~scripts.verification.framework.CheckRegistry` at
import time.

Typical usage::

    from scripts.verification.framework import CheckRegistry
    report = CheckRegistry.run_all(timeout_s=60, parallelism=4)
    print(report.pass_rate)

See :mod:`scripts.verification.framework` for the check abstraction and
:mod:`scripts.verify_all` for the command-line driver.
"""

from __future__ import annotations

__all__ = ["framework"]
