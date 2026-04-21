"""Application-level authorization for Implicit Interaction Intelligence.

The `i3.authz` package wraps Cedar-based authorization policy evaluation.
See :mod:`i3.authz.cedar_adapter` for the primary entry point.
"""

from __future__ import annotations

from i3.authz.cedar_adapter import CedarAuthorizer, is_available

__all__: list[str] = ["CedarAuthorizer", "is_available"]
