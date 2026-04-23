"""Cedar adapter — application-level authorization for I³.

Wraps the `cedarpy <https://pypi.org/project/cedarpy/>`_ Python binding to
AWS's open-source `Cedar <https://www.cedarpolicy.com/>`_ policy engine.

The adapter is a *soft* import: if ``cedarpy`` is not installed, the
:class:`CedarAuthorizer` still instantiates but every call to
:meth:`CedarAuthorizer.is_authorized` raises :class:`ImportError` with an
install hint.  All public callers are typed end-to-end.

Typical usage::

    from i3.authz import CedarAuthorizer

    authz = CedarAuthorizer.from_default_policy_file()
    allowed = authz.is_authorized(
        principal=("User", "alice"),
        action="read",
        resource=("Diary", "diary-alice"),
        context={},
        entities=entities,
    )

References:
    Cutler, E. *et al.* (2024). *Cedar: A New Language for Expressive, Fast,
    Safe, and Analyzable Authorization*. OOPSLA 2024.
    https://www.cedarpolicy.com/

Install hint::

    pip install "cedarpy>=4.0"
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import of cedarpy
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    import cedarpy  # type: ignore[import-not-found]

    _CEDAR_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    cedarpy = None  # type: ignore[assignment]
    _CEDAR_AVAILABLE = False


_INSTALL_HINT: str = (
    "cedarpy is not installed. Install it with: "
    '`pip install "cedarpy>=4.0"`  '
    "(or add it to the [tool.poetry.group.policy.dependencies] group)."
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return ``True`` iff the ``cedarpy`` binding is importable.

    Useful for ``pytest.skip`` guards and for toggling feature-flag
    behaviour in services that want to fall back to in-code RBAC.
    """

    return _CEDAR_AVAILABLE


def _default_policy_path() -> Path:
    """Return the repo-relative default Cedar policy path.

    Resolves to ``<repo-root>/deploy/policy/cedar/i3.cedar`` by walking
    up from this module's location.
    """

    # i3/authz/cedar_adapter.py -> i3/authz -> i3 -> <repo-root>
    repo_root: Path = Path(__file__).resolve().parents[2]
    return repo_root / "deploy" / "policy" / "cedar" / "i3.cedar"


def _default_schema_path() -> Path:
    """Return the repo-relative default Cedar schema path."""

    repo_root: Path = Path(__file__).resolve().parents[2]
    return repo_root / "deploy" / "policy" / "cedar" / "schema.cedarschema.json"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


EntityRef = tuple[str, str]
"""An ``(EntityType, entity_id)`` tuple, e.g. ``("User", "alice")``."""

Entity = Mapping[str, Any]
"""A fully-expanded entity record in Cedar JSON form."""


@dataclass(frozen=True, slots=True)
class AuthzDecision:
    """Outcome of a single Cedar authorization request.

    Attributes:
        allowed: ``True`` iff the decision was ``Allow``.
        determining_policies: IDs of the policies that produced the
            decision (helpful for audit trails).
        errors: Any evaluator errors reported by cedarpy.
    """

    allowed: bool
    determining_policies: tuple[str, ...]
    errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Authorizer
# ---------------------------------------------------------------------------


class CedarAuthorizer:
    """Application-level authorizer backed by Cedar.

    The authorizer loads the I³ policy set on construction and exposes
    :meth:`is_authorized`, which returns a Boolean decision in the common
    case and :meth:`authorize` which returns the richer
    :class:`AuthzDecision` structure.

    Args:
        policy_src: Raw Cedar policy text.
        schema_src: Optional schema JSON (stringified).  When provided,
            cedarpy will enforce schema-based validation on every
            request.
    """

    def __init__(
        self,
        policy_src: str,
        schema_src: str | None = None,
    ) -> None:
        self._policy_src: str = policy_src
        self._schema_src: str | None = schema_src

    # -- Constructors ------------------------------------------------------

    @classmethod
    def from_default_policy_file(cls) -> CedarAuthorizer:
        """Load the canonical I³ policy set from the repo.

        Returns:
            Authorizer initialised with ``deploy/policy/cedar/i3.cedar``
            and the matching schema.

        Raises:
            FileNotFoundError: If either file is missing.
        """

        policy_path: Path = _default_policy_path()
        schema_path: Path = _default_schema_path()
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Cedar policy file not found at {policy_path}"
            )
        policy_src: str = policy_path.read_text(encoding="utf-8")
        schema_src: str | None
        if schema_path.is_file():
            schema_src = schema_path.read_text(encoding="utf-8")
        else:
            schema_src = None
        return cls(policy_src=policy_src, schema_src=schema_src)

    @classmethod
    def from_path(
        cls,
        policy_path: Path,
        schema_path: Path | None = None,
    ) -> CedarAuthorizer:
        """Load a policy set from an arbitrary file path.

        Args:
            policy_path: Path to a ``.cedar`` file.
            schema_path: Optional path to a Cedar schema JSON file.

        Returns:
            New :class:`CedarAuthorizer`.
        """

        policy_src: str = Path(policy_path).read_text(encoding="utf-8")
        schema_src: str | None = None
        if schema_path is not None:
            schema_src = Path(schema_path).read_text(encoding="utf-8")
        return cls(policy_src=policy_src, schema_src=schema_src)

    # -- Public API --------------------------------------------------------

    def is_authorized(
        self,
        principal: EntityRef,
        action: str,
        resource: EntityRef,
        context: Mapping[str, Any] | None = None,
        entities: Sequence[Entity] | None = None,
    ) -> bool:
        """Return whether the request is allowed by the policy set.

        Args:
            principal: ``(type, id)`` tuple, e.g. ``("User", "alice")``.
            action: Action name without the ``Action::"..."`` wrapper.
            resource: ``(type, id)`` tuple of the target resource.
            context: Optional attribute map (``mfa``, ``reason`` …).
            entities: Optional pre-expanded Cedar entity list.  When
                omitted, a minimal entity list is synthesized from the
                principal and resource references.

        Returns:
            ``True`` on an explicit ``Allow``, ``False`` otherwise
            (including implicit deny).

        Raises:
            ImportError: When ``cedarpy`` is not installed.
        """

        decision: AuthzDecision = self.authorize(
            principal=principal,
            action=action,
            resource=resource,
            context=context,
            entities=entities,
        )
        return decision.allowed

    def authorize(
        self,
        principal: EntityRef,
        action: str,
        resource: EntityRef,
        context: Mapping[str, Any] | None = None,
        entities: Sequence[Entity] | None = None,
    ) -> AuthzDecision:
        """Evaluate a request and return the full :class:`AuthzDecision`.

        See :meth:`is_authorized` for arg semantics.

        Raises:
            ImportError: When ``cedarpy`` is not installed.
        """

        if not _CEDAR_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        principal_uid: str = self._format_uid(principal)
        resource_uid: str = self._format_uid(resource)
        action_uid: str = f'Action::"{action}"'

        request_payload: dict[str, Any] = {
            "principal": principal_uid,
            "action": action_uid,
            "resource": resource_uid,
            "context": dict(context or {}),
        }

        entity_list: list[Any] = list(entities or self._fallback_entities(principal, resource))

        try:
            raw: Any = cedarpy.is_authorized(  # type: ignore[union-attr]
                request=request_payload,
                policies=self._policy_src,
                entities=entity_list,
                schema=self._schema_src,
            )
        except Exception as exc:  # pragma: no cover - binding errors
            logger.exception("Cedar evaluation failed: %s", exc)
            return AuthzDecision(
                allowed=False,
                determining_policies=tuple(),
                errors=(str(exc),),
            )

        return self._parse_decision(raw)

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _format_uid(ref: EntityRef) -> str:
        """Format an ``(type, id)`` tuple as a Cedar UID string."""

        entity_type, entity_id = ref
        # Cedar UID literal form: `Type::"id"`
        return f'{entity_type}::"{entity_id}"'

    @staticmethod
    def _fallback_entities(
        principal: EntityRef,
        resource: EntityRef,
    ) -> list[dict[str, Any]]:
        """Return a minimal entity list covering only the request UIDs.

        Callers that reference resource attributes (e.g. ``resource.owner``)
        must pass a fully-expanded entity list; otherwise Cedar's record
        attribute lookups will resolve to ``null`` and all owner-based
        rules short-circuit to deny.
        """

        return [
            {
                "uid": {"type": principal[0], "id": principal[1]},
                "attrs": {},
                "parents": [],
            },
            {
                "uid": {"type": resource[0], "id": resource[1]},
                "attrs": {},
                "parents": [],
            },
        ]

    @staticmethod
    def _parse_decision(raw: Any) -> AuthzDecision:
        """Normalise a cedarpy result into :class:`AuthzDecision`.

        cedarpy returns a dict or object across versions; we handle both.
        """

        # Dict-shaped response (common in 4.x).
        if isinstance(raw, dict):
            decision_str: str = str(raw.get("decision", "")).lower()
            policies: Sequence[Any] = raw.get("diagnostics", {}).get("reason", []) or []
            errors: Sequence[Any] = raw.get("diagnostics", {}).get("errors", []) or []
        else:  # pragma: no cover - object-shaped response
            decision_str = str(getattr(raw, "decision", "")).lower()
            diagnostics: Any = getattr(raw, "diagnostics", None)
            policies = getattr(diagnostics, "reason", []) if diagnostics else []
            errors = getattr(diagnostics, "errors", []) if diagnostics else []

        allowed: bool = decision_str == "allow"
        return AuthzDecision(
            allowed=allowed,
            determining_policies=tuple(str(p) for p in policies),
            errors=tuple(str(e) for e in errors),
        )

    # -- Dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CedarAuthorizer(policy_bytes={len(self._policy_src)}, "
            f"schema={'yes' if self._schema_src else 'no'})"
        )
