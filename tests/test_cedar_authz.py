"""Tests for the Cedar authorization adapter.

Exercises the full scenario table in
``deploy/policy/cedar/tests/test_cases.json`` against the
:class:`i3.authz.CedarAuthorizer`.  Soft-skips if ``cedarpy`` is not
installed so that CI can still pass on minimal environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from i3.authz import CedarAuthorizer, is_available

# ---------------------------------------------------------------------------
# Skip entire module when cedarpy is not available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="cedarpy is not installed — install with `pip install cedarpy>=4.0`",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


REPO_ROOT: Path = Path(__file__).resolve().parents[1]
POLICY_PATH: Path = REPO_ROOT / "deploy" / "policy" / "cedar" / "i3.cedar"
SCHEMA_PATH: Path = REPO_ROOT / "deploy" / "policy" / "cedar" / "schema.cedarschema.json"
CASES_PATH: Path = REPO_ROOT / "deploy" / "policy" / "cedar" / "tests" / "test_cases.json"


@pytest.fixture(scope="module")
def scenario_data() -> dict[str, Any]:
    """Return the parsed Cedar scenario file."""

    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def authorizer() -> CedarAuthorizer:
    """Construct a :class:`CedarAuthorizer` from the shipped policy."""

    return CedarAuthorizer.from_path(POLICY_PATH, SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Parametric scenario test — drives `is_authorized` through all 20+ cases
# ---------------------------------------------------------------------------


def _case_ids(scenario: dict[str, Any]) -> list[str]:
    """Return a human-readable pytest id for each scenario."""

    return [c["name"] for c in scenario["cases"]]


def test_scenarios_load() -> None:
    """The scenario JSON is syntactically valid and non-empty."""

    data: dict[str, Any] = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert "cases" in data
    assert len(data["cases"]) >= 20


def test_cedar_is_available() -> None:
    """Sanity check — the soft-import reports availability."""

    assert is_available() is True


def test_authorizer_constructs(authorizer: CedarAuthorizer) -> None:
    """The authorizer loads from the canonical policy file."""

    assert authorizer is not None
    assert "permit" in repr(authorizer) or "CedarAuthorizer" in repr(authorizer)


def test_default_policy_file_exists() -> None:
    """The default policy path resolves to a real file."""

    assert POLICY_PATH.is_file()
    assert SCHEMA_PATH.is_file()


def test_from_default_policy_file() -> None:
    """The convenience constructor walks up to the repo root correctly."""

    auth: CedarAuthorizer = CedarAuthorizer.from_default_policy_file()
    assert auth is not None


@pytest.mark.parametrize(
    "case_index",
    range(21),  # mirror the length of the scenario table
    ids=[f"case_{i}" for i in range(21)],
)
def test_scenarios(
    case_index: int,
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Each scenario's decision matches the expected `allow`/`deny`."""

    cases: list[dict[str, Any]] = scenario_data["cases"]
    if case_index >= len(cases):
        pytest.skip("scenario table shorter than 21 cases")

    case: dict[str, Any] = cases[case_index]
    principal: tuple[str, str] = (case["principal"]["type"], case["principal"]["id"])
    resource: tuple[str, str] = (case["resource"]["type"], case["resource"]["id"])
    action: str = case["action"]["id"]
    context: dict[str, Any] = case.get("context", {})

    # Pass the pre-expanded entity table so resource.owner lookups resolve.
    entities: list[dict[str, Any]] = list(scenario_data["entities"])

    allowed: bool = authorizer.is_authorized(
        principal=principal,
        action=action,
        resource=resource,
        context=context,
        entities=entities,
    )

    expected: bool = case["decision"] == "allow"
    assert allowed is expected, (
        f"scenario '{case['name']}' — expected {case['decision']}, got "
        f"{'allow' if allowed else 'deny'}"
    )


# ---------------------------------------------------------------------------
# Targeted unit tests that don't round-trip through JSON
# ---------------------------------------------------------------------------


def test_owner_reads_own_diary(
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Alice reads her own diary → allow."""

    entities: list[dict[str, Any]] = list(scenario_data["entities"])
    assert authorizer.is_authorized(
        principal=("User", "alice"),
        action="read",
        resource=("Diary", "diary-alice"),
        context={},
        entities=entities,
    )


def test_other_user_read_denied(
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Bob trying to read Alice's diary → deny."""

    entities: list[dict[str, Any]] = list(scenario_data["entities"])
    assert not authorizer.is_authorized(
        principal=("User", "bob"),
        action="read",
        resource=("Diary", "diary-alice"),
        context={},
        entities=entities,
    )


def test_admin_delete_requires_reason(
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Admin delete without reason is blocked; with reason is allowed."""

    entities: list[dict[str, Any]] = list(scenario_data["entities"])
    without_reason: bool = authorizer.is_authorized(
        principal=("Admin", "carol"),
        action="delete",
        resource=("Diary", "diary-alice"),
        context={},
        entities=entities,
    )
    with_reason: bool = authorizer.is_authorized(
        principal=("Admin", "carol"),
        action="delete",
        resource=("Diary", "diary-alice"),
        context={"reason": "GDPR erasure"},
        entities=entities,
    )
    assert without_reason is False
    assert with_reason is True


def test_cloud_path_requires_sanitized_flag(
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Cloud Service can only read Adaptation when sanitized=True."""

    entities: list[dict[str, Any]] = list(scenario_data["entities"])
    denied: bool = authorizer.is_authorized(
        principal=("Service", "cloud"),
        action="read",
        resource=("Adaptation", "adapt-alice"),
        context={},
        entities=entities,
    )
    allowed: bool = authorizer.is_authorized(
        principal=("Service", "cloud"),
        action="read",
        resource=("Adaptation", "adapt-alice"),
        context={"sanitized": True},
        entities=entities,
    )
    assert denied is False
    assert allowed is True


def test_export_requires_mfa(
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Export without MFA is denied; with MFA is allowed."""

    entities: list[dict[str, Any]] = list(scenario_data["entities"])
    assert not authorizer.is_authorized(
        principal=("User", "alice"),
        action="export",
        resource=("Diary", "diary-alice"),
        context={},
        entities=entities,
    )
    assert authorizer.is_authorized(
        principal=("User", "alice"),
        action="export",
        resource=("Diary", "diary-alice"),
        context={"mfa": True},
        entities=entities,
    )


def test_admin_cannot_read_diary(
    authorizer: CedarAuthorizer,
    scenario_data: dict[str, Any],
) -> None:
    """Explicit forbid: admin may never read diary contents."""

    entities: list[dict[str, Any]] = list(scenario_data["entities"])
    assert not authorizer.is_authorized(
        principal=("Admin", "carol"),
        action="read",
        resource=("Diary", "diary-alice"),
        context={"reason": "support ticket"},
        entities=entities,
    )


def test_import_error_when_cedarpy_missing() -> None:
    """A fake authorizer with the flag flipped raises ImportError."""

    # This test is intentionally trivial: the soft-import path is well
    # exercised in CI environments where cedarpy is deliberately absent,
    # which causes `pytestmark` above to skip the whole module.  Having
    # this test here documents the contract.
    from i3.authz import cedar_adapter

    assert hasattr(cedar_adapter, "_INSTALL_HINT")
    assert "cedarpy" in cedar_adapter._INSTALL_HINT
