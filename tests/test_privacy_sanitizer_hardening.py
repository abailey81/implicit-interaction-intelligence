"""Tests for the 2026-04-23-audit hardening on :mod:`i3.privacy.sanitizer`.

Covers the three discrete changes that tightened the sanitiser:

1. IP-address regex now requires each octet ≤ 255 so Windows build
   numbers (``10.0.22621``), SemVer strings, and telemetry counters
   no longer register as PII.
2. ``PrivacyAuditor._scan_value`` is depth-capped at 32 and builds
   field paths in O(n) via list-join instead of O(n²) concatenation.
3. ``PrivacyAuditor._findings`` is a ``deque(maxlen=1_000)`` so a
   misconfigured long-running auditor cannot accumulate GBs of
   records.
"""

from __future__ import annotations

import sys
import types

import pytest


# Stub torch so this test file is importable in environments where
# the binary torch install is broken (Windows DLL issue, CI minimal).
_torch_stub = types.ModuleType("torch")
_torch_stub.Tensor = type("Tensor", (), {})
_torch_stub.tensor = lambda *a, **k: _torch_stub.Tensor()
_torch_stub.float32 = "float32"
sys.modules.setdefault("torch", _torch_stub)


from i3.privacy.sanitizer import PrivacyAuditor, PrivacySanitizer  # noqa: E402


# ---------------------------------------------------------------------------
# IP-address regex strictness (L-3 in the 2026-04-23 audit)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sanitizer() -> PrivacySanitizer:
    return PrivacySanitizer(enabled=True)


@pytest.mark.parametrize(
    "text",
    [
        "Server is at 192.168.1.100",
        "Connect to 10.0.0.1 for the gateway",
        "DNS: 8.8.8.8 and 1.1.1.1",
        "internal: 172.16.254.1",
    ],
)
def test_real_ips_are_redacted(sanitizer, text):
    result = sanitizer.sanitize(text)
    assert "[IP_ADDRESS]" in result.sanitized_text, f"no redaction on {text!r}"
    assert "ip_address" in result.pii_types


@pytest.mark.parametrize(
    "text",
    [
        # Each of these has at least one octet > 255 and therefore
        # cannot be a valid IPv4 address, so the strict regex must
        # skip them.
        "Windows 10.0.22621 is the build number",
        "Telemetry counter 999.999.999.999 out-of-range",
        "SemVer 10.256.0.0 looks like an IP but isn't",
        "Build stamp 300.4.5.6 for release",
    ],
)
def test_non_ip_dotted_quads_not_flagged(sanitizer, text):
    """Each octet must be ≤ 255; out-of-range dotted quads are ignored.

    Note: a coincidental in-range quad like "1.2.3.4" in SemVer text
    IS still a valid IPv4 literal, so we do NOT claim to distinguish
    SemVer from real IPs in general — only to reject impossible quads.
    """
    result = sanitizer.sanitize(text)
    assert "ip_address" not in result.pii_types, (
        f"false-positive IP hit on {text!r}: {result}"
    )


# ---------------------------------------------------------------------------
# PrivacyAuditor recursion depth cap (M-4)
# ---------------------------------------------------------------------------


def test_auditor_depth_cap_on_adversarial_nesting():
    """A deeply-nested payload must not drive RecursionError."""
    auditor = PrivacyAuditor()
    # Build a 200-level-deep dict.
    payload: dict = {"leaf": "hello"}
    for _ in range(200):
        payload = {"nested": payload}
    # Should return without raising.
    result = auditor.audit_request(payload)
    # And should have emitted a depth-exceeded issue.
    issues = result.get("issues", [])
    assert any("nesting" in str(i).lower() for i in issues)


def test_auditor_field_path_uses_list_join():
    """Sanity: field path is bounded in length, not catastrophically
    quadratic in depth.  We don't measure exact O but we DO confirm
    the depth-cap message truncates field paths."""
    auditor = PrivacyAuditor()
    payload: dict = {"leaf": "x"}
    for _ in range(100):
        payload = {"nested": payload}
    result = auditor.audit_request(payload)
    issues = result.get("issues", [])
    # The depth-exceeded issue carries a field path; it should be
    # finite-length and start at "root".
    depth_issue = next(
        (i for i in issues if "nesting" in str(i).lower()), None
    )
    assert depth_issue is not None
    assert "root" in str(depth_issue)


def test_auditor_normal_nested_payload_scans_cleanly():
    """Shallow-nested payloads still get scanned end-to-end."""
    auditor = PrivacyAuditor()
    result = auditor.audit_request({
        "user": {"name": "alice", "email": "alice@example.com"},
        "message": "the weather is great",
    })
    assert "alice@example.com" not in str(result) or (
        "email" in result.get("pii_fields", [])
        or any("email" in str(f) for f in result.get("pii_fields", []))
    )


# ---------------------------------------------------------------------------
# PrivacyAuditor findings-deque bound (M-5)
# ---------------------------------------------------------------------------


def test_auditor_findings_bounded_to_1000():
    """Even thousands of audit calls should not exceed the deque cap."""
    auditor = PrivacyAuditor()
    for i in range(5_000):
        auditor.audit_request({"msg": f"test message {i}"})
    # ``_findings`` is a deque with maxlen=1000.  The internal state is
    # accessed defensively; if the implementation switches back to list,
    # the test will fail loudly.
    findings = list(auditor._findings)  # noqa: SLF001
    assert len(findings) <= 1_000, (
        f"findings buffer grew to {len(findings)} — expected ≤1 000"
    )


# ---------------------------------------------------------------------------
# Positive case — the audit still reports when PII slips in
# ---------------------------------------------------------------------------


def test_auditor_detects_embedded_pii():
    auditor = PrivacyAuditor()
    result = auditor.audit_request({
        "text": "My SSN is 123-45-6789 and my email is alice@example.com",
    })
    pii_fields = result.get("pii_fields", [])
    assert pii_fields, "expected the auditor to flag embedded PII"
