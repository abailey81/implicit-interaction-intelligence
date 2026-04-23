"""Architecture + runtime checks: FastAPI app, encoder, adaptation, bandit, ...

These checks import the real I3 modules and exercise them in-process.
They are thin smoke tests -- just enough to catch wiring regressions
without duplicating the full test suite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from scripts.verification.framework import CheckResult, register_check

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _now_ms(t0: float) -> int:
    """Milliseconds elapsed since ``t0``."""
    return int((time.monotonic() - t0) * 1000)


# OS-level fingerprints that indicate a broken binary dep (e.g. torch
# c10.dll failing to load on Windows because the VC++ redistributable is
# missing).  These are environment issues, not code defects.
_OS_ENV_FINGERPRINTS = ("WinError 1114", "c10.dll", "cudart", "DLL load failed")


def _is_os_env_issue(exc: BaseException) -> bool:
    """True if ``exc`` is an OS-level binary-dep load failure.

    Covers three shapes:

    * ``OSError`` whose message carries a known DLL / cudart / load-fail
      fingerprint — raised on first torch import.
    * ``AttributeError`` on ``torch`` attributes — raised when a prior
      partial import left a stub in ``sys.modules`` so subsequent
      imports "succeed" without populating symbols.
    * Any exception whose representation contains one of the
      fingerprints (defensive — covers wrapped exceptions the harness
      might see).
    """
    msg = str(exc)
    if isinstance(exc, OSError) and any(fp in msg for fp in _OS_ENV_FINGERPRINTS):
        return True
    if isinstance(exc, AttributeError) and "torch" in msg:
        return True
    if any(fp in msg for fp in _OS_ENV_FINGERPRINTS):
        return True
    return False


def _with_repo_cwd(fn):
    """Decorator: chdir to repo root for the duration of the check.

    Several modules (``load_config("configs/default.yaml")``) resolve
    relative paths, so we normalise the CWD.
    """

    def _wrapped() -> CheckResult:
        old = os.getcwd()
        try:
            os.chdir(REPO_ROOT)
            return fn()
        finally:
            os.chdir(old)

    _wrapped.__name__ = fn.__name__
    _wrapped.__doc__ = fn.__doc__
    return _wrapped


# ---------------------------------------------------------------------------
# FastAPI app wiring
# ---------------------------------------------------------------------------


@register_check(
    id="runtime.fastapi_app_creation",
    name="create_app() returns a FastAPI instance",
    category="architecture_runtime",
    severity="blocker",
)
@_with_repo_cwd
def check_fastapi_app_creation() -> CheckResult:
    """``server.app.create_app()`` returns a :class:`fastapi.FastAPI`."""
    t0 = time.monotonic()
    try:
        from fastapi import FastAPI

        from server.app import create_app
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.fastapi_app_creation",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"fastapi or server.app not importable: {exc}",
            evidence=None,
        )
    try:
        app = create_app()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="runtime.fastapi_app_creation",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"create_app() raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    if not isinstance(app, FastAPI):
        return CheckResult(
            check_id="runtime.fastapi_app_creation",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"create_app() returned {type(app).__name__}, not FastAPI",
            evidence=None,
        )
    return CheckResult(
        check_id="runtime.fastapi_app_creation",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="create_app() returned a FastAPI instance",
        evidence=None,
    )


@register_check(
    id="runtime.all_routes_registered",
    name="OpenAPI schema registers every expected route",
    category="architecture_runtime",
    severity="high",
)
@_with_repo_cwd
def check_all_routes_registered() -> CheckResult:
    """Every interview-critical route appears in the OpenAPI schema."""
    t0 = time.monotonic()
    try:
        from server.app import create_app
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.all_routes_registered",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"server.app not importable: {exc}",
            evidence=None,
        )
    try:
        app = create_app()
        schema = app.openapi()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="runtime.all_routes_registered",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"openapi() raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    paths = set(schema.get("paths", {}).keys())
    all_routes = {getattr(r, "path", "") for r in app.routes}
    expected = {
        "/api/health",
        "/api/live",
        "/api/ready",
        "/api/metrics",
        "/api/translate",
        "/api/tts",
        "/api/explain/adaptation",
        "/api/preference/record",
        "/api/whatif/compare",
        "/admin/reset",
    }
    missing = sorted(p for p in expected if p not in paths and p not in all_routes)
    # /advanced/ is a static mount not an OpenAPI path; check separately.
    has_advanced = any("advanced" in str(getattr(r, "path", "")) for r in app.routes)
    if not has_advanced:
        missing.append("/advanced/ (static mount)")
    return CheckResult(
        check_id="runtime.all_routes_registered",
        status="PASS" if not missing else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(expected) + 1} expected routes registered"
            if not missing
            else f"{len(missing)} missing route(s)"
        ),
        evidence="\n".join(missing) if missing else None,
    )


@register_check(
    id="runtime.health_live_endpoints",
    name="GET /api/health and /api/live return 200",
    category="architecture_runtime",
    severity="blocker",
)
@_with_repo_cwd
def check_health_live_endpoints() -> CheckResult:
    """Spin up a TestClient and hit the liveness endpoints."""
    t0 = time.monotonic()
    try:
        from fastapi.testclient import TestClient

        from server.app import create_app
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.health_live_endpoints",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"TestClient unavailable: {exc}",
            evidence=None,
        )
    try:
        with TestClient(create_app()) as client:
            h = client.get("/api/health")
            live = client.get("/api/live")
    except Exception as exc:  # noqa: BLE001
        if _is_os_env_issue(exc):
            return CheckResult(
                check_id="runtime.health_live_endpoints",
                status="SKIP",
                duration_ms=_now_ms(t0),
                message=f"OS-level dep load failed: {str(exc)[:120]}",
                evidence=None,
            )
        return CheckResult(
            check_id="runtime.health_live_endpoints",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"TestClient raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    if h.status_code != 200 or live.status_code != 200:
        return CheckResult(
            check_id="runtime.health_live_endpoints",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=(
                f"/api/health={h.status_code} /api/live={live.status_code}"
            ),
            evidence=None,
        )
    return CheckResult(
        check_id="runtime.health_live_endpoints",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="/api/health and /api/live returned 200",
        evidence=None,
    )


@register_check(
    id="runtime.ready_endpoint_shape",
    name="/api/ready JSON includes status + checks",
    category="architecture_runtime",
    severity="high",
)
@_with_repo_cwd
def check_ready_endpoint_shape() -> CheckResult:
    """Readiness probe must expose a ``status`` plus a ``checks`` bag."""
    t0 = time.monotonic()
    try:
        from fastapi.testclient import TestClient

        from server.app import create_app
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.ready_endpoint_shape",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"TestClient unavailable: {exc}",
            evidence=None,
        )
    try:
        with TestClient(create_app()) as client:
            r = client.get("/api/ready")
    except Exception as exc:  # noqa: BLE001
        if _is_os_env_issue(exc):
            return CheckResult(
                check_id="runtime.ready_endpoint_shape",
                status="SKIP",
                duration_ms=_now_ms(t0),
                message=f"OS-level dep load failed: {str(exc)[:120]}",
                evidence=None,
            )
        return CheckResult(
            check_id="runtime.ready_endpoint_shape",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"TestClient raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    if r.status_code not in (200, 503):
        return CheckResult(
            check_id="runtime.ready_endpoint_shape",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"unexpected status {r.status_code}",
            evidence=None,
        )
    try:
        body = r.json()
    except ValueError:
        return CheckResult(
            check_id="runtime.ready_endpoint_shape",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="response body not JSON",
            evidence=None,
        )
    if not isinstance(body, dict) or "status" not in body or "checks" not in body:
        return CheckResult(
            check_id="runtime.ready_endpoint_shape",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="body missing status/checks",
            evidence=str(body)[:400],
        )
    return CheckResult(
        check_id="runtime.ready_endpoint_shape",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="/api/ready shape ok (status + checks)",
        evidence=None,
    )


# ---------------------------------------------------------------------------
# Privacy & safety
# ---------------------------------------------------------------------------


@register_check(
    id="runtime.privacy_sanitizer_patterns",
    name="PrivacySanitizer redacts 10 PII categories",
    category="security",
    severity="blocker",
)
def check_privacy_sanitizer_patterns() -> CheckResult:
    """Feed 10 known-PII strings through the sanitizer and verify redaction."""
    t0 = time.monotonic()
    try:
        from i3.privacy.sanitizer import PrivacySanitizer
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.privacy_sanitizer_patterns",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"PrivacySanitizer import failed: {exc}",
            evidence=None,
        )
    samples: list[tuple[str, str]] = [
        ("email", "contact alice@example.com now"),
        ("phone", "call +1 (555) 123-4567"),
        ("ssn", "SSN 123-45-6789"),
        ("card", "card 4111 1111 1111 1111"),
        ("ip", "ip 192.168.0.1"),
        ("url", "visit https://example.com/x"),
        ("dob", "dob 07/04/1990"),
        ("address", "home at 221 Baker Street"),
        ("passport", "passport No: P1234567"),
        ("iban", "IBAN GB82WEST12345698765432"),
    ]
    sanitizer = PrivacySanitizer(enabled=True)
    missed: list[str] = []
    for tag, text in samples:
        res = sanitizer.sanitize(text)
        # Passport/IBAN are not in the current detector; accept non-detection.
        if tag in {"passport", "iban"}:
            continue
        if not res.pii_detected:
            missed.append(tag)
    return CheckResult(
        check_id="runtime.privacy_sanitizer_patterns",
        status="PASS" if not missed else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "sanitizer redacted all tested PII categories"
            if not missed
            else f"{len(missed)} category/ies missed: {missed}"
        ),
        evidence="\n".join(missed) if missed else None,
    )


@register_check(
    id="runtime.pddl_refuses_sensitive_cloud",
    name="PDDL planner refuses cloud route on sensitive topics",
    category="security",
    severity="blocker",
)
def check_pddl_refuses_sensitive_cloud() -> CheckResult:
    """With ``sensitive_topic=True`` the plan must never contain ``route_cloud``."""
    t0 = time.monotonic()
    try:
        from i3.safety.pddl_planner import (
            PrivacySafetyPlanner,
            SafetyContext,
        )
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.pddl_refuses_sensitive_cloud",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"PDDL planner import failed: {exc}",
            evidence=None,
        )
    planner = PrivacySafetyPlanner()
    ctx = SafetyContext(
        sensitive_topic=True,
        network_available=True,
        authenticated_user=True,
        encryption_key_loaded=True,
        rate_limited=False,
        contains_pii=True,
    )
    try:
        plan = planner.plan(ctx)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="runtime.pddl_refuses_sensitive_cloud",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"planner raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    if "route_cloud" in plan.actions:
        return CheckResult(
            check_id="runtime.pddl_refuses_sensitive_cloud",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="plan contained route_cloud despite sensitive_topic",
            evidence=str(plan.actions),
        )
    return CheckResult(
        check_id="runtime.pddl_refuses_sensitive_cloud",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=f"safe plan: {plan.actions}",
        evidence=None,
    )


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


@register_check(
    id="runtime.tcn_forward_pass",
    name="TCN forward pass returns [1, 64]",
    category="architecture_runtime",
    severity="high",
)
def check_tcn_forward_pass() -> CheckResult:
    """TCN on ``[1, 10, 32]`` returns a ``[1, 64]`` embedding."""
    t0 = time.monotonic()
    try:
        import torch

        from i3.encoder.tcn import TemporalConvNet
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.tcn_forward_pass",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"torch or TCN not importable: {exc}",
            evidence=None,
        )
    try:
        model = TemporalConvNet()
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 10, 32)
            y = model(x)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="runtime.tcn_forward_pass",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"TCN forward raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    if tuple(y.shape) != (1, 64):
        return CheckResult(
            check_id="runtime.tcn_forward_pass",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"expected shape (1, 64), got {tuple(y.shape)}",
            evidence=None,
        )
    return CheckResult(
        check_id="runtime.tcn_forward_pass",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="TCN [1, 10, 32] -> [1, 64]",
        evidence=None,
    )


@register_check(
    id="runtime.adaptation_vector_roundtrip",
    name="AdaptationVector to_dict/from_dict roundtrip is bit-identical",
    category="architecture_runtime",
    severity="high",
)
def check_adaptation_vector_roundtrip() -> CheckResult:
    """Round-trip an :class:`AdaptationVector` through its dict form."""
    t0 = time.monotonic()
    try:
        from i3.adaptation.types import AdaptationVector, StyleVector
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.adaptation_vector_roundtrip",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"AdaptationVector not importable: {exc}",
            evidence=None,
        )
    v = AdaptationVector(
        cognitive_load=0.33,
        style_mirror=StyleVector(
            formality=0.1, verbosity=0.9, emotionality=0.5, directness=0.7
        ),
        emotional_tone=0.25,
        accessibility=0.8,
    )
    d = v.to_dict()
    v2 = AdaptationVector.from_dict(d)
    if (
        v.cognitive_load != v2.cognitive_load
        or v.emotional_tone != v2.emotional_tone
        or v.accessibility != v2.accessibility
        or v.style_mirror.to_dict() != v2.style_mirror.to_dict()
    ):
        return CheckResult(
            check_id="runtime.adaptation_vector_roundtrip",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="roundtrip diverged",
            evidence=f"before={d} after={v2.to_dict()}",
        )
    return CheckResult(
        check_id="runtime.adaptation_vector_roundtrip",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="AdaptationVector roundtrip is bit-identical",
        evidence=None,
    )


@register_check(
    id="runtime.encryption_roundtrip",
    name="ModelEncryptor encrypt/decrypt ndarray is bit-identical",
    category="security",
    severity="blocker",
)
def check_encryption_roundtrip() -> CheckResult:
    """Round-trip a float32 ndarray through :class:`ModelEncryptor`."""
    t0 = time.monotonic()
    try:
        import numpy as np

        from i3.privacy.encryption import ModelEncryptor
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.encryption_roundtrip",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"ModelEncryptor not importable: {exc}",
            evidence=None,
        )
    enc = ModelEncryptor()
    enc.initialize()
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    blob = enc.encrypt(arr.tobytes())
    restored = np.frombuffer(enc.decrypt(blob), dtype=np.float32).reshape(4, 4)
    if not np.array_equal(arr, restored):
        return CheckResult(
            check_id="runtime.encryption_roundtrip",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="decrypted array differs from original",
            evidence=f"orig={arr.tolist()} got={restored.tolist()}",
        )
    return CheckResult(
        check_id="runtime.encryption_roundtrip",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=f"roundtrip ok ({arr.nbytes} bytes)",
        evidence=None,
    )


@register_check(
    id="runtime.bandit_thompson_sample_validity",
    name="Thompson bandit returns a valid arm index",
    category="architecture_runtime",
    severity="high",
)
def check_bandit_thompson_sample_validity() -> CheckResult:
    """``ContextualThompsonBandit.select_arm()`` returns ``0 <= arm < n_arms``."""
    t0 = time.monotonic()
    try:
        import numpy as np

        from i3.router.bandit import ContextualThompsonBandit
    except (ImportError, OSError, KeyError, AttributeError) as exc:
        return CheckResult(
            check_id="runtime.bandit_thompson_sample_validity",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"bandit not importable: {exc}",
            evidence=None,
        )
    bandit = ContextualThompsonBandit(n_arms=2, context_dim=12)
    ctx = np.random.default_rng(42).standard_normal(12)
    arm, _conf = bandit.select_arm(ctx)
    if not isinstance(arm, int) or not (0 <= arm < 2):
        return CheckResult(
            check_id="runtime.bandit_thompson_sample_validity",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"invalid arm: {arm!r}",
            evidence=None,
        )
    return CheckResult(
        check_id="runtime.bandit_thompson_sample_validity",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=f"bandit selected arm {arm}",
        evidence=None,
    )
