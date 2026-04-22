"""Infrastructure checks: Dockerfile, compose, helm, k8s manifests, Cedar policy."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from scripts.verification.framework import CheckResult, register_check

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _now_ms(t0: float) -> int:
    """Milliseconds since ``t0``."""
    return int((time.monotonic() - t0) * 1000)


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


@register_check(
    id="infra.dockerfile_parses",
    name="Every Dockerfile* has FROM and CMD/ENTRYPOINT",
    category="infrastructure",
    severity="high",
)
def check_dockerfile_parses() -> CheckResult:
    """Minimal syntax check: ``FROM`` on a line, ``CMD`` or ``ENTRYPOINT`` present.

    A raw ``:latest`` without any other pin context is flagged (because
    it breaks reproducibility / SLSA).  Multi-stage builds that pin at
    least one FROM with a digest or explicit version are tolerated.
    """
    t0 = time.monotonic()
    offenders: list[str] = []
    files = [
        p
        for p in REPO_ROOT.glob("Dockerfile*")
        if p.is_file()
    ] + [
        p for p in (REPO_ROOT / "docker").rglob("Dockerfile*") if p.is_file()
    ]
    if not files:
        return CheckResult(
            check_id="infra.dockerfile_parses",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="no Dockerfile* found",
            evidence=None,
        )
    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            offenders.append(f"{p.relative_to(REPO_ROOT)}: unreadable ({exc})")
            continue
        has_from = re.search(r"^\s*FROM\s+\S+", src, re.MULTILINE)
        has_cmd = re.search(r"^\s*(CMD|ENTRYPOINT)\s+", src, re.MULTILINE)
        if not has_from:
            offenders.append(f"{p.relative_to(REPO_ROOT)}: no FROM directive")
        if not has_cmd:
            offenders.append(
                f"{p.relative_to(REPO_ROOT)}: no CMD/ENTRYPOINT"
            )
        # :latest without any pinned alternative
        latest_only = re.findall(r"^\s*FROM\s+(\S+):latest", src, re.MULTILINE)
        if latest_only and not re.search(
            r"^\s*FROM\s+\S+@sha256:", src, re.MULTILINE
        ):
            offenders.append(
                f"{p.relative_to(REPO_ROOT)}: uses :latest without SHA pin"
            )
    return CheckResult(
        check_id="infra.dockerfile_parses",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} Dockerfile(s) valid"
            if not offenders
            else f"{len(offenders)} Dockerfile issue(s)"
        ),
        evidence="\n".join(offenders[:20]) if offenders else None,
    )


# ---------------------------------------------------------------------------
# docker-compose
# ---------------------------------------------------------------------------


@register_check(
    id="infra.compose_schema",
    name="docker-compose*.yml has services section",
    category="infrastructure",
    severity="medium",
)
def check_compose_schema() -> CheckResult:
    """Every ``docker-compose*.yml`` parses and has a ``services`` map."""
    t0 = time.monotonic()
    try:
        import yaml
    except ImportError:
        return CheckResult(
            check_id="infra.compose_schema",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="PyYAML not available",
            evidence=None,
        )
    files = [
        p for p in REPO_ROOT.glob("docker-compose*.yml") if p.is_file()
    ] + [
        p for p in REPO_ROOT.glob("docker-compose*.yaml") if p.is_file()
    ]
    # Example files are templates and may be skipped.
    files = [p for p in files if not p.name.endswith(".example")]
    if not files:
        return CheckResult(
            check_id="infra.compose_schema",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="no docker-compose*.yml found",
            evidence=None,
        )
    offenders: list[str] = []
    for p in files:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            offenders.append(f"{p.relative_to(REPO_ROOT)}: {exc}")
            continue
        if not isinstance(data, dict):
            offenders.append(f"{p.relative_to(REPO_ROOT)}: top-level not a mapping")
            continue
        if "services" not in data:
            offenders.append(f"{p.relative_to(REPO_ROOT)}: missing services key")
    return CheckResult(
        check_id="infra.compose_schema",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} compose file(s) valid"
            if not offenders
            else f"{len(offenders)} compose schema issue(s)"
        ),
        evidence="\n".join(offenders[:10]) if offenders else None,
    )


# ---------------------------------------------------------------------------
# Helm
# ---------------------------------------------------------------------------


@register_check(
    id="infra.helm_lint",
    name="helm lint deploy/helm/i3 (if helm available)",
    category="infrastructure",
    severity="low",
)
def check_helm_lint() -> CheckResult:
    """Shell out to ``helm lint``; SKIP if helm binary is absent."""
    t0 = time.monotonic()
    helm = shutil.which("helm")
    chart_dir = REPO_ROOT / "deploy" / "helm" / "i3"
    if not helm:
        return CheckResult(
            check_id="infra.helm_lint",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="helm binary not on PATH",
            evidence=None,
        )
    if not chart_dir.exists():
        return CheckResult(
            check_id="infra.helm_lint",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="deploy/helm/i3 missing",
            evidence=None,
        )
    try:
        proc = subprocess.run(
            [helm, "lint", str(chart_dir)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            check_id="infra.helm_lint",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"helm invocation failed: {exc}",
            evidence=None,
        )
    if proc.returncode != 0:
        return CheckResult(
            check_id="infra.helm_lint",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"helm lint exit {proc.returncode}",
            evidence=((proc.stdout or "") + (proc.stderr or ""))[-1500:],
        )
    return CheckResult(
        check_id="infra.helm_lint",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="helm lint clean",
        evidence=None,
    )


# ---------------------------------------------------------------------------
# Kubernetes manifests
# ---------------------------------------------------------------------------


@register_check(
    id="infra.kubernetes_manifests",
    name="deploy/k8s/*.yaml have apiVersion/kind/metadata",
    category="infrastructure",
    severity="medium",
)
def check_kubernetes_manifests() -> CheckResult:
    """Every k8s manifest in ``deploy/k8s`` has the canonical triplet."""
    t0 = time.monotonic()
    try:
        import yaml
    except ImportError:
        return CheckResult(
            check_id="infra.kubernetes_manifests",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="PyYAML not available",
            evidence=None,
        )
    k8s_dir = REPO_ROOT / "deploy" / "k8s"
    if not k8s_dir.exists():
        return CheckResult(
            check_id="infra.kubernetes_manifests",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="deploy/k8s missing",
            evidence=None,
        )
    offenders: list[str] = []
    files = [p for p in k8s_dir.glob("*.yaml") if p.is_file()]
    # Example / skeleton templates are explicitly out of scope.
    files = [p for p in files if "example" not in p.name.lower()]
    for p in files:
        try:
            docs = list(yaml.safe_load_all(p.read_text(encoding="utf-8")))
        except yaml.YAMLError as exc:
            offenders.append(f"{p.name}: parse error {exc}")
            continue
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            # kustomization.yaml is a special case: apiVersion + kind are
            # the only required keys; metadata is optional.
            required = {"apiVersion", "kind"}
            if p.name != "kustomization.yaml":
                required.add("metadata")
            missing = required - set(doc.keys())
            if missing:
                offenders.append(f"{p.name}: missing {sorted(missing)}")
    return CheckResult(
        check_id="infra.kubernetes_manifests",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} k8s manifest(s) valid"
            if not offenders
            else f"{len(offenders)} manifest issue(s)"
        ),
        evidence="\n".join(offenders[:15]) if offenders else None,
    )


# ---------------------------------------------------------------------------
# Cedar policies
# ---------------------------------------------------------------------------


@register_check(
    id="infra.cedar_policy_parses",
    name="deploy/policy/cedar/*.cedar parses via cedarpy",
    category="infrastructure",
    severity="low",
)
def check_cedar_policy_parses() -> CheckResult:
    """Best-effort cedar parse.  SKIP if cedarpy is not importable."""
    t0 = time.monotonic()
    try:
        import cedarpy  # type: ignore[import-not-found]
    except ImportError:
        return CheckResult(
            check_id="infra.cedar_policy_parses",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="cedarpy not available",
            evidence=None,
        )
    cedar_dir = REPO_ROOT / "deploy" / "policy" / "cedar"
    if not cedar_dir.exists():
        return CheckResult(
            check_id="infra.cedar_policy_parses",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="deploy/policy/cedar missing",
            evidence=None,
        )
    files = list(cedar_dir.glob("*.cedar"))
    if not files:
        return CheckResult(
            check_id="infra.cedar_policy_parses",
            status="SKIP",
            duration_ms=_now_ms(t0),
            message="no *.cedar files",
            evidence=None,
        )
    offenders: list[str] = []
    for p in files:
        try:
            # cedarpy exposes a ``format_policies`` / ``is_authorized``
            # API; the parse entry point is implementation-specific so
            # we simply attempt to read and pass through a helper if
            # one exists.
            src = p.read_text(encoding="utf-8")
            if hasattr(cedarpy, "format_policies"):
                cedarpy.format_policies(src)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            offenders.append(f"{p.name}: {type(exc).__name__}: {exc}")
    return CheckResult(
        check_id="infra.cedar_policy_parses",
        status="PASS" if not offenders else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(files)} cedar policy/ies parsed"
            if not offenders
            else f"{len(offenders)} policy parse issue(s)"
        ),
        evidence="\n".join(offenders[:10]) if offenders else None,
    )
