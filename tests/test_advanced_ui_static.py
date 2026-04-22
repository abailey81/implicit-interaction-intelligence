"""Static-file tests for the advanced command-center UI (Batch G9).

These tests build a minimal FastAPI app that mounts the advanced
UI exactly the way ``server/app.py`` does, then verifies the
on-disk layout, MIME types, internal references, CSP / SRI
attributes, and path-traversal resistance.

We intentionally DO NOT import ``server.app`` because that module
loads the full pipeline (heavy). The mount is 4 lines — verifying
it in isolation keeps the test fast and hermetic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
ADV_DIR = REPO_ROOT / "web" / "advanced"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.mount(
        "/advanced",
        StaticFiles(directory=str(ADV_DIR), html=True),
        name="advanced_ui",
    )
    return TestClient(app)


@pytest.fixture(scope="module")
def index_html() -> str:
    return (ADV_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Smoke: directory exists
# ---------------------------------------------------------------------------


def test_advanced_dir_layout() -> None:
    assert ADV_DIR.is_dir(), "web/advanced/ must exist on D:"
    for must in [
        "index.html",
        "README.md",
        "css/command_center.css",
        "js/main.js",
        "js/ws_bridge.js",
        "js/chat_panel.js",
        "js/embedding_3d.js",
        "js/radial_gauges.js",
        "js/attention_heatmap.js",
        "js/router_dashboard.js",
        "js/interpretability_strip.js",
        "js/guided_tour.js",
        "js/keyboard_shortcuts.js",
        "js/screen_recording_preset.js",
        "js/loading_states.js",
        "js/a11y.js",
        "vendor/README.md",
    ]:
        p = ADV_DIR / must
        assert p.exists(), f"missing required file: {must}"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_advanced_index_ok(client: TestClient) -> None:
    resp = client.get("/advanced/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "I3 Command Center" in resp.text or "Command Center" in resp.text


def test_advanced_css_ok(client: TestClient) -> None:
    resp = client.get("/advanced/css/command_center.css")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/css")
    # Palette vars are present.
    for token in ["--bg", "--panel", "--accent", "--hot", "--muted", "--active"]:
        assert token in resp.text, f"palette var missing: {token}"


def test_advanced_main_js_ok(client: TestClient) -> None:
    resp = client.get("/advanced/js/main.js")
    assert resp.status_code == 200
    ctype = resp.headers["content-type"].lower()
    assert ("javascript" in ctype) or ctype.startswith("text/plain"), (
        f"main.js served with unexpected MIME: {ctype}"
    )


# ---------------------------------------------------------------------------
# Reference integrity
# ---------------------------------------------------------------------------


_SRC_HREF_RE = re.compile(
    r"""(?:src|href)\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)


def test_all_local_refs_exist(index_html: str) -> None:
    """Every ``src=`` and ``href=`` pointing to a relative path must
    resolve to an actual file inside ``web/advanced/``.
    """
    missing: list[str] = []
    for match in _SRC_HREF_RE.finditer(index_html):
        ref = match.group(1).strip()
        # Skip absolute URLs and fragment/anchor-only links.
        if ref.startswith(("http://", "https://", "data:", "#", "mailto:")):
            continue
        target = (ADV_DIR / ref.lstrip("/")).resolve()
        if not target.exists():
            missing.append(ref)
    assert missing == [], f"broken local refs in index.html: {missing}"


def test_palette_css_vars_in_html_or_linked_css(index_html: str) -> None:
    """The HTML must link to ``command_center.css`` which defines the
    closing-pulse-friendly palette custom properties."""
    assert "css/command_center.css" in index_html
    css = (ADV_DIR / "css" / "command_center.css").read_text(encoding="utf-8")
    for colour in ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#a0a0b0", "#f0f0f0"]:
        assert colour.lower() in css.lower(), f"palette colour missing: {colour}"


# ---------------------------------------------------------------------------
# Third-party origin policy
# ---------------------------------------------------------------------------


_ALLOWED_THIRD_PARTY_HOSTS = {"cdn.jsdelivr.net", "unpkg.com"}
_URL_RE = re.compile(r"""https?://([A-Za-z0-9.\-]+)""")


def test_only_documented_cdn_hosts(index_html: str) -> None:
    hosts = set(_URL_RE.findall(index_html))
    # Strip hosts that only appear in comments / policy strings (harmless).
    # We still require all found hosts to be in the allow-list OR localhost.
    bad = {h for h in hosts if h not in _ALLOWED_THIRD_PARTY_HOSTS
           and not h.endswith("localhost")
           and h not in {"127.0.0.1"}}
    assert not bad, f"disallowed third-party hosts referenced: {bad}"


def test_third_party_scripts_have_sri(index_html: str) -> None:
    """Each <script src="https://..."> must have an integrity attr."""
    # Extract <script ...> tags with src starting with http(s).
    tag_re = re.compile(r"<script\b[^>]*>", re.IGNORECASE)
    for tag in tag_re.findall(index_html):
        src_m = re.search(r'src\s*=\s*["\'](https?://[^"\']+)', tag, re.IGNORECASE)
        if not src_m:
            continue
        assert re.search(r'integrity\s*=\s*["\']sha384-', tag, re.IGNORECASE), (
            f"CDN <script> without sha384 SRI: {tag}"
        )
        assert re.search(r'crossorigin\s*=', tag, re.IGNORECASE), (
            f"CDN <script> without crossorigin: {tag}"
        )


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(client: TestClient) -> None:
    """``/advanced/../server/routes.py`` must not escape the mount."""
    resp = client.get("/advanced/../server/routes.py")
    # Starlette's StaticFiles normalises the path and returns 404 for
    # anything outside the mount root. Anything non-2xx is acceptable
    # here; specifically we must NOT leak file contents.
    assert resp.status_code != 200, (
        "path traversal returned 200; contents may have been leaked"
    )


def test_csp_meta_present(index_html: str) -> None:
    """A CSP meta tag must be present and restrict script-src."""
    assert "Content-Security-Policy" in index_html
    assert "script-src" in index_html
    # Must name the allowed CDN hosts explicitly.
    assert "cdn.jsdelivr.net" in index_html
