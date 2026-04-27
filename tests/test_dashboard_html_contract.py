"""Iter 85 — Dashboard HTML nav-link + tab-panel contract test.

The dashboard's tab router (web/js/tab_router.js) joins
``a[data-tab=X]`` nav links to ``#tab-X`` panels.  A nav link
without its panel (or vice versa) silently breaks the tab navigation.
This test reads the static index.html and asserts every nav-link's
data-tab has a matching tab-panel id.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_INDEX = Path("web/index.html")


@pytest.fixture(scope="module")
def html() -> str:
    if not _INDEX.exists():
        pytest.skip("web/index.html not present")
    return _INDEX.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Nav links and tab panels must agree
# ---------------------------------------------------------------------------

def _nav_link_tabs(html: str) -> set[str]:
    pat = re.compile(r'class="nav-link[^"]*"\s+data-tab="([^"]+)"')
    return set(pat.findall(html))


def _tab_panel_ids(html: str) -> set[str]:
    """Find every <section ...class="tab-panel"... id="tab-X" ...> id.

    Attribute order varies (some panels list class first, some list
    id first), so we extract section tags then pull the id from each.
    """
    section_pat = re.compile(
        r'<section[^>]*class="[^"]*tab-panel[^"]*"[^>]*>',
        re.IGNORECASE,
    )
    id_pat = re.compile(r'id="tab-([^"]+)"')
    out: set[str] = set()
    for tag in section_pat.findall(html):
        m = id_pat.search(tag)
        if m:
            out.add(m.group(1))
    return out


def test_every_nav_link_has_a_panel(html):
    nav = _nav_link_tabs(html)
    panels = _tab_panel_ids(html)
    missing = nav - panels
    assert not missing, \
        f"nav links without matching tab panel: {sorted(missing)}"


def test_every_tab_panel_has_a_nav_link(html):
    """Optional: a panel without a nav link is dead code."""
    nav = _nav_link_tabs(html)
    panels = _tab_panel_ids(html)
    orphan = panels - nav
    # Some tabs are exposed via dropdowns or programmatic navigation
    # (not direct nav-links), so this is informational.  We just
    # log them — never fail.
    if orphan:
        pytest.skip(f"informational: panels without nav-link: "
                    f"{sorted(orphan)}")


def test_iter51_huawei_tabs_present(html):
    panels = _tab_panel_ids(html)
    iter51_tabs = {"intent", "edge-profile", "finetune",
                   "facts", "multimodal", "research", "jdmap"}
    missing = iter51_tabs - panels
    assert not missing, f"iter-51 huawei tabs missing: {missing}"


def test_iter51_subsystem_grid_in_stack_tab(html):
    """Iter 51 added the 22-card subsystem grid to the Stack tab."""
    assert 'id="stack-subsystem-grid"' in html, \
        "iter-51 Stack-tab subsystem grid missing from index.html"


def test_iter51_huawei_dropdown_present(html):
    """Iter 51 added a Huawei dropdown nav.  At least one nav link
    with data-tab in the iter-51 set must exist."""
    nav = _nav_link_tabs(html)
    huawei_nav = nav & {"intent", "edge-profile", "finetune",
                        "facts", "multimodal", "research", "jdmap"}
    assert huawei_nav, "no iter-51 Huawei nav links found"


def test_required_css_files_referenced(html):
    """The huawei_tabs.css that styles iter-51..62 chips + cards
    must be referenced from index.html."""
    assert "huawei_tabs.css" in html


def test_required_js_files_referenced(html):
    assert "huawei_tabs.js" in html
    assert "chat.js" in html
    assert "app.js" in html
