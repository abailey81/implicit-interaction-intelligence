"""Interview-readiness checks: slide count, Q&A count, closing line, ADRs, docs."""

from __future__ import annotations

import re
import time
from pathlib import Path

from scripts.verification.framework import CheckResult, register_check

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _now_ms(t0: float) -> int:
    """Milliseconds since ``t0``."""
    return int((time.monotonic() - t0) * 1000)


def _read(path: Path) -> str | None:
    """Return the file content as UTF-8, or ``None`` on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# Slides / presentation
# ---------------------------------------------------------------------------


@register_check(
    id="interview.slide_count",
    name="docs/slides/presentation.md has exactly 15 slides (frontmatter-aware)",
    category="interview_readiness",
    severity="high",
)
def check_slide_count() -> CheckResult:
    """Marp slide decks use ``---`` on its own line as a separator.

    A Marp deck contains:
      * 1 leading `---` on line 1 (frontmatter opener),
      * 1 closing `---` after the YAML frontmatter (slide 1 boundary),
      * N-1 `---` between slides 1..N.

    So for N = 15 slides we expect **either** 16 `---` separators
    (with Marp frontmatter) **or** 15 (without frontmatter). Accept both.
    """
    t0 = time.monotonic()
    p = REPO_ROOT / "docs" / "slides" / "presentation.md"
    src = _read(p)
    if src is None:
        return CheckResult(
            check_id="interview.slide_count",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="presentation.md missing or unreadable",
            evidence=None,
        )
    # Count lines that are exactly '---' (with optional trailing whitespace).
    count = sum(1 for line in src.splitlines() if line.strip() == "---")
    has_frontmatter = src.lstrip().startswith("---")
    expected = 16 if has_frontmatter else 15
    return CheckResult(
        check_id="interview.slide_count",
        status="PASS" if count == expected else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{count} '---' separator(s); "
            f"expected {expected} (Marp frontmatter={has_frontmatter})"
        ),
        evidence=None,
    )


@register_check(
    id="interview.qa_pair_count",
    name="docs/slides/qa_prep.md has exactly 52 Q&A pairs",
    category="interview_readiness",
    severity="high",
)
def check_qa_pair_count() -> CheckResult:
    """Q&A pairs are counted by their ``### `` headings."""
    t0 = time.monotonic()
    p = REPO_ROOT / "docs" / "slides" / "qa_prep.md"
    src = _read(p)
    if src is None:
        return CheckResult(
            check_id="interview.qa_pair_count",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="qa_prep.md missing or unreadable",
            evidence=None,
        )
    count = sum(1 for line in src.splitlines() if line.startswith("### "))
    return CheckResult(
        check_id="interview.qa_pair_count",
        status="PASS" if count == 52 else "FAIL",
        duration_ms=_now_ms(t0),
        message=f"{count} '### ' Q&A heading(s), expected 52",
        evidence=None,
    )


@register_check(
    id="interview.closing_line_verbatim",
    name="closing_lines.md contains the verbatim closing line",
    category="interview_readiness",
    severity="blocker",
)
def check_closing_line_verbatim() -> CheckResult:
    """Two sentences combined must appear verbatim somewhere in the file."""
    t0 = time.monotonic()
    p = REPO_ROOT / "docs" / "slides" / "closing_lines.md"
    src = _read(p)
    if src is None:
        return CheckResult(
            check_id="interview.closing_line_verbatim",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="closing_lines.md missing or unreadable",
            evidence=None,
        )
    # The file may wrap the line across two markdown lines; normalise
    # whitespace/newlines before matching.
    collapsed = re.sub(r"\s+", " ", src)
    needle = (
        "I build intelligent systems that adapt to people. "
        "I'd like to do that in your lab."
    )
    return CheckResult(
        check_id="interview.closing_line_verbatim",
        status="PASS" if needle in collapsed else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "verbatim closing line present"
            if needle in collapsed
            else "verbatim closing line NOT found"
        ),
        evidence=None,
    )


@register_check(
    id="interview.honesty_slide_title_case",
    name="presentation.md contains the 'What This Prototype Is Not' slide",
    category="interview_readiness",
    severity="high",
)
def check_honesty_slide_title_case() -> CheckResult:
    """Exact Title Case string must appear (the honesty slide)."""
    t0 = time.monotonic()
    p = REPO_ROOT / "docs" / "slides" / "presentation.md"
    src = _read(p)
    if src is None:
        return CheckResult(
            check_id="interview.honesty_slide_title_case",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="presentation.md missing",
            evidence=None,
        )
    needle = "What This Prototype Is Not"
    return CheckResult(
        check_id="interview.honesty_slide_title_case",
        status="PASS" if needle in src else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "honesty slide title present"
            if needle in src
            else "honesty slide title NOT found"
        ),
        evidence=None,
    )


# ---------------------------------------------------------------------------
# ADRs + supporting docs
# ---------------------------------------------------------------------------


@register_check(
    id="interview.adr_count",
    name="docs/adr has >= 10 numbered ADRs",
    category="interview_readiness",
    severity="medium",
)
def check_adr_count() -> CheckResult:
    """Architecture decisions keep the harness honest."""
    t0 = time.monotonic()
    d = REPO_ROOT / "docs" / "adr"
    if not d.exists():
        return CheckResult(
            check_id="interview.adr_count",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="docs/adr missing",
            evidence=None,
        )
    # Numbered ADRs: 0001-foo.md, 0002-bar.md, ...
    adrs = [
        p
        for p in d.glob("*.md")
        if re.match(r"^\d{4}-", p.name)
    ]
    return CheckResult(
        check_id="interview.adr_count",
        status="PASS" if len(adrs) >= 10 else "FAIL",
        duration_ms=_now_ms(t0),
        message=f"{len(adrs)} numbered ADR(s) (>=10 required)",
        evidence=None,
    )


@register_check(
    id="interview.changelog_unreleased_nonempty",
    name="CHANGELOG.md [Unreleased] section > 500 chars",
    category="interview_readiness",
    severity="low",
)
def check_changelog_unreleased_nonempty() -> CheckResult:
    """Validates an Unreleased section with real content exists."""
    t0 = time.monotonic()
    p = REPO_ROOT / "CHANGELOG.md"
    src = _read(p)
    if src is None:
        return CheckResult(
            check_id="interview.changelog_unreleased_nonempty",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="CHANGELOG.md missing",
            evidence=None,
        )
    # Capture content between '## [Unreleased]' and the next '## [' heading.
    match = re.search(
        r"##\s*\[Unreleased\](.*?)(?=^##\s*\[|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        return CheckResult(
            check_id="interview.changelog_unreleased_nonempty",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="[Unreleased] section not found",
            evidence=None,
        )
    body = match.group(1).strip()
    ok = len(body) > 500
    return CheckResult(
        check_id="interview.changelog_unreleased_nonempty",
        status="PASS" if ok else "FAIL",
        duration_ms=_now_ms(t0),
        message=f"[Unreleased] body is {len(body)} chars (>500 required)",
        evidence=None,
    )


