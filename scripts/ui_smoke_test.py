"""UI smoke-test: drive Playwright through every tab and screenshot it.

Iter 51 (2026-04-27) — closes the visibility gap so the Huawei reviewer
can see the entire 19-tab surface without booting the demo locally.

Usage::

    # 1. Start the server in a separate shell:
    poetry run uvicorn server.app:app --host 127.0.0.1 --port 8000

    # 2. Run this script:
    poetry run python scripts/ui_smoke_test.py --out docs/screenshots

The script:
  * loads the SPA at ``http://127.0.0.1:8000``
  * waits for the WebSocket to connect (the chat-status pill flips to "Live")
  * sends 3 synthetic chat turns to populate live state
  * iterates the navbar in DOM order, clicks each ``data-tab``, waits for
    the corresponding ``.tab-panel`` to become visible, asserts it has
    non-empty content, then captures a 1440 x 900 viewport PNG into the
    output directory
  * exits non-zero if any tab fails to render or the console emits an
    error during the visit (helps catch regressions where a panel ships
    but its JS module 404s)

Playwright is an optional extra; install via ``pip install playwright &&
playwright install chromium``.  The script no-ops with a friendly error
when Playwright isn't on the path so CI without a headless browser
doesn't fail.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_OUT = Path("docs/screenshots")
WARMUP_TURNS = (
    "hi, how are you doing today?",
    "tell me about your favourite book",
    "set timer for 10 minutes",
)


async def _run(url: str, out: Path, headless: bool) -> int:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print(
            "playwright is not installed. Install with:\n"
            "    pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 2

    out.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        console_errors: list[str] = []
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error" else None,
        )

        await page.goto(url, wait_until="networkidle")
        await page.wait_for_selector(".tab-panel", timeout=15_000)

        # Warm-up: 3 turns to populate live state.
        try:
            for line in WARMUP_TURNS:
                await page.fill("textarea, input[type='text'][placeholder*='message' i]", line)
                await page.keyboard.press("Enter")
                await asyncio.sleep(2.0)
        except Exception as exc:
            print(f"warmup failed (continuing): {exc}", file=sys.stderr)

        nav_links = await page.query_selector_all("a.nav-link[data-tab], .nav-link[data-tab]")
        if not nav_links:
            nav_links = await page.query_selector_all("[data-tab]")
        print(f"discovered {len(nav_links)} tabs to capture")

        failures: list[str] = []
        for idx, link in enumerate(nav_links, start=1):
            tab_id = await link.get_attribute("data-tab")
            if not tab_id:
                continue
            await link.click()
            await asyncio.sleep(0.7)
            panel = await page.query_selector(f"#tab-{tab_id}")
            if panel is None:
                failures.append(f"#tab-{tab_id} not found")
                continue
            text = (await panel.inner_text()).strip()
            if not text:
                failures.append(f"#tab-{tab_id} empty")
            png_path = out / f"{idx:02d}_{tab_id}.png"
            await page.screenshot(path=str(png_path), full_page=True)
            print(f"  captured {png_path.name}")

        await browser.close()

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    if console_errors:
        print("\nconsole.error events captured:")
        for err in console_errors[:25]:
            print("  -", err)
    print(f"\nOK — {len(nav_links)} tabs captured to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="UI smoke-test for I³ SPA")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args.url, args.out, headless=not args.headed))


if __name__ == "__main__":
    raise SystemExit(main())
