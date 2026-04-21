/**
 * wcag_audit.js -- I3 Advanced Panels
 *
 * Self-running accessibility audit. Activate by appending `?a11y=1`
 * to the URL. Walks every focusable element, checks contrast ratios
 * against WCAG 2.2 AA thresholds (4.5:1 normal, 3:1 large), flags
 * missing aria-labels on interactive elements, detects heading-level
 * gaps, and checks tab-order sanity. Prints a pass/fail table to the
 * devtools console and logs an overall AA/AAA score.
 *
 * No third-party dependencies.
 */

(function () {
    'use strict';

    function isAuditRequested() {
        try {
            const p = new URLSearchParams(window.location.search);
            return p.get('a11y') === '1';
        } catch (e) {
            return false;
        }
    }

    if (!isAuditRequested()) return;

    // ----- Colour utilities ------------------------------------------------

    function parseColour(str) {
        if (!str) return null;
        const m = str.match(/rgba?\(([^)]+)\)/i);
        if (!m) return null;
        const parts = m[1].split(',').map((x) => parseFloat(x.trim()));
        if (parts.length < 3) return null;
        return {
            r: parts[0] | 0,
            g: parts[1] | 0,
            b: parts[2] | 0,
            a: parts.length >= 4 ? parts[3] : 1,
        };
    }

    function channelLuminance(c) {
        const v = c / 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    }

    function relativeLuminance(rgb) {
        return (
            0.2126 * channelLuminance(rgb.r) +
            0.7152 * channelLuminance(rgb.g) +
            0.0722 * channelLuminance(rgb.b)
        );
    }

    function contrastRatio(fg, bg) {
        if (!fg || !bg) return null;
        const l1 = relativeLuminance(fg);
        const l2 = relativeLuminance(bg);
        const lighter = Math.max(l1, l2);
        const darker = Math.min(l1, l2);
        return (lighter + 0.05) / (darker + 0.05);
    }

    // Walk up the DOM until we find an element with a non-transparent bg.
    function effectiveBackground(el) {
        let cur = el;
        while (cur && cur.nodeType === 1) {
            const bg = parseColour(getComputedStyle(cur).backgroundColor);
            if (bg && bg.a > 0) return bg;
            cur = cur.parentElement;
        }
        return { r: 26, g: 26, b: 46, a: 1 }; // page default #1a1a2e
    }

    // ----- Focusable selector ---------------------------------------------

    const FOCUSABLE = [
        'a[href]',
        'area[href]',
        'input:not([disabled]):not([type="hidden"])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        'button:not([disabled])',
        'iframe',
        'object',
        'embed',
        '[tabindex]:not([tabindex="-1"])',
        '[contenteditable="true"]',
    ].join(',');

    function isVisible(el) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return false;
        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none') return false;
        return true;
    }

    function isLargeText(el) {
        const cs = getComputedStyle(el);
        const size = parseFloat(cs.fontSize);
        const weight = parseInt(cs.fontWeight, 10) || 400;
        // WCAG: 18pt = 24px, or 14pt = 18.66px + bold.
        if (size >= 24) return true;
        if (size >= 18.66 && weight >= 700) return true;
        return false;
    }

    function hasAccessibleName(el) {
        const aria = el.getAttribute('aria-label');
        if (aria && aria.trim()) return true;
        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy && document.getElementById(labelledBy)) return true;
        const title = el.getAttribute('title');
        if (title && title.trim()) return true;
        const txt = (el.textContent || '').trim();
        if (txt) return true;
        if (el.tagName === 'INPUT') {
            const id = el.getAttribute('id');
            if (id && document.querySelector(`label[for="${id}"]`)) return true;
            if (el.closest('label')) return true;
            if (el.getAttribute('placeholder')) return true;
        }
        return false;
    }

    // ----- Audit ----------------------------------------------------------

    function auditContrast(results) {
        const all = document.querySelectorAll('body *');
        all.forEach((el) => {
            if (!isVisible(el)) return;
            const txt = Array.from(el.childNodes)
                .filter((n) => n.nodeType === 3 && (n.textContent || '').trim())
                .map((n) => n.textContent)
                .join('');
            if (!txt) return;
            const cs = getComputedStyle(el);
            const fg = parseColour(cs.color);
            const bg = effectiveBackground(el);
            const ratio = contrastRatio(fg, bg);
            if (ratio === null) return;
            const large = isLargeText(el);
            const aaThreshold = large ? 3.0 : 4.5;
            const aaaThreshold = large ? 4.5 : 7.0;
            const passAA = ratio >= aaThreshold;
            const passAAA = ratio >= aaaThreshold;
            results.push({
                check: 'contrast',
                pass: passAA,
                passAAA,
                element: el.tagName.toLowerCase() +
                    (el.id ? '#' + el.id : '') +
                    (el.className && typeof el.className === 'string'
                        ? '.' + el.className.split(/\s+/).slice(0, 2).join('.')
                        : ''),
                detail: `ratio=${ratio.toFixed(2)} min=${aaThreshold}`,
            });
        });
    }

    function auditAriaLabels(results) {
        const focusables = document.querySelectorAll(FOCUSABLE);
        focusables.forEach((el) => {
            if (!isVisible(el)) return;
            const ok = hasAccessibleName(el);
            results.push({
                check: 'aria-label',
                pass: ok,
                passAAA: ok,
                element: el.tagName.toLowerCase() +
                    (el.id ? '#' + el.id : ''),
                detail: ok ? 'has accessible name' : 'MISSING accessible name',
            });
        });
    }

    function auditTabOrder(results) {
        const focusables = Array.from(document.querySelectorAll(FOCUSABLE))
            .filter(isVisible);
        let prev = -Infinity;
        let clean = true;
        let offender = null;
        for (const el of focusables) {
            const ti = parseInt(el.getAttribute('tabindex') || '0', 10);
            // tabindex > 0 is an anti-pattern.
            if (ti > 0 && ti < prev) {
                clean = false;
                offender = el;
                break;
            }
            if (ti > 0) prev = ti;
        }
        results.push({
            check: 'tab-order',
            pass: clean,
            passAAA: clean,
            element: offender
                ? offender.tagName.toLowerCase()
                : `${focusables.length} focusable elements`,
            detail: clean ? 'no descending positive tabindex' : 'descending tabindex detected',
        });
    }

    function auditHeadingGaps(results) {
        const headings = document.querySelectorAll('h1,h2,h3,h4,h5,h6');
        let last = 0;
        let clean = true;
        let offender = null;
        headings.forEach((h) => {
            const lvl = parseInt(h.tagName.substring(1), 10);
            if (last > 0 && lvl > last + 1) {
                clean = false;
                if (!offender) offender = h;
            }
            last = lvl;
        });
        results.push({
            check: 'heading-hierarchy',
            pass: clean,
            passAAA: clean,
            element: offender
                ? offender.tagName.toLowerCase() +
                  (offender.id ? '#' + offender.id : '')
                : `${headings.length} headings`,
            detail: clean ? 'no skipped levels' : 'skipped heading level detected',
        });
    }

    function auditImageAlts(results) {
        const imgs = document.querySelectorAll('img');
        imgs.forEach((img) => {
            if (!isVisible(img)) return;
            const alt = img.getAttribute('alt');
            const ok = alt !== null; // empty alt is valid for decorative
            results.push({
                check: 'img-alt',
                pass: ok,
                passAAA: ok,
                element: 'img' + (img.id ? '#' + img.id : ''),
                detail: ok ? `alt="${(alt || '').slice(0, 40)}"` : 'missing alt attribute',
            });
        });
    }

    function auditLanguage(results) {
        const lang = document.documentElement.getAttribute('lang');
        const ok = !!(lang && lang.trim());
        results.push({
            check: 'html-lang',
            pass: ok,
            passAAA: ok,
            element: 'html',
            detail: ok ? `lang="${lang}"` : 'missing lang attribute on <html>',
        });
    }

    function run() {
        const results = [];
        auditLanguage(results);
        auditContrast(results);
        auditAriaLabels(results);
        auditTabOrder(results);
        auditHeadingGaps(results);
        auditImageAlts(results);

        const total = results.length;
        const passedAA = results.filter((r) => r.pass).length;
        const passedAAA = results.filter((r) => r.passAAA).length;
        const aa = total > 0 ? (passedAA / total) * 100 : 100;
        const aaa = total > 0 ? (passedAAA / total) * 100 : 100;

        console.groupCollapsed(
            `%c[I3 a11y] WCAG 2.2 audit -- ${total} checks, AA ${aa.toFixed(1)}%, AAA ${aaa.toFixed(1)}%`,
            'color:#e94560;font-weight:700'
        );
        try {
            if (console.table) {
                console.table(results.map((r) => ({
                    check: r.check,
                    AA: r.pass ? 'PASS' : 'FAIL',
                    AAA: r.passAAA ? 'PASS' : 'FAIL',
                    element: r.element,
                    detail: r.detail,
                })));
            } else {
                results.forEach((r) => console.log(r));
            }
        } catch (e) {
            results.forEach((r) => console.log(r));
        }
        console.log(
            `%cOverall: AA ${aa.toFixed(1)}% | AAA ${aaa.toFixed(1)}%`,
            'color:#0f3460;font-weight:700;background:#f0f0f0;padding:2px 8px;border-radius:3px'
        );
        console.groupEnd();

        // Make results inspectable after the fact.
        window.__i3_a11y_report = {
            total, passedAA, passedAAA,
            aaScore: aa, aaaScore: aaa,
            results,
        };
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(run, 300));
    } else {
        setTimeout(run, 300);
    }
})();
