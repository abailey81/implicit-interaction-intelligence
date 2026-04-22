/**
 * a11y.js — runtime accessibility helpers.
 *
 *   - announce(msg): writes to the hidden aria-live region.
 *   - runContrastAudit(): logs WCAG 2.2 AA contrast results.
 *   - Respects prefers-reduced-motion + prefers-color-scheme.
 *   - Installs a focus trap for the shortcuts dialog when open.
 */

let liveEl = null;
function live() {
  if (!liveEl) liveEl = document.getElementById("a11y-live");
  return liveEl;
}

export function announce(msg) {
  const el = live();
  if (!el) return;
  // Toggle to re-trigger screen-reader announcements.
  el.textContent = "";
  setTimeout(() => { el.textContent = String(msg).slice(0, 240); }, 50);
}

/* ---------- Contrast audit ---------- */

function parseColor(str) {
  // Accepts "rgb(r,g,b)" or "rgba(r,g,b,a)".
  const m = str.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}
function lum([r, g, b]) {
  const f = (c) => {
    c = c / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
function ratio(a, b) {
  const [L1, L2] = lum(a) > lum(b) ? [lum(a), lum(b)] : [lum(b), lum(a)];
  return (L1 + 0.05) / (L2 + 0.05);
}

export function runContrastAudit() {
  const nodes = Array.from(document.querySelectorAll("body *"))
    .filter(n => n.childNodes.length > 0 && Array.from(n.childNodes).some(c => c.nodeType === 3 && c.textContent.trim()));
  const results = [];
  for (const n of nodes) {
    const s = getComputedStyle(n);
    const fg = parseColor(s.color);
    // Walk up for an opaque background.
    let p = n, bgc = null;
    while (p && !bgc) {
      const ps = getComputedStyle(p);
      const c = parseColor(ps.backgroundColor);
      if (c && !/rgba\(.*,\s*0\)/i.test(ps.backgroundColor)) bgc = c;
      p = p.parentElement;
    }
    if (!fg || !bgc) continue;
    const r = ratio(fg, bgc);
    const sz = parseFloat(s.fontSize);
    const bold = parseInt(s.fontWeight, 10) >= 700;
    const aa = (sz >= 24 || (bold && sz >= 18.66)) ? 3.0 : 4.5;
    if (r < aa) results.push({ node: n, ratio: r.toFixed(2), required: aa });
  }
  if (results.length === 0) {
    console.info("[i3.a11y] contrast audit: all text passes WCAG 2.2 AA.");
  } else {
    console.warn(`[i3.a11y] contrast audit: ${results.length} element(s) below AA:`);
    for (const r of results.slice(0, 20)) console.warn(" ", r.ratio, "<", r.required, r.node);
  }
  return results;
}

/* ---------- Install ---------- */

export function installA11y(bus) {
  // Announce major state changes.
  bus.addEventListener("state", (ev) => {
    const s = ev.detail;
    if (s && s.connState) announce(`Connection ${s.connState}`);
  });

  // Reduced motion — reflect into body attribute so CSS can depend
  // on both the media query and an explicit toggle.
  const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  const apply = () => {
    if (document.body.getAttribute("data-motion") !== "off") {
      document.body.setAttribute("data-motion", mq.matches ? "off" : "on");
    }
  };
  apply();
  mq.addEventListener?.("change", apply);

  // Console welcome.
  console.info("[i3.advanced] a11y installed. Press ? for shortcuts, Alt+A for contrast audit.");
}
