/**
 * radial_gauges.js — 8 animated radial SVG gauges.
 *
 * Each gauge renders:
 *   - background arc      (accent)
 *   - uncertainty band    (muted, width = 2 * CI)
 *   - foreground fill     (hot)
 *   - target marker tick  (active)
 *
 * Animation is pure CSS (stroke-dashoffset transition), so the
 * reduced-motion media query in command_center.css disables it
 * automatically when the user prefers reduced motion.
 */

const GAUGE_NAMES = [
  "cognitive_load",
  "style.formality",
  "style.verbosity",
  "style.emotionality",
  "style.directness",
  "emotional_tone",
  "accessibility",
  "reserved",
];

const LABEL = {
  "cognitive_load":    "Cog. load",
  "style.formality":   "Formality",
  "style.verbosity":   "Verbosity",
  "style.emotionality":"Emotion",
  "style.directness":  "Directness",
  "emotional_tone":    "Tone",
  "accessibility":     "A11y",
  "reserved":          "Reserved",
};

const SIZE = 84;        // svg viewBox size
const R = 32;           // arc radius
const CIRC = 2 * Math.PI * R;
// Use only 270 deg of the circle so the dial has a visible gap at the bottom.
const ARC_FRAC = 0.75;

function arcPath(valueFrac) {
  const v = Math.max(0, Math.min(1, valueFrac));
  const dash = CIRC * ARC_FRAC * v;
  const gap = CIRC - dash;
  return { dash, gap };
}

function svgTemplate(name) {
  const cx = SIZE / 2, cy = SIZE / 2;
  const start = 135; // deg, rotates the arc so gap faces bottom
  const full = arcPath(1);
  return `
    <svg viewBox="0 0 ${SIZE} ${SIZE}" role="img" aria-labelledby="lbl-${name}">
      <title id="lbl-${name}">${LABEL[name] || name}</title>
      <g transform="rotate(${start} ${cx} ${cy})">
        <circle class="gauge-arc-bg"  cx="${cx}" cy="${cy}" r="${R}"
                fill="none" stroke-width="6"
                stroke-dasharray="${full.dash} ${full.gap}"
                stroke-linecap="round" />
        <circle class="gauge-arc-unc" cx="${cx}" cy="${cy}" r="${R}"
                fill="none" stroke-width="6"
                stroke-dasharray="0 ${CIRC}"
                stroke-linecap="butt"
                data-kind="unc" />
        <circle class="gauge-arc-fg"  cx="${cx}" cy="${cy}" r="${R}"
                fill="none" stroke-width="6"
                stroke-dasharray="0 ${CIRC}"
                stroke-linecap="round"
                data-kind="fg" />
        <line class="gauge-target" x1="${cx + R - 5}" y1="${cy}" x2="${cx + R + 5}" y2="${cy}"
              stroke-width="2" data-kind="target" />
      </g>
    </svg>
  `;
}

export function mountGauges({ root, bus }) {
  root.innerHTML = "";
  const gauges = {};

  for (const name of GAUGE_NAMES) {
    const wrap = document.createElement("div");
    wrap.className = "gauge";
    wrap.setAttribute("data-gauge", name);
    wrap.innerHTML = `
      <div class="gauge-label">${LABEL[name] || name}</div>
      ${svgTemplate(name)}
      <div class="gauge-value mono" data-kind="value">—</div>
    `;
    root.appendChild(wrap);
    gauges[name] = {
      wrap,
      fg: wrap.querySelector('[data-kind="fg"]'),
      unc: wrap.querySelector('[data-kind="unc"]'),
      target: wrap.querySelector('[data-kind="target"]'),
      value: wrap.querySelector('[data-kind="value"]'),
    };
  }

  function updateGauge(name, value, uncertainty = 0, target = null) {
    const g = gauges[name];
    if (!g) return;
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    const { dash } = arcPath(v);
    g.fg.setAttribute("stroke-dasharray", `${dash} ${CIRC}`);

    const u = Math.max(0, Math.min(0.5, Number(uncertainty) || 0));
    // uncertainty band spans 2u, centred on v
    const lo = Math.max(0, v - u);
    const hi = Math.min(1, v + u);
    const band = arcPath(hi).dash - arcPath(lo).dash;
    const offset = arcPath(lo).dash;
    g.unc.setAttribute("stroke-dasharray", `0 ${offset} ${band} ${CIRC}`);

    if (target != null) {
      const t = Math.max(0, Math.min(1, Number(target) || 0));
      const ang = 135 + t * 270;
      g.target.setAttribute("transform", `rotate(${ang - 135} ${SIZE / 2} ${SIZE / 2})`);
      g.target.style.display = "";
    } else {
      g.target.style.display = "none";
    }
    g.value.textContent = v.toFixed(2);
  }

  // Listen to adaptation frames.
  bus.addEventListener("adaptation", (ev) => {
    const data = (ev.detail && ev.detail.data) || {};
    const unc = (ev.detail && ev.detail.uncertainty) || {};
    for (const name of GAUGE_NAMES) {
      const v = readNested(data, name);
      const u = readNested(unc, name);
      if (v != null) updateGauge(name, v, u || 0);
    }
  });

  // Initial idle display.
  for (const name of GAUGE_NAMES) updateGauge(name, 0.5, 0.05);

  return { updateGauge };
}

function readNested(obj, path) {
  if (!obj) return null;
  const parts = path.split(".");
  let cur = obj;
  for (const p of parts) {
    if (cur == null) return null;
    cur = cur[p];
  }
  return (typeof cur === "number") ? cur : null;
}
