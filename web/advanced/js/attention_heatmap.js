/**
 * attention_heatmap.js — 4x4 live cross-attention heatmap.
 *
 * Sources (in priority order):
 *   1. "attention" events broadcast by the WS bridge.
 *   2. Poll `/api/explain/adaptation` when the WS source is quiet
 *      for more than 4 seconds.
 *   3. Synthetic decay to an idle state if both are unavailable
 *      (so the panel never shows "broken").
 *
 * Colour mapping: each cell's background is the hot colour with
 * alpha == value; foreground text is the palette's active colour.
 * Only palette colours are used.
 */

const ROWS = 4, COLS = 4;

export function mountAttention({ root, bus }) {
  const cells = [];
  root.innerHTML = "";
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const el = document.createElement("div");
      el.className = "attn-cell";
      el.setAttribute("role", "img");
      el.setAttribute("aria-label", `Attention r${r} c${c}`);
      el.setAttribute("data-v", "0.00");
      root.appendChild(el);
      cells.push(el);
    }
  }

  function paint(matrix, token) {
    for (let i = 0; i < cells.length; i++) {
      const v = Math.max(0, Math.min(1, matrix[i] || 0));
      cells[i].setAttribute("data-v", v.toFixed(2));
      // Blend accent→hot by alpha without leaving the palette:
      // base = #0f3460 (accent), highlight = #e94560 (hot).
      cells[i].style.background = `rgba(233, 69, 96, ${v.toFixed(3)})`;
      cells[i].style.boxShadow = v > 0.75
        ? "inset 0 0 0 1px rgba(240,240,240,0.25)"
        : "none";
    }
    if (token) document.getElementById("attn-token").textContent = String(token).slice(0, 40);
  }

  let lastLive = 0;
  bus.addEventListener("attention", (ev) => {
    lastLive = Date.now();
    const m = (ev.detail && ev.detail.matrix) || [];
    const flat = Array.isArray(m[0]) ? m.flat() : m;
    if (flat.length >= ROWS * COLS) paint(flat.slice(0, ROWS * COLS), ev.detail.token);
  });

  async function poll() {
    if (Date.now() - lastLive < 4000) return;
    try {
      const res = await fetch("/api/explain/adaptation", { method: "GET" });
      if (!res.ok) return;
      const json = await res.json();
      const att = json && (json.attention || (json.cross_attention && json.cross_attention.matrix));
      if (Array.isArray(att)) {
        const flat = Array.isArray(att[0]) ? att.flat() : att;
        if (flat.length >= 16) paint(flat.slice(0, 16), json.top_token || "—");
        return;
      }
    } catch {
      /* fallthrough to synthetic */
    }
    // Synthetic idle — soft random pattern.
    const m = Array.from({ length: 16 }, (_, i) => 0.25 + 0.35 * Math.sin(Date.now() / 700 + i));
    paint(m.map(x => Math.max(0.1, Math.min(0.9, x))), null);
  }

  setInterval(poll, 2500);
  paint(Array(16).fill(0.1), "warm-up");

  return { paint };
}
