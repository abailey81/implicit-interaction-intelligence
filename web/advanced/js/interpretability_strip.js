/**
 * interpretability_strip.js — bottom strip.
 *
 *   - Counterfactual "why this response" snippet.
 *   - Uncertainty pills (one per top-k uncertain dim).
 *   - Refusal / safety badges.
 *   - Active-preference A/B status.
 *
 * Data hooks:
 *   - "interp" bus events (pushed by WS frames of type
 *     counterfactual / refusal / preference).
 *   - Lazy poll of /api/preference/query/:user_id for A/B.
 *   - Lazy poll of /api/explain/adaptation for counterfactuals.
 */

const MAX_TOASTS = 4;

export function mountInterp({ cf, unc, refusal, ab, bus, getState }) {
  // Toast container (counterfactuals only).
  let toastStack = document.querySelector(".toast-stack");
  if (!toastStack) {
    toastStack = document.createElement("div");
    toastStack.className = "toast-stack";
    toastStack.setAttribute("aria-live", "polite");
    document.body.appendChild(toastStack);
  }

  function pushToast(kicker, body) {
    const t = document.createElement("div");
    t.className = "toast";
    t.innerHTML = `<span class="toast-kicker"></span><span class="toast-body"></span>`;
    t.querySelector(".toast-kicker").textContent = kicker;
    t.querySelector(".toast-body").textContent = body;
    toastStack.appendChild(t);
    // Enforce cap.
    while (toastStack.children.length > MAX_TOASTS) toastStack.firstChild.remove();
    setTimeout(() => t.remove(), 7000);
  }

  function renderPills(root, items, cls = "pill") {
    root.innerHTML = "";
    for (const it of items) {
      const span = document.createElement("span");
      span.className = cls + (it.hot ? " pill--hot" : "");
      span.textContent = it.label;
      root.appendChild(span);
    }
  }

  bus.addEventListener("interp", (ev) => {
    const m = ev.detail || {};
    if (m.type === "counterfactual") {
      cf.textContent = m.text || cf.textContent;
      pushToast("Counterfactual", m.text || "Adjusted response style");
    } else if (m.type === "refusal") {
      renderPills(refusal, [{ label: m.reason || "safety", hot: true }]);
    } else if (m.type === "preference") {
      ab.textContent = m.status || JSON.stringify(m).slice(0, 60);
    }
  });

  bus.addEventListener("adaptation", (ev) => {
    const d = ev.detail || {};
    const unc_obj = d.uncertainty || {};
    const items = Object.entries(unc_obj)
      .filter(([, v]) => typeof v === "number")
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k, v]) => ({ label: `${k}: ±${v.toFixed(2)}`, hot: v > 0.2 }));
    renderPills(unc, items);
    if (d.counterfactual) cf.textContent = d.counterfactual;
  });

  /* Lazy polls. */
  async function pollAB() {
    const uid = getState().userId || "demo_user";
    try {
      const res = await fetch(`/api/preference/query/${encodeURIComponent(uid)}`);
      if (!res.ok) return;
      const j = await res.json();
      if (j && j.active_comparison) {
        ab.textContent = `A vs B — ${j.active_comparison.status || "pending"}`;
      } else {
        ab.textContent = "idle";
      }
    } catch { /* endpoint absent */ }
  }
  setInterval(pollAB, 5000);

  async function pollExplain() {
    try {
      const res = await fetch("/api/explain/adaptation");
      if (!res.ok) return;
      const j = await res.json();
      if (j && j.counterfactual) cf.textContent = j.counterfactual;
      if (j && j.refusal_reasons && j.refusal_reasons.length) {
        renderPills(refusal, j.refusal_reasons.slice(0, 3).map(r => ({ label: r, hot: true })));
      }
    } catch { /* endpoint absent */ }
  }
  setInterval(pollExplain, 4000);

  // Initial state.
  renderPills(unc, [{ label: "warming up", hot: false }]);
  renderPills(refusal, [{ label: "OK", hot: false }]);
  ab.textContent = "idle";

  return { pushToast };
}
