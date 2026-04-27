/**
 * Benchmarks tab.
 *
 * Loads the latest benchmark report from /api/benchmarks/latest and
 * renders four hand-written SVG plots plus a 4-card headline row.
 * The Run button POSTs to /api/benchmarks/run and polls until the
 * timestamp changes.
 */

(() => {
    "use strict";

    const PLOT_NAMES = {
        latency: "latency_breakdown.svg",
        ppl: "perplexity_curve.svg",
        coh: "coherence_categories.svg",
        ada: "adaptation_faithfulness.svg",
    };

    let lastTimestamp = null;
    let pollHandle = null;

    function fmt(value, unit, digits) {
        if (value === null || value === undefined || Number.isNaN(value)) {
            return "—";
        }
        const d = (digits === undefined) ? 1 : digits;
        const n = Number(value);
        if (unit === "params" && n >= 1e6) {
            return `${(n / 1e6).toFixed(1)} M`;
        }
        if (unit === "params" && n >= 1e3) {
            return `${(n / 1e3).toFixed(1)} k`;
        }
        return n.toFixed(d);
    }

    function setHeadline(report) {
        const head = report.headline || {};
        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        set(
            "bench-latency",
            head.latency_p50_ms ? `${fmt(head.latency_p50_ms, null, 0)} ms` : "—"
        );
        set("bench-perplexity", fmt(head.perplexity, null, 2));
        set(
            "bench-coherence",
            head.coherence_pct ? `${fmt(head.coherence_pct, null, 1)}%` : "—"
        );
        set("bench-params", head.slm_params ? fmt(head.slm_params, "params") : "—");
        const tsEl = document.getElementById("bench-timestamp");
        if (tsEl) {
            tsEl.textContent = `Last run: ${report.timestamp || "—"}`;
        }
    }

    function setPlots() {
        // SEC: switched from <object data="..."> to <img src="..."> in
        // April 2026 — the <object> tag triggers Chrome's `frame-ancestors`
        // CSP check (treats embedded SVG as a frame), which clashed with
        // our `frame-ancestors 'none'` policy.  <img> embeds are simpler,
        // honour the `img-src 'self'` directive cleanly, and still scale
        // SVG correctly inside the figcaption layout.
        Object.entries(PLOT_NAMES).forEach(([key, fname]) => {
            const img = document.getElementById(`bench-svg-${key}`);
            if (img) {
                // Cache-bust on every load.
                img.src = `/api/benchmarks/svg/${fname}?_=${Date.now()}`;
            }
        });
    }

    async function load() {
        try {
            const res = await fetch("/api/benchmarks/latest");
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const empty = document.getElementById("bench-empty");
            if (data.status === "not_run") {
                if (empty) empty.hidden = false;
                const tsEl = document.getElementById("bench-timestamp");
                if (tsEl) tsEl.textContent = "No benchmarks yet.";
                return null;
            }
            if (empty) empty.hidden = true;
            setHeadline(data);
            setPlots();
            lastTimestamp = data.timestamp || null;
            return data;
        } catch (err) {
            console.warn("benchmarks load failed:", err);
            return null;
        }
    }

    async function runBench() {
        const btn = document.getElementById("bench-run");
        if (!btn) return;
        btn.disabled = true;
        btn.textContent = "Running…";
        try {
            const res = await fetch("/api/benchmarks/run", { method: "POST" });
            if (res.status === 429) {
                btn.textContent = "Cooldown — wait";
                setTimeout(() => {
                    btn.disabled = false;
                    btn.textContent = "Run benchmarks";
                }, 60_000);
                return;
            }
            if (res.status === 409) {
                btn.textContent = "Already running…";
            }
            if (pollHandle) clearInterval(pollHandle);
            pollHandle = setInterval(async () => {
                const fresh = await load();
                if (fresh && fresh.timestamp && fresh.timestamp !== lastTimestamp) {
                    clearInterval(pollHandle);
                    pollHandle = null;
                    btn.disabled = false;
                    btn.textContent = "Run benchmarks";
                }
            }, 5000);
            // Stop polling after 5 minutes regardless.
            setTimeout(() => {
                if (pollHandle) {
                    clearInterval(pollHandle);
                    pollHandle = null;
                    btn.disabled = false;
                    btn.textContent = "Run benchmarks";
                }
            }, 300_000);
        } catch (err) {
            btn.disabled = false;
            btn.textContent = "Run benchmarks";
            console.warn("bench run failed:", err);
        }
    }

    function init() {
        const btn = document.getElementById("bench-run");
        if (btn) btn.addEventListener("click", runBench);
        // Lazy-load when the user actually opens the tab so we don't
        // hit the endpoint on every page load.
        const checkAndLoad = () => {
            const panel = document.querySelector(
                ".tab-panel[data-tab='benchmarks']"
            );
            if (panel && !panel.hasAttribute("hidden")) {
                load();
            }
        };
        window.addEventListener("hashchange", checkAndLoad);
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", checkAndLoad);
        } else {
            checkAndLoad();
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
