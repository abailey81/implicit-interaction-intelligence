/**
 * Edge-deployment proof dashboard wiring.
 *
 * Fetches ``/api/edge/profile`` when the Edge tab becomes active and
 * populates the four stat tiles, the size-vs-precision bar chart, the
 * deployability table, and the footer timestamp line.
 *
 * Graceful degradation: if the endpoint returns 404 the tiles show a
 * "run the profiler" prompt and the bar chart / table stay empty.
 *
 * No framework, no build step.
 */

(() => {
    "use strict";

    // Deployment budgets used in the table — match the ones in
    // i3/edge/profiler.py so the checkmarks line up with deployable_to.
    const BUDGETS = [
        { label: "Mid-range phone",  budget_mb: 300, sub: "(4 GB RAM app budget: 300 MB)" },
        { label: "Budget phone",     budget_mb: 100, sub: "(2 GB RAM app budget: 100 MB)" },
        { label: "Wearable",         budget_mb: 50,  sub: "(smartwatch, <50 MB budget)" },
    ];

    let lastFetchAt = 0;
    let cachedData = null;

    function qs(id) { return document.getElementById(id); }

    function fmtMB(value) {
        if (value === null || value === undefined) return "—";
        if (value < 1) return `${value.toFixed(2)} MB`;
        return `${value.toFixed(1)} MB`;
    }

    function fmtMs(value) {
        if (value === null || value === undefined) return "—";
        if (value >= 100) return `${value.toFixed(0)} ms`;
        if (value >= 10)  return `${value.toFixed(1)} ms`;
        return `${value.toFixed(2)} ms`;
    }

    function setText(id, text, title) {
        const el = qs(id);
        if (!el) return;
        el.textContent = text;
        if (title) el.setAttribute("title", title);
    }

    function renderEmpty(reason) {
        setText("edge-slm-int8", "—");
        setText("edge-tcn-p50",  "—");
        setText("edge-slm-p50",  "—");
        setText("edge-memory",   "—");
        const bars = qs("edge-bars");
        if (bars) bars.innerHTML = "";
        const deploy = qs("edge-deploy");
        if (deploy) deploy.innerHTML = "";
        const footer = qs("edge-footer");
        if (footer) {
            footer.textContent = reason ||
                "Run `python scripts/measure_edge.py` to see measurements.";
        }
    }

    function renderBars(data) {
        const host = qs("edge-bars");
        if (!host) return;
        const rows = [
            { label: "fp32", value: data.slm_size_fp32_mb },
            { label: "bf16", value: data.slm_size_bf16_mb },
            { label: "int8", value: data.slm_size_int8_mb },
            { label: "onnx", value: data.onnx_size_mb },
        ];
        const values = rows.map((r) => r.value).filter((v) => v !== null && v !== undefined);
        const max = values.length ? Math.max(...values) : 1;

        host.innerHTML = "";
        rows.forEach((r) => {
            const row = document.createElement("div");
            row.className = "edge-bar-row";

            const lbl = document.createElement("div");
            lbl.className = "edge-bar-label";
            lbl.textContent = r.label;

            const track = document.createElement("div");
            track.className = "edge-bar-track";
            const fill = document.createElement("div");
            fill.className = "edge-bar-fill";
            if (r.value === null || r.value === undefined) {
                fill.classList.add("is-missing");
                fill.style.width = "100%";
                fill.title = "export not available";
            } else {
                fill.style.width = `${Math.max(2, (r.value / max) * 100)}%`;
            }
            track.appendChild(fill);

            const val = document.createElement("div");
            val.className = "edge-bar-value";
            val.textContent = r.value === null || r.value === undefined
                ? "n/a"
                : fmtMB(r.value);

            row.appendChild(lbl);
            row.appendChild(track);
            row.appendChild(val);
            host.appendChild(row);
        });
    }

    function renderDeploy(data) {
        const host = qs("edge-deploy");
        if (!host) return;
        const int8 = data.slm_size_int8_mb;
        host.innerHTML = "";
        BUDGETS.forEach((b) => {
            const row = document.createElement("div");
            row.className = "edge-deploy-row";

            const left = document.createElement("div");
            const name = document.createElement("span");
            name.textContent = b.label;
            const sub = document.createElement("span");
            sub.className = "budget";
            sub.textContent = b.sub;
            left.appendChild(name);
            left.appendChild(sub);

            const mark = document.createElement("div");
            const fits = typeof int8 === "number" && int8 <= b.budget_mb;
            mark.className = `edge-deploy-mark ${fits ? "ok" : "no"}`;
            mark.textContent = fits ? "✓" : "✗";
            mark.setAttribute("aria-label", fits ? "fits budget" : "exceeds budget");

            row.appendChild(left);
            row.appendChild(mark);
            host.appendChild(row);
        });
    }

    function render(data) {
        setText("edge-slm-int8", fmtMB(data.slm_size_int8_mb));
        setText("edge-tcn-p50",  fmtMs(data.latency_ms_encoder_p50));
        setText("edge-slm-p50",  fmtMs(data.latency_ms_p50));
        setText("edge-memory",   fmtMB(data.memory_peak_mb));

        renderBars(data);
        renderDeploy(data);

        const footer = qs("edge-footer");
        if (footer) {
            footer.innerHTML =
                `Measured <strong>${data.timestamp || "—"}</strong> on ` +
                `<code>${data.device || "cpu"}</code>. ` +
                `Re-run <code>python scripts/measure_edge.py</code> to refresh.`;
        }
    }

    async function fetchAndRender() {
        // Avoid hammering the endpoint on rapid tab flips — 30 s soft TTL.
        const now = Date.now();
        if (cachedData && (now - lastFetchAt) < 30_000) {
            render(cachedData);
            return;
        }

        try {
            const res = await fetch("/api/edge/profile", {
                credentials: "same-origin",
                cache: "no-store",
            });
            if (res.status === 404) {
                renderEmpty("No edge profile yet — run `python scripts/measure_edge.py`.");
                return;
            }
            if (!res.ok) {
                renderEmpty(`Edge profile endpoint returned ${res.status}.`);
                return;
            }
            const data = await res.json();
            cachedData = data;
            lastFetchAt = now;
            render(data);
        } catch (err) {
            renderEmpty("Failed to load edge profile (network error).");
        }
    }

    function onTabChange() {
        const panel = qs("tab-edge");
        if (panel && !panel.hasAttribute("hidden")) {
            fetchAndRender();
        }
    }

    // Fetch on initial load if the hash already points at #edge.
    document.addEventListener("DOMContentLoaded", () => {
        if ((location.hash || "").toLowerCase() === "#edge") {
            fetchAndRender();
        }
    });

    // Re-fetch whenever the hash changes to #edge — tab_router.js
    // handles panel show/hide on hashchange, so we piggy-back on that.
    window.addEventListener("hashchange", () => {
        if ((location.hash || "").toLowerCase() === "#edge") {
            fetchAndRender();
        }
    });

    // Also wire a click listener on the Edge nav link so the panel
    // gets populated without needing a hash change (covers the case
    // where the user is already on #edge and clicks the brand link).
    document.addEventListener("click", (ev) => {
        const link = ev.target.closest(".nav-link");
        if (!link) return;
        if (link.dataset.tab === "edge") {
            // Let tab_router.js toggle visibility first.
            setTimeout(onTabChange, 0);
        }
    });
})();
