/**
 * router_dashboard.js — router pie + P50/P95 latency line + live cost.
 *
 * Chart.js is loaded from CDN (UMD) with SRI; we read it lazily from
 * window.Chart and fail soft to a hand-drawn canvas if it's missing.
 *
 * Data sources:
 *   - `router` bus events append samples to in-memory buffer.
 *   - Every 2 s we GET `/api/metrics` (if it exists) for ground-truth
 *     cost/latency. Absent endpoint → we stay with the buffer only.
 */

export function mountRouter({ donut, line, costEl, breakdownEl, bus, getState }) {
  const Chart = window.Chart || null;

  /* Buffers. */
  const latencies = []; // ms
  const armCounts = { local_slm: 0, cloud: 0, local_reflect: 0 };
  const costByProvider = Object.create(null);
  let totalCost = 0;

  /* ---- charts ---- */
  let donutChart = null, lineChart = null;
  if (Chart) {
    donutChart = new Chart(donut.getContext("2d"), {
      type: "doughnut",
      data: {
        labels: ["local_slm", "cloud", "local_reflect"],
        datasets: [{
          data: [0, 0, 0],
          backgroundColor: ["#0f3460", "#e94560", "#a0a0b0"],
          borderColor: "#16213e",
          borderWidth: 2,
        }],
      },
      options: {
        responsive: false,
        animation: { duration: 600, easing: "easeOutCubic" },
        plugins: { legend: { labels: { color: "#a0a0b0", font: { size: 10 } } } },
        cutout: "62%",
      },
    });

    lineChart = new Chart(line.getContext("2d"), {
      type: "line",
      data: {
        labels: [],
        datasets: [
          { label: "P50", data: [], borderColor: "#f0f0f0", backgroundColor: "rgba(240,240,240,0.1)", tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
          { label: "P95", data: [], borderColor: "#e94560", backgroundColor: "rgba(233,69,96,0.12)", tension: 0.3, pointRadius: 0, borderWidth: 1.5 },
        ],
      },
      options: {
        responsive: false,
        animation: { duration: 250, easing: "easeOutCubic" },
        scales: {
          x: { ticks: { color: "#a0a0b0", font: { size: 9 } }, grid: { color: "rgba(160,160,176,0.12)" } },
          y: { ticks: { color: "#a0a0b0", font: { size: 9 } }, grid: { color: "rgba(160,160,176,0.12)" }, beginAtZero: true },
        },
        plugins: { legend: { labels: { color: "#a0a0b0", font: { size: 10 } } } },
      },
    });
  } else {
    paintFallback(donut, "router");
    paintFallback(line, "latency");
  }

  /* ---- helpers ---- */
  function pct(arr, p) {
    if (arr.length === 0) return 0;
    const s = [...arr].sort((a, b) => a - b);
    const idx = Math.min(s.length - 1, Math.floor((p / 100) * s.length));
    return s[idx];
  }

  function renderCost() {
    costEl.textContent = totalCost.toFixed(4);
    const bits = Object.entries(costByProvider)
      .map(([p, v]) => `<span class="cost-chip">${p}: £${v.toFixed(4)}</span>`);
    breakdownEl.innerHTML = bits.join(" ");
  }

  function pushLatency(ms) {
    latencies.push(ms);
    if (latencies.length > 60) latencies.shift();
    if (lineChart) {
      const p50 = pct(latencies, 50);
      const p95 = pct(latencies, 95);
      lineChart.data.labels.push("");
      if (lineChart.data.labels.length > 60) {
        lineChart.data.labels.shift();
        lineChart.data.datasets[0].data.shift();
        lineChart.data.datasets[1].data.shift();
      }
      lineChart.data.datasets[0].data.push(p50);
      lineChart.data.datasets[1].data.push(p95);
      lineChart.update("none");
    }
  }

  function pushArm(arm) {
    if (!(arm in armCounts)) armCounts[arm] = 0;
    armCounts[arm] += 1;
    if (donutChart) {
      donutChart.data.labels = Object.keys(armCounts);
      donutChart.data.datasets[0].data = Object.values(armCounts);
      donutChart.update("none");
    }
  }

  /* ---- bus ---- */
  bus.addEventListener("router", (ev) => {
    const e = ev.detail;
    if (!e) return;
    pushLatency(e.latency_ms);
    pushArm(e.arm);
    totalCost += e.cost_gbp || 0;
    costByProvider[e.provider] = (costByProvider[e.provider] || 0) + (e.cost_gbp || 0);
    renderCost();
  });

  /* ---- metrics poll ---- */
  async function pollMetrics() {
    try {
      const res = await fetch("/api/metrics", { method: "GET" });
      if (!res.ok) return;
      const j = await res.json();
      if (j && typeof j.cost_total_gbp === "number") {
        totalCost = j.cost_total_gbp;
        if (j.cost_by_provider && typeof j.cost_by_provider === "object") {
          for (const k of Object.keys(costByProvider)) delete costByProvider[k];
          Object.assign(costByProvider, j.cost_by_provider);
        }
        renderCost();
      }
    } catch { /* endpoint absent — fine */ }
  }
  setInterval(pollMetrics, 2000);
  renderCost();

  return { pushLatency, pushArm };
}

function paintFallback(canvas, kind) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#16213e";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#a0a0b0";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(`${kind} (chart.js unavailable)`, canvas.width / 2, canvas.height / 2);
}
