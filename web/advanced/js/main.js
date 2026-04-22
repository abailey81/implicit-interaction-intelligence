/**
 * I3 Command Center — main.js
 * ---------------------------------------------------------------
 * Entry module. Orchestrates panels, holds app state, wires the
 * WebSocket, dispatches DOM events, and boots the a11y subsystem.
 *
 * This is vanilla ES modules. No build step. No npm. No TypeScript.
 * Third-party libs (Three.js, Chart.js) are loaded from CDN with
 * SRI attributes in index.html and attached to window.* — we read
 * them lazily with feature detection so the page still works if a
 * CDN is unreachable.
 */

import { createWSBridge } from "./ws_bridge.js";
import { mountChat } from "./chat_panel.js";
import { mountEmbedding3D } from "./embedding_3d.js";
import { mountGauges } from "./radial_gauges.js";
import { mountAttention } from "./attention_heatmap.js";
import { mountRouter } from "./router_dashboard.js";
import { mountInterp } from "./interpretability_strip.js";
import { installShortcuts } from "./keyboard_shortcuts.js";
import { installRecordPreset } from "./screen_recording_preset.js";
import { installA11y, announce } from "./a11y.js";
import { createTour } from "./guided_tour.js";

/** Immutable-ish app state. Panels subscribe via the "state" event. */
const state = {
  userId: "demo_user",
  adaptation: null,
  uncertainty: null,
  embeddingBuffer: [],   // rolling window of last 200 embeddings
  routerHistory: [],     // [{t, latency_ms, arm, cost_gbp}]
  tourActive: false,
  connState: "connecting",
};

const bus = new EventTarget();

/** Replace state and emit. */
function setState(patch) {
  Object.assign(state, patch);
  bus.dispatchEvent(new CustomEvent("state", { detail: state }));
}

function pushEmbedding(vec, personaLabel = "current") {
  const buf = state.embeddingBuffer;
  buf.push({ vec, label: personaLabel, t: Date.now() });
  if (buf.length > 200) buf.shift();
  bus.dispatchEvent(new CustomEvent("embedding", { detail: { buf, latest: buf[buf.length - 1] } }));
}

function pushRouter(entry) {
  state.routerHistory.push(entry);
  if (state.routerHistory.length > 120) state.routerHistory.shift();
  bus.dispatchEvent(new CustomEvent("router", { detail: entry }));
}

/** Live clock. */
function startClock() {
  const el = document.getElementById("live-clock");
  const pad = (n) => String(n).padStart(2, "0");
  const tick = () => {
    const d = new Date();
    el.textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    el.setAttribute("datetime", d.toISOString());
  };
  tick();
  setInterval(tick, 1000);
}

/** Connection indicator. */
function wireConnIndicator(ws) {
  const dot = document.getElementById("conn-dot");
  const lbl = document.getElementById("conn-label");
  ws.on("state", (st) => {
    dot.setAttribute("data-state", st);
    lbl.textContent = ({
      connecting: "Connecting…",
      open: "Live",
      closed: "Reconnecting…",
      error: "Offline",
    })[st] || st;
    setState({ connState: st });
    announce(`Connection ${st}`);
  });
}

/** WS frame router. */
function wireFrames(ws) {
  ws.on("frame", (msg) => {
    if (!msg || typeof msg !== "object") return;
    switch (msg.type) {
      case "adaptation":
        setState({ adaptation: msg.data, uncertainty: msg.uncertainty || null });
        bus.dispatchEvent(new CustomEvent("adaptation", { detail: msg }));
        break;
      case "embedding":
        if (Array.isArray(msg.vec)) pushEmbedding(msg.vec, msg.persona || "current");
        break;
      case "router":
        pushRouter({
          t: Date.now(),
          latency_ms: Number(msg.latency_ms) || 0,
          arm: msg.arm || "local_slm",
          cost_gbp: Number(msg.cost_gbp) || 0,
          provider: msg.provider || "local",
        });
        break;
      case "attention":
        bus.dispatchEvent(new CustomEvent("attention", { detail: msg }));
        break;
      case "chat":
        bus.dispatchEvent(new CustomEvent("chat:incoming", { detail: msg }));
        break;
      case "counterfactual":
      case "refusal":
      case "preference":
        bus.dispatchEvent(new CustomEvent("interp", { detail: msg }));
        break;
      default:
        // Unknown frame types are ignored — the server can evolve
        // without breaking the UI.
    }
  });
}

/** Boot. */
function boot() {
  startClock();
  installA11y(bus);

  const ws = createWSBridge({
    url: (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws",
    heartbeatMs: 15000,
    maxQueue: 100,
  });
  wireConnIndicator(ws);
  wireFrames(ws);

  mountChat({ root: document.querySelector(".panel--chat"), bus, ws, getState: () => state });
  mountEmbedding3D({ root: document.getElementById("embed-canvas-host"), bus, getState: () => state });
  mountGauges({ root: document.getElementById("gauges-grid"), bus });
  mountAttention({ root: document.getElementById("attn-host"), bus });
  mountRouter({
    donut: document.getElementById("router-donut"),
    line: document.getElementById("router-latency"),
    costEl: document.getElementById("cost-total"),
    breakdownEl: document.getElementById("router-cost-breakdown"),
    bus,
    getState: () => state,
  });
  mountInterp({
    cf: document.getElementById("cf-body"),
    unc: document.getElementById("unc-pills"),
    refusal: document.getElementById("refusal-badges"),
    ab: document.getElementById("ab-panel"),
    bus,
    getState: () => state,
  });

  const tour = createTour({ bus, ws, setActive: (b) => setState({ tourActive: b }) });
  document.getElementById("btn-tour").addEventListener("click", () => tour.toggle());

  installRecordPreset(document.getElementById("btn-record"));
  installShortcuts({ bus, tour });

  document.getElementById("btn-shortcuts").addEventListener("click", () => {
    const dlg = document.getElementById("shortcuts-dialog");
    if (dlg && typeof dlg.showModal === "function") dlg.showModal();
  });

  ws.connect();
  announce("Command center ready");
}

// Expose a minimal debug handle for devtools (read-only).
Object.defineProperty(window, "__i3", {
  value: Object.freeze({ state, bus, version: "g9-1.0.0" }),
  writable: false,
  configurable: false,
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}
