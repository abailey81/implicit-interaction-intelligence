/**
 * chat_panel.js — chat UI with live composition-intensity indicator.
 *
 * Composition intensity is the rolling mean inter-key interval,
 * inverted and clipped into [0, 1]. When typing is fast, the bar
 * pulses at full brightness; as keystrokes slow, the fill shrinks.
 * A decay timer gradually fades the bar down to zero when idle.
 */

import { announce } from "./a11y.js";
import { showEmpty } from "./loading_states.js";

export function mountChat({ root, bus, ws, getState }) {
  const log = root.querySelector("#chat-log");
  const form = root.querySelector("#chat-form");
  const input = root.querySelector("#chat-input");
  const fill = root.querySelector("#intensity-fill");
  const meter = root.querySelector("#intensity-meter");

  showEmpty(log, "Send a message to begin.");

  /** ---- composition intensity ---- */
  const keyTimes = [];
  const WINDOW = 8;
  const MIN_IKI = 40;      // keypress faster than this is noise
  const MAX_IKI = 800;     // slower than this counts as idle
  let intensity = 0;

  function updateBar() {
    // intensity in [0,1]; 1 means max speed.
    const pct = Math.round(Math.min(1, Math.max(0, intensity)) * 100);
    fill.style.right = (100 - pct) + "%";
    fill.style.opacity = (0.35 + 0.65 * intensity).toFixed(2);
    meter.setAttribute("aria-valuenow", String(pct));
  }

  function onKey(e) {
    if (e.key.length !== 1 && e.key !== "Backspace") return;
    const now = performance.now();
    if (keyTimes.length > 0) {
      const iki = now - keyTimes[keyTimes.length - 1];
      if (iki > MIN_IKI) {
        const norm = 1 - Math.min(1, Math.max(0, (iki - MIN_IKI) / (MAX_IKI - MIN_IKI)));
        intensity = intensity * 0.6 + norm * 0.4;
      }
    }
    keyTimes.push(now);
    if (keyTimes.length > WINDOW) keyTimes.shift();
    updateBar();
  }

  // Idle decay.
  setInterval(() => {
    intensity *= 0.85;
    if (intensity < 0.01) intensity = 0;
    updateBar();
  }, 250);

  input.addEventListener("keydown", onKey);

  /** ---- send ---- */
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    appendMessage({ role: "user", text });
    const sent = ws && ws.send({ type: "user_input", user_id: getState().userId, text, client_ts: Date.now() });
    if (!sent) {
      appendMessage({ role: "meta", text: "(offline — queued locally)" });
    }
    input.value = "";
    intensity = 0; updateBar();
  });

  /** ---- receive ---- */
  bus.addEventListener("chat:incoming", (ev) => {
    const msg = ev.detail || {};
    if (msg.role && msg.text) appendMessage({ role: msg.role, text: msg.text });
  });

  /** ---- render ---- */
  function appendMessage({ role, text }) {
    // Strip any pre-existing empty state.
    const emp = log.querySelector(".state-empty");
    if (emp) emp.remove();

    const el = document.createElement("div");
    const cls = role === "user" ? "chat-msg--user" : (role === "system" || role === "sys" || role === "assistant") ? "chat-msg--sys" : "chat-msg--meta";
    el.className = "chat-msg " + cls;
    el.textContent = text; // textContent — XSS-safe.
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    if (role !== "user") announce(text.slice(0, 140));
  }

  return { appendMessage };
}
