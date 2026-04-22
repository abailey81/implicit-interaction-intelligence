/**
 * guided_tour.js — cinematic 4-phase guided demo.
 *
 * Phases (60s each):
 *   1. COLD START      — no prior preferences, 60 WPM typing.
 *   2. ENERGETIC       — fast 100 WPM burst, high emotional_tone.
 *   3. FATIGUE         — slow 30 WPM, high cognitive_load.
 *   4. ACCESSIBILITY   — 15 WPM, accessibility gauge maxed.
 *
 * Safety: the tour runs against a dedicated `demo_user` id so the
 * real user's state is never touched. Exit cleanly on Escape.
 */

const PHASES = [
  {
    name: "Cold Start",
    caption: "No prior preferences. The system is observing.",
    wpm: 60,
    lines: [
      "hello, can you help me with a quick question",
      "what's the fastest way to set this up",
      "thanks, that's useful",
    ],
    adaptation: { cognitive_load: 0.3, "style.formality": 0.5, "style.verbosity": 0.5, accessibility: 0.2 },
  },
  {
    name: "Energetic",
    caption: "Fast typing, high energy — system dials up emotionality and directness.",
    wpm: 100,
    lines: [
      "ok now show me something cool!!",
      "wow that's great keep going",
      "can you do something even more impressive",
    ],
    adaptation: { cognitive_load: 0.25, "style.formality": 0.25, "style.verbosity": 0.45, emotional_tone: 0.8, "style.directness": 0.75 },
  },
  {
    name: "Fatigue",
    caption: "Slower inter-key intervals — cognitive load rises, verbosity drops.",
    wpm: 30,
    lines: [
      "im kind of tired",
      "just give me the short answer",
      "that's fine, simple is better",
    ],
    adaptation: { cognitive_load: 0.85, "style.verbosity": 0.2, emotional_tone: 0.35, "style.directness": 0.5 },
  },
  {
    name: "Accessibility",
    caption: "Very slow typing + long pauses — accessibility mode activates.",
    wpm: 15,
    lines: [
      "please  can  you  speak  more  slowly",
      "use  clearer  words",
    ],
    adaptation: { cognitive_load: 0.7, "style.verbosity": 0.6, accessibility: 0.95, "style.formality": 0.55 },
  },
];

export function createTour({ bus, ws, setActive }) {
  let running = false;
  let cancel = null;
  const overlay = document.getElementById("tour-overlay");
  const phaseEl = document.getElementById("tour-phase");
  const capEl   = document.getElementById("tour-caption");

  function showOverlay(phase, cap) {
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    phaseEl.textContent = phase;
    capEl.textContent = cap;
  }
  function hideOverlay() {
    overlay.hidden = true;
    overlay.setAttribute("aria-hidden", "true");
  }

  function sleep(ms, signal) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(resolve, ms);
      if (signal) signal.addEventListener("abort", () => { clearTimeout(t); reject(new Error("aborted")); }, { once: true });
    });
  }

  async function typeLine(text, wpm, signal) {
    const input = document.getElementById("chat-input");
    const charsPerSec = (wpm * 5) / 60;
    const intervalMs = Math.max(40, Math.floor(1000 / charsPerSec));
    input.value = "";
    for (const ch of text) {
      if (signal && signal.aborted) throw new Error("aborted");
      input.value += ch;
      input.dispatchEvent(new KeyboardEvent("keydown", { key: ch, bubbles: true }));
      await sleep(intervalMs, signal);
    }
    document.getElementById("chat-form").dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
  }

  async function run() {
    if (running) return;
    running = true;
    setActive(true);
    const controller = new AbortController();
    cancel = () => controller.abort();
    const signal = controller.signal;

    // Isolate: swap user id for the tour.
    try { ws && ws.send({ type: "set_user", user_id: "demo_user" }); } catch { /* ignore */ }

    try {
      for (const phase of PHASES) {
        if (signal.aborted) break;
        showOverlay(phase.name, phase.caption);
        await sleep(1400, signal);
        hideOverlay();

        // Synthesise an adaptation frame so the gauges react even
        // when the backend is offline.
        bus.dispatchEvent(new CustomEvent("adaptation", {
          detail: { type: "adaptation", data: phase.adaptation, uncertainty: { cognitive_load: 0.1 } },
        }));

        for (const line of phase.lines) {
          if (signal.aborted) break;
          await typeLine(line, phase.wpm, signal);
          await sleep(1800, signal);
        }
        await sleep(900, signal);
      }
    } catch (e) {
      if (e && e.message !== "aborted") console.warn("[guided_tour] error", e);
    } finally {
      hideOverlay();
      running = false;
      cancel = null;
      setActive(false);
    }
  }

  function stop() {
    if (cancel) cancel();
    hideOverlay();
  }

  function toggle() {
    if (running) stop(); else run();
  }

  // Esc cancels.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && running) { e.preventDefault(); stop(); }
  });

  return { toggle, run, stop, isRunning: () => running };
}
