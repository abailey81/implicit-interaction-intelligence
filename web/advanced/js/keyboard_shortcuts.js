/**
 * keyboard_shortcuts.js — global keyboard bindings.
 *
 *   Alt+T  Guided tour
 *   Alt+R  Reset demo user
 *   Alt+W  What-if panel (fires a bus event — panel subscribes)
 *   Alt+A  A11y audit (writes to console)
 *   Alt+M  Toggle reduced-motion
 *   Alt+S  Screen-recording preset
 *   ?      Show shortcut help overlay
 *   Esc    Cancel / close
 *
 * Everything is additive and respects focus — when the active
 * element is an <input> or <textarea>, modifier-less shortcuts
 * (like "?") are ignored.
 */

import { runContrastAudit } from "./a11y.js";

export function installShortcuts({ bus, tour }) {
  document.addEventListener("keydown", (e) => {
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    const typing = tag === "INPUT" || tag === "TEXTAREA" || (document.activeElement && document.activeElement.isContentEditable);

    if (e.altKey && !e.ctrlKey && !e.metaKey) {
      switch (e.key.toLowerCase()) {
        case "t": e.preventDefault(); tour.toggle(); return;
        case "r": e.preventDefault(); resetDemo(); return;
        case "w": e.preventDefault(); bus.dispatchEvent(new CustomEvent("shortcut:whatif")); return;
        case "a": e.preventDefault(); runContrastAudit(); return;
        case "m": e.preventDefault(); toggleMotion(); return;
        case "s": e.preventDefault(); document.getElementById("btn-record").click(); return;
      }
    }

    if (!typing && e.key === "?") {
      e.preventDefault();
      const dlg = document.getElementById("shortcuts-dialog");
      if (dlg && typeof dlg.showModal === "function") dlg.showModal();
    }

    if (e.key === "Escape") {
      const dlg = document.getElementById("shortcuts-dialog");
      if (dlg && dlg.open) dlg.close();
    }
  });
}

function resetDemo() {
  fetch("/api/admin/reset", { method: "POST" }).catch(() => { /* admin may be disabled */ });
  const log = document.getElementById("chat-log");
  if (log) log.innerHTML = "";
}

function toggleMotion() {
  const body = document.body;
  const off = body.getAttribute("data-motion") === "off";
  body.setAttribute("data-motion", off ? "on" : "off");
}
