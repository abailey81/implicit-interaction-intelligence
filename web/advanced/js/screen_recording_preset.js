/**
 * screen_recording_preset.js — toggle a CSS class that:
 *   - Hides dev/ghost controls.
 *   - Bumps base font size by ~15 % (via CSS custom property).
 *   - Slows transitions by ~30 %.
 *   - Uses a custom cursor that survives screen capture.
 *   - Freezes grid track sizes to prevent layout jitter on capture.
 *
 * Most of the above is driven by the `.record-mode` class in
 * command_center.css. This module just flips the class and tracks
 * the aria-pressed state.
 */

export function installRecordPreset(btn) {
  if (!btn) return;
  const KEY = "i3.advanced.record_mode";
  const restore = sessionStorage.getItem(KEY) === "1";
  if (restore) enable();

  btn.addEventListener("click", () => {
    if (document.body.classList.contains("record-mode")) disable();
    else enable();
  });

  function enable() {
    document.body.classList.add("record-mode");
    btn.setAttribute("aria-pressed", "true");
    btn.textContent = "Record On";
    sessionStorage.setItem(KEY, "1");
  }
  function disable() {
    document.body.classList.remove("record-mode");
    btn.setAttribute("aria-pressed", "false");
    btn.textContent = "Record Mode";
    sessionStorage.setItem(KEY, "0");
  }
}
