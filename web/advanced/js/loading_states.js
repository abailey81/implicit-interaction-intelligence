/**
 * loading_states.js — shared skeleton / error / empty helpers.
 *
 * Every panel uses these so the UI has a single visual vocabulary
 * for "loading", "oops", and "nothing yet".
 */

export function showSkeleton(element, rows = 3) {
  if (!element) return;
  element.innerHTML = "";
  for (let i = 0; i < rows; i++) {
    const bar = document.createElement("div");
    bar.className = "skeleton";
    bar.style.height = "12px";
    bar.style.margin = "8px 0";
    bar.style.width = (60 + Math.floor(Math.random() * 35)) + "%";
    element.appendChild(bar);
  }
}

export function showError(element, msg, retry) {
  if (!element) return;
  element.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "state-error";
  wrap.setAttribute("role", "alert");
  wrap.textContent = msg || "Something went wrong.";
  element.appendChild(wrap);
  if (typeof retry === "function") {
    const btn = document.createElement("button");
    btn.className = "retry-btn";
    btn.type = "button";
    btn.textContent = "Retry";
    btn.addEventListener("click", () => retry());
    wrap.appendChild(document.createElement("br"));
    wrap.appendChild(btn);
  }
}

export function showEmpty(element, msg) {
  if (!element) return;
  element.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "state-empty";
  wrap.textContent = msg || "No data yet.";
  element.appendChild(wrap);
}
