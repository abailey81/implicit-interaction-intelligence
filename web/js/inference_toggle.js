/**
 * inference_toggle.js
 * ───────────────────
 * UI toggle that lets the user pick between server-side inference
 * (default) and in-browser inference.  The preference is persisted in
 * `localStorage.i3InferenceLocation`.
 *
 * Exports
 *   * `renderInferenceToggle(container, options)` — injects the toggle
 *     widget into a DOM node.
 *   * `getInferenceLocation()` — returns `'browser'` or `'server'`.
 *   * `setInferenceLocation(location)` — persists the choice and
 *     dispatches an `i3:inference-location` custom event on `window`.
 *
 * The toggle is the ONLY piece of UI state that controls whether the
 * WebSocket plumbing sends the raw feature vector (server mode) or the
 * already-encoded embedding with `client_encoded: true` (browser mode).
 * The wire-format change is additive: server code does not need to
 * know the flag exists.
 */

const STORAGE_KEY = 'i3InferenceLocation';
const EVENT_NAME = 'i3:inference-location';
const DEFAULT_LOCATION = 'server';

/**
 * @returns {'server'|'browser'}
 */
export function getInferenceLocation() {
    try {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (raw === 'browser' || raw === 'server') return raw;
    } catch (_err) {
        // localStorage can throw in private-browsing contexts.
    }
    return DEFAULT_LOCATION;
}

/**
 * Persist the chosen location and fire a custom event so other modules
 * can react without tight coupling.
 * @param {'server'|'browser'} location
 */
export function setInferenceLocation(location) {
    const safe = location === 'browser' ? 'browser' : 'server';
    try {
        window.localStorage.setItem(STORAGE_KEY, safe);
    } catch (_err) { /* ignore */ }
    try {
        window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: { location: safe } }));
    } catch (_err) { /* ignore */ }
    return safe;
}

/**
 * Render the toggle inside `container`.  Idempotent: calling a second
 * time removes the previous widget and re-renders.
 *
 * @param {HTMLElement} container
 * @param {{ onChange?: (location: string) => void }} [options]
 */
export function renderInferenceToggle(container, options) {
    if (!container) return;
    const opts = options || {};

    // Remove any prior instance so subsequent renders are clean.
    const prior = container.querySelector('[data-i3-ref="inference-toggle"]');
    if (prior && prior.parentNode) prior.parentNode.removeChild(prior);

    const wrap = document.createElement('div');
    wrap.className = 'i3-inference-toggle';
    wrap.setAttribute('data-i3-ref', 'inference-toggle');
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'Inference location');

    const label = document.createElement('label');
    label.className = 'i3-inference-toggle-label';
    label.setAttribute('for', 'i3-inference-toggle-input');
    label.textContent = 'Run inference in browser';

    const subtitle = document.createElement('span');
    subtitle.className = 'i3-inference-toggle-subtitle';
    subtitle.textContent = 'keystrokes stay on-device';

    const input = document.createElement('input');
    input.type = 'checkbox';
    input.id = 'i3-inference-toggle-input';
    input.className = 'i3-inference-toggle-input';
    input.checked = getInferenceLocation() === 'browser';
    input.setAttribute('aria-describedby', 'i3-inference-toggle-subtitle');

    subtitle.id = 'i3-inference-toggle-subtitle';

    const slider = document.createElement('span');
    slider.className = 'i3-inference-toggle-slider';
    slider.setAttribute('aria-hidden', 'true');

    input.addEventListener('change', () => {
        const loc = input.checked ? 'browser' : 'server';
        setInferenceLocation(loc);
        if (typeof opts.onChange === 'function') {
            try {
                opts.onChange(loc);
            } catch (err) {
                console.error('[i3] inference toggle onChange failed:', err);
            }
        }
    });

    // Layout: [label + subtitle] [slider].
    const textBlock = document.createElement('div');
    textBlock.className = 'i3-inference-toggle-text';
    textBlock.appendChild(label);
    textBlock.appendChild(subtitle);

    const sliderWrap = document.createElement('label');
    sliderWrap.className = 'i3-inference-toggle-sliderwrap';
    sliderWrap.setAttribute('for', 'i3-inference-toggle-input');
    sliderWrap.appendChild(input);
    sliderWrap.appendChild(slider);

    wrap.appendChild(textBlock);
    wrap.appendChild(sliderWrap);
    container.appendChild(wrap);
}
