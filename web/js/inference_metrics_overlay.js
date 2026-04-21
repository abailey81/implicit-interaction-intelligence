/**
 * inference_metrics_overlay.js
 * ────────────────────────────
 * Floating overlay that reports in-browser inference health:
 *   * P50 / P95 latency over the last N samples.
 *   * Active backend (WebGPU / WASM-SIMD-threaded / WASM).
 *   * A "Verified local inference" pill that lights up green when
 *     inference is confirmed to be running on-device.
 *
 * Exports
 *   * `renderMetricsOverlay(container)` — mount the overlay.
 *   * `recordSample(latency_ms)` — feed a latency sample.  Also
 *     exposes `.p50()`, `.p95()`, `.count()`, `.reset()` helpers.
 *   * `setBackendLabel(backend, adapter)` — swap the backend text.
 *   * `setActive(isActive)` — toggles the green verified pill.
 *
 * The overlay is deliberately decoupled from the worker so tests and
 * other modules can pipe samples in without a Worker dependency.
 */

const MAX_SAMPLES = 256;
const samples = [];
let _backendLabel = 'idle';
let _adapterDescription = '';
let _active = false;

// DOM handles — populated by renderMetricsOverlay().
let _root = null;
let _p50Node = null;
let _p95Node = null;
let _countNode = null;
let _backendNode = null;
let _pillNode = null;

// ─────────────────────────────────────────────────────────────────────
// Sample statistics
// ─────────────────────────────────────────────────────────────────────
function _quantile(arr, q) {
    if (arr.length === 0) return 0;
    const sorted = arr.slice().sort((a, b) => a - b);
    const idx = Math.min(
        sorted.length - 1,
        Math.max(0, Math.floor(q * (sorted.length - 1))),
    );
    return sorted[idx];
}

/**
 * Feed a latency sample into the rolling window.  Also updates the
 * overlay DOM if it is mounted.
 * @param {number} latency_ms
 */
export function recordSample(latency_ms) {
    if (typeof latency_ms !== 'number' || !isFinite(latency_ms)) return;
    samples.push(latency_ms);
    if (samples.length > MAX_SAMPLES) samples.shift();
    _refreshDom();
}

recordSample.p50 = () => _quantile(samples, 0.5);
recordSample.p95 = () => _quantile(samples, 0.95);
recordSample.count = () => samples.length;
recordSample.reset = () => {
    samples.length = 0;
    _refreshDom();
};

// ─────────────────────────────────────────────────────────────────────
// DOM state
// ─────────────────────────────────────────────────────────────────────
function _refreshDom() {
    if (!_root) return;
    if (_p50Node) _p50Node.textContent = `${recordSample.p50().toFixed(1)} ms`;
    if (_p95Node) _p95Node.textContent = `${recordSample.p95().toFixed(1)} ms`;
    if (_countNode) _countNode.textContent = String(recordSample.count());
    if (_backendNode) {
        const adapterSuffix = _adapterDescription ? ` (${_adapterDescription})` : '';
        _backendNode.textContent = `${_backendLabel}${adapterSuffix}`;
    }
    if (_pillNode) {
        _pillNode.classList.toggle('is-active', _active);
        _pillNode.setAttribute('aria-hidden', _active ? 'false' : 'true');
    }
}

/**
 * Swap the backend label shown in the overlay.
 * @param {string} backend  — e.g. 'webgpu', 'wasm', 'wasm-simd-threaded'.
 * @param {object|null} [adapter] — optional adapter info (WebGPU only).
 */
export function setBackendLabel(backend, adapter) {
    _backendLabel = String(backend || 'idle');
    if (adapter && typeof adapter === 'object') {
        const vendor = adapter.vendor || '';
        const arch = adapter.architecture || '';
        _adapterDescription = [vendor, arch].filter(Boolean).join(' · ');
    } else {
        _adapterDescription = '';
    }
    _refreshDom();
}

/**
 * Flip the "Verified local inference" pill on or off.
 * @param {boolean} isActive
 */
export function setActive(isActive) {
    _active = Boolean(isActive);
    _refreshDom();
}

/**
 * Mount the overlay DOM inside `container`.  Idempotent.
 * @param {HTMLElement} container
 */
export function renderMetricsOverlay(container) {
    if (!container) return;

    // Remove any prior overlay in this container.
    const prior = container.querySelector('[data-i3-ref="metrics-overlay"]');
    if (prior && prior.parentNode) prior.parentNode.removeChild(prior);

    const root = document.createElement('div');
    root.className = 'i3-metrics-overlay';
    root.setAttribute('data-i3-ref', 'metrics-overlay');
    root.setAttribute('role', 'status');
    root.setAttribute('aria-live', 'polite');

    const header = document.createElement('div');
    header.className = 'i3-metrics-header';

    const title = document.createElement('span');
    title.className = 'i3-metrics-title';
    title.textContent = 'In-browser inference';
    header.appendChild(title);

    const pill = document.createElement('span');
    pill.className = 'i3-metrics-pill';
    pill.textContent = 'Verified local inference';
    pill.setAttribute('aria-hidden', 'true');
    header.appendChild(pill);

    const grid = document.createElement('div');
    grid.className = 'i3-metrics-grid';

    const cell = (labelText, valueClass) => {
        const c = document.createElement('div');
        c.className = 'i3-metrics-cell';
        const l = document.createElement('div');
        l.className = 'i3-metrics-label';
        l.textContent = labelText;
        const v = document.createElement('div');
        v.className = `i3-metrics-value ${valueClass}`;
        v.textContent = '—';
        c.appendChild(l);
        c.appendChild(v);
        return { cell: c, value: v };
    };

    const p50 = cell('P50', 'i3-metrics-p50');
    const p95 = cell('P95', 'i3-metrics-p95');
    const count = cell('Samples', 'i3-metrics-count');
    const backend = cell('Backend', 'i3-metrics-backend');

    grid.appendChild(p50.cell);
    grid.appendChild(p95.cell);
    grid.appendChild(count.cell);
    grid.appendChild(backend.cell);

    root.appendChild(header);
    root.appendChild(grid);
    container.appendChild(root);

    _root = root;
    _p50Node = p50.value;
    _p95Node = p95.value;
    _countNode = count.value;
    _backendNode = backend.value;
    _pillNode = pill;

    _refreshDom();
}
