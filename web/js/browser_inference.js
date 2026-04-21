/**
 * browser_inference.js
 * ────────────────────
 * Main-thread coordinator for the in-browser inference pipeline.
 *
 * Responsibilities
 *   * Detect the best backend (WebGPU vs WASM).
 *   * Spawn the `encoder_worker.js` module worker.
 *   * Own the request/response plumbing (promises per encode call).
 *   * Surface latency samples to the metrics overlay.
 *   * Render the toggle + overlay into `#i3-browser-inference`.
 *
 * Exports
 *   * `initBrowserInference(options)` — idempotent, returns the
 *     singleton `BrowserInferenceController`.
 *   * `encodeFeatureVector(fv)` — convenience wrapper.
 *   * `dispose()` — tears the worker down.
 *   * class `BrowserInferenceController`.
 *
 * This module is pure ES (no build step) and does NOT depend on — and
 * does NOT modify — any of `app.js`, `dashboard.js`, `embedding_viz.js`,
 * `websocket.js`, or `chat.js`.  It is entirely opt-in behind the
 * toggle rendered by `inference_toggle.js`.
 */

import { detectBackend, executionProvidersFor } from './webgpu_probe.js';
import { renderInferenceToggle, getInferenceLocation } from './inference_toggle.js';
import { renderMetricsOverlay, recordSample, setBackendLabel, setActive } from './inference_metrics_overlay.js';

const MODEL_URL = '/api/onnx/encoder_int8.onnx';
const WORKER_URL = '/js/encoder_worker.js';
const CONTAINER_ID = 'i3-browser-inference';

/**
 * Pending inference request record.  We keep one per in-flight
 * `encodeFeatureVector` call so the worker's single message channel
 * can service several overlapping requests.
 * @typedef {{ resolve: Function, reject: Function, t0: number }} PendingRequest
 */

/** @type {BrowserInferenceController|null} */
let _singleton = null;

export class BrowserInferenceController {
    constructor() {
        /** @type {Worker|null} */
        this.worker = null;
        /** @type {string|null} */
        this.backend = null;
        /** @type {object|null} */
        this.adapterInfo = null;
        /** @type {boolean} */
        this.ready = false;
        /** @type {Promise<void>|null} */
        this._initPromise = null;
        /** @type {PendingRequest[]} */
        this._pending = [];
        /** @type {{p50: number, p95: number, count: number}} */
        this.metrics = { p50: 0, p95: 0, count: 0 };
    }

    /**
     * Initialise the worker + load the encoder model.  Idempotent.
     * @returns {Promise<void>}
     */
    async init() {
        if (this._initPromise) return this._initPromise;
        this._initPromise = (async () => {
            const det = await detectBackend();
            this.backend = det.backend;
            this.adapterInfo = det.adapter;
            setBackendLabel(det.backend, det.adapter);

            // Workers spawned as modules can `import` our loader.
            this.worker = new Worker(WORKER_URL, { type: 'module' });
            this.worker.onmessage = (ev) => this._onWorkerMessage(ev);
            this.worker.onerror = (err) => {
                // Reject every pending request so callers unblock.
                this._failAllPending(err && err.message ? err.message : 'worker error');
            };

            // Send the init message and wait for 'ready'.
            await new Promise((resolve, reject) => {
                const onReady = (ev) => {
                    const msg = ev.data || {};
                    if (msg.type === 'ready') {
                        this.worker.removeEventListener('message', onReady);
                        resolve();
                    } else if (msg.type === 'error') {
                        this.worker.removeEventListener('message', onReady);
                        reject(new Error(msg.message || 'worker init failed'));
                    }
                };
                this.worker.addEventListener('message', onReady);
                this.worker.postMessage({
                    type: 'init',
                    modelUrl: MODEL_URL,
                    executionProviders: executionProvidersFor(det.backend),
                });
            });

            this.ready = true;
            setActive(true);
        })().catch((err) => {
            // Wipe the cached promise so a second call can retry.
            this._initPromise = null;
            this.ready = false;
            setActive(false);
            throw err;
        });
        return this._initPromise;
    }

    /**
     * Encode a 32-dim feature vector into a 64-dim embedding.
     * @param {Float32Array|number[]} fv
     * @returns {Promise<{embedding: Float32Array, latency_ms: number}>}
     */
    async encodeFeatureVector(fv) {
        if (!this.ready) await this.init();
        if (!this.worker) throw new Error('browser_inference: worker missing');

        // Normalise input.  Float32Array avoids structured-clone copies
        // and matches the worker's expectation.
        const payload = fv instanceof Float32Array ? fv : Float32Array.from(fv);
        if (payload.length !== 32) {
            throw new RangeError(`feature_vector length must be 32, got ${payload.length}`);
        }

        const t0 = performance.now();
        return new Promise((resolve, reject) => {
            this._pending.push({ resolve, reject, t0 });
            // Transfer the buffer to the worker so we avoid a copy.
            this.worker.postMessage(
                { type: 'encode', feature_vector: payload },
                [payload.buffer],
            );
        });
    }

    /**
     * Tear down the worker.  Idempotent.
     * @returns {Promise<void>}
     */
    async dispose() {
        this._failAllPending('disposed');
        if (this.worker) {
            try {
                this.worker.postMessage({ type: 'close' });
            } catch (_err) { /* ignore */ }
            this.worker.terminate();
            this.worker = null;
        }
        this.ready = false;
        this._initPromise = null;
        setActive(false);
    }

    /**
     * Handle one message from the worker.
     * @param {MessageEvent} ev
     */
    _onWorkerMessage(ev) {
        const msg = ev.data || {};
        if (msg.type === 'embedding') {
            // FIFO: the worker is single-threaded so responses arrive
            // in request order.
            const pending = this._pending.shift();
            if (!pending) return;
            const latency = typeof msg.latency_ms === 'number'
                ? msg.latency_ms
                : performance.now() - pending.t0;
            this._recordLatency(latency);
            pending.resolve({ embedding: msg.embedding, latency_ms: latency });
            return;
        }
        if (msg.type === 'error') {
            const pending = this._pending.shift();
            const err = new Error(msg.message || 'worker error');
            if (pending) pending.reject(err);
            return;
        }
        // `ready` and `closed` are handled inline in init()/dispose();
        // any other message types are currently ignored.
    }

    /**
     * Book-keep a latency sample and forward to the overlay.
     * @param {number} latency_ms
     */
    _recordLatency(latency_ms) {
        recordSample(latency_ms);
        this.metrics = {
            p50: recordSample.p50(),
            p95: recordSample.p95(),
            count: recordSample.count(),
        };
    }

    /**
     * Reject every pending request with the same error.
     * @param {string} message
     */
    _failAllPending(message) {
        const pending = this._pending.splice(0);
        for (const p of pending) {
            p.reject(new Error(message));
        }
    }
}

/**
 * Initialise the singleton controller and mount the UI surface (toggle
 * + overlay) into `#i3-browser-inference`.  Safe to call many times —
 * the second invocation is a no-op.
 *
 * @param {{autoInitWhenLocal?: boolean}} [options]
 * @returns {BrowserInferenceController}
 */
export function initBrowserInference(options) {
    if (_singleton) return _singleton;
    const opts = options || {};
    _singleton = new BrowserInferenceController();

    const container = typeof document !== 'undefined'
        ? document.getElementById(CONTAINER_ID)
        : null;

    if (container) {
        renderInferenceToggle(container, {
            onChange: async (location) => {
                if (location === 'browser') {
                    try {
                        await _singleton.init();
                    } catch (err) {
                        console.error('[i3] browser inference init failed:', err);
                        setActive(false);
                    }
                } else {
                    await _singleton.dispose();
                }
            },
        });
        renderMetricsOverlay(container);
    }

    // Auto-init when the user had previously opted-in (persisted in
    // localStorage).  The default is OFF so privacy-by-default wins.
    const location = getInferenceLocation();
    if (opts.autoInitWhenLocal !== false && location === 'browser') {
        _singleton.init().catch((err) => {
            console.error('[i3] browser inference auto-init failed:', err);
        });
    }

    return _singleton;
}

/**
 * Encode using the singleton.  Auto-initialises on first call.
 * @param {Float32Array|number[]} fv
 */
export async function encodeFeatureVector(fv) {
    if (!_singleton) initBrowserInference();
    return _singleton.encodeFeatureVector(fv);
}

/**
 * Dispose the singleton.  Safe to call if never initialised.
 */
export async function dispose() {
    if (!_singleton) return;
    await _singleton.dispose();
    _singleton = null;
}

// Auto-mount on DOM ready so the toggle appears even if nobody
// explicitly calls `initBrowserInference()`.  This keeps the feature
// fully self-contained — the only touchpoint in `index.html` is the
// `<div id="i3-browser-inference"></div>` anchor.
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => initBrowserInference(), { once: true });
    } else {
        initBrowserInference();
    }
}
