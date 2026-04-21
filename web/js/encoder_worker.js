/**
 * encoder_worker.js
 * ─────────────────
 * Web Worker that owns the ONNX Runtime Web session for the TCN encoder
 * and runs inference off the main thread.  The main thread talks to the
 * worker via `postMessage`.
 *
 * Wire format (incoming):
 *   { type: 'init',    modelUrl: string, executionProviders: string[] }
 *   { type: 'encode',  feature_vector: Float32Array(32) }
 *   { type: 'close' }
 *
 * Wire format (outgoing):
 *   { type: 'ready',     backend: string }
 *   { type: 'embedding', embedding: Float32Array(64), latency_ms: number }
 *   { type: 'error',     message: string }
 *   { type: 'closed' }
 *
 * The worker is spawned with `{ type: 'module' }` from
 * `browser_inference.js` so it can `import` the ort loader directly.
 * We never hold references to DOM objects (there is no DOM here) — the
 * worker is pure compute.
 */

// Workers started with `{ type: 'module' }` can use ES module imports.
// The loader itself caches ort-web on `self.__I3_ORT_CACHE__` so two
// workers running side by side would each have their own runtime, which
// is what we want (workers are isolated heaps).
import { loadOrtSession } from './ort_loader.js';

/** @type {object|null} the active InferenceSession */
let _session = null;

/** @type {string|null} the currently-configured backend name */
let _backend = null;

/** @type {string|null} the encoder's expected input tensor name */
let _inputName = null;

/** @type {string|null} the encoder's expected output tensor name */
let _outputName = null;

/** Shape of a single feature vector sample.  [batch, time, features]. */
const INPUT_SHAPE = [1, 1, 32];

/** Expected embedding dimensionality — matches i3/encoder/tcn.py. */
const EMBEDDING_DIM = 64;

/**
 * Lookup the first input/output name from the session.  ONNX graphs
 * produced by `i3/encoder/onnx_export.py` always expose exactly one of
 * each, but we resolve dynamically to avoid brittleness.
 * @param {object} session
 */
function _resolveIO(session) {
    const inputs = session.inputNames;
    const outputs = session.outputNames;
    if (!Array.isArray(inputs) || inputs.length === 0) {
        throw new Error('ONNX session has no declared inputs');
    }
    if (!Array.isArray(outputs) || outputs.length === 0) {
        throw new Error('ONNX session has no declared outputs');
    }
    _inputName = inputs[0];
    _outputName = outputs[0];
}

/**
 * Run one inference step.
 * @param {Float32Array} fv  input feature vector (length 32)
 * @returns {Promise<{embedding: Float32Array, latency_ms: number}>}
 */
async function _encode(fv) {
    if (!_session) {
        throw new Error('encoder_worker: session not initialised');
    }
    if (!(fv instanceof Float32Array) || fv.length !== 32) {
        throw new Error(
            `encoder_worker: feature_vector must be Float32Array(32), got length=${fv && fv.length}`,
        );
    }

    // Lazy import the runtime namespace (already cached inside the
    // ort_loader module) so we can construct tensors.
    const ort = self.__I3_ORT__;
    if (!ort || !ort.Tensor) {
        throw new Error('encoder_worker: ort namespace unavailable');
    }

    const inputTensor = new ort.Tensor('float32', fv, INPUT_SHAPE);

    const t0 = performance.now();
    const feeds = {};
    feeds[_inputName] = inputTensor;
    const results = await _session.run(feeds);
    const t1 = performance.now();

    const outTensor = results[_outputName];
    if (!outTensor || !outTensor.data) {
        throw new Error('encoder_worker: output tensor missing from session.run result');
    }

    // Copy into a fresh Float32Array so the buffer is transferable
    // back to the main thread without retaining an ort-internal ref.
    const srcData = outTensor.data;
    const embedding = new Float32Array(EMBEDDING_DIM);
    const copyLen = Math.min(EMBEDDING_DIM, srcData.length);
    for (let i = 0; i < copyLen; i += 1) {
        embedding[i] = srcData[i];
    }

    return {
        embedding,
        latency_ms: t1 - t0,
    };
}

self.onmessage = async (event) => {
    const msg = event.data || {};
    try {
        if (msg.type === 'init') {
            if (_session) {
                // Already initialised — treat as a no-op success.
                self.postMessage({ type: 'ready', backend: _backend });
                return;
            }
            const providers =
                Array.isArray(msg.executionProviders) && msg.executionProviders.length > 0
                    ? msg.executionProviders
                    : ['wasm'];
            _backend = providers[0];
            _session = await loadOrtSession(msg.modelUrl, providers);
            _resolveIO(_session);
            self.postMessage({ type: 'ready', backend: _backend });
            return;
        }

        if (msg.type === 'encode') {
            const { embedding, latency_ms } = await _encode(msg.feature_vector);
            // Transfer the Float32Array's underlying buffer to avoid a
            // structured-clone copy on the way back.
            self.postMessage(
                { type: 'embedding', embedding, latency_ms },
                [embedding.buffer],
            );
            return;
        }

        if (msg.type === 'close') {
            try {
                if (_session && typeof _session.release === 'function') {
                    await _session.release();
                }
            } catch (_err) {
                // Best-effort cleanup; do not fail the worker on close.
            }
            _session = null;
            _inputName = null;
            _outputName = null;
            self.postMessage({ type: 'closed' });
            return;
        }

        throw new Error(`encoder_worker: unknown message type '${msg.type}'`);
    } catch (err) {
        const message = err && err.message ? err.message : String(err);
        self.postMessage({ type: 'error', message });
    }
};
