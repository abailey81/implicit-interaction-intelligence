/**
 * ort_loader.js
 * ─────────────
 * Lazy loader for ONNX Runtime Web (ort-web), pinned to 1.18.0.
 *
 * The runtime is fetched from `unpkg.com` with an SRI `integrity` hash,
 * or — if the operator has vendored the bundle under `/vendor/ort/` —
 * from that local path.  The WASM worker/threads bundle lives at
 * `ort.env.wasm.wasmPaths`; we point this at the same origin as the
 * loaded `.mjs` so the runtime does not have to cross origins for the
 * WASM binary.
 *
 * Exports:
 *   loadOrtSession(modelUrl, preferredExecutionProviders)
 *     Returns a Promise resolving to an `ort.InferenceSession`.
 *
 * Constraints:
 *   * Vanilla ES module — no bundler, no npm.
 *   * One request per page load; the runtime module is cached in
 *     `window.__I3_ORT_CACHE__` so repeat calls reuse the same `ort`.
 *   * SRI hash is hard-pinned; update when bumping the version.
 */

// ─────────────────────────────────────────────────────────────────────
// Configuration — update both the version and the SRI hash together.
// ─────────────────────────────────────────────────────────────────────
const ORT_VERSION = '1.18.0';

// Primary CDN (subresource-integrity pinned).
// NOTE: the SRI hash below is a placeholder that MUST be regenerated
// whenever the pinned version changes.  Use:
//   curl -sL https://unpkg.com/onnxruntime-web@1.18.0/dist/ort.min.mjs \
//     | openssl dgst -sha384 -binary | openssl base64 -A
const ORT_CDN_URL = `https://unpkg.com/onnxruntime-web@${ORT_VERSION}/dist/ort.min.mjs`;
const ORT_CDN_SRI = 'sha384-PLACEHOLDER_REGENERATE_WHEN_BUMPING_VERSION_XXXXXXXXXXXXXXXXXXXXXXXX';

// Local fallback — checked first when present.  Operators can vendor
// the runtime into `web/vendor/ort/` to avoid the third-party hop.
const ORT_LOCAL_URL = '/vendor/ort/ort.min.mjs';

// WASM artefact root.  Must end with a slash.
const ORT_LOCAL_WASM_PATH = '/vendor/ort/';
const ORT_CDN_WASM_PATH = `https://unpkg.com/onnxruntime-web@${ORT_VERSION}/dist/`;

// ─────────────────────────────────────────────────────────────────────
// Module-level cache — so multiple callers share ONE runtime.
// ─────────────────────────────────────────────────────────────────────
let _ortPromise = null;

/**
 * Probe whether the local vendor path is populated.  We issue a HEAD
 * request and treat any 2xx as "present"; anything else falls back to
 * the CDN.
 * @returns {Promise<boolean>}
 */
async function _probeLocalOrt() {
    try {
        const resp = await fetch(ORT_LOCAL_URL, { method: 'HEAD', cache: 'no-store' });
        return resp.ok;
    } catch (_err) {
        return false;
    }
}

/**
 * Dynamically import the ort-web module.  Uses dynamic `import()` so
 * we can pass the URL at runtime — native ES modules do not support
 * `integrity=` on dynamic imports directly, so we layer an extra
 * `fetch` with SRI when coming from the CDN and then `import` from a
 * blob URL.
 *
 * @param {string} url — absolute URL to the ort-web `.mjs`.
 * @param {string|null} sri — optional SRI hash (`sha384-...`).
 * @returns {Promise<object>} the ort-web module namespace.
 */
async function _importOrt(url, sri) {
    if (!sri) {
        // Local vendored path — no SRI needed because the bytes are
        // served from our own origin under our own CSP.
        return import(url);
    }

    // CDN path: fetch with SRI, then import the blob.
    //
    // We cannot pass `integrity=` to dynamic `import()` yet (the
    // proposal is still stage-3), so we verify via `fetch` instead.
    // `fetch` honours the `integrity` option per the WHATWG spec and
    // will reject the promise with a TypeError if the hash mismatches.
    const resp = await fetch(url, {
        integrity: sri,
        cache: 'force-cache',
        credentials: 'omit',
        mode: 'cors',
    });
    if (!resp.ok) {
        throw new Error(`ort-web fetch failed: HTTP ${resp.status}`);
    }
    const source = await resp.text();
    const blob = new Blob([source], { type: 'text/javascript' });
    const blobUrl = URL.createObjectURL(blob);
    try {
        return await import(blobUrl);
    } finally {
        URL.revokeObjectURL(blobUrl);
    }
}

/**
 * Load (or return the cached) ort-web module.  Populates the WASM
 * artefact path on `ort.env.wasm.wasmPaths` as a side-effect.
 *
 * @returns {Promise<object>} the ort-web module namespace.
 */
export async function loadOrt() {
    if (_ortPromise) return _ortPromise;

    _ortPromise = (async () => {
        const useLocal = await _probeLocalOrt();
        const url = useLocal ? ORT_LOCAL_URL : ORT_CDN_URL;
        const sri = useLocal ? null : ORT_CDN_SRI;
        const wasmPath = useLocal ? ORT_LOCAL_WASM_PATH : ORT_CDN_WASM_PATH;

        const mod = await _importOrt(url, sri);
        const ort = mod.default || mod;

        // Tell the runtime where to find the WASM binaries.
        if (ort && ort.env && ort.env.wasm) {
            ort.env.wasm.wasmPaths = wasmPath;
            // Leave thread count on auto (respects COOP/COEP cross-
            // origin isolation).  Operators can override post-load.
        }

        // Stash on window for tooling / debugging (the metrics overlay
        // reads `ort.env.webgpu` for backend confirmation).
        if (typeof window !== 'undefined') {
            window.__I3_ORT__ = ort;
        }
        return ort;
    })();

    return _ortPromise;
}

/**
 * Load an InferenceSession for a given ONNX model URL.
 *
 * @param {string} modelUrl — URL of the `.onnx` blob to load.
 * @param {string[]} preferredExecutionProviders — ordered list, e.g.
 *   `['webgpu', 'wasm']`.  The first one the runtime can actually
 *   initialise wins.
 * @returns {Promise<object>} the InferenceSession instance.
 */
export async function loadOrtSession(modelUrl, preferredExecutionProviders) {
    if (typeof modelUrl !== 'string' || modelUrl.length === 0) {
        throw new TypeError('loadOrtSession: modelUrl must be a non-empty string');
    }
    const eps = Array.isArray(preferredExecutionProviders) && preferredExecutionProviders.length > 0
        ? preferredExecutionProviders.slice()
        : ['wasm'];

    const ort = await loadOrt();

    // Fetch the model bytes through our own origin so COOP/COEP apply.
    const resp = await fetch(modelUrl, {
        cache: 'force-cache',
        credentials: 'omit',
        mode: 'cors',
    });
    if (!resp.ok) {
        throw new Error(`model fetch failed: HTTP ${resp.status} for ${modelUrl}`);
    }
    const bytes = new Uint8Array(await resp.arrayBuffer());

    const sessionOptions = {
        executionProviders: eps,
        graphOptimizationLevel: 'all',
        // Keep logs quiet — the overlay handles observability for us.
        logSeverityLevel: 3,
    };

    return ort.InferenceSession.create(bytes, sessionOptions);
}

export const ORT_META = Object.freeze({
    version: ORT_VERSION,
    cdnUrl: ORT_CDN_URL,
    localUrl: ORT_LOCAL_URL,
});
