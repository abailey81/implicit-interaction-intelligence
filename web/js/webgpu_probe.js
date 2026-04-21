/**
 * webgpu_probe.js
 * ───────────────
 * Feature-detection for WebGPU with a graceful fallback path to
 * WebAssembly (SIMD + threads where cross-origin isolated, plain WASM
 * otherwise).
 *
 * Exports:
 *   detectBackend() -> Promise<{
 *     backend: 'webgpu' | 'wasm-simd-threaded' | 'wasm',
 *     adapter: { vendor, architecture, features: string[], limits } | null,
 *     reason: string
 *   }>
 *
 * Design notes
 *   * We probe in this order: WebGPU → threaded-SIMD WASM → plain WASM.
 *   * WebGPU requires `navigator.gpu` AND a usable `GPUAdapter`.  We
 *     explicitly request `requestAdapter({ powerPreference: 'high-
 *     performance' })` because the default adapter on laptops is often
 *     an integrated GPU that is no faster than WASM SIMD.
 *   * Threaded WASM requires `crossOriginIsolated === true`, which in
 *     turn requires the server to emit COOP/COEP headers.  See
 *     `server/routes_inference.py`.
 *   * On fallback we `console.warn` so the cause is visible to devs
 *     inspecting the browser console.
 */

/**
 * Best-effort adapter info serialisation.  The `adapter.info` property
 * is in progress at the W3C and is not yet available in every browser.
 * @param {GPUAdapter} adapter
 * @returns {Promise<object>}
 */
async function _serialiseAdapter(adapter) {
    const out = {
        vendor: '',
        architecture: '',
        description: '',
        features: [],
        limits: {},
    };
    try {
        // `adapter.info` (or the older `requestAdapterInfo`) is the
        // canonical way to get GPU vendor info.  Both are guarded in
        // try/catch because the shape differs across browsers.
        if (adapter.info && typeof adapter.info === 'object') {
            out.vendor = adapter.info.vendor || '';
            out.architecture = adapter.info.architecture || '';
            out.description = adapter.info.description || '';
        } else if (typeof adapter.requestAdapterInfo === 'function') {
            const info = await adapter.requestAdapterInfo();
            out.vendor = info.vendor || '';
            out.architecture = info.architecture || '';
            out.description = info.description || '';
        }
    } catch (_err) {
        // Some browsers gate adapter info behind a permission prompt;
        // ignore and leave the fields blank.
    }

    try {
        if (adapter.features && typeof adapter.features.forEach === 'function') {
            adapter.features.forEach((feat) => out.features.push(String(feat)));
        }
    } catch (_err) { /* ignore */ }

    try {
        if (adapter.limits) {
            // Copy a handful of useful numeric limits for the overlay.
            const keys = [
                'maxBufferSize',
                'maxStorageBufferBindingSize',
                'maxComputeWorkgroupSizeX',
                'maxComputeInvocationsPerWorkgroup',
            ];
            for (const k of keys) {
                if (typeof adapter.limits[k] === 'number') {
                    out.limits[k] = adapter.limits[k];
                }
            }
        }
    } catch (_err) { /* ignore */ }

    return out;
}

/**
 * Detect the best inference backend available in this browser.
 * @returns {Promise<{backend: string, adapter: object|null, reason: string}>}
 */
export async function detectBackend() {
    // ── 1. WebGPU ────────────────────────────────────────────────────
    if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance',
            });
            if (adapter) {
                const info = await _serialiseAdapter(adapter);
                return {
                    backend: 'webgpu',
                    adapter: info,
                    reason: 'navigator.gpu available and adapter obtained',
                };
            }
            console.warn(
                '[i3] WebGPU: navigator.gpu present but requestAdapter() returned null; falling back to WASM.',
            );
        } catch (err) {
            console.warn('[i3] WebGPU: requestAdapter() threw; falling back to WASM.', err);
        }
    } else {
        // Silent — most non-Chromium browsers land here and we don't
        // want to spam the console on every page load.
    }

    // ── 2. Threaded SIMD WebAssembly ─────────────────────────────────
    // Requires cross-origin isolation, `SharedArrayBuffer`, AND
    // `WebAssembly` itself.  We only advertise this when ALL three hold.
    const hasCrossOriginIsolation =
        typeof globalThis !== 'undefined' && globalThis.crossOriginIsolated === true;
    const hasSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
    const hasWasm = typeof WebAssembly !== 'undefined';
    if (hasWasm && hasSharedArrayBuffer && hasCrossOriginIsolation) {
        return {
            backend: 'wasm-simd-threaded',
            adapter: null,
            reason: 'WASM + threads + SIMD (crossOriginIsolated)',
        };
    }

    // ── 3. Plain WASM (baseline) ─────────────────────────────────────
    if (hasWasm) {
        const missing = [];
        if (!hasSharedArrayBuffer) missing.push('SharedArrayBuffer');
        if (!hasCrossOriginIsolation) missing.push('crossOriginIsolated');
        console.warn(
            `[i3] WASM threads disabled (missing: ${missing.join(', ')}); ` +
                'single-threaded WASM will be used.  ' +
                'Ensure COOP/COEP headers are set on the model route.',
        );
        return {
            backend: 'wasm',
            adapter: null,
            reason: `single-threaded WASM; missing: ${missing.join(', ')}`,
        };
    }

    // No WebAssembly support at all — extremely rare, pre-2017
    // browsers.  We still return a sentinel so the caller can show a
    // friendly message.
    return {
        backend: 'wasm',
        adapter: null,
        reason: 'WebAssembly unavailable — browser too old',
    };
}

/**
 * Map a backend name to the ORT `executionProviders` array.
 * @param {string} backend
 * @returns {string[]}
 */
export function executionProvidersFor(backend) {
    switch (backend) {
        case 'webgpu':
            return ['webgpu', 'wasm'];
        case 'wasm-simd-threaded':
        case 'wasm':
        default:
            return ['wasm'];
    }
}
