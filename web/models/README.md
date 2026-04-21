# `web/models/` — Browser-side ONNX artefact drop point

This directory is the **artefact drop point** for the ONNX models that power
**in-browser inference** of the Implicit Interaction Intelligence (I³) system.
When the browser toggle "Run inference in browser" is enabled the client
loads these files with [ONNX Runtime Web](https://onnxruntime.ai/) and runs
the TCN encoder (and, optionally, a minimal SLM prefill) directly on the
user's device.  No keystrokes, no feature vectors, no embeddings ever leave
the browser in that mode — **privacy by architecture**, enforced by the
network boundary.

## Expected artefacts

| File                                | Source script                     | Size budget |
| ----------------------------------- | --------------------------------- | ----------- |
| `encoder_int8.onnx`                 | `i3/encoder/onnx_export.py`       | < 2 MB      |
| `slm_prefill_int8.onnx` (optional)  | `i3/slm/onnx_export.py`           | < 40 MB     |

Both files are expected to be INT8-quantised with per-channel weight
quantisation so that they run well under WebAssembly SIMD / WebGPU.  The
`.gitkeep` file exists so the empty directory survives in git even before
you have trained a model.

## How to produce them

```bash
# TCN encoder (output: encoder_int8.onnx)
python -m i3.encoder.onnx_export \
    --checkpoint checkpoints/encoder.ckpt \
    --out web/models/encoder_int8.onnx \
    --quantize int8

# Optional SLM prefill (output: slm_prefill_int8.onnx)
python -m i3.slm.onnx_export \
    --checkpoint checkpoints/slm.ckpt \
    --out web/models/slm_prefill_int8.onnx \
    --quantize int8
```

## How the browser fetches them

The server exposes `GET /api/onnx/{model}` (see `server/routes_inference.py`)
which streams the raw bytes with:

* `Content-Type: application/octet-stream`
* `Cross-Origin-Opener-Policy: same-origin`
* `Cross-Origin-Embedder-Policy: require-corp`

The COOP + COEP pair is **required** to unlock cross-origin isolation, which
in turn unlocks:

1. `SharedArrayBuffer` (threaded WebAssembly for `onnxruntime-web`)
2. A stable `performance.now()` resolution (high-resolution timers)

Without these headers the runtime will silently fall back to single-threaded
WASM (≈2× slower on desktop, ≈4× slower on mobile).

## Why not ship them in git?

The weights are model IP and can run into tens of megabytes.  Instead we
treat them as **release artefacts**: train → export → drop into
`web/models/` at deployment time (or via a CI release job that uploads to
this path on the container).

## Path-traversal safety

The route handler validates the `{model}` path parameter against a strict
regex (`^[a-zA-Z0-9_.-]+\.onnx$`) **and** re-resolves the target with
`Path.resolve()` to ensure it stays inside `web/models/`.  Requests like
`GET /api/onnx/../../etc/passwd` are rejected with a 404.

## Versioning

When a new model is rolled out the filename should either (a) stay the same
(the browser will pick up the new bytes on the next page load because the
server emits `ETag` / `Last-Modified`) or (b) be renamed with a suffix
(`encoder_int8_v2.onnx`) and the loader code in `web/js/browser_inference.js`
updated accordingly.
