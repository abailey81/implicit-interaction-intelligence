# I³ — edge inference profile

> Run on: 2026-04-28 (laptop CPU, no GPU acceleration).  Snapshot of
> the actual artefacts shipped to `web/models/` for in-browser
> inference via ONNX Runtime Web.

## Encoder (TCN, 64-d user-state embedding)

| Variant | Size on disk | p50 latency | p95 latency | p99 latency | Throughput |
|---|---:|---:|---:|---:|---:|
| **INT8 ONNX** (`web/models/encoder_int8.onnx`) | **162.2 KB** | 460 µs | 637 µs | 718 µs | 2 176 enc/s |
| FP32 ONNX (`checkpoints/encoder/tcn.onnx`)    | 441.4 KB    | 200 µs | 285 µs | 351 µs | 4 990 enc/s |

**INT8 vs FP32 trade-off**

- **−63.3 % size on disk** (441 → 162 KB).  Fits in L2 cache on a
  Cortex-A55 watch (typically 128 KB shared L2, plenty of headroom
  with code).  FP32 doesn't.
- **Latency cost on x86 CPU: ~2.3×.**  Dynamic INT8 quantisation adds
  per-op dequant + requant; on x86 with no INT8 SIMD path this is
  pure overhead.  On a Kirin watch's NPU (or any chip with INT8
  hardware) the relationship inverts and INT8 is the faster path.
- **Parity vs FP32: MAE 0.00055, max abs err 0.0018** on the same
  10-step feature window.  The 64-d embedding is preserved across
  quantisation — adaptation downstream is identical.

## In-browser inference path

The browser-side runtime lives at `web/js/browser_inference.js` and
auto-mounts a "Run inference in browser" toggle inside the **State
tab** (under "Edge inference · Run on this device.").  Toggle ON and
the page:

1. Detects WebGPU / WASM SIMD support (`web/js/webgpu_probe.js`).
2. Spawns `web/js/encoder_worker.js` as a module worker.
3. Fetches `/api/onnx/encoder_int8.onnx` (162 KB) once and caches it.
4. Routes every keystroke-feature encode call through the worker
   instead of the server.

**Demo-visible proof:** open Chrome DevTools → Network panel.  With
the toggle **OFF**, every typed message fires a `/api/encode` request
to the server.  With the toggle **ON**, **zero** `/api/encode`
requests fire — the encoder ran inside the browser tab.  Keystrokes
never leave the device.

## Wearable budget — Kirin A2 watch

Huawei's Kirin A2 watch class (typical specs we'd target):

| Constraint | Budget | Our footprint | Headroom |
|---|---:|---:|---:|
| Encoder model size (RAM-resident) | ≤ 2 MB | **0.16 MB INT8** | 12.5 × |
| Encoder peak resident memory | ≤ 8 MB | ~1 MB working set | 8 × |
| Encoder inference latency | ≤ 50 ms | ~0.5 ms p50 (CPU x86); est. 2-5 ms on watch | well under |
| Wire bytes per turn | ≤ 1 KB | 8 floats × 4 B = 32 B (the adaptation vector) | 32 × |

The SLM (204 M params, ~780 MB FP32 / ~200 MB INT8) is **not** a
watch target — that's a phone-class artefact (Kirin 9000 series).
The architecture is split intentionally: encoder + retrieval +
intent parser fit on a watch; the SLM lives on the paired phone or
in a cloud arm; the cascade decides routing per turn.

## What this demonstrates for the JD

The job description specifically asks: *"Have you ever deployed ML
models to low-compute devices (e.g., wearables or IoT), where memory
and power are strictly limited?"*

The honest answer for I³:

- **Yes** — the encoder is INT8-quantised, ONNX-exported, and runs
  client-side in the browser via ONNX Runtime Web (WASM / WebGPU).
  This is a real deployment, demonstrable live in the demo.
- **No** — the SLM has not been shipped to a Kirin device.  It has
  the export plumbing and a quantisation spec, but I haven't done
  the on-device latency / power profiling.  That would be the first
  thing I'd close as an intern.

## Reproduce

```bash
# Quantise the FP32 encoder export to INT8 (drops 441 → 162 KB).
.venv/Scripts/python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('checkpoints/encoder/tcn.onnx',
                 'web/models/encoder_int8.onnx',
                 weight_type=QuantType.QInt8)"

# Verify parity vs FP32 (MAE should be < 0.001).
.venv/Scripts/python scripts/verify_onnx.py

# Re-run this profile (1 000 timed inferences each, with warm-up).
.venv/Scripts/python scripts/profile_edge_inference.py \
    --runs 1000 --warmup 50 --out reports/edge_profile_$(date +%F).md
```
