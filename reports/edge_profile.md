# Edge profile

*Measured 2026-04-25T02:26:08+00:00 on `cpu`.*

## SLM (v1 Adaptive decoder)

- Parameters: **53,307,392**
- fp32 size: 203.35 MB
- bf16 size: 101.68 MB
- int8 size: **110.19 MB** (dynamic quantisation)
- ONNX prefill graph: (export failed — see logs)

## TCN encoder

- Parameters: **106,112**
- fp32 size: 0.405 MB
- int8 size: 0.400 MB

## Latency (CPU, 100 runs)

- SLM greedy decode (32 prompt → 16 new tokens): p50 **612.8 ms**, p95 692.4 ms
- TCN encoder (single 10×32 window): p50 **3.679 ms**, p95 4.707 ms

## Memory

- Peak process RSS during measurements: **1311.1 MB**

## Deployability

- [x] mid-range phone (int8 110.2 MB <= 300 MB budget)
- [ ] budget phone (int8 110.2 MB > 100 MB budget - too big)
- [ ] wearable (int8 110.2 MB > 50 MB budget - too big)

*SLM checkpoint*: `D:\implicit-interaction-intelligence\checkpoints\slm\best_model.pt`  *TCN checkpoint*: `D:\implicit-interaction-intelligence\checkpoints\encoder\best_model.pt`
