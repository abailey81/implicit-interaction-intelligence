# I³ Edge Deployment — Overview

This folder collects all I³ documentation about on-device ("edge")
deployment. The I³ product targets memory- and power-constrained
hardware (AI Glasses, Kirin-class handsets, Apple Silicon laptops,
Meteor Lake notebooks) and therefore evaluates the full 2026 edge
runtime landscape, not a single vendor stack.

## Contents

- [`alternative_runtimes.md`](./alternative_runtimes.md) — detailed
  comparison of every edge runtime that I³ has evaluated: Apple MLX,
  llama.cpp + GGUF, Apache TVM, IREE, Core ML, TensorRT-LLM, Intel
  OpenVINO, Google MediaPipe, and the default ExecuTorch path. Includes
  a decision matrix, per-runtime usage notes, benchmark tables, and
  deployment-scenario recommendations.

## Related

- [`../edge_profiling_report.md`](../edge_profiling_report.md) — raw
  profiling numbers for the ExecuTorch baseline.
- [`../huawei/kirin_deployment.md`](../huawei/kirin_deployment.md) —
  Huawei Kirin / HarmonyOS-specific deployment notes, including the
  MindSpore Lite / NNRt path.
- [`../../i3/edge/`](../../i3/edge/) — source for the edge exporters
  described in these docs.

## Which runtime should I use?

| Scenario                             | Recommended runtime                          |
|--------------------------------------|----------------------------------------------|
| Default on-device production build   | **ExecuTorch** (`.pte`)                      |
| Huawei Kirin / HarmonyOS             | ExecuTorch, then MindSpore Lite for NPU      |
| Apple Silicon developer laptop demo  | **MLX**                                      |
| iOS AI Glasses deployment            | **Core ML** (Neural Engine)                  |
| Smart Hanhan / Cortex-A76 ARM SoC    | **TVM** (`llvm -mcpu=cortex-a76`) + INT8     |
| Generic 2026 CPU LLM distribution    | **llama.cpp** + GGUF (`Q4_K_M`)              |
| Intel Meteor Lake NPU laptop         | **OpenVINO** (INT8)                          |
| Android mobile task bundle           | **MediaPipe** (on TFLite)                    |
| Cross-platform AOT GPU compile       | **IREE** (`vulkan-spirv`)                    |

See [`alternative_runtimes.md`](./alternative_runtimes.md) for the
full decision matrix and rationale.
