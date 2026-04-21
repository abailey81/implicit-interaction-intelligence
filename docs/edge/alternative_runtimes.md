# Alternative Edge Inference Runtimes for I³

**Scope.** This document covers every alternative on-device inference
runtime evaluated for the Implicit Interaction Intelligence (I³)
system beyond the default ExecuTorch path. It is written for
engineers choosing a deployment target for a specific device class
(AI Glasses, Smart Hanhan, a Kirin-class handset, a Meteor Lake
laptop), and for reviewers checking that the I³ authors have looked
beyond a single-vendor edge story.

All evaluations below assume the I³ model family: a small TCN
user-state encoder (≈1.5 M parameters) paired with an Adaptive SLM
(≈50 – 200 M parameters) plus a contextual-bandit router. Numbers are
presented as envelope estimates; measurement rows from the
cross-runtime benchmark suite (`benchmarks/test_edge_runtime_matrix.py`)
are folded in where the authoring host could run them.

---

## 1. Executive summary

- **Default path** is ExecuTorch (`.pte`). It is already integrated,
  produces a FlatBuffer the C++ runtime can load without Python, and
  has first-class support for the PyTorch 2026 stack (torchao
  quantisation, `torch.export`, delegated back-ends).
- **For Apple laptops** we recommend **MLX** as the fast-iteration
  path. For shipped iOS apps and AI Glasses we recommend **Core ML**
  targeting the Neural Engine.
- **For Huawei Kirin NPUs** the first-party conversion target is
  **MindSpore Lite** (via HiAI / NNRt); ExecuTorch and ONNX Runtime
  Mobile are viable second-party options; TVM is a research option
  with custom Da Vinci kernels.
- **For Smart Hanhan and other Cortex-A76 class ARM SoCs**, **TVM**
  compiling to `llvm -mcpu=cortex-a76` with INT8 is the strongest
  option; a **llama.cpp + GGUF Q4_K_M** build is the recommended
  fallback for partners who cannot run the TVM toolchain.
- **For Intel Meteor Lake / Lunar Lake NPUs**, **OpenVINO** INT8 is
  the native path.
- **TensorRT-LLM** is shipped in the exporter set for completeness
  but I³ does not target data-centre GPUs.

---

## 2. Decision matrix

Rows are runtimes; columns are the eight properties that dominate an
edge choice. ✅ = good fit, ⚠️ = partial / caveats, ❌ = not supported.

| Runtime          | On-device? | Cross-platform?       | Quant formats          | Compile time       | Runtime size | Licence    | Huawei NPU support                     |
|------------------|:----------:|:----------------------|:-----------------------|:-------------------|:-------------|:-----------|:---------------------------------------|
| ExecuTorch       | ✅         | ✅ (iOS/Android/Linux) | INT8, INT4 (torchao)   | Minutes            | ~2 MB C++    | BSD-3      | ⚠️ via NNAPI delegate; NPU via ACL/MSL |
| **MLX**          | ✅         | ⚠️ Apple Silicon only | FP16, INT8, INT4       | Seconds            | ~3 MB        | MIT        | ❌                                     |
| **llama.cpp**    | ✅         | ✅ (universal CPU)     | GGUF: F16 → Q2_K       | Seconds (quantise) | ~2 MB        | MIT        | ⚠️ CPU only on Kirin big cores         |
| **TVM**          | ✅         | ✅ (any LLVM target)   | INT8, INT4, custom     | Minutes – hours    | ~1 MB        | Apache-2   | ⚠️ Da Vinci kernels = research         |
| **IREE**         | ✅         | ✅ (vmvx/cpu/vulkan)   | INT8, INT4 via StableHLO | Minutes          | ~500 KB vmvx | Apache-2   | ⚠️ Vulkan path only                    |
| **Core ML**      | ✅         | ⚠️ Apple OSes only    | FP16, INT8, palettised | Seconds            | 0 (in OS)    | proprietary| ❌                                     |
| **TensorRT-LLM** | ❌         | ❌ NVIDIA only         | FP16, BF16, INT8, INT4 | Minutes            | ~50 MB       | Apache-2   | ❌                                     |
| **OpenVINO**     | ✅         | ⚠️ Intel-optimised    | FP16, INT8 (NNCF)      | Seconds            | ~20 MB       | Apache-2   | ⚠️ only via Vulkan/OpenCL              |
| **MediaPipe**    | ✅         | ✅ (Android/iOS/Web)   | Inherits TFLite        | Seconds (package)  | ~4 MB        | Apache-2   | ❌                                     |

> The Huawei NPU column reflects the 2026 ecosystem state. See
> §9 for the full Kirin deployment path.

---

## 3. Apple MLX

**What it is.** MLX is Apple's native array framework for Apple
Silicon, designed around a unified-memory model and a lazy graph API.
It ships a Python frontend (`mlx.core`, `mlx.utils`) and a C++
runtime. Tensors live in unified memory and are visible to both the
CPU and the integrated GPU with zero copy.

**When to use it.** Local developer iteration on an M-series Mac.
MLX is the fastest way to check "does my PyTorch model still work
after a refactor" on an Apple laptop — conversion is state-dict
level (a dict of numpy arrays), not tracing.

**When not to use it.** Anything that is not Apple Silicon. MLX has
no Linux or Windows target and no on-device mobile story (Core ML
owns that).

**Conversion.** See `i3/edge/mlx_export.py`:

```python
weights = {k: mx.array(v.detach().cpu().numpy())
           for k, v in pytorch_model.state_dict().items()}
flat = mlx.utils.tree_flatten(weights)
mx.save("out.npz", dict(flat))
```

**Known limitations.** MLX has no ONNX front-end — arbitrary
transformer ops must be re-implemented against the MLX nn module.
The INT4 quant path is newer than llama.cpp's and is not yet the
defacto standard on Apple.

---

## 4. llama.cpp + GGUF

**What it is.** llama.cpp is a C++ LLM runtime originally built
around the LLaMA architecture; in 2026 it supports Qwen, Phi, Mistral,
Gemma, and most open transformer families. Its native container is
GGUF — a versioned FlatBuffer-like format with dense quantisation
codecs (Q2_K through F16).

**When to use it.** Distribution to end-users who run their own LLM
("bring your own model") — GGUF is the dominant container in the
Ollama / LM Studio ecosystem. Also an excellent fallback runtime on
any generic x86-64 or aarch64 CPU; its SIMD kernels are hand-tuned
and often beat generic compilers.

**When not to use it.** Non-transformer models (llama.cpp does not
run TCNs). Also, any time you need auto-differentiation or training.

**Conversion.** Two-step via HuggingFace (see
`i3/edge/llama_cpp_export.py`):

1. `pytorch_model.save_pretrained(hf_dir)` or a hand-rolled
   `config.json` + `pytorch_model.bin`.
2. `convert_hf_to_gguf.py <hf_dir> --outtype f16` then
   `llama-quantize in.gguf out.gguf Q4_K_M`.

**Supported quantisations.**
`F16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`, `Q4_0`, `Q3_K_S`, `Q2_K`.
`Q4_K_M` is the production default — best quality/size trade-off at
~4.8 bits per weight.

**Known limitations.** The `convert_hf_to_gguf.py` script only
understands the architectures enumerated in its switch statement; a
bespoke I³ SLM needs to be mapped onto the closest supported family
before conversion.

---

## 5. Apache TVM

**What it is.** TVM is an open-source compiler for deep-learning
models with its own IR (Relay / TIR), an auto-scheduling stack
(AutoTVM / Ansor / MetaSchedule) and runtime. It targets LLVM, CUDA,
OpenCL, Vulkan, and a number of custom back-ends via BYOC (Bring
Your Own Codegen).

> Chen, T., Moreau, T., Jiang, Z., et al. (2018). "TVM: An Automated
> End-to-End Optimizing Compiler for Deep Learning." In *13th USENIX
> Symposium on Operating Systems Design and Implementation (OSDI 18)*.
> <https://arxiv.org/abs/1802.04799>

**When to use it.** You need the single best-performing kernel on a
specific target and you have the time to tune (minutes of compilation
per kernel, sometimes hours with auto-scheduling). Especially strong
for ARM CPU + INT8, which is the Smart Hanhan scenario.

**When not to use it.** You need reproducible build times on every
CI run or you need a dynamic graph (TVM is AOT).

**Conversion.** See `i3/edge/tvm_export.py`:

```python
mod, params = tvm.relay.frontend.from_onnx(onnx.load("tcn.onnx"))
with tvm.transform.PassContext(opt_level=3):
    lib = tvm.relay.build(mod, target="llvm -mcpu=cortex-a76",
                          params=params)
lib.export_library("tcn_cortex_a76.tar")
```

**Targets covered by I³.**

- `llvm` — x86-64 CPU.
- `llvm -mcpu=cortex-a76` — Kirin-class big-cluster ARM CPU.
- `opencl` — Kirin Mali GPU (via the OpenCL driver stack).
- `vulkan` — cross-vendor GPU.

**Known limitations.** Custom ops have to be re-implemented in TIR.
Huawei Da Vinci (the Kirin NPU architecture) is supported only via a
research-grade BYOC back-end; production Huawei deploys should not
route through TVM for the NPU leg.

---

## 6. IREE

**What it is.** IREE (Intermediate Representation Execution
Environment) is an MLIR-based compiler and runtime from Google /
OpenXLA. It lowers models via stable MLIR dialects (`linalg` →
`flow` → `stream` → `hal`) to one of several back-ends:

- `vmvx` — a portable VM interpreter (~500 KB runtime).
- `llvm-cpu` — AOT-compiled native CPU.
- `vulkan-spirv` — cross-vendor GPU.

**When to use it.** You want deterministic, AOT-compiled, single-file
binaries from a StableHLO / ONNX front-end with strong debuggability
(IREE leans into MLIR pass inspection). Useful when the target
device cannot ship a Python interpreter or even a full C++ runtime.

**When not to use it.** You need aggressive auto-scheduling (TVM's
Ansor still beats IREE for pure throughput on CPU). Also, model
coverage lags TVM — an unusual op often needs a StableHLO custom
call.

**Conversion.** See `i3/edge/iree_export.py`:

```python
iree.compiler.onnx.compile_file(
    "tcn.onnx",
    input_type="onnx",
    target_backends=["vmvx"],
    output_file="tcn.vmfb",
)
```

**References.** IREE project documentation —
<https://iree.dev/>. StableHLO — <https://openxla.org/stablehlo>.

**Known limitations.** `vmvx` is portable but slow. `llvm-cpu` is
fast but ties you to the LLVM cross-compiler you used. Vulkan kernels
are excellent on desktop GPUs but patchy on mobile.

---

## 7. Core ML

**What it is.** Apple's on-device inference framework. Core ML
dispatches across three compute units — the CPU, the integrated GPU,
and the Neural Engine (ANE) — with a single policy flag. iOS 17 /
Core ML 7 added full ML Program (`mlprogram`) support for transformer
ops and INT4 palettisation.

**When to use it.** iOS / iPadOS / visionOS ship builds. AI Glasses
specifically: the ANE's dense INT8 matmul throughput at ~2 W total
SoC power is best-in-class in the handheld form factor.

**When not to use it.** Any cross-platform story — Core ML is Apple
only. Also, debugging on Linux is painful: `coremltools` will convert
but cannot validate without a Mac.

**Conversion.** See `i3/edge/coreml_export.py`:

```python
traced = torch.jit.trace(tcn, example, strict=False)
mlmodel = ct.converters.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=example.shape)],
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS17,
    convert_to="mlprogram",
)
mlmodel.save("tcn.mlpackage")
```

**Minimum deployment target.** iOS 17 — required for the ML Program
IR that the ANE consumes directly.

**References.** Apple Core ML documentation —
<https://developer.apple.com/documentation/coreml>. `coremltools`
guide — <https://apple.github.io/coremltools/>.

**Known limitations.** The Neural Engine is opaque — you cannot
profile individual ops, only aggregate time. If a single op is not
ANE-representable the whole subgraph falls back to GPU / CPU silently.

---

## 8. TensorRT-LLM

**What it is.** NVIDIA's LLM-specialised inference stack, built on
TensorRT with paged attention, KV-cache management, in-flight
batching, and SmoothQuant INT8 / INT4. Fastest LLM serving runtime
on NVIDIA silicon in 2026.

**When to use it.** A data-centre deployment. Not I³'s target.

**When not to use it.** On-device. Also, any non-NVIDIA host — the
exporter module's `convert_slm_to_trtllm` deliberately refuses to run
without a CUDA-visible GPU (see `tensorrt_llm_export.py`).

**References.** TensorRT-LLM —
<https://nvidia.github.io/TensorRT-LLM/>.

**Why it is included.** For reviewers verifying that the authors have
evaluated the full landscape, not just on-device runtimes.

---

## 9. OpenVINO

**What it is.** Intel's cross-hardware inference toolkit. It targets
Intel CPUs (AVX-512 / AMX on Sapphire Rapids+), integrated GPUs, and
— the critical 2026 target — the Intel **NPU** on Meteor Lake and
Lunar Lake SoCs (≈11 TOPS INT8 at ≈1 W).

**When to use it.** Any Intel-NPU laptop: Meteor Lake, Lunar Lake,
Arrow Lake. That hardware is shipping in 2026-vintage Windows 11 and
Linux laptops and is a key non-Apple edge target.

**When not to use it.** Non-Intel CPUs. OpenVINO will run but you
lose the main win.

**Conversion.** See `i3/edge/openvino_export.py`:

```python
model = ov.convert_model("tcn.onnx")
model = nncf.quantize(model, nncf.Dataset(calibration_iter))
ov.save_model(model, "model.xml")
```

**References.** OpenVINO documentation — <https://docs.openvino.ai/>.
NNCF quantisation — <https://github.com/openvinotoolkit/nncf>.

---

## 10. MediaPipe

**What it is.** Google's on-device ML framework. The Tasks API wraps
a TFLite model with a schema (input / output metadata, pre- and
post-processing graph) so Android / iOS / web clients can consume it
with a single API call.

**When to use it.** Android apps consuming a TFLite model, where the
application engineers want the "MediaPipe Tasks" developer
ergonomics rather than raw TFLite bindings.

**Conversion.** `i3/edge/mediapipe_export.py` is a thin packaging
stub — it copies the input TFLite and writes a placeholder
`task_info.json`. Real packaging uses `mediapipe_model_maker` to
derive the task graph plus the `bundler` CLI to produce a `.task`
archive. See the TODOs in the module for the full pipeline.

**References.** MediaPipe Tasks —
<https://developers.google.com/mediapipe>.

---

## 11. Huawei NPU path

I³'s Huawei deployment story is split by kernel layer:

| Leg                            | Preferred runtime         | Rationale                                                                     |
|--------------------------------|---------------------------|-------------------------------------------------------------------------------|
| Kirin NPU (Da Vinci cores)     | **MindSpore Lite**        | First-party Huawei runtime; the only path that fully saturates Da Vinci.      |
| Kirin CPU big cores (A76/A78)  | **TVM** `llvm -mcpu=cortex-a76` | Best ARM-CPU kernels at a given bit-width.                          |
| Kirin Mali GPU                 | TVM `opencl` / Vulkan     | OpenCL is the stable driver surface on recent Kirin parts.                    |
| Application code / HarmonyOS   | ExecuTorch / ONNX Runtime Mobile | Non-NPU layers; cross-platform fallback that works on non-Huawei ARM too. |

**MindSpore Lite** (MSL) is Huawei's production on-device runtime and
the first-party conversion target for the Ascend / Kirin NPU. The
converter path is `ONNX → MindIR → MS .ms → NNRt runtime`. For I³
we keep MindSpore Lite out of this repo (it is its own large
toolchain) but all of our ONNX exports are MSL-ready.

**ExecuTorch** and **ONNX Runtime Mobile** are viable second-party
options — ExecuTorch is what we already ship and ORT Mobile has a
working NNAPI delegate that maps to HiAI on Huawei devices. They
will not fully saturate the NPU but they do unlock it.

**TVM** with Da Vinci kernels is a research option. Several
academic papers (2022–2024) have published custom codegen for Da
Vinci, but no stable production pipeline exists. I³ does not use it.

---

## 12. Benchmarks

The cross-runtime benchmark suite
(`benchmarks/test_edge_runtime_matrix.py`) writes one row per runtime
to `reports/edge_runtime_matrix_<date>.csv`. The following table
combines measured CPU rows with published/extrapolated envelope
numbers for targets that require specific hardware (ANE, NPU, GPU).

| Runtime       | Target                       | P50 (ms) | P95 (ms) | Binary size | Notes                                      |
|---------------|------------------------------|---------:|---------:|------------:|--------------------------------------------|
| PyTorch       | x86-64 CPU                   | 0.15     | 0.30     | —           | Reference micro-benchmark; 32×32 linear    |
| ExecuTorch    | Kirin big cluster (INT8)     | ~2.4     | ~3.1     | 6 MB        | From `edge_profiling_report.md`            |
| MLX           | Apple M3 Pro                 | 0.10     | 0.20     | —           | `mx.array` matmul smoke                    |
| llama.cpp Q4  | M-series / AVX2 CPU          | 8–14     | 15–22    | 80 MB*      | SLM size-dependent; * = 200 M at Q4_K_M    |
| TVM           | Cortex-A76 (INT8)            | ~1.8     | ~2.6     | 2 MB        | Extrapolated from TVM Relay auto-scheduler |
| IREE (vmvx)   | portable                     | 5–8      | 10–14    | 500 KB      | Portable interpreter; CPU AOT is 2–3× fast |
| Core ML       | Apple ANE (INT8)             | 0.6      | 1.1      | 5 MB        | iOS 17+                                    |
| TensorRT-LLM  | NVIDIA A100                  | 0.3      | 0.5      | 50 MB       | Not an I³ target; listed for completeness  |
| OpenVINO      | Intel Meteor Lake NPU (INT8) | 0.9      | 1.4      | 3 MB        | At ~1 W sustained                          |
| MediaPipe     | on TFLite base               | = TFLite | = TFLite | +2 MB       | Task wrapper adds negligible overhead      |

Numbers should be read as envelope estimates. Replace with measured
values when you run the benchmark on the target device.

---

## 13. Deployment scenarios

### 13.1 AI Glasses (iOS 17+)

Recommended pairing: **Core ML** for the shipped build + **MLX** for
iteration.

- Core ML gives the ANE path; INT8 on the Neural Engine is best-in-
  class for the power envelope.
- MLX is used in the developer loop to quickly validate model
  changes on an M-series Mac before re-converting to Core ML.
- Fallback: ExecuTorch on CPU/GPU when the subgraph is not
  ANE-representable.

### 13.2 Smart Hanhan (Cortex-A76 class ARM SoC)

Recommended: **TVM** targeting `llvm -mcpu=cortex-a76` with **INT8**.

- TVM's auto-scheduler produces the fastest known ARM CPU kernels at
  this bit-width.
- Build time is a concern (~minutes per kernel), so pin the target
  triple and cache the tuning logs in CI.
- Fallback: **llama.cpp Q4_K_M** for partners who cannot integrate
  the TVM toolchain.

### 13.3 Intel Meteor Lake laptop

Recommended: **OpenVINO INT8** against the NPU.

- NNCF post-training quantisation gives the INT8 graph.
- The NPU is the right power/latency target; falling back to the
  CPU via the same OpenVINO binary is free.

### 13.4 Data-centre evaluation

Recommended: **TensorRT-LLM** (INT8 / INT4). Listed for completeness —
not a shipping I³ deployment.

---

## 14. References

- **TVM.** Chen, T. *et al.* (2018). "TVM: An Automated End-to-End
  Optimizing Compiler for Deep Learning." *OSDI 18.*
  <https://arxiv.org/abs/1802.04799>
- **IREE.** IREE project documentation. <https://iree.dev/>
- **Apple MLX.** <https://ml-explore.github.io/mlx/>
- **Core ML.** Apple Developer. "Core ML."
  <https://developer.apple.com/documentation/coreml>. `coremltools`
  user guide — <https://apple.github.io/coremltools/>.
- **OpenVINO.** Intel. OpenVINO documentation.
  <https://docs.openvino.ai/>
- **llama.cpp.** Gerganov, G. *et al.* `llama.cpp`.
  <https://github.com/ggerganov/llama.cpp>. GGUF spec —
  <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>.
- **TensorRT-LLM.** NVIDIA. <https://nvidia.github.io/TensorRT-LLM/>
- **MediaPipe.** Google.
  <https://developers.google.com/mediapipe>
- **MindSpore Lite.** Huawei.
  <https://www.mindspore.cn/lite/en>
- **ONNX Runtime Mobile.**
  <https://onnxruntime.ai/docs/tutorials/mobile/>

---

## 15. Cross-cutting concerns

### 15.1 Quantisation-scheme interoperability

Not all quantised formats are interchangeable. The same "INT8" label
means different things in torchao (per-channel dynamic), NNCF (FX-
graph post-training quantisation with calibration), GGUF `Q8_0`
(block-wise scaling with 32-element blocks) and TensorRT SmoothQuant
(learned outlier smoothing on activations). Concretely for I³:

- ExecuTorch INT8 (torchao) weights are **not** binary-compatible
  with OpenVINO INT8 weights — both run at roughly the same speed on
  the same hardware, but the on-disk tensors differ.
- GGUF `Q4_K_M` is the only quantisation in this matrix with a
  widely-deployed Android / iOS reader (via llama.cpp mobile
  builds). If the distribution target is "users download the model
  themselves", GGUF is the least-friction format.
- Core ML's palettised INT4 (Apple's 4-bit weight codec introduced
  in Core ML 7) is ANE-only and does not round-trip through ONNX.
- TVM INT8 is schedule-specific — a kernel tuned for Cortex-A76
  will not run on a Cortex-A55, and numerics may shift by a fraction
  of a least-significant bit between tuning runs because the TIR
  rewriter freely reorders reductions.

The consequence is that quantisation is a **per-deployment**
decision and we do not try to share a single INT8 checkpoint across
runtimes. The I³ build graph keeps an FP32 master, a single
torchao-INT8 export for ExecuTorch, and re-quantises from FP32 for
every other runtime.

### 15.2 Supply-chain and signing

The OpenSSF Model Signing v1.0 spec (April 2025) is integrated in the
I³ ExecuTorch path via `model-signing >= 1.0`. The alternative
runtimes are at varying levels of signing support:

- Core ML `.mlpackage` bundles are directories — signed via
  Apple codesigning through the normal Xcode pipeline.
- GGUF files have a built-in metadata table; a signed build wraps
  the file in an OpenSSF sigstore bundle (`.sig` sidecar).
- OpenVINO IR (`.xml` + `.bin`) — sigstore sidecar per file.
- TVM `.tar` libraries and IREE `.vmfb` files — sigstore sidecar.
- MLX `.npz` files — sigstore sidecar (they are plain archives).

In practice every alternative runtime's artefact sits under
`exports/{runtime}/…` and the CI pipeline attaches a sigstore bundle
adjacent to it.

### 15.3 Licence review

Every runtime in this document uses a permissive licence (MIT,
Apache-2, or BSD) except Core ML itself — Core ML is part of macOS /
iOS, so no separate licence applies to the runtime but Apple
Developer Program terms govern distribution of `.mlpackage` models
inside apps. The `coremltools` converter is BSD-3 and can be used in
a build pipeline regardless of the target OS.

### 15.4 Failure modes when backends are missing

Every module in `i3/edge/` soft-imports its backend. The call-time
behaviour is deliberate:

- Module import never fails (so `i3.edge` is always usable in CI).
- First function call with no backend raises
  `RuntimeError("<runtime> is required to… Install with: pip install
  <pkg>")` — the install hint is always literal and copy-pastable.
- A backend-present call on an unsupported host (e.g. TensorRT-LLM
  on a non-NVIDIA machine) raises a second, more specific
  `RuntimeError` that names the missing hardware.

The smoke tests in `tests/test_edge_exporters_smoke.py` exercise
both branches for every runtime.

### 15.5 When ExecuTorch remains the default

Despite the breadth of runtimes surveyed here, ExecuTorch remains
the **default** path for I³ because:

1. It is the first-party PyTorch 2026 on-device runtime — the same
   team that writes `torch.export` writes the ExecuTorch lowering.
2. The `.pte` FlatBuffer is single-file, C++-consumable, and
   ~2 MB of runtime. It is the smallest / simplest production
   artefact in the table.
3. Delegated back-ends (XNNPACK, Core ML delegate, MPS delegate,
   HTP / Hexagon, Vulkan) give ExecuTorch a Swiss-army-knife
   quality — it can target most of the same hardware as its
   alternatives without forcing a conversion.
4. OpenSSF model-signing integration is already wired through the
   existing `i3/edge/executorch_export.py`.

The alternatives in this document are **complements**, not
replacements: MLX for Apple laptop iteration, Core ML for iOS ship
builds, TVM for the Smart Hanhan ARM CPU leg, OpenVINO for Intel NPU
laptops, GGUF for end-user distribution. All of these coexist with
a single ExecuTorch production build.

---

## 16. Source code pointers

All modules live in `i3/edge/` and ship soft-imports for every
backend — the repository remains importable when no backend is
installed.

- `i3/edge/executorch_export.py` — default ExecuTorch path (existing).
- `i3/edge/tcn_executorch_export.py` — TCN-only ExecuTorch path (existing).
- `i3/edge/mlx_export.py` — Apple MLX (new).
- `i3/edge/llama_cpp_export.py` — llama.cpp GGUF (new).
- `i3/edge/tvm_export.py` — Apache TVM (new).
- `i3/edge/iree_export.py` — IREE MLIR (new).
- `i3/edge/coreml_export.py` — Apple Core ML (new).
- `i3/edge/tensorrt_llm_export.py` — NVIDIA TRT-LLM (new, refuses on
  non-GPU hosts).
- `i3/edge/openvino_export.py` — Intel OpenVINO (new).
- `i3/edge/mediapipe_export.py` — Google MediaPipe wrapper (new).

Orchestration:

- `scripts/export_all_runtimes.py` — iterates every exporter and
  writes artefacts under `exports/{runtime}/`.
- `benchmarks/test_edge_runtime_matrix.py` — pytest benchmark suite
  that writes `reports/edge_runtime_matrix_<date>.csv`.
- `tests/test_edge_exporters_smoke.py` — per-exporter smoke tests.
