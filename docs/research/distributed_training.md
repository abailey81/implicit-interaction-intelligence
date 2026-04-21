# Distributed Training and High-Throughput Serving for I³

> Design note covering the scale-up path from single-box prototype to
> multi-node production deployment. This document is a companion to the
> new modules under `i3/serving/` and `training/train_*_fabric.py`,
> `training/train_with_accelerate.py`, `training/train_with_deepspeed.py`.

## 1. Motivation: why distributed matters for an "implicit" system

The Implicit Interaction Intelligence (I³) prototype was deliberately
built CPU-first. Keystroke dynamics, linguistic complexity, and temporal
patterns are inexpensive to encode; the TCN encoder is 6-8M parameters
and the Adaptive SLM is of the same order. For a single user on a single
device, all of this fits on a laptop, and that is by design — I³ is
meant to be runnable at the edge.

But three independent pressures push us toward a distributed stack:

1. **Federated long-term profile updates.** The long-term memory of each
   user's interaction patterns is stored on-device, but the *model
   structure* that interprets those patterns (the TCN encoder weights)
   improves fastest when evidence from many users can inform training,
   without any single user's data leaving their device. The canonical
   solution is a federated aggregation loop: each device produces a
   gradient (or model delta) against its own data; a coordinator
   averages them (DP-FedAvg, FedProx, secure aggregation) and redistributes
   the updated encoder. The coordinator must scale with the number of
   participating devices; that coordinator is a distributed training
   job.
2. **SLM retraining on curated corpora.** The Adaptive SLM's adaptation
   layer benefits from joint re-training whenever we add a new
   conditioning signal (e.g. wearable telemetry, explicit accessibility
   flags) or extend the vocabulary. The SLM itself stays small, but the
   retraining corpus — conversational data filtered for the traits
   encoded in the 8-dimensional AdaptationVector — has grown well beyond
   what fits in one GPU's memory with a reasonable batch size. FSDP2 and
   DeepSpeed ZeRO exist precisely for this shape of problem.
3. **Concurrent serving at scale.** Even a very small model becomes
   throughput-bound once the server is serving tens of thousands of
   concurrent WebSocket connections. That is an inference-side, not
   training-side, concern, but the same engineering team typically owns
   both, and the choices are coupled (the model format that is cheapest
   to train is often not the cheapest to serve).

The rest of this note argues for specific pieces of the stack, gives a
decision matrix, and discusses where each one stops being worth the
complexity.

## 2. Training framework trade-offs

### 2.1 The options

| Framework | Model wrapping | Loop ownership | Best fit |
|---|---|---|---|
| Lightning **Trainer** | `LightningModule` subclass | Framework | Greenfield projects where the team accepts Lightning conventions |
| Lightning **Fabric** | Plain `nn.Module` | Caller | Existing PyTorch loops that want DDP / FSDP / precision without rewrite |
| HF **Accelerate** | Plain `nn.Module` | Caller | HF-heavy stacks (transformers, datasets, PEFT) |
| **FSDP2** (native) | Plain `nn.Module` | Caller | Teams who want the underlying primitive with no third-party abstraction |
| **DeepSpeed** | `DeepSpeedEngine` | Engine | Very large models, aggressive ZeRO-3 + offload, curriculum / pipelines |

Our existing loops are pure PyTorch with a bespoke NT-Xent implementation
in `i3/encoder/loss.py` and a bespoke transformer block with cross-
attention conditioning in `i3/slm/transformer.py`. We do *not* want to
rewrite these as `LightningModule`s because (a) the cross-attention
conditioning doesn't fit cleanly into the Lightning forward contract and
(b) we want the training code to read like textbook PyTorch for people
learning the system.

That eliminates the Lightning Trainer. Everything else keeps the loop
in user code.

### 2.2 Why Fabric first

Fabric is the minimal common denominator: `fabric.setup(model, optimizer)`,
`fabric.setup_dataloaders(...)`, `fabric.backward(loss)`, `fabric.clip_gradients(...)`,
`fabric.save(...)`. It supports DDP, FSDP, and DeepSpeed as strategies.
Crucially, it does not *own* your loop.

We use it as the default path in `training/train_encoder_fabric.py`
(DDP) and `training/train_slm_fabric.py` (FSDP with `torch.compile(mode='max-autotune')`).
Both scripts fall back to a single-CPU run when `lightning` is not installed,
so the files never break a fresh clone.

### 2.3 When Accelerate wins

Accelerate earns its place when the training script must interoperate
with `transformers.Trainer`, `peft`, or the HuggingFace `datasets`
library. Even though we don't use transformers today, the SLM retraining
path is likely to pull in curated conversational corpora hosted on the
HuggingFace Hub, at which point `accelerate.Accelerator` becomes the
least-friction way to route data. `training/train_with_accelerate.py`
is maintained as a parallel implementation so operators can pick.

### 2.4 When DeepSpeed (ZeRO-3) wins

DeepSpeed is overkill for the current 6-8M-parameter SLM — the optimizer
state fits comfortably on any single accelerator, and ZeRO's
communication overhead would dominate. But two planned experiments make
ZeRO-3 the natural choice:

* **Wider variants.** A 100M-1B parameter SLM still inferences cheaply
  after INT4 quantization (`torchao`), but trains far more comfortably
  under ZeRO-3 with CPU optimizer offload.
* **Joint optimization with the long-term-memory compressor.** If we
  end up co-training a compressor that summarizes a user's profile into
  a fixed-size embedding, the memory budget for the optimizer states of
  two models exceeds what even an A100 80G holds at reasonable batch
  sizes. CPU offload of optimizer + parameter state is exactly ZeRO-3's
  value proposition.

`configs/distributed/ds_config_zero3.json` ships the config; the
training script `training/train_with_deepspeed.py` writes it on first
run if missing.

### 2.5 FSDP2 (native)

FSDP2 (`torch.distributed.fsdp` in PyTorch 2.4+) is now the PyTorch-native
answer to ZeRO. Fabric's `strategy='fsdp'` uses it under the hood. If
we ever want the lowest possible abstraction ceiling, we can drop Fabric
and call FSDP2 directly, but doing so forfeits the device-selection,
precision, and launcher ergonomics that Fabric provides. The
recommendation is to stay on Fabric until there is a specific need FSDP2
exposes that Fabric hides.

### 2.6 Decision matrix

| Scenario | Recommended tool |
|---|---|
| Daily TCN re-training on a laptop / single GPU box | `train_encoder.py` (the original) |
| Multi-GPU TCN training, one host | `train_encoder_fabric.py` with `strategy=ddp` |
| SLM retraining on new conditioning signals, one host | `train_slm_fabric.py` with `strategy=fsdp` |
| SLM retraining, multi-node, wide variant | `train_with_deepspeed.py` (ZeRO-3 + CPU offload) |
| Data hosted on HuggingFace Hub, retrained w/ HF datasets | `train_with_accelerate.py` |
| Federated profile update coordinator | Fabric + a thin aggregation loop |

## 3. Serving: autoscaling, batching, memory

### 3.1 FastAPI single-host (the reference)

`server/app.py` is the reference implementation. It hosts the pipeline,
the WebSocket layer, the admin routes, and the what-if routes in one
Uvicorn worker. For a laptop demo, a single tenant, or a small team,
this is the right answer. It is also the path we recommend for operators
who do not yet need horizontal scaling — the operational complexity of
everything that follows is real.

### 3.2 Ray Serve — replica autoscaling

Ray Serve changes the unit of scale from "worker process" to
"deployment replica". Each replica is a Python actor; the scheduler
routes requests to the least loaded one and can expand the replica
count based on target ongoing requests per replica.

`i3/serving/ray_serve_app.py` defines two deployments:

* `I3ServeDeployment` — the worker, holding a warm `AdaptiveSLM` per
  replica;
* `I3BanditRouter` — the ingress that runs bandit arm selection before
  delegating.

Separating the two lets us scale bandit routing (cheap) and generation
(expensive) independently. The KubeRay `RayService` manifest at
`deploy/serving/ray_serve_manifest.yaml` wires up two-tier autoscaling:
Ray Serve's own replica autoscaler, and cluster-level worker-node
autoscaling via the Ray Autoscaler v2. A Kubernetes HPA sits on top for
CPU / memory pressure signals, which are orthogonal to Ray's
per-deployment metrics.

### 3.3 NVIDIA Triton Inference Server

Triton is unbeatable when the deployment must serve *multiple* models
in *multiple* frameworks with *dynamic batching* and *versioning* out
of the box. For I³ the TCN encoder is the natural Triton tenant: it
runs continuously on every packet of interaction data, it benefits
enormously from batched ONNX Runtime on CPU, and its output shape is
fixed.

`i3/serving/triton_config.py` generates a `config.pbtxt` that enables:

* dynamic batching with `preferred_batch_size: [4, 8, 16]` and a 10ms
  queue delay (the tail-latency penalty is negligible because the TCN
  forward is sub-millisecond at batch 16 on any modern CPU);
* CPU and GPU instance groups (the latter is auto-suppressed when CUDA
  is not detected);
* a `model_repository/` layout documented in a generated `README.md`.

We explicitly do not move the SLM to Triton. The SLM's conditioning
surface (AdaptationVector + UserStateEmbedding via cross-attention) is
non-standard, and the generation loop is autoregressive. Triton's
support for LLM decoding through the TensorRT-LLM backend exists but it
adds substantial engineering cost relative to the current SLM's
latency budget.

### 3.4 vLLM — PagedAttention

vLLM's value proposition is two-fold:

1. **PagedAttention** removes KV-cache fragmentation for concurrent
   users, raising throughput under load by ~2-5×.
2. **Continuous batching** lets new requests join a running batch
   without waiting for its slowest member to finish.

These benefits scale with (a) concurrency and (b) context length. At
the I³ SLM's current size of ~6-8M parameters, a single request takes
under 50ms on CPU and a fraction of that on GPU. vLLM's per-request
overhead — Python API boundary, scheduler, block manager — is
measurable relative to that total, so a naive switch to vLLM would
regress single-request latency even as it improved aggregate throughput.

The right time to turn on vLLM is when at least one of the following is
true:

* the SLM grows to 100M+ parameters (at which point PagedAttention's
  memory savings become decisive);
* we see many concurrent users per serving replica (the continuous-
  batching throughput wins compound);
* long-context personalization lands (KV fragmentation becomes the
  bottleneck).

Until then, `i3/serving/vllm_server.py` is a scaffold. The file includes
an explicit `TODO: PagedAttention requires a format export step`
comment that spells out the two migration options (custom vLLM model
class vs. rewriting the SLM to match an existing vLLM-supported
architecture). Neither path is free, so we want them on the table before
a vLLM switch is scheduled.

## 4. Where each component stops being worth it

| Component | Turn on when... | Turn off when... |
|---|---|---|
| Fabric | Second GPU appears, or mixed precision is needed | Single-process single-device CPU dev loop |
| Accelerate | HF datasets / transformers land in the training data path | Not needed — Fabric covers the same ground |
| DeepSpeed ZeRO-3 | Model + optimizer state exceeds one accelerator | Small models; ZeRO overhead > benefit |
| FSDP2 direct | You need a primitive Fabric hides | Default case; Fabric is simpler |
| Ray Serve | You need horizontal replica autoscaling with dynamic cluster shape | Static one-host deployment |
| Triton | You must serve multi-model / multi-framework with dynamic batching | Single model + FastAPI is simpler |
| vLLM | 100M+ parameter model **and** many concurrent users | SLM stays at 6-8M params |

## 5. Federated learning and Huawei's MindSpore

I³ is research-motivated in part by Huawei's long-standing interest in
on-device AI. Huawei's first-party ML framework, **MindSpore**, offers
equivalents to most of what the PyTorch stack above provides:

* `MindSpore.Federated` parallels TensorFlow Federated and Flower for
  the aggregator side of the federated profile-update loop.
* `MindSpore.Distributed` provides a data-parallel and model-parallel
  equivalent to FSDP/DeepSpeed with Ascend-NPU aware collectives
  (HCCL).
* `MindSpore Serving` plays the role of Ray Serve / Triton on Ascend
  hardware.

The architectural ideas transfer cleanly — federated averaging, ZeRO-
style optimizer sharding, replica-level autoscaling are framework-
agnostic. A future port to MindSpore is mostly a matter of swapping the
tensor / module APIs and reusing the training loop structure. The
soft-import pattern we adopt throughout means a MindSpore-specific path
can be added alongside the PyTorch one without destabilising existing
installs.

For Huawei's own stack, the single most relevant piece of this work is
the **federated profile-update coordinator**: the combination of
`Fabric` (or MindSpore.Distributed), a thin secure-aggregation protocol,
and the per-user long-term-memory compressor. The serving-side pieces
(Ray Serve, Triton, vLLM) are table stakes for any inference provider;
the federated training path is what makes an adaptive companion system
deployable without pulling user data off-device.

## 6. What we are *not* doing

Out of scope for this work:

* **Model parallelism beyond FSDP/ZeRO.** The SLM is not large enough
  to need tensor parallelism or pipeline parallelism, and introducing
  either would make debugging significantly harder.
* **Custom collective backends.** We rely on NCCL (CUDA) and Gloo (CPU);
  HCCL support arrives with the MindSpore port, not before.
* **Continuous retraining from production traffic.** The loop
  `prod traffic → aggregator → retrain → deploy` requires a separate
  set of MLOps pieces (data validation, feature-store write-through,
  shadow deployment, online-eval gate) which belong in a follow-up
  design note.

## 7. References

* Falcon, W. et al. *PyTorch Lightning: Fabric*. Lightning AI, 2024.
  https://lightning.ai/docs/fabric/stable/
* Rasley, J., Rajbhandari, S., Ruwase, O., He, Y. *DeepSpeed: System
  optimizations enable training deep learning models with over 100
  billion parameters.* KDD 2020.
* Zhao, Y. et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded
  Data Parallel.* VLDB 2023.
* Kwon, W. et al. *Efficient Memory Management for Large Language Model
  Serving with PagedAttention.* SOSP 2023.
* Moritz, P. et al. *Ray: A Distributed Framework for Emerging AI
  Applications.* OSDI 2018.
* Bonawitz, K. et al. *Practical Secure Aggregation for Privacy-
  Preserving Machine Learning.* CCS 2017.
* McMahan, H. B. et al. *Communication-Efficient Learning of Deep
  Networks from Decentralized Data.* AISTATS 2017.
* Huawei MindSpore team. *MindSpore Federated Learning.* Huawei, 2023.
* NVIDIA. *Triton Inference Server Documentation.*
  https://github.com/triton-inference-server/server.
