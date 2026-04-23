# I3 MLOps

This document describes the MLOps plumbing added under `i3/mlops/` and
the supporting scripts, benchmarks, and DVC pipeline.  It is a
standalone reference; the main MkDocs site does not link to it so that
the public docs stay focused on architecture and APIs.

## Components

| Concern                | Module / File                         |
|------------------------|---------------------------------------|
| Experiment tracking    | `i3/mlops/tracking.py`                |
| Model registry         | `i3/mlops/registry.py`                |
| Checkpoint integrity   | `i3/mlops/checkpoint.py`              |
| ONNX export dispatch   | `i3/mlops/export.py`                  |
| TCN ONNX exporter      | `i3/encoder/onnx_export.py`           |
| SLM ONNX exporter      | `i3/slm/onnx_export.py`               |
| CLI: ONNX export       | `scripts/export/onnx.py`              |
| CLI: ONNX verify       | `scripts/verify_onnx.py`              |
| CLI: inference bench   | `scripts/benchmarks/inference.py`      |
| CLI: edge profile      | `scripts/benchmarks/profile_edge.py`             |
| CLI: tracked training  | `scripts/training/train_with_tracking.py`      |
| pytest benchmarks      | `benchmarks/`                         |
| Load tests             | `benchmarks/locustfile.py`, `benchmarks/k6/load.js` |
| SLOs                   | `benchmarks/slos.yaml`                |
| Data + model pipeline  | `dvc.yaml`, `.dvc/config`, `.dvcignore` |

All MLflow and W&B dependencies are **optional**.  The `ExperimentTracker`
and `ModelRegistry` silently degrade to no-op mode when the
corresponding packages are absent -- training and evaluation never
break because of missing MLOps tooling.

## Experiment tracking

```python
from i3.mlops.tracking import ExperimentTracker

tracker = ExperimentTracker(config=my_training_config)

with tracker.run("encoder-v7", tags={"arch": "tcn", "stage": "staging"}):
    tracker.log_params({"lr": 1e-3, "batch_size": 64})
    for step, loss in enumerate(losses):
        tracker.log_metrics({"loss": loss}, step=step)
    tracker.log_artifact("models/encoder/best.pt")
```

Environment variables:

* `MLFLOW_TRACKING_URI` (default `file:./mlruns`)
* `MLFLOW_EXPERIMENT_NAME` (default `i3-experiments`)

The tracker automatically attaches these tags on every run:

* `i3.git_sha`      -- short git SHA of `HEAD`
* `i3.python`       -- `platform.python_version()`
* `i3.torch`        -- `torch.__version__`
* `i3.config_hash`  -- SHA-256 of the caller-supplied config mapping

### Reproducing a run

1. Open the MLflow UI: `mlflow ui --backend-store-uri file:./mlruns`.
2. Locate the target run and copy the `i3.git_sha` + `i3.config_hash`
   tags.
3. `git checkout <sha>` to restore the exact source tree.
4. `dvc repro` (or `dvc pull && dvc repro`) to rebuild the data and
   models from the tracked pipeline.
5. Re-run with `scripts/training/train_with_tracking.py`, passing any
   hyperparameters that were originally supplied on the CLI.

The combination of git SHA + DVC lock + config hash gives us
bit-reproducible artefacts for any committed training run.

## Model registry

`ModelRegistry` writes a filesystem-first registry under `registry/`
and best-effort mirrors each registration to MLflow and / or W&B:

```python
from i3.mlops.registry import ModelRegistry

reg = ModelRegistry(use_mlflow=True, use_wandb=False)
entry = reg.register(
    name="encoder",
    source_path="models/encoder/best_model.pt",
    metrics={"val_loss": 0.123, "knn_accuracy": 0.81},
    tags={"stage": "staging", "dataset": "v3"},
)
reg.promote("encoder", entry.version, stage="production")
```

Every entry has a SHA-256 sidecar so integrity is verifiable without
loading torch.

## Checkpoint integrity

`save_with_hash` writes three files:

* `path`              -- the torch state dict.
* `path.sha256`       -- `<digest>  <basename>` textual checksum.
* `path.json`         -- metadata (timestamp, torch/python versions,
                         user-supplied keys).

`load_verified` re-hashes the checkpoint, compares against the sidecar
(or an explicit `expected_hash` argument), and then calls
`torch.load(..., weights_only=True)` by default.  A mismatch raises
`ChecksumError`; an unreadable or tampered signature raises
`SignatureError`.

### Signing

`_verify_signature` in `i3/mlops/checkpoint.py` is a stub today: if
`<path>.sig` exists it is treated as "trust on first use" with a
warning.  Replace the body with your preferred sigstore / KMS verifier
in production deployments.

## ONNX export

```bash
python scripts/export/onnx.py \
    --encoder checkpoints/encoder/best.pt \
    --slm     checkpoints/slm/best.pt \
    --out     exports/ \
    --opset   17
```

The SLM exporter emits the **prefill-only** graph.  Token-by-token
decode requires stable `past_key_values` wiring and is exported as a
separate graph once that landed; see the comment in
`i3/slm/onnx_export.py` for details.

Parity is enforced via `np.allclose(atol=1e-4)` against PyTorch before
the file is considered valid.  CI should chain the verifier:

```bash
python scripts/verify_onnx.py --encoder exports/encoder.onnx --slm exports/slm.onnx
```

## Benchmarks and SLOs

`benchmarks/` uses `pytest-benchmark`.  Each benchmark runs 3 warmup
rounds + 20 measured rounds with `time.perf_counter`.  Targets live in
`benchmarks/slos.yaml` and must be respected by CI.

End-to-end load testing uses Locust (`benchmarks/locustfile.py`) or
k6 (`benchmarks/k6/load.js`).  Both scripts ramp 10 users over 30 s,
sustain for 5 min, then ramp down for 30 s.

## DVC pipeline

`dvc.yaml` declares the four stages of the I3 training lifecycle:

1. `generate_synthetic` -- synthetic data generation.
2. `train_encoder`      -- TCN encoder training.
3. `train_slm`          -- Adaptive SLM training.
4. `evaluate`           -- end-to-end evaluation on the held-out split.

Reproduce the full pipeline with:

```bash
dvc repro
```

The default remote is `s3://i3-dvc-storage`; swap it in `.dvc/config`
for your preferred backend.

## Dependencies

Add the following optional dependency groups to `pyproject.toml`:

```toml
[tool.poetry.group.mlops.dependencies]
mlflow           = "^2.11"
onnx             = "^1.16"
onnxruntime      = "^1.17"
dvc              = { version = "^3.50", extras = ["s3"] }
pytest-benchmark = "^4.0"
locust           = "^2.24"
wandb            = { version = "^0.17", optional = true }
```

`hypothesis` is already covered by the dev group.
