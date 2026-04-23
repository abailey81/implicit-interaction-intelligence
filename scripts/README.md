# Scripts

Operator and researcher entry points. The top level holds shell scripts most
people run directly; subdirectories group the Python tooling by purpose so
unfamiliar readers can narrow their search quickly.

## Top level — common entry points

| File | What it does |
|---|---|
| [`setup.sh`](setup.sh) | One-shot developer onboarding (Poetry install + pre-commit hooks). |
| [`uv_bootstrap.sh`](uv_bootstrap.sh) | Alternate onboarding for [`uv`](https://github.com/astral-sh/uv) users. |
| [`run_all_tests.sh`](run_all_tests.sh) | Full pytest suite with coverage. |
| [`run_demo.sh`](run_demo.sh) | Start the FastAPI server + open the demo UI. |
| [`run_preflight.sh`](run_preflight.sh) | CI preflight: lint, types, unit tests. |
| [`verify_all.py`](verify_all.py) | Main verification harness (46 checks across 7 categories). |
| [`verify_onnx.py`](verify_onnx.py) | Verify exported ONNX artefacts are byte-identical to the PyTorch source. |
| [`verify_reproducibility.sh`](verify_reproducibility.sh) | Re-run a training config and compare artefact hashes. |

## Subdirectories

| Directory | Purpose |
|---|---|
| [`benchmarks/`](benchmarks/) | Latency, throughput, and edge-profiling micro-benchmarks. |
| [`demos/`](demos/) | Standalone feature demos (TTS, PPG/HRV, EWC, federated, guardrails, multimodal, MCP, …). |
| [`experiments/`](experiments/) | Research runs — ablation, closed-loop eval, DPO, LLM-as-judge, interpretability, fairness, sparse-autoencoder analysis. |
| [`export/`](export/) | Model/data export tooling (ONNX, ExecuTorch, ExecuTorch-all-runtimes, diary→Parquet, GDPR user export). |
| [`security/`](security/) | Red-team runner, Sigstore model signing, encryption-key generation. |
| [`training/`](training/) | Training entry points (SAE, MLflow-tracked runs). |
| [`verification/`](verification/) | The 46 registered checks invoked by `verify_all.py` (code integrity, config, runtime, providers, infra, interview, security). |

## Conventions

- **One script, one job.** Every script exposes an `argparse` CLI with a
  short `--help`. No script mutates repository state; all writes go under
  `reports/`, `checkpoints/`, or paths you pass on the command line.
- **Importable.** Script bodies sit under `if __name__ == "__main__":` so
  their helpers can be imported from tests without side-effects.
- **Failing loudly.** Non-zero exit codes for any precondition failure;
  never swallow exceptions at the outer layer.
- **Dependency-light at import time.** Heavy deps (`torch`, `transformers`,
  `dspy`, …) are imported lazily inside the relevant function so
  `--help` never requires a full environment.

## Examples

```bash
# Validate the whole repo (46 checks, strict mode)
python scripts/verify_all.py --strict

# Run the red-team harness on torch-free surfaces
python scripts/security/run_redteam_notorch.py --targets sanitizer,pddl,guardrails

# Export the encoder + SLM to ONNX
python scripts/export/onnx.py --output web/models/

# Run the closed-loop evaluation experiment
python scripts/experiments/closed_loop_eval.py --config configs/default.yaml
```
