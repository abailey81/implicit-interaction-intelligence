# Troubleshooting

Symptom-first fixes for the most common problems encountered in
development, deployment, and day-2 operations.

!!! tip "For production incidents"
    See the [Runbook](runbook.md) — this page is for ordinary breakage.

## Install / setup { #install }

### `ModuleNotFoundError: No module named 'torch'`

Poetry did not install the Torch wheel because the lock file and lock
resolver disagreed.

```bash
poetry lock --no-update
poetry install --only main
```

If the problem persists on Windows, install Torch manually with the
CPU wheel:

```bash
poetry run pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
```

### `RuntimeError: Cannot find a working triton installation`

You enabled `torch.compile()` on Windows.  Triton has no supported
Windows wheel today
([triton-lang/triton#1640](https://github.com/triton-lang/triton/issues/1640)).

The orchestrator auto-detects this and skips compile gracefully on
modern versions (you'll see `torch.compile skipped: Triton not
available (common on Windows).  AMP + TF32 still active.` in the
stage log).  If you need the 1.2–1.6× compile speed-up anyway, run
the project from WSL2 Ubuntu — see [WSL2 on Windows](wsl.md).

```powershell
# Disable compile on native Windows
poetry run python training/train_slm.py --compile off
# Or run the orchestrator with --compile off (also accepted by
# train_encoder.py).  AMP + TF32 + cuDNN benchmark + bigger batch
# + prefetch all remain active.
```

### CUDA wheel missing for torch ≥ 2.6 on Windows

PyTorch's `cu121` wheel stream stopped at torch 2.5.1.  For torch
≥ 2.6 with CUDA, use `cu124` (or `cu126` for very new torch):

```powershell
poetry run pip uninstall -y torch
poetry run pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

The driver on your Windows host (`nvidia-smi` should report CUDA 12.x)
is forward-compatible — you do not need to match the wheel's minor
version to the driver exactly.

### `cryptography.fernet.InvalidToken` on startup

Your `I3_ENCRYPTION_KEY` does not match the key used to write the
existing SQLite data.

```bash
# Option 1 — you still have the old key
export I3_ENCRYPTION_KEY="old,new"      # MultiFernet tries old first
python -m i3.privacy.encryption --rotate

# Option 2 — you do not have the old key
rm data/profiles.sqlite                  # data loss; rebuild from next session
python scripts/security/generate_encryption_key.py
```

### `RuntimeError: Fernet key must be 32 url-safe base64-encoded bytes`

`I3_ENCRYPTION_KEY` is missing, empty, or malformed. Regenerate:

```bash
python scripts/security/generate_encryption_key.py
```

### `ValidationError: Config is frozen`

You tried to mutate an `I3Config` at runtime. Rebuild instead:

```python
new = cfg.model_copy(update={"router": cfg.router.model_copy(update={"cold_start_n": 10})})
```

See [Configuration § Immutability](../getting-started/configuration.md#immutability).

## Runtime { #runtime }

### `/health` returns 503 "Service unavailable"

The FastAPI lifespan initialisation failed (usually a config or checkpoint
load error). Inspect:

```bash
kubectl -n i3 logs deploy/i3 --tail=200
# or, for systemd
journalctl -u i3.service -n 200
```

Common causes:

| Message | Cause |
|:--------|:------|
| `FileNotFoundError: checkpoints/slm.pt` | Checkpoint volume not mounted |
| `ValidationError: server.cors_origins` | Empty / unset `I3_CORS_ORIGINS` |
| `InvalidToken` | Fernet key mismatch (see above) |

### REST returns `422 Unprocessable Entity`

Your request failed Pydantic validation. `detail` field names the
offending path. Common offenders:

- `limit` outside `[1, 100]`.
- `user_id` containing `/`, `.`, or whitespace.

### WebSocket closes immediately with `1008`

One of:

- `Origin` header not in the allow-list. Set `I3_CORS_ORIGINS` or serve
  the page from an allowed origin.
- `user_id` fails the regex `^[a-zA-Z0-9_-]{1,64}$`.

### WebSocket closes with `1009` mid-session

A client frame exceeded 64 KiB (or message text > 8 KiB). Split the
message or reduce keystroke buffering on the client.

### `/metrics` returns 404

`ObservabilityConfig.metrics_enabled = false`, or the middleware was
disabled. Check `configs/default.yaml::observability`.

## Performance { #perf }

### Generation latency > 300 ms P95

1. Confirm INT8 quantization is on:
   ```bash
   curl -s .../metrics | grep i3_slm_quantized
   # Expect 1
   ```
2. CPU throttled? In Kubernetes, check CFS throttling:
   ```bash
   kubectl -n i3 top pod
   ```
3. Other workloads on the node?
4. Is `max_seq_len` too large for your use case? Reduce in config.

### Memory keeps growing

1. Confirm `_MAX_KEYSTROKE_BUFFER` is enforced (check `tests/test_pipeline.py`).
2. Profiles not being evicted? Check `user_model.lru_cache_size`.
3. Torch grad enabled accidentally? Ensure `torch.no_grad()` wraps
   inference paths.

## Training { #training }

### NT-Xent loss collapses to zero

Usually the positives are indistinguishable from negatives at
\(\tau = 0.07\). Check:

- Augmentation is actually augmenting (set `--save-augmented-batch` to
  dump a batch).
- `batch_size * 2` distinct sessions — the batch collation must not
  duplicate.

### SLM training NaNs

Pre-LN models are stable but not bulletproof. Usually:

- Learning rate too high — reduce by 2×.
- Positional encodings wrong at the first step — confirm
  `embeddings.py::PositionalEncoding` is seeded and not re-initialised
  between steps.
- Gradient clipping missing — ensure `torch.nn.utils.clip_grad_norm_`
  with `max_norm=1.0`.

### Perplexity floor above 25

- Vocabulary coverage — check UNK rate on the training set; if > 3 %,
  retrain the tokenizer.
- Too-aggressive PII redaction replaced useful tokens with sentinel —
  inspect the sanitizer's output on a sample.
- Encoder frozen when it should not be (epoch > 1).

## Docker / Kubernetes { #k8s }

### Pod stuck `CrashLoopBackOff`

```bash
kubectl -n i3 describe pod <name>
kubectl -n i3 logs      <name> --previous
```

Look for:

- OOMKilled → raise `resources.limits.memory`.
- Secret not mounted → check `i3-secrets` exists.
- Probe failing → raise `startupProbe.failureThreshold` for the first
  load of the checkpoint.

### `503` at the ingress, pods healthy

- `Service` selector mismatch — `app: i3` everywhere.
- Sticky session misconfigured — verify cookie name and TTL.
- Backend draining — `kubectl rollout status`.

### Cloud egress failing

The `NetworkPolicy` egress block must allow `0.0.0.0/0:443` to reach the
Anthropic API. Check the manifest's `egress` list.

## Development loop { #dev }

### `pre-commit` complains about line length

`ruff` line length is 100. If a single string must exceed it, use
`# noqa: E501` sparingly — don't blanket-exempt files.

### `pytest` hangs

Usually an `async` test forgot to use `pytest-asyncio`. Ensure the
test function is `async def test_…` and the module has no sync-mode
markers.

### `mkdocs serve` shows broken links

Run with `strict: true` in CI to fail the build on any broken anchor:

```bash
poetry run mkdocs build --strict
```

## When to escalate { #escalate }

Escalate to the maintainer immediately if:

- The privacy auditor triggered at any time.
- An unencrypted `UserProfile` ever landed on disk.
- A cloud response was logged verbatim.
- An unauthorised client reached `/demo/reset` in production.

See the [Runbook](runbook.md) and [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md).
