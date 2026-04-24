# WSL2 Setup on Windows

I³ runs on Windows natively, but some GPU features — notably
`torch.compile()` via the Triton backend — have no supported Windows
build today (see
[triton-lang/triton#1640](https://github.com/triton-lang/triton/issues/1640)).
Running the project inside WSL2 gives you a real Linux environment
with full Triton support while still using your Windows NVIDIA driver.

This page covers **setup, usage, performance, and gotchas**. It
assumes Windows 10 build 19041+ or Windows 11.

---

## What you gain by using WSL2

| Capability | Native Windows | WSL2 Ubuntu |
|---|---|---|
| AMP (fp16 / bf16 autocast) | ✓ | ✓ |
| TF32 fast matmul | ✓ | ✓ |
| cuDNN benchmark | ✓ | ✓ |
| Enlarged batch sizes on CUDA | ✓ | ✓ |
| DataLoader prefetching | ✓ | ✓ |
| **`torch.compile()` (Triton)** | **✗** | **✓** |
| **GNU `make` on PATH** | ✗ (use `scoop`/`choco`) | **✓** |
| **ext4 filesystem speed** | — | ✓ (if project cloned inside WSL, not on `/mnt/d/`) |
| Bash / zsh shell | — | ✓ |
| `apt` package manager | — | ✓ |

The one tradeoff is that running from `/mnt/d/` (the Windows disk) is
slower at small-file I/O than WSL's native ext4. See
[Where to put the project files](#where-to-put-the-project-files) below.

---

## Quick install

### 1. Enable WSL2 + install Ubuntu

From an **admin** PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
```

On Windows 11 this is one command; no reboot needed on fresh systems
because the Virtual Machine Platform feature is already on. Older
builds may reboot once.

Verify:

```powershell
wsl --list --verbose
```

You should see:

```
  NAME            STATE           VERSION
  Ubuntu-22.04    Running         2
  docker-desktop  Running         2
```

### 2. Confirm GPU passthrough

The NVIDIA driver on the Windows side (≥ 496.76) exposes CUDA inside
WSL automatically. No separate Linux CUDA install is required.

```powershell
wsl -d Ubuntu-22.04 -u root -- nvidia-smi | head -5
```

You should see your GPU listed with the same driver version as on
Windows. If you see "command not found" or CUDA errors, update your
Windows NVIDIA driver and reinstall WSL.

### 3. Install Python 3.12 + project deps

Project files can stay on `D:` — WSL mounts Windows drives at `/mnt/`.
This means your existing checkpoints, synthetic data, tokenizers,
and `.env` are all preserved; you just build a second venv alongside
the Windows one.

```powershell
wsl -d Ubuntu-22.04 -u root -- bash -c "
  apt-get update -qq &&
  apt-get install -y -qq software-properties-common &&
  add-apt-repository -y ppa:deadsnakes/ppa &&
  apt-get update -qq &&
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev \
                        build-essential git make
"
```

```powershell
wsl -d Ubuntu-22.04 -u root -- bash -c "
  cd /mnt/d/implicit-interaction-intelligence &&
  python3.12 -m venv .venv-wsl &&
  .venv-wsl/bin/pip install --upgrade pip poetry &&
  .venv-wsl/bin/poetry config virtualenvs.create false &&
  .venv-wsl/bin/poetry install --with dev,security
"
```

The `.venv-wsl` suffix keeps the Linux venv separate from the Windows
`.venv` so you can jump between them without reinstalling either.

### 4. Verify CUDA + Triton

```powershell
wsl -d Ubuntu-22.04 -u root -- bash -c "
  cd /mnt/d/implicit-interaction-intelligence &&
  .venv-wsl/bin/python -c '
    import torch, triton
    print(\"torch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available(), \"GPU:\", torch.cuda.get_device_name(0))
    print(\"triton:\", triton.__version__)
    m = torch.compile(torch.nn.Linear(10, 10).cuda())
    out = m(torch.randn(2, 10).cuda())
    print(\"torch.compile OK:\", out.shape)
  '
"
```

Expected:

```
torch: 2.6.0+cu124   CUDA: True   GPU: NVIDIA GeForce RTX 4050 Laptop GPU
triton: 3.x.x
torch.compile OK: torch.Size([2, 10])
```

---

## Running the project inside WSL

### One-shot run from PowerShell

```powershell
wsl -d Ubuntu-22.04 -u root -- bash -c "
  cd /mnt/d/implicit-interaction-intelligence &&
  .venv-wsl/bin/python scripts/run_everything.py --mode full --with-docker
"
```

### Long interactive session

Drop into a shell and stay there:

```powershell
wsl -d Ubuntu-22.04
```

Then:

```bash
cd /mnt/d/implicit-interaction-intelligence
source .venv-wsl/bin/activate
python scripts/run_everything.py --mode full --with-docker
```

---

## Performance tips

### Where to put the project files

| Path | Location | I/O speed | Syncs with Windows? |
|---|---|---|---|
| `/mnt/d/implicit-interaction-intelligence` | Windows NTFS (mounted) | slower (~40% of native) | yes — your Windows `.venv` sees the same files |
| `/root/implicit-interaction-intelligence` (WSL-native) | ext4 | fastest | no — git clone separately |
| `/home/<user>/implicit-interaction-intelligence` | ext4 | fastest | no |

**Recommendation for dev:** keep the repo on `D:` (so you can edit in
Windows VS Code and run in WSL) — the I/O hit is only noticeable on
the very first data pipeline pass.

**Recommendation for benchmarking:** clone a separate copy into WSL's
ext4 home and run there — removes the mount overhead.

### Sidecar containers — reuse your Windows Docker Desktop

Don't run a second Docker inside WSL2. Docker Desktop for Windows
already exposes its daemon to WSL distros. Just verify:

```bash
# Inside WSL
docker ps
```

If you see the same containers you see in Windows, you're good. Your
Prometheus / OTel sidecars started on Windows stay reachable from
WSL via `host.docker.internal`.

### `.env` + caches

The orchestrator's `.env` auto-loader works identically. Cache env
vars defined on Windows (`POETRY_CACHE_DIR`, `HF_HOME`, etc.) are
**not** inherited by WSL — set them inside WSL if you want the same
layout:

```bash
# Inside WSL, in ~/.bashrc
export HF_HOME=/mnt/d/caches/huggingface
export TORCH_HOME=/mnt/d/caches/torch
```

Sharing caches between Windows and WSL through `/mnt/d/` avoids
re-downloading multi-GB model weights.

---

## Gotchas

| Symptom | Fix |
|---|---|
| `wsl --install -d Ubuntu-22.04` fails with "invalid argument" | Enable Virtual Machine Platform and WSL in Windows Features, then `wsl --set-default-version 2`, then retry. |
| `nvidia-smi` missing inside WSL | Upgrade Windows NVIDIA driver to 496.76+. Don't install CUDA Toolkit inside WSL. |
| Slow file ops on `/mnt/d/` | Clone into ext4 home (`/root/…`) or accept the ~40% penalty — it's dominated by small-file I/O only. |
| `torch.compile` first call extremely slow | Normal — Inductor compiles once per shape and caches. Subsequent calls hit the Triton kernel cache in `~/.triton/`. |
| Two `.venv` directories confusing | `.venv` = Windows; `.venv-wsl` = Linux. Activate whichever matches your shell. |
| Docker builds fail with "cannot connect to daemon" | Docker Desktop setting: Resources → WSL Integration → enable for Ubuntu-22.04. |

---

## Shutting WSL down cleanly

```powershell
wsl --shutdown
```

Frees the VM memory. Next command that targets a distro starts a
fresh kernel (~1 s cold start).

---

## Related reading

- [`README.md` → Installation](../../README.md#installation) — covers
  the native Windows path.
- [`docs/operations/troubleshooting.md`](troubleshooting.md) — general
  issues that aren't WSL-specific.
- [Microsoft's WSL docs](https://learn.microsoft.com/en-us/windows/wsl/install)
  — authoritative reference for WSL itself.
