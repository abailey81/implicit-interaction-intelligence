"""CLI to train one SAE per cross-attention block of the I3 SLM.

For each :class:`~i3.slm.transformer.AdaptiveTransformerBlock` layer of a
loaded :class:`~i3.slm.model.AdaptiveSLM`, the script:

1. Collects residual-stream activations on a synthetic dataset that
   pairs the 50 canonical prompts from Batch A with the 8 archetype
   adaptation vectors — 400 (prompt, adaptation) combinations.
2. Trains a :class:`SparseAutoencoder` with ``d_dict = 8 * d_model`` on
   the cached activations.
3. Saves each SAE to ``checkpoints/sae/layer_{i}.pt``.
4. Emits a Markdown report at ``reports/sae_training_<ts>.md`` with the
   per-layer training summary.

The script tolerates the absence of a trained SLM checkpoint: if
``I3_CHECKPOINT_PATH`` is unset (or the file does not exist) the SAE is
trained on a random-init SLM, which matches the Batch B precedent of
measuring *architectural capacity* rather than learned behaviour.

Usage::

    python scripts/train_sae.py --seed 42 --epochs 50 \\
        --out-dir checkpoints/sae --report reports/sae_training.md
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

# Allow running the script directly without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.eval.ablation_experiment import (  # noqa: E402
    canonical_archetypes,
    canonical_prompts,
)
from i3.interpretability.activation_cache import (  # noqa: E402
    ActivationCache,
)
from i3.interpretability.sparse_autoencoder import (  # noqa: E402
    SAETrainer,
    SparseAutoencoder,
)
from i3.slm.model import AdaptiveSLM  # noqa: E402
from i3.slm.tokenizer import SimpleTokenizer  # noqa: E402


logger = logging.getLogger("train_sae")


# ---------------------------------------------------------------------------
# Arg parsing.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SAE training pipeline."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Train one sparse autoencoder per cross-attention block.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity-coef", type=float, default=1e-3)
    parser.add_argument(
        "--d-dict-mult",
        type=int,
        default=8,
        help="Overcomplete-basis multiplier; d_dict = mult * d_model.",
    )
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-cross-heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4096,
        help="Per-layer cap on cached activation rows.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/sae",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=f"reports/sae_training_{ts}.md",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    """Return HEAD SHA or ``"unknown"`` when git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


def _build_model(args: argparse.Namespace) -> AdaptiveSLM:
    """Instantiate an SLM, loading a checkpoint if one is available."""
    torch.manual_seed(args.seed)
    model = AdaptiveSLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 2,
        max_seq_len=max(32, args.seq_len),
        n_cross_heads=args.n_cross_heads,
    )

    ckpt_env = os.environ.get("I3_CHECKPOINT_PATH")
    if ckpt_env:
        ckpt_path = Path(ckpt_env)
        if ckpt_path.is_file():
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                logger.info("Loaded SLM checkpoint from %s", ckpt_path)
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                logger.warning(
                    "Failed to load checkpoint %s (%s) -- using random init",
                    ckpt_path,
                    exc,
                )
        else:
            logger.warning(
                "I3_CHECKPOINT_PATH=%s does not exist -- using random init",
                ckpt_env,
            )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Synthetic dataset builder.
# ---------------------------------------------------------------------------


def _build_dataset(
    model: AdaptiveSLM,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Cartesian product of the 50 canonical prompts × 8 archetypes.

    Uses a small word-level tokenizer fitted over the prompts; prompts
    longer than ``--seq-len`` are truncated with the canonical pad/unk
    handling that :class:`SimpleTokenizer` already provides.
    """
    prompts = list(canonical_prompts())
    archetypes = canonical_archetypes()

    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(prompts)

    records: list[dict[str, Any]] = []
    for prompt in prompts:
        ids = tokenizer.encode(
            prompt,
            add_special=True,
            max_length=args.seq_len,
            padding=True,
        )
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        for _arche_name, adapt_vec in archetypes.items():
            adapt_t = adapt_vec.to_tensor().unsqueeze(0)
            user_t = torch.zeros(1, 64)
            records.append(
                {
                    "input_ids": ids_tensor,
                    "adaptation_vector": adapt_t,
                    "user_state": user_t,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------


def _train_per_layer(
    model: AdaptiveSLM,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    """Collect activations and train one SAE per cross-attention block."""
    cache = ActivationCache(max_samples=args.max_samples)
    for i, layer in enumerate(model.layers):
        cache.register(layer.cross_attn, f"cross_attn_{i}")

    logger.info("Collecting activations across %d prompt/adapt pairs", len(records))
    counts = cache.collect(model, iter(records), max_samples=args.max_samples)
    logger.info("Captured row counts: %s", counts)

    trainer = SAETrainer(seed=args.seed)
    d_dict = args.d_dict_mult * args.d_model

    results: dict[int, dict[str, Any]] = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(model.layers)):
        layer_name = f"cross_attn_{i}"
        try:
            activations = cache.get(layer_name)
        except KeyError:
            logger.warning("no activations for %s; skipping", layer_name)
            continue

        sae = SparseAutoencoder(
            d_model=args.d_model,
            d_dict=d_dict,
            sparsity_coef=args.sparsity_coef,
        )

        if not args.no_compile and hasattr(torch, "compile"):
            try:
                sae = torch.compile(sae)  # type: ignore[assignment]
            except (RuntimeError, TypeError) as exc:  # pragma: no cover - env-dependent
                logger.warning("torch.compile failed (%s) -- proceeding uncompiled", exc)

        trained, report = trainer.fit(
            activations=activations,
            sae=sae if isinstance(sae, SparseAutoencoder) else None,
            d_dict=d_dict,
            sparsity_coef=args.sparsity_coef,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

        ckpt_path = out_dir / f"layer_{i}.pt"
        torch.save(
            {
                "state_dict": trained.state_dict(),
                "d_model": trained.d_model,
                "d_dict": trained.d_dict,
                "sparsity_coef": trained.sparsity_coef,
                "normalise_input": trained.normalise_input,
                "tied_weights": trained.tied_weights,
                "layer_name": layer_name,
            },
            ckpt_path,
        )
        results[i] = {
            "layer_name": layer_name,
            "checkpoint": str(ckpt_path),
            "n_samples": int(activations.size(0)),
            "initial_loss": report.initial_loss,
            "final_loss": report.final_loss,
            "final_mse": report.final_reconstruction_mse,
            "final_sparsity": report.final_mean_sparsity,
            "loss_history": report.loss_history,
        }
        logger.info(
            "layer %d: loss %.4f -> %.4f, sparsity %.3f",
            i,
            report.initial_loss,
            report.final_loss,
            report.final_mean_sparsity,
        )
    return results


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def _write_report(
    args: argparse.Namespace,
    results: dict[int, dict[str, Any]],
    git_sha: str,
) -> None:
    """Write a Markdown summary of the run."""
    ts = datetime.now(tz=timezone.utc).isoformat()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# SAE training report\n")
    lines.append(f"- timestamp: `{ts}`\n")
    lines.append(f"- git sha: `{git_sha}`\n")
    lines.append(f"- seed: `{args.seed}`\n")
    lines.append(f"- d_model: `{args.d_model}`\n")
    lines.append(f"- d_dict: `{args.d_dict_mult * args.d_model}`\n")
    lines.append(f"- sparsity_coef: `{args.sparsity_coef}`\n")
    lines.append(f"- epochs: `{args.epochs}`\n")
    lines.append(f"- batch_size: `{args.batch_size}`\n")
    lines.append(f"- lr: `{args.lr}`\n")
    lines.append("\n## Per-layer summary\n")
    lines.append("| layer | n_samples | initial_loss | final_loss | final_mse | sparsity | checkpoint |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        lines.append(
            f"| {layer_idx} | {r['n_samples']} | "
            f"{r['initial_loss']:.4f} | {r['final_loss']:.4f} | "
            f"{r['final_mse']:.4f} | {r['final_sparsity']:.3f} | "
            f"`{r['checkpoint']}` |\n"
        )
    lines.append("\n## Citations\n\n")
    lines.append(
        "- Bricken, T. et al. (2023). *Towards Monosemanticity.*\n"
        "- Templeton, A. et al. (2024). *Scaling Monosemanticity.*\n"
        "- Cunningham, H. et al. (2023). *Sparse Autoencoders Find Highly "
        "Interpretable Features in Language Models.*\n"
    )
    report_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the SAE training pipeline and return a process exit code."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = _build_model(args)
    records = _build_dataset(model, args)
    results = _train_per_layer(model, records, args)
    _write_report(args, results, _git_sha(_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
