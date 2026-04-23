"""CLI to analyse trained SAEs and produce the SAE analysis report.

Given the per-layer SAE checkpoints written by ``scripts/training/train_sae.py``,
this tool:

1. Re-loads each SAE from ``checkpoints/sae/layer_{i}.pt``.
2. Regenerates the synthetic (prompt, adaptation vector) dataset used
   at training time and re-collects the activation cache.
3. Runs :func:`compute_per_feature_semantics` and
   :func:`identify_monosemantic_features` for every layer.
4. Emits ``reports/sae_analysis_<ts>.md`` with the per-feature
   semantic table, the monosemantic-feature count per layer, the top-k
   features per :class:`AdaptationVector` dimension, and a decoder-
   column cosine-similarity heatmap summary.

Usage::

    python scripts/analyse_sae.py --seed 42 \\
        --sae-dir checkpoints/sae \\
        --report reports/sae_analysis.md
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.adaptation.types import AdaptationVector  # noqa: E402
from i3.eval.ablation_experiment import (  # noqa: E402
    canonical_archetypes,
    canonical_prompts,
)
from i3.interpretability.activation_cache import (  # noqa: E402
    ActivationCache,
)
from i3.interpretability.feature_attribution import ADAPTATION_DIMS  # noqa: E402
from i3.interpretability.sae_analysis import (  # noqa: E402
    compute_per_feature_semantics,
    decoder_cosine_similarity_matrix,
    identify_monosemantic_features,
    top_features_per_dimension,
)
from i3.interpretability.sparse_autoencoder import (  # noqa: E402
    FeatureDictionary,
    SparseAutoencoder,
)
from i3.slm.model import AdaptiveSLM  # noqa: E402
from i3.slm.tokenizer import SimpleTokenizer  # noqa: E402


logger = logging.getLogger("analyse_sae")


# ---------------------------------------------------------------------------
# Arg parsing.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Analyse trained SAEs and emit the semantics report.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae-dir", type=str, default="checkpoints/sae")
    parser.add_argument(
        "--report",
        type=str,
        default=f"reports/sae_analysis_{ts}.md",
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-cross-heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument(
        "--cos-heatmap-features",
        type=int,
        default=32,
        help="Number of leading features summarised in the cosine heatmap.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
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
    model.eval()
    return model


def _build_records(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[AdaptationVector]]:
    """Rebuild the deterministic (prompt, adaptation) dataset."""
    prompts = list(canonical_prompts())
    archetypes = canonical_archetypes()

    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(prompts)

    records: list[dict[str, Any]] = []
    adapt_per_record: list[AdaptationVector] = []
    for prompt in prompts:
        ids = tokenizer.encode(
            prompt,
            add_special=True,
            max_length=args.seq_len,
            padding=True,
        )
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        for _name, adapt_vec in archetypes.items():
            adapt_t = adapt_vec.to_tensor().unsqueeze(0)
            user_t = torch.zeros(1, 64)
            records.append(
                {
                    "input_ids": ids_tensor,
                    "adaptation_vector": adapt_t,
                    "user_state": user_t,
                }
            )
            adapt_per_record.append(adapt_vec)
    return records, adapt_per_record


def _load_sae(path: Path) -> SparseAutoencoder:
    """Instantiate a :class:`SparseAutoencoder` from a saved checkpoint."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a valid SAE checkpoint")
    sae = SparseAutoencoder(
        d_model=int(payload["d_model"]),
        d_dict=int(payload["d_dict"]),
        sparsity_coef=float(payload.get("sparsity_coef", 1e-3)),
        normalise_input=bool(payload.get("normalise_input", True)),
        tied_weights=bool(payload.get("tied_weights", False)),
    )
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def _adapt_iter_per_sample(
    adapt_per_record: list[AdaptationVector],
    activations_per_record_avg_rows: int,
) -> list[AdaptationVector]:
    """Repeat each record's adaptation vector for its captured-row count.

    Activation capture flattens ``[batch, seq, d_model]`` to
    ``[batch*seq, d_model]``; the adaptation vector is shared across the
    sequence so we repeat it the same number of times.
    """
    repeated: list[AdaptationVector] = []
    for v in adapt_per_record:
        for _ in range(activations_per_record_avg_rows):
            repeated.append(v)
    return repeated


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def _write_report(
    args: argparse.Namespace,
    per_layer_analysis: dict[int, dict[str, Any]],
    git_sha: str,
) -> None:
    ts = datetime.now(tz=timezone.utc).isoformat()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# SAE analysis report\n")
    lines.append(f"- timestamp: `{ts}`\n")
    lines.append(f"- git sha: `{git_sha}`\n")
    lines.append(f"- seed: `{args.seed}`\n")
    lines.append(f"- monosemanticity threshold: `{args.threshold}`\n")
    lines.append(f"- top-k per dimension: `{args.top_k}`\n\n")

    lines.append("## Monosemantic feature count per layer\n\n")
    lines.append("| layer | d_dict | monosemantic | fraction |\n")
    lines.append("|---|---|---|---|\n")
    for layer_idx in sorted(per_layer_analysis.keys()):
        info = per_layer_analysis[layer_idx]
        total = int(info["d_dict"])
        mono = int(info["monosemantic_count"])
        frac = (mono / total) if total else 0.0
        lines.append(
            f"| {layer_idx} | {total} | {mono} | {frac:.3f} |\n"
        )

    for layer_idx in sorted(per_layer_analysis.keys()):
        info = per_layer_analysis[layer_idx]
        lines.append(f"\n## Layer {layer_idx}\n\n")
        lines.append("### Top features per AdaptationVector dimension\n\n")
        lines.append("| dimension | top features (idx, |r|) |\n")
        lines.append("|---|---|\n")
        per_dim_top = info["per_dim_top"]
        for dim in ADAPTATION_DIMS:
            entries = per_dim_top.get(dim, [])
            rendered = ", ".join(
                f"({idx}, {abs(r):.2f})" for idx, r in entries
            )
            lines.append(f"| `{dim}` | {rendered} |\n")

        lines.append("\n### Monosemantic features (|r| >= threshold)\n\n")
        mono_entries = info["monosemantic_entries"]
        if not mono_entries:
            lines.append("*No features cleared the threshold at this layer.*\n")
        else:
            lines.append("| feature_idx | label | top correlations | sparsity | mean_act |\n")
            lines.append("|---|---|---|---|---|\n")
            for sem in mono_entries[: max(10, args.top_k * 2)]:
                top_str = "; ".join(
                    f"{name}={r:+.2f}"
                    for name, r in sem["top_dimension_correlations"]
                )
                lines.append(
                    f"| {sem['feature_idx']} | `{sem['dimension_label'] or '-'}` "
                    f"| {top_str} | {sem['sparsity']:.2f} | "
                    f"{sem['mean_activation']:.3f} |\n"
                )

        lines.append(
            "\n### Decoder-column cosine heatmap summary "
            f"(first {args.cos_heatmap_features} features)\n\n"
        )
        cos = info["cos_summary"]
        lines.append(
            f"- off-diagonal mean cosine: `{cos['mean_off_diag']:.3f}`\n"
        )
        lines.append(
            f"- off-diagonal max cosine: `{cos['max_off_diag']:.3f}`\n"
        )
        lines.append(
            f"- fraction |cos| > 0.5: `{cos['high_sim_fraction']:.3f}`\n"
        )

    lines.append("\n## Citations\n\n")
    lines.append(
        "- Bricken, T. et al. (2023). *Towards Monosemanticity.*\n"
        "- Templeton, A. et al. (2024). *Scaling Monosemanticity.*\n"
        "- Cunningham, H. et al. (2023). *Sparse Autoencoders Find Highly "
        "Interpretable Features in Language Models.*\n"
        "- Turner, A. et al. (2023). *Activation Addition.*\n"
    )
    report_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Analysis report written to %s", report_path)


# ---------------------------------------------------------------------------
# Core pipeline.
# ---------------------------------------------------------------------------


def _analyse_layer(
    layer_idx: int,
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    adapt_per_row: list[AdaptationVector],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Produce the analysis payload for a single layer."""
    semantics = compute_per_feature_semantics(
        sae,
        activations,
        adapt_per_row,
        monosemanticity_threshold=args.threshold,
    )
    mono = identify_monosemantic_features(semantics, threshold=args.threshold)
    per_dim_top = top_features_per_dimension(semantics, k=args.top_k)

    dictionary = FeatureDictionary(sae=sae, activations=activations)
    cos = decoder_cosine_similarity_matrix(
        dictionary, max_features=args.cos_heatmap_features
    )
    m = cos.size(0)
    off_diag_mask = ~torch.eye(m, dtype=torch.bool)
    off = cos[off_diag_mask]
    cos_summary = {
        "mean_off_diag": float(off.mean().item()) if off.numel() else 0.0,
        "max_off_diag": float(off.abs().max().item()) if off.numel() else 0.0,
        "high_sim_fraction": (
            float((off.abs() > 0.5).float().mean().item())
            if off.numel()
            else 0.0
        ),
    }

    return {
        "d_dict": sae.d_dict,
        "monosemantic_count": len(mono),
        "monosemantic_entries": [sem.model_dump() for sem in mono],
        "per_dim_top": per_dim_top,
        "cos_summary": cos_summary,
    }


def _record_iterator(records: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield from records


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sae_dir = Path(args.sae_dir)
    if not sae_dir.exists():
        logger.error("SAE directory %s does not exist", sae_dir)
        return 1

    model = _build_model(args)
    records, adapt_per_record = _build_records(args)

    cache = ActivationCache(max_samples=args.max_samples)
    for i, layer in enumerate(model.layers):
        cache.register(layer.cross_attn, f"cross_attn_{i}")
    counts = cache.collect(model, _record_iterator(records), max_samples=args.max_samples)
    logger.info("Captured row counts: %s", counts)

    per_layer_analysis: dict[int, dict[str, Any]] = {}
    for i in range(len(model.layers)):
        ckpt_path = sae_dir / f"layer_{i}.pt"
        if not ckpt_path.exists():
            logger.warning("checkpoint missing for layer %d: %s", i, ckpt_path)
            continue
        try:
            sae = _load_sae(ckpt_path)
        except (ValueError, RuntimeError) as exc:
            logger.error("failed to load %s: %s", ckpt_path, exc)
            continue

        try:
            activations = cache.get(f"cross_attn_{i}")
        except KeyError:
            logger.warning("no activations for layer %d", i)
            continue

        # Align adaptation vectors to the flattened sample rows. Since the
        # cache flattens ``[batch, seq, d]`` into rows, the per-sample
        # adaptation vector is repeated ``seq_len`` times per record. The
        # cache may also have truncated trailing rows against max_samples,
        # so we truncate the adaptation list to match.
        rows_per_record = max(1, activations.size(0) // max(1, len(records)))
        adapt_per_row = _adapt_iter_per_sample(adapt_per_record, rows_per_record)
        adapt_per_row = adapt_per_row[: activations.size(0)]
        if len(adapt_per_row) != activations.size(0):
            # Final-fallback alignment: pad with the first record's vector.
            while len(adapt_per_row) < activations.size(0):
                adapt_per_row.append(adapt_per_record[0])

        per_layer_analysis[i] = _analyse_layer(
            i, sae, activations, adapt_per_row, args
        )

    _write_report(args, per_layer_analysis, _git_sha(_ROOT))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
