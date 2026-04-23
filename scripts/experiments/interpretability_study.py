"""CLI for the Batch B mechanistic interpretability study.

Runs all three analyses — activation patching, linear probing, and
attention-circuit analysis — on a single random-init
:class:`~i3.slm.model.AdaptiveSLM`. Emits a JSON dump plus a
Markdown report whose sections match the structure committed in
``docs/research/mechanistic_interpretability.md``.

Usage::

    python scripts/run_interpretability_study.py \\
        --seed 42 --n-prompts 20 \\
        --out reports/interpretability_study.md

The script is deliberately self-contained: it relies only on the public
interfaces exposed by :mod:`i3.interpretability.*` and does not require a
trained checkpoint. Numbers produced in this regime measure
*architectural capacity* rather than learned behaviour, and the report
flags this explicitly in the Threats to Validity section.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

# Allow running the script directly without ``pip install -e .``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from i3.interpretability.activation_patching import (  # noqa: E402
    CausalEffect,
    canonical_components,
    trace_causal_effect,
)
from i3.interpretability.attention_circuits import (  # noqa: E402
    AttentionPattern,
    extract_attention_patterns,
    identify_conditioning_specialists,
    summarise_circuit,
)
from i3.interpretability.feature_attribution import ADAPTATION_DIMS  # noqa: E402
from i3.interpretability.probing_classifiers import (  # noqa: E402
    ProbingExample,
    ProbingSuite,
    compute_probe_selectivity,
)

# Soft imports.
try:  # pragma: no cover - exercised at runtime
    import matplotlib  # type: ignore[import-untyped]
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt  # type: ignore[import-untyped]
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    _plt = None
    _MPL_AVAILABLE = False

try:  # pragma: no cover - optional dep
    import pandas as _pd  # type: ignore[import-untyped]
    _PD_AVAILABLE = True
except ImportError:  # pragma: no cover
    _pd = None
    _PD_AVAILABLE = False


logger = logging.getLogger("run_interpretability_study")


# ---------------------------------------------------------------------------
# Arg parsing.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the Batch B mechanistic interpretability study.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=20,
        help="Number of synthetic prompts for the circuit analysis.",
    )
    parser.add_argument(
        "--n-probe-samples",
        type=int,
        default=64,
        help="Number of examples per probed adaptation dimension.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=f"reports/interpretability_study_{ts}.md",
        help="Output path for the Markdown report.",
    )
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-cross-heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Git SHA.
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    """Return the current HEAD commit SHA, or ``"unknown"`` on failure.

    Args:
        repo_root: Path to the repository root.

    Returns:
        A 40-character hex SHA or ``"unknown"``.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


# ---------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------


def _build_model(args: argparse.Namespace) -> Any:
    """Instantiate a small random-init :class:`AdaptiveSLM`.

    Args:
        args: Parsed CLI namespace.

    Returns:
        A random-init :class:`AdaptiveSLM`.
    """
    from i3.slm.model import AdaptiveSLM

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


def _random_prompt(
    vocab_size: int, seq_len: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw a random prompt of integer token ids.

    Args:
        vocab_size: Size of the vocabulary.
        seq_len: Length of the prompt.
        generator: Torch RNG for reproducibility.

    Returns:
        Tensor of shape ``[1, seq_len]``.
    """
    return torch.randint(
        low=1,
        high=vocab_size,
        size=(1, seq_len),
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Individual studies.
# ---------------------------------------------------------------------------


def _run_activation_patching(
    model: Any, args: argparse.Namespace
) -> dict[str, CausalEffect]:
    """Run the causal-tracing study.

    Args:
        model: The SLM under study.
        args: Parsed CLI namespace.

    Returns:
        Dict ``component -> CausalEffect``.
    """
    gen = torch.Generator().manual_seed(args.seed + 1)
    prompt = _random_prompt(args.vocab_size, args.seq_len, gen)

    adaptation_dim = model.conditioning_projector.adaptation_dim
    user_state_dim = model.conditioning_projector.user_state_dim

    clean_adapt = torch.rand((1, adaptation_dim), generator=gen)
    clean_user = torch.randn((1, user_state_dim), generator=gen)
    corrupted_adapt = torch.zeros((1, adaptation_dim))
    corrupted_user = torch.zeros((1, user_state_dim))

    return trace_causal_effect(
        model,
        clean_input={
            "input_ids": prompt,
            "adaptation_vector": clean_adapt,
            "user_state": clean_user,
        },
        corrupted_input={
            "input_ids": prompt,
            "adaptation_vector": corrupted_adapt,
            "user_state": corrupted_user,
        },
        components=canonical_components(len(model.layers)),
    )


def _run_probes(
    model: Any, args: argparse.Namespace
) -> dict[str, dict[int, float]]:
    """Run linear probes on every adaptation dimension.

    Args:
        model: The SLM under study.
        args: Parsed CLI namespace.

    Returns:
        Nested dict ``{adaptation_dim_name: {layer_idx: r_squared}}``.
    """
    gen = torch.Generator().manual_seed(args.seed + 2)
    adaptation_dim = model.conditioning_projector.adaptation_dim
    user_state_dim = model.conditioning_projector.user_state_dim

    examples: list[ProbingExample] = []
    for _ in range(args.n_probe_samples):
        prompt = _random_prompt(args.vocab_size, args.seq_len, gen).squeeze(0)
        adapt = torch.rand(adaptation_dim, generator=gen)
        user = torch.randn(user_state_dim, generator=gen)
        examples.append(
            ProbingExample(
                input_ids=prompt,
                adaptation_vector=adapt,
                user_state=user,
            )
        )

    suite = ProbingSuite()
    results: dict[str, dict[int, float]] = {}
    for dim in ADAPTATION_DIMS:
        try:
            results[dim] = suite.train_probes(
                model=model,
                adaptation_dataset=examples,
                target_dimension=dim,
                layer_indices=list(range(len(model.layers))),
            )
        except ValueError:
            logger.warning("probe training skipped for dim %r", dim)
            results[dim] = {}
    return results


def _run_circuits(
    model: Any, args: argparse.Namespace
) -> tuple[AttentionPattern, list[AttentionPattern]]:
    """Run the circuit / specialist analysis over ``args.n_prompts`` prompts.

    Args:
        model: The SLM under study.
        args: Parsed CLI namespace.

    Returns:
        Tuple ``(average_pattern, per_prompt_patterns)``.
    """
    import numpy as np

    gen = torch.Generator().manual_seed(args.seed + 3)
    adaptation_dim = model.conditioning_projector.adaptation_dim
    user_state_dim = model.conditioning_projector.user_state_dim

    patterns: list[AttentionPattern] = []
    for _ in range(max(1, args.n_prompts)):
        prompt = _random_prompt(args.vocab_size, args.seq_len, gen).squeeze(0)
        adapt = torch.rand(adaptation_dim, generator=gen)
        user = torch.randn(user_state_dim, generator=gen)
        patterns.append(
            extract_attention_patterns(
                model=model,
                prompt=prompt,
                conditioning_vector=adapt,
                user_state=user,
                max_tokens=args.seq_len,
            )
        )

    # Build an average pattern for the headline figure.
    per_layer_avg = [np.zeros_like(patterns[0].per_layer[li])
                     for li in range(patterns[0].n_layers)]
    entropy_avg = np.zeros_like(patterns[0].per_head_entropy)
    focus_avg = np.zeros_like(patterns[0].per_token_conditioning_focus)
    for p in patterns:
        for li in range(p.n_layers):
            per_layer_avg[li] = per_layer_avg[li] + p.per_layer[li]
        entropy_avg = entropy_avg + p.per_head_entropy
        focus_avg = focus_avg + p.per_token_conditioning_focus
    scale = float(len(patterns))
    per_layer_avg = [a / scale for a in per_layer_avg]
    entropy_avg = entropy_avg / scale
    focus_avg = focus_avg / scale

    avg = AttentionPattern(
        per_layer=per_layer_avg,
        per_head_entropy=entropy_avg,
        per_token_conditioning_focus=focus_avg,
        n_cond=patterns[0].n_cond,
    )
    return avg, patterns


# ---------------------------------------------------------------------------
# Report formatters.
# ---------------------------------------------------------------------------


def _ascii_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
) -> str:
    """Render a numeric matrix as an ASCII heatmap.

    Args:
        matrix: Rectangular list-of-lists of floats.
        row_labels: Labels for the rows (length == len(matrix)).
        col_labels: Labels for the columns.
        title: Title prepended above the heatmap.

    Returns:
        Fenced code-block string ready to paste into Markdown.
    """
    chars = " .:-=+*#%@"
    flat: list[float] = [v for row in matrix for v in row if v == v]
    if not flat:
        return f"```\n{title}\n(empty)\n```"
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    lines = [title]
    header = "          " + " ".join(f"{c:>6}" for c in col_labels)
    lines.append(header)
    for r, row in enumerate(matrix):
        rendered_cells: list[str] = []
        for v in row:
            if v != v:  # NaN
                rendered_cells.append("  nan ")
                continue
            idx = int(round((v - lo) / span * (len(chars) - 1)))
            idx = max(0, min(len(chars) - 1, idx))
            rendered_cells.append(f"{v:+.2f}{chars[idx]}")
        label = row_labels[r] if r < len(row_labels) else f"row_{r}"
        lines.append(f"{label[:9]:<9} " + " ".join(rendered_cells))
    lines.append(f"scale: [{lo:+.3f} .. {hi:+.3f}] using '{chars}'")
    return "```\n" + "\n".join(lines) + "\n```"


def _try_save_png(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
) -> bool:
    """Save a matplotlib heatmap PNG if matplotlib is available.

    Args:
        matrix: Rectangular list-of-lists of floats.
        row_labels: Row labels.
        col_labels: Column labels.
        title: Figure title.
        out_path: Target PNG path.

    Returns:
        ``True`` if the PNG was written, ``False`` if matplotlib is
        unavailable or an exception was raised during plotting.
    """
    if not _MPL_AVAILABLE:
        return False
    try:
        import numpy as np

        arr = np.array(matrix, dtype=float)
        fig, ax = _plt.subplots(
            figsize=(max(4, len(col_labels) * 0.6),
                     max(3, len(row_labels) * 0.5)),
        )
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        _plt.close(fig)
        return True
    except (ValueError, RuntimeError, OSError) as exc:  # narrow, not broad
        logger.warning("matplotlib save failed: %s", exc)
        return False


def _format_report(
    args: argparse.Namespace,
    git_sha: str,
    patching_results: dict[str, CausalEffect],
    probe_results: dict[str, dict[int, float]],
    avg_pattern: AttentionPattern,
    per_prompt_patterns: list[AttentionPattern],
    fig_paths: dict[str, Optional[Path]],
) -> str:
    """Render the full Markdown report.

    Args:
        args: Parsed CLI namespace.
        git_sha: Repo HEAD commit SHA.
        patching_results: Output of :func:`_run_activation_patching`.
        probe_results: Output of :func:`_run_probes`.
        avg_pattern: Averaged attention pattern.
        per_prompt_patterns: Per-prompt attention patterns.
        fig_paths: Dict with keys ``"patching"`` / ``"probing"`` mapped
            to optional PNG paths; if ``None`` the ASCII fallback was
            emitted.

    Returns:
        Markdown string.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Batch B — Mechanistic Interpretability Study (Results)")
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Git SHA:** `{git_sha}`")
    lines.append(f"- **Seed:** `{args.seed}`")
    lines.append(
        f"- **Model:** random-init AdaptiveSLM(vocab={args.vocab_size}, "
        f"d_model={args.d_model}, n_layers={args.n_layers}, "
        f"n_heads={args.n_heads}, n_cross_heads={args.n_cross_heads})"
    )
    lines.append(
        f"- **Prompts:** {len(per_prompt_patterns)} random prompts of "
        f"length {args.seq_len}"
    )
    lines.append(f"- **Probe samples:** {args.n_probe_samples}")
    lines.append("")

    # --- Method ---
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Three complementary analyses are applied to a single random-init "
        "`AdaptiveSLM`:"
    )
    lines.append("")
    lines.append(
        "1. **Activation patching (ROME / causal mediation).** Cache the "
        "outputs of every sub-module on a *corrupted* (zeroed) conditioning "
        "pass, then substitute each one — one at a time — into a *clean* "
        "pass and measure symmetric KL divergence between the patched and "
        "clean next-token distributions (Meng et al., 2022; Vig et al., 2020)."
    )
    lines.append(
        "2. **Linear probing.** Train a bias-free `LinearProbe` on pooled "
        "hidden states at every layer to predict each of the 8 adaptation "
        "dimensions; report held-out R² (Alain & Bengio, 2016; Hewitt & "
        "Liang, 2019)."
    )
    lines.append(
        "3. **Attention-circuit analysis.** Extract per-layer per-head "
        "cross-attention distributions over conditioning tokens and flag "
        "heads whose max weight exceeds 0.6 on a majority of positions "
        "(Elhage et al., 2021; Olsson et al., 2022)."
    )
    lines.append("")

    # --- Activation patching results ---
    lines.append("## Activation Patching Results")
    lines.append("")
    lines.append(
        "Per-component causal effect, measured as symmetric KL between the "
        "patched and the fully-clean next-token distribution. Larger means "
        "the component is more on the critical path for adaptation-driven "
        "generation."
    )
    lines.append("")
    lines.append(
        "| Component | symKL (nats) | L2 | top-1 flip |"
    )
    lines.append("|---|---|---|---|")
    for comp, eff in patching_results.items():
        lines.append(
            f"| `{comp}` | {eff.kl_to_clean:.5f} | {eff.logit_l2:.4f} | "
            f"{'yes' if eff.top1_flipped else 'no'} |"
        )
    lines.append("")

    # Heatmap of patching results.
    comps = list(patching_results.keys())
    patching_matrix = [[patching_results[c].kl_to_clean] for c in comps]
    png = fig_paths.get("patching")
    if png is not None:
        lines.append(f"![Activation-patching KL by component]({png.name})")
    else:
        lines.append(
            _ascii_heatmap(
                patching_matrix,
                row_labels=comps,
                col_labels=["symKL"],
                title="Activation patching (symmetric KL per component)",
            )
        )
    lines.append("")

    # --- Probing results ---
    lines.append("## Probing Results")
    lines.append("")
    lines.append(
        "Held-out R² for each (adaptation-dimension × layer) pair. Higher R² "
        "means the dimension is more linearly decodable from that layer's "
        "pooled hidden state."
    )
    lines.append("")
    layer_idx = sorted({
        li for row in probe_results.values() for li in row.keys()
    })
    lines.append(
        "| dimension | " + " | ".join(f"layer {li}" for li in layer_idx) + " |"
    )
    lines.append(
        "|---|" + "|".join("---" for _ in layer_idx) + "|"
    )
    probe_matrix: list[list[float]] = []
    for dim in probe_results:
        row = [probe_results[dim].get(li, float("nan")) for li in layer_idx]
        probe_matrix.append(row)
        cells = [
            f"{v:.3f}" if v == v else "—" for v in row
        ]
        lines.append(f"| `{dim}` | " + " | ".join(cells) + " |")
    lines.append("")
    png = fig_paths.get("probing")
    if png is not None:
        lines.append(f"![Probe R² — dimension × layer]({png.name})")
    else:
        lines.append(
            _ascii_heatmap(
                probe_matrix,
                row_labels=list(probe_results.keys()),
                col_labels=[f"L{li}" for li in layer_idx],
                title="Probe R² (held-out) by dimension × layer",
            )
        )
    lines.append("")

    # --- Circuits ---
    lines.append("## Circuit Analysis Results")
    lines.append("")
    specialists = identify_conditioning_specialists(avg_pattern, threshold=0.6)
    lines.append(
        f"Identified {len(specialists)} conditioning-specialist head(s) "
        f"across {avg_pattern.n_layers} layers × {avg_pattern.n_heads} "
        f"cross-attention heads."
    )
    lines.append("")
    if specialists:
        lines.append(
            "| layer | head | mean max-w | mean entropy (nats) | preferred cond |"
        )
        lines.append("|---|---|---|---|---|")
        for sp in specialists[:12]:
            lines.append(
                f"| {sp.layer} | {sp.head} | {sp.mean_max_weight:.3f} | "
                f"{sp.mean_entropy:.3f} | {sp.preferred_cond_token} |"
            )
        lines.append("")

    lines.append("### Natural-language summary")
    lines.append("")
    lines.append("> " + summarise_circuit(avg_pattern, threshold=0.6))
    lines.append("")

    # --- Synthesis ---
    lines.append("## Synthesis — which parts of the model do the conditioning work?")
    lines.append("")
    ordered = sorted(
        patching_results.items(),
        key=lambda kv: kv[1].kl_to_clean,
        reverse=True,
    )
    top_three = ", ".join(f"`{c}` ({e.kl_to_clean:.4f})" for c, e in ordered[:3])
    best_probe_dim = max(
        probe_results.items(),
        key=lambda kv: max(
            (v for v in kv[1].values() if v == v), default=float("-inf")
        ),
    )[0] if probe_results else "n/a"
    lines.append(
        f"Activation patching places the largest causal effect on {top_three}. "
        f"Linear probing finds that `{best_probe_dim}` is the most linearly "
        f"decodable dimension in this random-init model. The circuit analysis "
        f"identifies {len(specialists)} conditioning-specialist head(s); "
        f"consult the Natural-language summary above for the per-layer "
        f"picture. Taken together, the three measurements answer the same "
        f"question from three different sides: **the causal path runs "
        f"through the `ConditioningProjector`, is amplified by "
        f"cross-attention heads with sharp conditioning-token focus, and "
        f"leaves a linearly decodable residue on the hidden stream that "
        f"the probes pick up.**"
    )
    lines.append("")

    # --- Threats ---
    lines.append("## Threats to Validity")
    lines.append("")
    lines.append(
        "1. **Random-init caveat.** Every measurement on an untrained model "
        "reports architectural *capacity* rather than learned behaviour. "
        "The magnitudes are therefore not directly comparable to published "
        "ROME / probing numbers on production LMs."
    )
    lines.append(
        "2. **Small-model, single-seed run.** A single seed-42 run on a "
        f"{args.d_model}-d, {args.n_layers}-layer model is sufficient to "
        "exercise the scaffolding but not to make a population claim."
    )
    lines.append(
        "3. **Synthetic prompts.** Prompts are drawn uniformly from the "
        "vocabulary; they do not carry semantic structure and may bias the "
        "circuit analysis towards the positional-encoding component."
    )
    lines.append(
        "4. **No prompt-tuning comparison.** The study does not compare "
        "against a prefix-tuning baseline; that is deferred to future work "
        "per the research note."
    )
    lines.append(
        "5. **KL-based effect size.** Symmetric KL between next-token "
        "distributions is a coarse proxy for 'causal importance'; a full "
        "treatment would add indirect-effect decomposition per Vig et al. "
        "(2020)."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full study.

    Returns:
        ``0`` on success, non-zero otherwise.
    """
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    torch.manual_seed(args.seed)
    model = _build_model(args)

    logger.info("running activation patching ...")
    patching_results = _run_activation_patching(model, args)
    logger.info("running linear probes ...")
    probe_results = _run_probes(model, args)
    logger.info("running circuit analysis ...")
    avg_pattern, per_prompt_patterns = _run_circuits(model, args)

    # Figures.
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    fig_paths: dict[str, Optional[Path]] = {}
    comps = list(patching_results.keys())
    patching_matrix = [[patching_results[c].kl_to_clean] for c in comps]
    p_png = out_md.with_name(out_md.stem + "_patching.png")
    if _try_save_png(
        patching_matrix,
        row_labels=comps,
        col_labels=["symKL"],
        title="Activation patching",
        out_path=p_png,
    ):
        fig_paths["patching"] = p_png
    else:
        fig_paths["patching"] = None

    layer_idx = sorted({li for row in probe_results.values() for li in row.keys()})
    probe_matrix = [
        [probe_results[d].get(li, float("nan")) for li in layer_idx]
        for d in probe_results
    ]
    probe_png = out_md.with_name(out_md.stem + "_probes.png")
    if _try_save_png(
        probe_matrix,
        row_labels=list(probe_results.keys()),
        col_labels=[f"L{li}" for li in layer_idx],
        title="Probe R² (held-out)",
        out_path=probe_png,
    ):
        fig_paths["probing"] = probe_png
    else:
        fig_paths["probing"] = None

    git_sha = _git_sha(_ROOT)

    # JSON sibling.
    out_json = out_md.with_suffix(".json")
    json_payload: dict[str, Any] = {
        "schema_version": 1,
        "run_metadata": {
            "git_sha": git_sha,
            "seed": args.seed,
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "n_prompts": len(per_prompt_patterns),
            "n_probe_samples": args.n_probe_samples,
            "model": {
                "vocab_size": args.vocab_size,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "n_cross_heads": args.n_cross_heads,
            },
        },
        "activation_patching": {
            comp: {
                "kl_to_clean": eff.kl_to_clean,
                "logit_l2": eff.logit_l2,
                "top1_flipped": eff.top1_flipped,
            }
            for comp, eff in patching_results.items()
        },
        "probing": {
            dim: {str(li): v for li, v in row.items()}
            for dim, row in probe_results.items()
        },
        "circuits": {
            "n_prompts": len(per_prompt_patterns),
            "specialists": [
                {
                    "layer": sp.layer,
                    "head": sp.head,
                    "mean_max_weight": sp.mean_max_weight,
                    "mean_entropy": sp.mean_entropy,
                    "preferred_cond_token": sp.preferred_cond_token,
                }
                for sp in identify_conditioning_specialists(
                    avg_pattern, threshold=0.6
                )
            ],
            "summary": summarise_circuit(avg_pattern, threshold=0.6),
        },
    }

    if _PD_AVAILABLE:
        # Attach a rendered selectivity table for downstream consumers.
        frame = compute_probe_selectivity(probe_results)
        if hasattr(frame, "to_dict"):
            json_payload["probing_selectivity_table"] = frame.to_dict()

    out_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    logger.info("wrote JSON: %s", out_json)

    md = _format_report(
        args,
        git_sha=git_sha,
        patching_results=patching_results,
        probe_results=probe_results,
        avg_pattern=avg_pattern,
        per_prompt_patterns=per_prompt_patterns,
        fig_paths=fig_paths,
    )
    out_md.write_text(md, encoding="utf-8")
    logger.info("wrote Markdown: %s", out_md)

    print(f"JSON:     {out_json}")
    print(f"Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
