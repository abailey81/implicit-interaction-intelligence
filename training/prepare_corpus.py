"""Combine every corpus source into one training corpus + train BPE.

Pipeline stage that replaces the narrower ``prepare_dialogue.py`` for
the upgraded training run:

1. Call every loader in :mod:`training.corpus_sources`.
2. Merge with the existing hand-crafted intent buckets from
   :mod:`training.prepare_dialogue` (greetings, factoids, HMI-tailored
   responses) so the curated demo prompts still resolve cleanly.
3. Deduplicate on the exact ``(history, response)`` pair.
4. Shuffle deterministically and write to
   ``data/processed/dialogue/triples.json``.
5. Train a from-scratch byte-level BPE tokenizer on a sample of the
   combined corpus and save to
   ``checkpoints/slm/tokenizer_bpe.json``.

Everything is idempotent — loaders cache raw downloads under
``D:/caches/corpus/``, BPE training re-uses the existing tokenizer
file unless ``--force-bpe`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

# Wire stdout / stderr to UTF-8 so the Windows console stops choking
# on em-dashes and smart quotes inside the corpus.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.corpus_sources.cornell_movies import load_cornell_movies
from training.corpus_sources.squad_qa import load_squad_qa
from training.corpus_sources.daily_dialog import load_daily_dialog
from training.corpus_sources.persona_chat import load_persona_chat
from training.corpus_sources.wikitext import load_wikitext
from training.corpus_sources.open_subtitles_en import load_open_subtitles

from i3.slm.bpe_tokenizer import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("prepare_corpus")


# ---------------------------------------------------------------------------
# Hand-crafted intent buckets (kept for demo-critical prompts)
# ---------------------------------------------------------------------------

def _load_curated_intents() -> list[dict]:
    """Re-use the hand-crafted dialogues from the existing pipeline.

    Those entries are deliberately tuned to the HMI-lab pitch (self-
    description, Huawei mentions, edge-first framing) and the
    small-talk seeds a chatbot demo needs.  We keep them as a
    high-priority subset of the training mix so the canonical demo
    prompts still retrieve cleanly even after the big corpora drown
    them out by volume.
    """
    try:
        from training.prepare_dialogue import (
            generate_synthetic_dialogues,
            extract_triples,
        )
    except Exception:
        logger.warning("Could not import curated intents from prepare_dialogue.")
        return []
    try:
        dialogues = generate_synthetic_dialogues(n_dialogues=5000, seed=17)
        triples = extract_triples(dialogues)
    except Exception:
        logger.exception("curated-intent generation failed")
        return []
    out: list[dict] = []
    for t in triples:
        h = (t.get("history") or "").strip()
        r = (t.get("response") or "").strip()
        if not h or not r:
            continue
        out.append({"history": h, "response": r, "kind": t.get("kind", "qa")})
    logger.info("curated intents: %d pairs", len(out))
    return out


# ---------------------------------------------------------------------------
# Combine + dedupe
# ---------------------------------------------------------------------------

def _combine(cache_dir: Path) -> list[dict]:
    """Pull from every source, dedupe, return the combined list."""
    sources = [
        ("curated",      _load_curated_intents, ()),
        ("cornell",      load_cornell_movies,   (cache_dir,)),
        ("squad",        load_squad_qa,         (cache_dir,)),
        ("dailydialog",  load_daily_dialog,     (cache_dir,)),
        ("persona",      load_persona_chat,     (cache_dir,)),
        ("wikitext",     load_wikitext,         (cache_dir,)),
        ("opensubs",     load_open_subtitles,   (cache_dir,)),
    ]

    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    per_source_counts: dict[str, int] = {}
    for name, loader, args in sources:
        try:
            t0 = time.perf_counter()
            pairs = loader(*args) or []
            t_elapsed = time.perf_counter() - t0
            kept = 0
            for p in pairs:
                h = (p.get("history") or "").strip()
                r = (p.get("response") or "").strip()
                if not h or not r:
                    continue
                if len(r) > 2000:
                    continue
                key = (h, r)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "history": h,
                    "response": r,
                    "kind": p.get("kind", "qa"),
                    "source": name,
                })
                kept += 1
            per_source_counts[name] = kept
            logger.info(
                "source %-14s loaded=%d kept=%d elapsed=%.1fs",
                name, len(pairs), kept, t_elapsed,
            )
        except Exception:
            logger.exception("source %s failed; skipping", name)

    logger.info("=" * 64)
    logger.info("per-source unique pair counts:")
    for name, cnt in per_source_counts.items():
        logger.info("  %-14s %8d", name, cnt)
    logger.info("total unique pairs: %d", len(out))
    return out


# ---------------------------------------------------------------------------
# BPE training
# ---------------------------------------------------------------------------

def _train_bpe(
    pairs: list[dict],
    *,
    vocab_size: int,
    out_path: Path,
    sample_size: int = 200_000,
    seed: int = 17,
) -> None:
    """Train a byte-level BPE tokenizer on a sample of the corpus.

    A ~200 k-document sample is more than enough to learn good merges
    at 32 k vocab size — BPE statistics saturate quickly and the full
    1.2 M-pair corpus isn't worth the training time (~10 min extra for
    marginal vocab quality).
    """
    if out_path.exists():
        logger.info("BPE tokenizer already exists at %s — skipping training.", out_path)
        logger.info("(pass --force-bpe to retrain)")
        return
    rng = random.Random(seed)
    sample = pairs if len(pairs) <= sample_size else rng.sample(pairs, sample_size)
    docs: list[str] = []
    for p in sample:
        docs.append(p["history"])
        docs.append(p["response"])
    logger.info(
        "BPE: training on %d sampled documents (target vocab=%d)",
        len(docs), vocab_size,
    )
    t0 = time.perf_counter()
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(docs, verbose=True)
    elapsed = time.perf_counter() - t0
    logger.info(
        "BPE training done in %.1fs  (actual vocab=%d, merges=%d)",
        elapsed, len(tok), len(tok.merges),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)

    # Smoke test on a few demo prompts so we see at-a-glance that the
    # tokenizer behaves sanely (short ids on common phrases, clean
    # round-trips on nonsense).
    probes = [
        "hello",
        "what is photosynthesis",
        "I'm anxious",
        "1+2123",
        "asdfgh qwerty",
        "Huawei R&D UK",
    ]
    logger.info("BPE smoke test:")
    for text in probes:
        ids = tok.encode(text)
        back = tok.decode(ids)
        ok = "✓" if back == text else "✗"
        logger.info("  %s %-35r -> %2d ids -> %r", ok, text, len(ids), back)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("D:/caches/corpus"),
        help="Cache directory for raw downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "data" / "processed" / "dialogue" / "triples.json",
        help="Combined triples.json output path",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="BPE vocab size (default: %(default)d)",
    )
    parser.add_argument(
        "--tokenizer-out", type=Path,
        default=ROOT / "checkpoints" / "slm" / "tokenizer_bpe.json",
    )
    parser.add_argument(
        "--force-bpe", action="store_true",
        help="Re-train BPE even if tokenizer_bpe.json exists",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="If > 0, cap output corpus at this many pairs (for quick runs)",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combine sources.
    pairs = _combine(args.cache_dir)
    if not pairs:
        logger.error("No pairs produced — bailing out")
        return 1

    # 2. Deterministic shuffle + optional cap.
    rng = random.Random(42)
    rng.shuffle(pairs)
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]
        logger.info("capped corpus at %d pairs", len(pairs))

    # 3. Write triples.json.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    logger.info("wrote %d pairs to %s (%d MB)",
                len(pairs), args.out,
                args.out.stat().st_size / (1024 * 1024))

    # 4. Train BPE.
    if args.force_bpe and args.tokenizer_out.exists():
        args.tokenizer_out.unlink()
    _train_bpe(
        pairs,
        vocab_size=args.vocab_size,
        out_path=args.tokenizer_out,
    )

    logger.info("=" * 64)
    logger.info("prepare_corpus.py done.")
    logger.info("corpus: %s (%d pairs)", args.out, len(pairs))
    logger.info("tokenizer: %s", args.tokenizer_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
