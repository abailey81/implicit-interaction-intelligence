"""CLI driver for the :mod:`i3.data` pipeline.

Consumes one or more source adapters (JSONL, CSV, plain text,
DailyDialog, EmpatheticDialogues) and writes a cleaned, deduplicated,
deterministically-split dataset plus a quality / lineage report.

The original ``prepare_dialogue.py`` is preserved unchanged for
backwards compatibility; use this v2 driver for new work.

Example::

    # Process the bundled sample corpus
    python -m training.prepare_dialogue_v2 \\
        --jsonl data/corpora/sample_dialogues.jsonl \\
        --output-dir data/processed/sample

    # Process a real DailyDialog download
    python -m training.prepare_dialogue_v2 \\
        --dailydialog /path/to/ijcnlp_dailydialog \\
        --output-dir data/processed/dailydialog

    # Process several sources together
    python -m training.prepare_dialogue_v2 \\
        --jsonl data/corpora/sample_dialogues.jsonl \\
        --empathetic /path/to/empatheticdialogues/train.csv \\
        --output-dir data/processed/combined
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from i3.data import (
    CSVColumnMap,
    CSVSource,
    CleaningConfig,
    DailyDialogSource,
    DataPipeline,
    EmpatheticDialoguesSource,
    JSONLSource,
    PipelineConfig,
    PlainTextSource,
    SourceAdapter,
)

logger = logging.getLogger(__name__)


def _build_sources(args: argparse.Namespace) -> list[SourceAdapter]:
    sources: list[SourceAdapter] = []
    for p in args.jsonl or []:
        sources.append(JSONLSource(p))
    for p in args.txt or []:
        sources.append(PlainTextSource(p))
    for spec in args.csv or []:
        # Spec format: path:text_col[,label_col[,speaker_col[,conv_col]]]
        path, _, cols = spec.partition(":")
        parts = [p.strip() for p in (cols or "").split(",")] if cols else []
        if not parts:
            parser_error(
                f"--csv spec needs at least a text column: '{spec}'"
            )
        col_map = CSVColumnMap(
            text=parts[0],
            label=parts[1] if len(parts) > 1 else None,
            speaker=parts[2] if len(parts) > 2 else None,
            conv_id=parts[3] if len(parts) > 3 else None,
        )
        sources.append(CSVSource(path, columns=col_map))
    for p in args.dailydialog or []:
        sources.append(DailyDialogSource(p))
    for p in args.empathetic or []:
        sources.append(EmpatheticDialoguesSource(p))
    if not sources:
        parser_error(
            "at least one source required: "
            "--jsonl / --txt / --csv / --dailydialog / --empathetic"
        )
    return sources


def parser_error(msg: str) -> None:  # pragma: no cover
    sys.stderr.write(f"prepare_dialogue_v2: error: {msg}\n")
    sys.exit(2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean, dedup, split one or more dialogue corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Sources (repeatable)
    p.add_argument(
        "--jsonl", action="append", metavar="PATH",
        help="JSONL file with {text,label,speaker,conv_id} keys. Repeatable.",
    )
    p.add_argument(
        "--txt", action="append", metavar="PATH",
        help="Plain-text file, one record per line. Repeatable.",
    )
    p.add_argument(
        "--csv", action="append", metavar="PATH:COLS",
        help=(
            "CSV file with column spec. "
            "COLS = text[,label[,speaker[,conv_id]]]. Repeatable."
        ),
    )
    p.add_argument(
        "--dailydialog", action="append", metavar="DIR",
        help="DailyDialog corpus directory. Repeatable.",
    )
    p.add_argument(
        "--empathetic", action="append", metavar="CSV",
        help="EmpatheticDialogues CSV file. Repeatable.",
    )
    # Output + behaviour
    p.add_argument(
        "--output-dir", type=str, default="data/processed/dialogue",
        help="Directory for train.jsonl / val.jsonl / test.jsonl / report.json.",
    )
    p.add_argument(
        "--train-fraction", type=float, default=0.8,
        help="Share of records in the train split (default: 0.8).",
    )
    p.add_argument(
        "--dedup-threshold", type=float, default=0.85,
        help="Jaccard threshold for near-duplicate rejection (default: 0.85).",
    )
    p.add_argument(
        "--exact-dedup-only", action="store_true",
        help="Skip the near-duplicate LSH stage (faster).",
    )
    p.add_argument(
        "--seed", type=str, default="i3-data-v1",
        help="Hashing seed for the deterministic split (default: i3-data-v1).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Log at DEBUG level.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    cfg = PipelineConfig(
        output_dir=Path(args.output_dir),
        train_fraction=args.train_fraction,
        dedup_threshold=args.dedup_threshold,
        exact_dedup_only=args.exact_dedup_only,
        cleaning=CleaningConfig(),
        seed=args.seed,
    )
    sources = _build_sources(args)
    pipeline = DataPipeline(cfg)
    report = pipeline.run(sources)

    logger.info(
        "Finished: accepted=%d (train=%d, val=%d, test=%d), "
        "rejected=%d, exact_dupes=%d, near_dupes=%d, rejection_rate=%.2f%%",
        sum(report.splits.values()),
        report.splits["train"], report.splits["val"], report.splits["test"],
        sum(report.quality["rejected_by_rule"].values()),
        report.dedup["exact_dupes"], report.dedup["near_dupes"],
        report.quality["rejection_rate"] * 100.0,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
