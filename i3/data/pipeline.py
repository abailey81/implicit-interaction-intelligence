"""End-to-end data pipeline orchestrator.

Composes the individual stages in :mod:`i3.data.sources`,
:mod:`i3.data.cleaning`, :mod:`i3.data.quality`, and
:mod:`i3.data.dedup` into a single, deterministic, streamable
pipeline:

.. code-block:: text

    SourceAdapter → Cleaner → QualityFilter → Deduplicator
                 →  deterministic splitter  → JSONL output (train/val/test)

Every stage is memory-streaming — the pipeline works on datasets
larger than RAM because records are consumed lazily and written out
per-batch.

The :class:`DataPipeline` is the single entry point; CLI is in
:mod:`training.prepare_dialogue_v2` (preserves the original
`prepare_dialogue.py` for backwards compatibility).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from i3.data.cleaning import Cleaner, CleaningConfig
from i3.data.dedup import Deduplicator
from i3.data.lineage import RecordSchema, SCHEMA_VERSION
from i3.data.quality import QualityFilter, QualityReport, default_rules
from i3.data.sources import SourceAdapter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for a :class:`DataPipeline` run.

    Attributes:
        output_dir: Directory where ``train.jsonl`` / ``val.jsonl`` /
            ``test.jsonl`` / ``report.json`` will be written.
        train_fraction: Share of records routed to the train split.
            ``val`` and ``test`` split the remaining share evenly.
        dedup_threshold: Jaccard similarity above which a record is
            considered a near-duplicate.
        exact_dedup_only: Skip the min-hash LSH stage (faster but
            misses paraphrase duplicates).
        cleaning: Cleaning config.
        seed: Hashing seed used by the deterministic splitter.
    """

    output_dir: Path
    train_fraction: float = 0.8
    dedup_threshold: float = 0.85
    exact_dedup_only: bool = False
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    seed: str = "i3-data-v1"


@dataclass(slots=True)
class _RunReport:
    """Internal — assembled into the final ``report.json``."""

    config: dict
    schema_version: str
    started_at: str
    finished_at: str
    duration_s: float
    quality: dict
    dedup: dict
    splits: dict[str, int]
    per_source_counts: dict[str, int]
    per_label_counts: dict[str, int]
    per_language_signal: dict[str, int]


class DataPipeline:
    """Orchestrates the full clean → filter → dedup → split flow."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.cleaner = Cleaner(config.cleaning)
        self.filter = QualityFilter(list(default_rules()))
        self.deduper = Deduplicator(
            threshold=config.dedup_threshold,
            exact_only=config.exact_dedup_only,
        )
        self._per_source: dict[str, int] = {}
        self._per_label: dict[str, int] = {}
        self._per_language_signal: dict[str, int] = {"primarily_latin": 0, "other": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, sources: Iterable[SourceAdapter]) -> _RunReport:
        """Consume every source end-to-end, writing split files + report."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)

        split_paths = {
            "train": self.config.output_dir / "train.jsonl",
            "val":   self.config.output_dir / "val.jsonl",
            "test":  self.config.output_dir / "test.jsonl",
        }
        split_handles = {k: p.open("w", encoding="utf-8") for k, p in split_paths.items()}
        split_counts = {"train": 0, "val": 0, "test": 0}

        try:
            for record in self._stream(sources):
                split = self._route_to_split(record)
                split_handles[split].write(
                    json.dumps(self._to_wire(record), ensure_ascii=False) + "\n"
                )
                split_counts[split] += 1
                self._tally(record)
        finally:
            for h in split_handles.values():
                h.close()

        finished = datetime.now(timezone.utc)
        report = _RunReport(
            config={
                "output_dir": str(self.config.output_dir),
                "train_fraction": self.config.train_fraction,
                "dedup_threshold": self.config.dedup_threshold,
                "exact_dedup_only": self.config.exact_dedup_only,
                "seed": self.config.seed,
                "cleaning": asdict(self.config.cleaning),
            },
            schema_version=SCHEMA_VERSION,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_s=(finished - started).total_seconds(),
            quality=self.filter.report.as_dict(),
            dedup=self.deduper.stats,
            splits=split_counts,
            per_source_counts=dict(self._per_source),
            per_label_counts=dict(self._per_label),
            per_language_signal=dict(self._per_language_signal),
        )

        report_path = self.config.output_dir / "report.json"
        report_path.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "DataPipeline finished: %d train / %d val / %d test (%d dupes, %d rejected)",
            split_counts["train"], split_counts["val"], split_counts["test"],
            self.deduper.stats["exact_dupes"] + self.deduper.stats["near_dupes"],
            sum(self.filter.report.rejected_by_rule.values()),
        )
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stream(
        self, sources: Iterable[SourceAdapter]
    ) -> Iterator[RecordSchema]:
        """Yield cleaned, filtered, deduplicated records end-to-end."""
        for source in sources:
            logger.info("Ingesting source: %s", source.source_uri)
            for raw in source.iter_records():
                cleaned_text = self.cleaner.clean(raw.text)
                if not cleaned_text.strip():
                    self.filter.report.total_seen += 1
                    self.filter.report.rejected_by_rule["empty_after_clean"] += 1
                    continue
                record = raw.model_copy(
                    update={
                        "text": cleaned_text,
                        "lineage": raw.lineage.with_transform("Cleaner"),
                    }
                )
                if not self.filter.accept(record):
                    continue
                is_unique, reason = self.deduper.add_and_check(
                    record.lineage.source_uri, record.text
                )
                if not is_unique:
                    logger.debug("Dedup drop (%s): %s", reason, record.lineage.source_uri)
                    continue
                yield record.model_copy(update={
                    "lineage": record.lineage.with_transform("Deduplicator"),
                })

    def _route_to_split(self, record: RecordSchema) -> str:
        """Deterministic split by stable hash of ``conv_id`` or content.

        Using ``conv_id`` when available prevents leakage: two turns
        from the same conversation always land in the same split.
        """
        key = record.conv_id or record.lineage.original_hash
        # Mix in the pipeline seed so regenerating with a different
        # seed produces a different (but still deterministic) split.
        digest = hashlib.sha256((self.config.seed + "|" + key).encode()).digest()
        x = int.from_bytes(digest[:8], "big") / float(1 << 64)
        train_cut = self.config.train_fraction
        val_cut = train_cut + (1.0 - train_cut) / 2.0
        if x < train_cut:
            return "train"
        if x < val_cut:
            return "val"
        return "test"

    def _tally(self, record: RecordSchema) -> None:
        src = record.lineage.source_format
        self._per_source[src] = self._per_source.get(src, 0) + 1
        if record.label:
            self._per_label[record.label] = self._per_label.get(record.label, 0) + 1
        # Cheap language signal: fraction of Latin letters.
        letters = sum(1 for ch in record.text if ch.isalpha())
        if letters == 0:
            return
        ascii_letters = sum(1 for ch in record.text if "a" <= ch.lower() <= "z")
        key = "primarily_latin" if ascii_letters / letters >= 0.7 else "other"
        self._per_language_signal[key] = self._per_language_signal.get(key, 0) + 1

    def _to_wire(self, record: RecordSchema) -> dict:
        return {
            "text": record.text,
            "label": record.label,
            "speaker": record.speaker,
            "conv_id": record.conv_id,
            "source_format": record.lineage.source_format,
            "source_uri": record.lineage.source_uri,
            "schema_version": record.lineage.schema_version,
            "applied_transforms": list(record.lineage.applied_transforms),
            "extra": record.extra,
        }


__all__ = ["DataPipeline", "PipelineConfig"]
