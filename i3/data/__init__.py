"""Data pipeline for ingesting, cleaning, and validating real-world corpora.

The modules under :mod:`i3.data` are the load-bearing pipeline between
raw public datasets (DailyDialog, EmpatheticDialogues, user-supplied
text logs, …) and the training-ready ``.pt`` / Parquet artefacts
consumed by :mod:`training.train_encoder` and
:mod:`training.train_slm`.

Pipeline stages (in order):

1. **Source adapters** — parse a vendor-specific layout (raw folder of
   dialogue files, JSONL, CSV, …) into a uniform :class:`Record`
   stream.
2. **Normalisation** — Unicode normalisation (NFKC), whitespace
   collapse, HTML entity decoding, zero-width character removal,
   case-folding of stopwords only (preserving the casing of content
   words).
3. **Quality filtering** — reject records that are too short / too
   long, contain forbidden PII categories, fail a language-detection
   probe, or exceed a profanity threshold.
4. **Deduplication** — content-hash-based dedup plus near-dup removal
   via min-hash LSH shingles (pure Python, no external dep).
5. **Enrichment** — append derived features (token count, readability,
   valence, topic hints).
6. **Lineage + provenance** — every record carries a ``source_uri``,
   ``ingested_at``, and a ``schema_version`` so downstream auditors
   can reconstruct the cleaning trail.
7. **Split + export** — deterministic 80/10/10 train/val/test splits
   keyed by a stable hash of the record content, so re-runs don't
   leak across splits.
8. **Validation** — Pydantic v2 ``RecordSchema`` enforces invariants
   at the boundary; a :class:`DataQualityReport` summarises rejection
   reasons, duplicate rate, language distribution, and length
   histogram.

All modules are pure Python (no torch, numpy, or spaCy at import
time) so the pipeline runs in a minimal container before any ML
dependency is installed.
"""

from __future__ import annotations

from i3.data.cleaning import (
    CleaningConfig,
    Cleaner,
    normalise_unicode,
    strip_zero_width,
)
from i3.data.dedup import MinHashLSH, Deduplicator, content_hash, jaccard
from i3.data.lineage import Lineage, RecordSchema, SCHEMA_VERSION
from i3.data.pipeline import DataPipeline, PipelineConfig
from i3.data.quality import (
    QualityFilter,
    QualityReport,
    QualityRule,
    default_rules,
)
from i3.data.sources import (
    CSVColumnMap,
    CSVSource,
    DailyDialogSource,
    EmpatheticDialoguesSource,
    JSONLSource,
    PlainTextSource,
    SourceAdapter,
)

__all__ = [
    "CSVColumnMap",
    "CSVSource",
    "Cleaner",
    "CleaningConfig",
    "DailyDialogSource",
    "DataPipeline",
    "Deduplicator",
    "EmpatheticDialoguesSource",
    "JSONLSource",
    "Lineage",
    "MinHashLSH",
    "PipelineConfig",
    "PlainTextSource",
    "QualityFilter",
    "QualityReport",
    "QualityRule",
    "RecordSchema",
    "SCHEMA_VERSION",
    "SourceAdapter",
    "content_hash",
    "default_rules",
    "jaccard",
    "normalise_unicode",
    "strip_zero_width",
]
