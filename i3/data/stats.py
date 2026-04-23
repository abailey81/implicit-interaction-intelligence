"""Post-hoc statistics over a processed dataset.

Consumes the split-aware JSONL that :class:`i3.data.pipeline.DataPipeline`
writes (``train.jsonl`` / ``val.jsonl`` / ``test.jsonl``) and computes
standard corpus-quality diagnostics:

- **Token statistics** — total tokens, unique tokens, vocabulary
  growth curve, token-length histogram, sentence-length histogram.
- **Zipf coefficient** — slope of the log-rank / log-frequency line
  on the top-N tokens (healthy English corpora fall near ``-1``).
- **OOV rate** — fraction of tokens in a target split that are absent
  from the train split's vocabulary.
- **Label balance** — per-label counts, plus Shannon entropy and the
  Gini coefficient of the distribution.
- **Duplication fingerprint** — proportion of records sharing their
  exact content hash with another record (even within a single
  split, since the pipeline dedups at ingest time — this catches
  residual duplicates introduced by manual edits or secondary
  sources).

All computations are streaming and O(n) — suitable for datasets
larger than RAM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# Lightweight in-module tokenizer: split on whitespace after separating
# punctuation.  Kept local to the stats module so nothing else in the
# data pipeline has to depend on the SLM tokenizer.
_PUNCT_RE: re.Pattern[str] = re.compile(r"([^\w\s])", re.UNICODE)
_WS_RE: re.Pattern[str] = re.compile(r"\s+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = _PUNCT_RE.sub(r" \1 ", text)
    text = _WS_RE.sub(" ", text).strip().lower()
    return text.split(" ") if text else []


tokenize = _tokenize


@dataclass(slots=True)
class SplitStats:
    """Per-split diagnostics."""

    name: str
    records: int = 0
    tokens: int = 0
    chars: int = 0
    unique_tokens: int = 0
    token_counts: Counter[str] = field(default_factory=Counter)
    length_histogram: Counter[str] = field(default_factory=Counter)
    char_length_histogram: Counter[str] = field(default_factory=Counter)
    label_counts: Counter[str] = field(default_factory=Counter)
    speaker_counts: Counter[str] = field(default_factory=Counter)
    duplicate_hashes: int = 0
    _seen_hashes: set[str] = field(default_factory=set, repr=False)

    def absorb(self, text: str, label: str | None, speaker: str | None) -> None:
        self.records += 1
        self.chars += len(text)
        tokens = tokenize(text)
        self.tokens += len(tokens)
        self.token_counts.update(tokens)
        self.length_histogram[_bucket_tokens(len(tokens))] += 1
        self.char_length_histogram[_bucket_chars(len(text))] += 1
        if label:
            self.label_counts[label] += 1
        if speaker:
            self.speaker_counts[speaker] += 1
        # Same normalisation as :func:`i3.data.dedup.content_hash` — a
        # stats duplicate signal matches a pipeline duplicate exactly.
        normalised = _WS_RE.sub(" ", text.lower()).strip()
        h = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            self.duplicate_hashes += 1
        else:
            self._seen_hashes.add(h)
        self.unique_tokens = len(self.token_counts)

    def type_token_ratio(self) -> float:
        return (self.unique_tokens / self.tokens) if self.tokens else 0.0

    def zipf_slope(self, top_n: int = 500) -> float:
        """Slope of ``log(rank) vs log(freq)`` on the top-N tokens.

        Healthy natural-language distributions sit near ``-1``.  A
        slope much shallower than ``-0.6`` or steeper than ``-1.5``
        is a fingerprint for repetitive or highly-peaked corpora.
        """
        if not self.token_counts:
            return 0.0
        freqs = [c for _, c in self.token_counts.most_common(top_n) if c > 0]
        if len(freqs) < 2:
            return 0.0
        xs = [math.log(i + 1) for i in range(len(freqs))]
        ys = [math.log(f) for f in freqs]
        return _linear_slope(xs, ys)

    def label_entropy(self) -> float:
        """Shannon entropy of the label distribution in bits."""
        total = sum(self.label_counts.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * math.log2(c / total)
            for c in self.label_counts.values() if c > 0
        )

    def label_gini(self) -> float:
        """Gini coefficient of the label distribution (0 = balanced)."""
        values = sorted(self.label_counts.values())
        n = len(values)
        if n == 0:
            return 0.0
        cum = 0.0
        total = sum(values) or 1
        for i, v in enumerate(values, start=1):
            cum += i * v
        return ((2 * cum) / (n * total)) - ((n + 1) / n)

    def top_tokens(self, n: int = 20) -> list[tuple[str, int]]:
        return self.token_counts.most_common(n)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "records": self.records,
            "tokens": self.tokens,
            "chars": self.chars,
            "unique_tokens": self.unique_tokens,
            "type_token_ratio": round(self.type_token_ratio(), 6),
            "zipf_slope_top500": round(self.zipf_slope(500), 4),
            "length_histogram": dict(self.length_histogram),
            "char_length_histogram": dict(self.char_length_histogram),
            "label_counts": dict(self.label_counts),
            "label_entropy_bits": round(self.label_entropy(), 4),
            "label_gini": round(self.label_gini(), 4),
            "speaker_counts": dict(self.speaker_counts),
            "duplicate_hash_collisions": self.duplicate_hashes,
            "top_tokens": self.top_tokens(20),
        }


@dataclass(slots=True)
class DatasetStats:
    """Aggregate across every split in a processed-dataset directory."""

    splits: dict[str, SplitStats] = field(default_factory=dict)
    oov_rates: dict[str, float] = field(default_factory=dict)
    vocab_overlap: dict[str, float] = field(default_factory=dict)

    def compute_oov_and_overlap(self) -> None:
        """Populate :attr:`oov_rates` relative to the train vocabulary."""
        train = self.splits.get("train")
        if train is None:
            return
        train_vocab = set(train.token_counts.keys())
        for name, split in self.splits.items():
            if name == "train" or not split.token_counts:
                continue
            target_vocab = set(split.token_counts.keys())
            oov_tokens = sum(
                c for tok, c in split.token_counts.items()
                if tok not in train_vocab
            )
            self.oov_rates[name] = (
                oov_tokens / split.tokens if split.tokens else 0.0
            )
            self.vocab_overlap[name] = (
                len(target_vocab & train_vocab) / len(target_vocab)
                if target_vocab else 0.0
            )

    def to_dict(self) -> dict:
        return {
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
            "oov_rates": {k: round(v, 6) for k, v in self.oov_rates.items()},
            "vocab_overlap": {k: round(v, 6) for k, v in self.vocab_overlap.items()},
        }


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def compute_stats(dataset_dir: str | Path) -> DatasetStats:
    """Compute diagnostics for every split file found under ``dataset_dir``.

    Expected layout:

    ::

        dataset_dir/
          train.jsonl
          val.jsonl
          test.jsonl
          report.json   (ignored — from :class:`DataPipeline`)

    Records that fail JSON parse are skipped silently; malformed
    input is already filtered by the pipeline.
    """
    dataset_dir = Path(dataset_dir)
    stats = DatasetStats()
    for split_name in ("train", "val", "test"):
        split_path = dataset_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        ss = SplitStats(name=split_name)
        for record in _iter_jsonl(split_path):
            text = record.get("text", "")
            if not isinstance(text, str):
                continue
            ss.absorb(
                text,
                record.get("label"),
                record.get("speaker"),
            )
        stats.splits[split_name] = ss
    stats.compute_oov_and_overlap()
    return stats


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bucket_tokens(n: int) -> str:
    for cap in (5, 10, 25, 50, 100, 250, 500, 1000):
        if n <= cap:
            return f"<={cap}"
    return ">1000"


def _bucket_chars(n: int) -> str:
    for cap in (20, 50, 100, 250, 500, 1000, 2500):
        if n <= cap:
            return f"<={cap}"
    return ">2500"


def _linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    """Ordinary-least-squares slope of ys over xs.  O(n), no numpy."""
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


__all__ = ["DatasetStats", "SplitStats", "compute_stats"]
