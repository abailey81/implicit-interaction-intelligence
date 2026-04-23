"""Exact and near-duplicate detection for the data pipeline.

Two complementary strategies:

- :func:`content_hash` — SHA-256 of the lowercased, whitespace-normalised
  text.  Catches byte-for-byte reposts and reformattings.
- :class:`MinHashLSH` — shingle-based near-duplicate detection.  Uses
  min-hash permutations + locality-sensitive hashing to find records
  with high Jaccard similarity, without pair-wise comparisons.

The :class:`Deduplicator` composes both: exact hashes are checked
first (O(1)), then LSH-candidate pairs are tested with real Jaccard
similarity (O(k) per record where k = number of hash bands).

The implementation is pure Python (no datasketch, no scikit-learn) so
the pipeline has no runtime dependencies beyond the standard library.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

#: Shingle size for min-hash.  k=5 is the canonical value used in
#: both the near-duplicate Web-crawling literature
#: (Broder 1997, Manku et al. 2007) and Hugging Face's FineWeb /
#: Datatrove pipeline (Penedo et al., arXiv:2406.17557) —
#: see also the reference `minhash_deduplication.py` example in
#: https://github.com/huggingface/datatrove.
_DEFAULT_SHINGLE_SIZE: int = 5

#: Number of min-hash permutations.  128 is the canonical sweet spot
#: for production pipelines; FineWeb uses 112 (14 bands × 8 rows).
#: We pick 128 because 128 = 16 × 8 divides cleanly into the band
#: / row decomposition.
_DEFAULT_NUM_PERMUTATIONS: int = 128

#: Bands × rows per band = num_permutations.  (bands=16, rows=8) makes
#: the LSH collision probability P(collision | similarity=s) =
#: 1 - (1 - s⁸)¹⁶, which gives ≥ 0.81 recall at s=0.75 and ≥ 0.97
#: recall at s=0.85.  FineWeb's (14, 8) is tuned for the same
#: threshold band; this module's defaults are chosen for parity.
_DEFAULT_BANDS: int = 16

_TOKEN_RE: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)


def content_hash(text: str) -> str:
    """Deterministic content hash for exact-duplicate detection.

    Normalisation: lowercase, whitespace-collapsed, stripped.  This is
    a superset of the :class:`i3.data.cleaning.Cleaner` normalisation
    so records that differ only in casing or whitespace still collide.
    """
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Min-hash + LSH
# ---------------------------------------------------------------------------


def _shingles(tokens: list[str], k: int) -> list[str]:
    """Return the set of k-shingles (as space-joined strings).

    For short inputs we fall back to token-level shingles so the
    signature is still meaningful.
    """
    if len(tokens) < k:
        return list(tokens)
    return [" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def _minhash_signature(
    shingles: Iterable[str],
    num_perm: int,
    *,
    seed: int = 0xA11CE,
) -> list[int]:
    """Return the min-hash signature for ``shingles``.

    Each permutation is implemented as a universal hash family
    ``(a * h + b) mod p`` over the shingle's SHA-256 prefix, so the
    signature is deterministic for a given seed.
    """
    rng = random.Random(seed)
    # Pre-generate (a, b) coefficient pairs — one per permutation.
    coeffs = [
        (rng.randrange(1, 1 << 31), rng.randrange(0, 1 << 31))
        for _ in range(num_perm)
    ]
    prime = (1 << 61) - 1  # Mersenne prime
    min_hashes = [prime] * num_perm
    for sh in shingles:
        digest = int.from_bytes(
            hashlib.sha256(sh.encode("utf-8")).digest()[:8], "big"
        )
        for i, (a, b) in enumerate(coeffs):
            h = (a * digest + b) % prime
            if h < min_hashes[i]:
                min_hashes[i] = h
    return min_hashes


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two shingle sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


@dataclass(slots=True)
class MinHashLSH:
    """Locality-sensitive hashing over min-hash signatures.

    Adding a record indexes every band; querying returns candidate
    collisions.  The caller then verifies candidates with
    :func:`jaccard` against a configurable similarity threshold.
    """

    num_perm: int = _DEFAULT_NUM_PERMUTATIONS
    bands: int = _DEFAULT_BANDS
    shingle_size: int = _DEFAULT_SHINGLE_SIZE
    threshold: float = 0.85
    _buckets: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    _signatures: dict[str, list[int]] = field(default_factory=dict)
    _shingle_sets: dict[str, set[str]] = field(default_factory=dict)
    _rows_per_band: int = 0  # computed in __post_init__

    def __post_init__(self) -> None:
        if self.num_perm % self.bands != 0:
            raise ValueError(
                f"num_perm ({self.num_perm}) must be divisible by "
                f"bands ({self.bands})"
            )
        self._rows_per_band = self.num_perm // self.bands

    def _bands_of(self, sig: list[int]) -> Iterator[tuple[int, tuple[int, ...]]]:
        r = self._rows_per_band
        for b in range(self.bands):
            yield b, tuple(sig[b * r:(b + 1) * r])

    def add(self, record_id: str, text: str) -> None:
        """Index ``text`` under ``record_id``."""
        tokens = _TOKEN_RE.findall(text.lower())
        shingles = _shingles(tokens, self.shingle_size)
        shingle_set = set(shingles)
        sig = _minhash_signature(shingle_set, self.num_perm)
        self._signatures[record_id] = sig
        self._shingle_sets[record_id] = shingle_set
        for band_idx, band_sig in self._bands_of(sig):
            key = (band_idx, hash(band_sig))
            self._buckets.setdefault(key, []).append(record_id)

    def candidates(self, text: str) -> set[str]:
        """Return record IDs that share at least one LSH band with ``text``."""
        tokens = _TOKEN_RE.findall(text.lower())
        shingles = _shingles(tokens, self.shingle_size)
        sig = _minhash_signature(set(shingles), self.num_perm)
        cands: set[str] = set()
        for band_idx, band_sig in self._bands_of(sig):
            key = (band_idx, hash(band_sig))
            for rid in self._buckets.get(key, ()):
                cands.add(rid)
        return cands

    def is_near_duplicate(self, text: str) -> tuple[bool, str | None, float]:
        """Return ``(is_dup, matching_id, similarity)`` for ``text``."""
        tokens = _TOKEN_RE.findall(text.lower())
        shingles = _shingles(tokens, self.shingle_size)
        target = set(shingles)
        best_sim = 0.0
        best_id: str | None = None
        for rid in self.candidates(text):
            sim = jaccard(target, self._shingle_sets[rid])
            if sim > best_sim:
                best_sim = sim
                best_id = rid
        return best_sim >= self.threshold, best_id, best_sim


# ---------------------------------------------------------------------------
# Deduplicator — exact first, then near-dup
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Deduplicator:
    """Two-stage deduplicator: exact-hash + min-hash LSH.

    Attributes:
        threshold: Jaccard similarity threshold for near-duplicate
            rejection.  0.85 is a strict setting; drop to 0.7 for
            aggressive paraphrase removal.
    """

    threshold: float = 0.85
    num_perm: int = _DEFAULT_NUM_PERMUTATIONS
    bands: int = _DEFAULT_BANDS
    shingle_size: int = _DEFAULT_SHINGLE_SIZE
    exact_only: bool = False
    _seen_hashes: set[str] = field(default_factory=set)
    _lsh: MinHashLSH | None = None
    _exact_dupes: int = 0
    _near_dupes: int = 0

    def __post_init__(self) -> None:
        if not self.exact_only:
            self._lsh = MinHashLSH(
                num_perm=self.num_perm,
                bands=self.bands,
                shingle_size=self.shingle_size,
                threshold=self.threshold,
            )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "exact_dupes": self._exact_dupes,
            "near_dupes": self._near_dupes,
            "unique": len(self._seen_hashes),
        }

    def add_and_check(self, record_id: str, text: str) -> tuple[bool, str]:
        """Return ``(is_unique, reason)``.

        The record is added to the index iff it was unique.
        """
        h = content_hash(text)
        if h in self._seen_hashes:
            self._exact_dupes += 1
            return False, "exact"
        if self._lsh is not None:
            is_dup, _, sim = self._lsh.is_near_duplicate(text)
            if is_dup:
                self._near_dupes += 1
                return False, f"near({sim:.3f})"
        self._seen_hashes.add(h)
        if self._lsh is not None:
            self._lsh.add(record_id, text)
        return True, "unique"


__all__ = [
    "Deduplicator",
    "MinHashLSH",
    "content_hash",
    "jaccard",
]
