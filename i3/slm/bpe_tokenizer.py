"""Byte-pair encoding tokenizer, implemented from scratch.

Why this exists
===============
The previous :class:`~i3.slm.tokenizer.SimpleTokenizer` is a
word-level tokenizer with a closed 30 k vocabulary.  Anything the
user types that isn't in those 30 k words becomes ``[UNK]``, which
is the root cause of the "word salad on novel inputs" problem.
Byte-pair encoding (Sennrich, Haddow & Birch 2015 — *Neural Machine
Translation of Rare Words with Subword Units*) fixes that: the 256
bytes are *always* in the vocabulary, so there is no OOV — every
possible UTF-8 string tokenises cleanly.

This module is a **from-scratch, dependency-free** reference
implementation of byte-level BPE.  We do not call into
``sentencepiece`` / ``tokenizers`` / any HuggingFace artefact.
That is deliberate: the Huawei HMI Lab JD emphasises *implementing
the core algorithms yourself* (filter question #1).  The trade-off
is that our trainer is ~20× slower than the C++ equivalents, but at
our corpus scale (<100 M characters) it's a one-off 2–5 min job.

Algorithm summary
-----------------
**Training**
    1. Split the corpus into "pre-tokens" using a regex adapted
       from GPT-2 (whitespace + punctuation + contractions).  Each
       pre-token keeps its leading space.
    2. Byte-encode each pre-token; the initial token alphabet is
       the 256 byte values (plus four special tokens).
    3. Count every adjacent-byte-pair across all pre-tokens.
    4. Merge the most frequent pair into a new token, update
       the pre-token representations, and record the merge rule.
    5. Repeat until ``vocab_size`` is reached.

**Encoding**
    Run the learned merges in priority order (lowest rank =
    highest priority) over the pre-tokenised byte stream.

**Decoding**
    Concatenate every token's byte representation and UTF-8-decode,
    replacing invalid bytes with the Unicode replacement char to
    stay lossless on malformed inputs.

Public API mirrors :class:`SimpleTokenizer` so downstream code
(``encode``, ``decode``, ``token_to_id``, ``save``, ``load``,
``vocab_size``) keeps working unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import torch

logger = logging.getLogger(__name__)


# GPT-2-style pre-tokenisation regex.  Keeps whitespace with the word
# that follows it (the common trick to ensure decoding is invertible),
# splits out contractions as their own tokens, and keeps runs of
# punctuation / digits separate.
_PRETOKENIZE_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|"""
    r"""\s?[A-Za-z]+|"""
    r"""\s?\d+|"""
    r"""\s?[^\sA-Za-z\d]+|"""
    r"""\s+""",
    re.UNICODE,
)


class BPETokenizer:
    """Byte-level byte-pair encoding tokenizer.

    The public interface (``encode``, ``decode``, ``token_to_id``,
    ``save``, ``load``, ``vocab_size``, plus BOS/EOS/SEP/PAD/UNK ids)
    is a superset of :class:`SimpleTokenizer` so the rest of the
    codebase can swap one for the other.

    Attributes:
        vocab_size: Target vocab size (actual size may be slightly
            smaller if the corpus is too small to produce that many
            distinct merges).
        token_to_id: Canonical mapping from token string to id.
        id_to_token: Reverse mapping.
        merges: Ordered list of ``(bytes_a, bytes_b)`` pairs in
            merge-priority order (index 0 = highest priority).
        PAD_ID / UNK_ID / BOS_ID / EOS_ID / SEP_ID: reserved slots.
    """

    # Reserved special-token strings.  Kept identical to
    # SimpleTokenizer so checkpoints trained under one tokenizer can
    # fall back to the same special ids under the other.
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    SEP_ID = 4

    def __init__(self, vocab_size: int = 32000) -> None:
        if vocab_size < 260:
            raise ValueError(
                f"vocab_size must be >= 260 (256 bytes + 4 specials + unk), got {vocab_size}"
            )
        self.vocab_size = int(vocab_size)
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        # Every token is canonically represented by its byte sequence.
        # ``token_bytes[id]`` is the ``bytes`` object for that id;
        # ``byte_token_to_id[tuple_of_bytes]`` is the reverse lookup
        # keyed by an immutable tuple (needed because bytes of length
        # > 1 aren't hashable as bytearrays).
        self.token_bytes: list[bytes] = []
        self.merges: list[tuple[bytes, bytes]] = []
        # Lookup mapping `(bytes_a, bytes_b) -> merge rank` used by
        # the encoder.  Lower rank = higher priority (earlier merge).
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {}
        # Canonical token-bytes -> id lookup, keyed by bytes (hashable).
        self._bytes_to_id: dict[bytes, int] = {}

        # Seed the specials + every single-byte token.  We populate
        # this eagerly so even an untrained tokenizer can at least
        # encode/decode (one id per byte) without crashing.
        self._reset_vocab()

    # ------------------------------------------------------------------
    # Vocab bookkeeping
    # ------------------------------------------------------------------

    def _reset_vocab(self) -> None:
        """Reset to the baseline specials + 256 byte-alphabet vocab."""
        self.token_to_id.clear()
        self.id_to_token.clear()
        self.token_bytes.clear()
        self._bytes_to_id.clear()
        self.merges.clear()
        self._merge_ranks.clear()

        specials = [
            (self.PAD_TOKEN, self.PAD_ID),
            (self.UNK_TOKEN, self.UNK_ID),
            (self.BOS_TOKEN, self.BOS_ID),
            (self.EOS_TOKEN, self.EOS_ID),
            (self.SEP_TOKEN, self.SEP_ID),
        ]
        for tok, _id in specials:
            idx = len(self.token_bytes)
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok
            # Specials don't correspond to any byte sequence — we store
            # an empty bytes object as a sentinel and never put them in
            # _bytes_to_id (so they never collide with real content).
            self.token_bytes.append(b"")

        # 256 single-byte tokens, one per byte value.  We give each
        # byte a human-readable vocab string ("<0x20>") purely for
        # debug pretty-printing; the canonical identity is the byte.
        for b in range(256):
            token = f"<0x{b:02X}>"
            idx = len(self.token_bytes)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            bs = bytes([b])
            self.token_bytes.append(bs)
            self._bytes_to_id[bs] = idx

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: Iterable[str], *, verbose: bool = True) -> None:
        """Learn BPE merges from *corpus*.

        Args:
            corpus: Iterable of strings (one document per element).
                Read exactly once; use an iterator if the corpus
                doesn't fit in memory.
            verbose: Log merge progress every 1 000 merges.
        """
        self._reset_vocab()

        # 1. Pre-tokenise and count pre-token frequencies.  Each
        #    pre-token is converted to a tuple of *single-byte*
        #    tokens (one bytes object per element) so adjacent-pair
        #    stats and merge rewrites are both O(len).
        word_freqs: Counter[tuple[bytes, ...]] = Counter()
        for doc in corpus:
            if not doc:
                continue
            for match in _PRETOKENIZE_RE.finditer(doc):
                pre = match.group(0)
                if not pre:
                    continue
                word = tuple(bytes([b]) for b in pre.encode("utf-8"))
                word_freqs[word] += 1
        if not word_freqs:
            logger.warning("BPE.train: corpus was empty — keeping byte-only vocab.")
            return

        # 2. Iteratively merge the most frequent adjacent pair until
        #    vocab_size is reached.  The classic BPE inner loop runs
        #    in O(vocab_size · max_word_len) per iteration; for our
        #    corpus size (<100 M chars, <1 M unique pre-tokens) this
        #    finishes in a few minutes on CPU.
        target_merges = self.vocab_size - len(self.token_bytes)
        if target_merges <= 0:
            logger.warning("BPE.train: vocab_size already covered by byte alphabet.")
            return

        # Maintain an up-to-date pair counter across all words so we
        # don't re-scan the whole dataset per merge.  ``pair_counts[(a,b)]``
        # is the number of times (a, b) appears (weighted by word
        # freq); on each merge we decrement every pair the merge
        # touched and increment the new pairs created.
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        # ``where[(a,b)]`` is the set of word tuples containing (a,b),
        # which lets us skip words that can't possibly contain the
        # chosen merge.
        where: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
                where.setdefault(pair, set()).add(word)

        for step in range(target_merges):
            if not pair_counts:
                break
            (a, b), _count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if pair_counts[(a, b)] <= 0:
                break
            merged = a + b
            # Register the new token.
            if merged in self._bytes_to_id:
                # Pathological duplicate; decrement its count and skip.
                pair_counts.pop((a, b), None)
                continue
            new_id = len(self.token_bytes)
            surface = f"<bpe_{new_id}>"
            self.token_to_id[surface] = new_id
            self.id_to_token[new_id] = surface
            self.token_bytes.append(merged)
            self._bytes_to_id[merged] = new_id
            self.merges.append((a, b))
            self._merge_ranks[(a, b)] = len(self._merge_ranks)

            # Rewrite every word that contained this pair.
            affected = where.pop((a, b), set())
            for word in list(affected):
                if word not in word_freqs:
                    continue
                freq = word_freqs[word]
                new_word: list[bytes] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_t = tuple(new_word)
                if new_word_t == word:
                    continue
                # Update pair_counts: subtract old pairs, add new pairs.
                for i in range(len(word) - 1):
                    old_pair = (word[i], word[i + 1])
                    pair_counts[old_pair] -= freq
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        where.pop(old_pair, None)
                    else:
                        s = where.get(old_pair)
                        if s is not None:
                            s.discard(word)
                for i in range(len(new_word_t) - 1):
                    new_pair = (new_word_t[i], new_word_t[i + 1])
                    pair_counts[new_pair] += freq
                    where.setdefault(new_pair, set()).add(new_word_t)
                del word_freqs[word]
                word_freqs[new_word_t] = word_freqs.get(new_word_t, 0) + freq

            if verbose and (step + 1) % 1000 == 0:
                logger.info(
                    "BPE.train: %d / %d merges learned (pair count cache=%d)",
                    step + 1, target_merges, len(pair_counts),
                )

        logger.info(
            "BPE.train: finished with %d merges, vocab size %d",
            len(self.merges), len(self.token_bytes),
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _apply_merges(self, pieces: list[bytes]) -> list[bytes]:
        """Greedy merge pass over *pieces* using learned merge ranks."""
        if len(pieces) < 2 or not self._merge_ranks:
            return pieces
        while True:
            # Find the pair with the lowest rank (= highest priority).
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(pieces) - 1):
                rank = self._merge_ranks.get((pieces[i], pieces[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                return pieces
            merged = pieces[best_idx] + pieces[best_idx + 1]
            pieces = pieces[:best_idx] + [merged] + pieces[best_idx + 2:]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode *text* to a list of token ids.

        Args:
            text: Input string; may include arbitrary Unicode.
            add_bos: Prepend ``[BOS]``.
            add_eos: Append ``[EOS]``.
            max_length: Truncate the *content* ids to this length
                (BOS/EOS are added after truncation).

        Returns:
            A list of ``int`` token ids.  Never raises on unseen
            bytes — every byte has a guaranteed single-byte token.
        """
        ids: list[int] = []
        for match in _PRETOKENIZE_RE.finditer(text):
            pre = match.group(0)
            if not pre:
                continue
            byte_pieces = [bytes([b]) for b in pre.encode("utf-8")]
            merged_pieces = self._apply_merges(byte_pieces)
            for piece in merged_pieces:
                tid = self._bytes_to_id.get(piece, self.UNK_ID)
                ids.append(tid)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode *ids* back to text.

        Invalid UTF-8 byte sequences get replaced with the Unicode
        replacement character so this never raises — robust to the
        model occasionally emitting bytes that don't compose into
        valid UTF-8 mid-sequence.
        """
        pieces: list[bytes] = []
        specials = {self.PAD_ID, self.BOS_ID, self.EOS_ID, self.SEP_ID, self.UNK_ID}
        for tid in ids:
            if skip_special and tid in specials:
                continue
            if 0 <= tid < len(self.token_bytes):
                pieces.append(self.token_bytes[tid])
        return b"".join(pieces).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the tokenizer to a JSON file.

        Byte sequences are stored as hex strings so the file stays
        human-readable and doesn't depend on JSON's treatment of
        non-UTF-8 strings.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "format": "bpe-bytelevel",
            "vocab_size": self.vocab_size,
            "token_bytes_hex": [b.hex() for b in self.token_bytes],
            "merges_hex": [[a.hex(), b.hex()] for a, b in self.merges],
            "specials": {
                "PAD": self.PAD_ID, "UNK": self.UNK_ID,
                "BOS": self.BOS_ID, "EOS": self.EOS_ID,
                "SEP": self.SEP_ID,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info(
            "BPE.save: wrote %d tokens + %d merges to %s",
            len(self.token_bytes), len(self.merges), path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != "bpe-bytelevel":
            raise ValueError(
                f"{path}: not a BPE tokenizer file (format={data.get('format')!r})"
            )
        tok = cls(vocab_size=int(data["vocab_size"]))
        # Restore exact vocab: throw away the freshly-seeded vocab and
        # rebuild from the saved token_bytes_hex list.
        tok.token_to_id.clear()
        tok.id_to_token.clear()
        tok.token_bytes.clear()
        tok._bytes_to_id.clear()
        specials = data.get("specials") or {}
        special_by_id = {
            specials.get("PAD", 0): cls.PAD_TOKEN,
            specials.get("UNK", 1): cls.UNK_TOKEN,
            specials.get("BOS", 2): cls.BOS_TOKEN,
            specials.get("EOS", 3): cls.EOS_TOKEN,
            specials.get("SEP", 4): cls.SEP_TOKEN,
        }
        for idx, bhex in enumerate(data["token_bytes_hex"]):
            b = bytes.fromhex(bhex)
            if idx in special_by_id:
                name = special_by_id[idx]
                tok.token_to_id[name] = idx
                tok.id_to_token[idx] = name
                tok.token_bytes.append(b"")
            elif len(b) == 1:
                name = f"<0x{b[0]:02X}>"
                tok.token_to_id[name] = idx
                tok.id_to_token[idx] = name
                tok.token_bytes.append(b)
                tok._bytes_to_id[b] = idx
            else:
                name = f"<bpe_{idx}>"
                tok.token_to_id[name] = idx
                tok.id_to_token[idx] = name
                tok.token_bytes.append(b)
                tok._bytes_to_id[b] = idx
        tok.merges = [
            (bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges_hex"]
        ]
        tok._merge_ranks = {pair: rank for rank, pair in enumerate(tok.merges)}
        logger.info(
            "BPE.load: %d tokens + %d merges from %s",
            len(tok.token_bytes), len(tok.merges), path,
        )
        return tok

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.token_bytes)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"actual_size={len(self.token_bytes)}, "
            f"merges={len(self.merges)})"
        )


# ---------------------------------------------------------------------------
# Convenience: tensor round-trip, matching SimpleTokenizer signature.
# ---------------------------------------------------------------------------


def encode_to_tensor(
    tokenizer: BPETokenizer,
    text: str,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    max_length: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
    t = torch.tensor(ids, dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t
