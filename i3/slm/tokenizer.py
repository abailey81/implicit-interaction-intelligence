"""Word-level tokenizer with vocabulary management.

Built from scratch. No HuggingFace tokenizers, no SentencePiece, no BPE libraries.
Simple, transparent, and fully controllable.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    """Word-level tokenizer with vocabulary management.

    Built from scratch. Supports:
    - Vocabulary building from corpus with frequency-based pruning
    - Special tokens: [PAD], [BOS], [EOS], [SEP], [UNK]
    - Encoding text to token IDs with optional BOS/EOS wrapping
    - Decoding token IDs back to text
    - Batch encoding with padding and attention masks
    - Save/load vocabulary to/from JSON

    Attributes:
        SPECIAL_TOKENS: Reserved tokens inserted at the start of every vocabulary.
        PAD_ID: Index of the padding token (0).
        BOS_ID: Index of the beginning-of-sequence token (1).
        EOS_ID: Index of the end-of-sequence token (2).
        SEP_ID: Index of the separator token (3).
        UNK_ID: Index of the unknown token (4).
    """

    SPECIAL_TOKENS: list[str] = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[UNK]"]
    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    SEP_ID: int = 3
    UNK_ID: int = 4

    # Punctuation characters to separate from adjacent words.
    _PUNCT_PATTERN = re.compile(
        r"""([.,!?;:'"()\-\[\]{}<>/@#$%^&*_+=~`|\\])"""
    )
    # Collapse runs of whitespace (including newlines) into a single space.
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, vocab_size: int = 8000) -> None:
        """Initialise the tokenizer.

        Args:
            vocab_size: Maximum vocabulary size *including* special tokens.
                        The actual vocabulary may be smaller if the corpus
                        contains fewer unique tokens.
        """
        if vocab_size < len(self.SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= number of special "
                f"tokens ({len(self.SPECIAL_TOKENS)})"
            )
        self.vocab_size: int = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._built: bool = False

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from a corpus of texts.

        Counts word frequencies across all texts after preprocessing, then
        keeps the top ``(vocab_size - n_special)`` most frequent words.
        Special tokens are always assigned indices 0-4.

        Args:
            texts: List of raw text strings (documents / sentences).
        """
        # SEC: Validate input type to fail fast on misuse.
        if texts is None:
            texts = []
        if not isinstance(texts, (list, tuple)):
            raise TypeError(
                f"texts must be a list/tuple of str, got {type(texts).__name__}"
            )

        # 1. Initialise special tokens at fixed positions.
        # SEC: Iterate over a tuple snapshot of SPECIAL_TOKENS so the class
        # constant cannot be mutated by accident through the loop variable.
        self.token_to_id = {}
        self.id_to_token = {}
        for idx, token in enumerate(tuple(self.SPECIAL_TOKENS)):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # 2. Count word frequencies across the entire corpus.
        # SEC: build_vocab() does NOT mutate the input list — _preprocess
        # operates on each string and never writes back to `texts`.
        freq: Counter[str] = Counter()
        for text in texts:
            if text is None:
                continue
            if not isinstance(text, str):
                raise TypeError(
                    f"each item in texts must be str, got {type(text).__name__}"
                )
            tokens = self._preprocess(text)
            freq.update(tokens)

        # Remove any special token strings that may appear in the corpus
        # to avoid collisions with reserved indices.
        for special in self.SPECIAL_TOKENS:
            freq.pop(special, None)

        # 3. Keep the top-k most frequent words.
        n_slots = self.vocab_size - len(self.SPECIAL_TOKENS)
        most_common = freq.most_common(n_slots)

        next_id = len(self.SPECIAL_TOKENS)
        for token, _ in most_common:
            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1

        self._built = True

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> list[str]:
        """Preprocess a single text string into a list of word tokens.

        Steps:
        1. Normalise Unicode to NFC form (canonical composition).
        2. Lowercase the entire string.
        3. Separate punctuation from adjacent words with spaces so that
           ``"hello!"`` becomes ``["hello", "!"]``.
        4. Collapse all whitespace (including newlines, tabs) into single
           spaces and strip leading/trailing whitespace.
        5. Split on spaces.

        Numbers are kept as-is (e.g. ``"42"`` is a token).
        Contractions are kept as-is (e.g. ``"don't"`` remains ``"don't"``).

        Args:
            text: Raw input text.

        Returns:
            List of preprocessed word tokens. Returns an empty list for
            empty or whitespace-only input.
        """
        if not text or not text.strip():
            return []

        # Unicode normalisation (NFC).
        text = unicodedata.normalize("NFC", text)

        # Lowercase.
        text = text.lower()

        # Separate punctuation from words by inserting spaces around each
        # punctuation character.
        text = self._PUNCT_PATTERN.sub(r" \1 ", text)

        # Collapse whitespace.
        text = self._WHITESPACE_PATTERN.sub(" ", text).strip()

        if not text:
            return []

        return text.split(" ")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    # SEC: Hard upper bound on input text length to prevent
    # memory exhaustion from a single colossal string (DoS protection).
    MAX_INPUT_CHARS: int = 10_000_000  # 10 MB of text

    def encode(
        self,
        text: str,
        add_special: bool = True,
        max_length: int | None = None,
        padding: bool = False,
    ) -> list[int]:
        """Encode a single text string into a list of token IDs.

        Args:
            text: Raw input text.
            add_special: If ``True``, prepend ``[BOS]`` and append ``[EOS]``
                         to the encoded sequence.
            max_length: If set, truncate (or pad, when *padding* is True) the
                        sequence to exactly this length. Truncation removes
                        tokens from the end but always preserves ``[EOS]`` if
                        *add_special* is True.
            padding: If ``True`` **and** *max_length* is set, pad with
                     ``[PAD]`` tokens to reach *max_length*.

        Returns:
            List of integer token IDs.

        Raises:
            RuntimeError: If the vocabulary has not been built yet.
        """
        if not self._built:
            raise RuntimeError(
                "Vocabulary has not been built. Call build_vocab() first."
            )

        # SEC: Type and length validation. Refuse non-strings (would crash
        # later inside the regex) and refuse pathologically large inputs
        # which could OOM the worker. None becomes empty list.
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise TypeError(
                f"text must be str, got {type(text).__name__}"
            )
        if len(text) > self.MAX_INPUT_CHARS:
            raise ValueError(
                f"text length ({len(text)}) exceeds MAX_INPUT_CHARS "
                f"({self.MAX_INPUT_CHARS})"
            )
        # SEC: max_length validation — must be positive when set.
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")

        tokens = self._preprocess(text)
        ids = [self.token_to_id.get(t, self.UNK_ID) for t in tokens]

        if add_special:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]

        # Truncate if necessary.
        if max_length is not None and len(ids) > max_length:
            # SEC: Special-case max_length == 0 — return empty even if
            # add_special would otherwise insert BOS/EOS.
            if max_length == 0:
                ids = []
            elif add_special and max_length >= 2:
                # Keep BOS at start AND [EOS] at the end after truncation.
                ids = ids[: max_length - 1] + [self.EOS_ID]
            else:
                ids = ids[:max_length]

        # Pad if necessary.
        if padding and max_length is not None:
            pad_count = max_length - len(ids)
            if pad_count > 0:
                ids = ids + [self.PAD_ID] * pad_count

        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs back into a text string.

        Args:
            ids: List of integer token IDs.
            skip_special: If ``True``, omit special tokens (PAD, BOS, EOS,
                          SEP, UNK) from the output.

        Returns:
            Decoded text string with tokens joined by spaces. Out-of-range
            or unknown ids are rendered as ``[UNK]`` rather than crashing.
        """
        # SEC: Defensive type check — refuse non-iterable input early
        # rather than crashing inside the loop with a confusing message.
        if ids is None:
            return ""
        special_ids = set(range(len(self.SPECIAL_TOKENS)))
        tokens: list[str] = []
        for token_id in ids:
            # SEC: Coerce numpy / torch ints to plain Python int so dict
            # lookup works as expected.
            try:
                token_id = int(token_id)
            except (TypeError, ValueError):
                tokens.append("[UNK]")
                continue
            if skip_special and token_id in special_ids:
                continue
            # SEC: id_to_token.get() handles out-of-range ids gracefully
            # by returning "[UNK]" — never raises IndexError.
            token = self.id_to_token.get(token_id, "[UNK]")
            tokens.append(token)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def batch_encode(
        self,
        texts: list[str],
        max_length: int = 256,
        padding: bool = True,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encode a batch of texts with uniform length and attention masks.

        Each text is encoded with ``add_special=True``. Sequences are
        truncated or padded to *max_length*.

        Args:
            texts: List of raw text strings.
            max_length: Target sequence length (truncate or pad).
            padding: Whether to pad shorter sequences to *max_length*.

        Returns:
            A tuple ``(token_ids, attention_masks)`` where each element is a
            list of lists with shape ``[batch_size, max_length]``.
            ``attention_masks`` contain 1 for real tokens and 0 for padding.
        """
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []

        for text in texts:
            ids = self.encode(
                text,
                add_special=True,
                max_length=max_length,
                padding=padding,
            )
            # Build attention mask: 1 where token is not PAD.
            mask = [0 if tok == self.PAD_ID else 1 for tok in ids]
            all_ids.append(ids)
            all_masks.append(mask)

        return all_ids, all_masks

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save vocabulary and configuration to a JSON file.

        The file stores ``vocab_size``, ``token_to_id`` mapping, and the
        special token definitions so that the tokenizer can be fully
        reconstructed later.

        Args:
            path: Filesystem path for the JSON file.
        """
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> SimpleTokenizer:
        """Load a tokenizer from a previously saved JSON file.

        Args:
            path: Filesystem path to the JSON vocabulary file.

        Returns:
            A fully initialised ``SimpleTokenizer`` with the saved vocabulary.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the JSON schema is invalid or special tokens are
                        missing / mis-indexed.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # SEC: Strict schema validation. A maliciously crafted or corrupted
        # vocab JSON could otherwise produce a silently broken tokenizer
        # that mis-indexes special tokens, drops [PAD] from index 0, or
        # introduces duplicate ids that crash later inside the model.
        if not isinstance(data, dict):
            raise ValueError(
                f"Tokenizer JSON must be an object, got {type(data).__name__}"
            )
        if "vocab_size" not in data or "token_to_id" not in data:
            raise ValueError(
                "Tokenizer JSON missing required keys: 'vocab_size', 'token_to_id'"
            )
        vocab_size = data["vocab_size"]
        token_to_id = data["token_to_id"]
        if not isinstance(vocab_size, int) or vocab_size < len(cls.SPECIAL_TOKENS):
            raise ValueError(
                f"Invalid vocab_size: {vocab_size!r}"
            )
        if not isinstance(token_to_id, dict):
            raise ValueError(
                f"token_to_id must be a dict, got {type(token_to_id).__name__}"
            )
        # SEC: Verify every value is a non-negative int strictly less
        # than vocab_size, and that there are no duplicate ids.
        seen_ids: set[int] = set()
        for tok, tid in token_to_id.items():
            if not isinstance(tok, str):
                raise ValueError(
                    f"token_to_id key must be str, got {type(tok).__name__}"
                )
            if not isinstance(tid, int) or tid < 0 or tid >= vocab_size:
                raise ValueError(
                    f"token_to_id[{tok!r}] = {tid!r} out of range [0, {vocab_size})"
                )
            if tid in seen_ids:
                raise ValueError(
                    f"Duplicate token id {tid} for token {tok!r}"
                )
            seen_ids.add(tid)
        # SEC: Verify special tokens occupy their reserved indices exactly.
        for idx, special in enumerate(cls.SPECIAL_TOKENS):
            if token_to_id.get(special) != idx:
                raise ValueError(
                    f"Special token {special!r} must be at index {idx}, "
                    f"got {token_to_id.get(special)!r}"
                )

        tokenizer = cls(vocab_size=vocab_size)
        # SEC: Defensive copy so external mutation of `data` after load
        # cannot affect the tokenizer state.
        tokenizer.token_to_id = dict(token_to_id)
        # JSON keys are always strings; id_to_token needs int keys.
        tokenizer.id_to_token = {
            int(v): k for k, v in tokenizer.token_to_id.items()
        }
        tokenizer._built = True
        return tokenizer

    # ------------------------------------------------------------------
    # Dunder / properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of tokens currently in the vocabulary."""
        return len(self.token_to_id)

    def __repr__(self) -> str:
        return (
            f"SimpleTokenizer(vocab_size={self.vocab_size}, "
            f"built={self._built}, actual={len(self)})"
        )

    @property
    def pad_token_id(self) -> int:
        """Index of the ``[PAD]`` token."""
        return self.PAD_ID

    @property
    def vocab_actual_size(self) -> int:
        """Actual vocabulary size (may be less than max if corpus is small)."""
        return len(self.token_to_id)
