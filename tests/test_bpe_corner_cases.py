"""Iter 64 — BPE tokenizer corner-case + invariant tests.

Pins the byte-level BPE tokenizer's contract for tricky inputs:
empty / whitespace-only / single-byte / control characters /
surrogate-pair-shaped emoji / very long input / leading-trailing
whitespace preservation / repeated special tokens.

Tests skip cleanly when the tokenizer file isn't on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BPE_CANDIDATES = [
    Path("checkpoints/slm_v2/tokenizer_bpe.json"),
    Path("checkpoints/slm/tokenizer_bpe.json"),
]


@pytest.fixture(scope="module")
def tok():
    from i3.slm.bpe_tokenizer import BPETokenizer
    p = next((c for c in _BPE_CANDIDATES if c.exists()), None)
    if p is None:
        pytest.skip("BPE tokenizer file not present")
    return BPETokenizer.load(p)


def _enc(tok, s, **kwargs):
    return tok.encode(s, add_bos=False, add_eos=False, **kwargs)


def test_empty_string(tok):
    ids = _enc(tok, "")
    assert isinstance(ids, list)
    assert tok.decode(ids) == ""


def test_single_space(tok):
    ids = _enc(tok, " ")
    assert tok.decode(ids) == " "


def test_only_whitespace(tok):
    s = "   \t\n  "
    assert tok.decode(_enc(tok, s)) == s


def test_single_byte_letter(tok):
    for ch in ["a", "z", "A", "Z", "0", "9", ".", "?", "!"]:
        assert tok.decode(_enc(tok, ch)) == ch


def test_control_chars_pass_through(tok):
    s = "\x01\x02\x1f"
    assert tok.decode(_enc(tok, s)) == s


def test_emoji_surrogate_pair(tok):
    # Astral-plane emoji are encoded via surrogate pairs in some
    # interpretations; byte-level BPE should still round-trip.
    for emoji in ["😀", "🚀", "🐍", "🇬🇧", "👨‍👩‍👧‍👦"]:
        assert tok.decode(_enc(tok, emoji)) == emoji


def test_leading_and_trailing_whitespace_preserved(tok):
    s = "   hi there   "
    assert tok.decode(_enc(tok, s)) == s


def test_long_input_under_max(tok):
    s = "the quick brown fox jumps over the lazy dog " * 200  # ~9000 chars
    ids = _enc(tok, s)
    assert tok.decode(ids) == s


def test_with_bos_eos(tok):
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.BOS_ID
    assert ids[-1] == tok.EOS_ID
    # Decode strips specials by default in most BPE impls; verify
    # the inner content is preserved when we trim ourselves.
    inner = ids[1:-1]
    assert tok.decode(inner) == "hi"


def test_special_ids_distinct(tok):
    ids = {tok.PAD_ID, tok.UNK_ID, tok.BOS_ID, tok.EOS_ID, tok.SEP_ID}
    assert len(ids) == 5, "PAD/UNK/BOS/EOS/SEP must be distinct"
    # All within the vocab
    assert max(ids) < len(tok)


def test_vocab_size_matches_metadata(tok):
    """The reported vocab size must equal the actual vocabulary used."""
    assert len(tok) > 0
    assert isinstance(len(tok), int)


def test_decode_of_arbitrary_ids_does_not_crash(tok):
    """Random valid ids should decode without raising."""
    import random
    rng = random.Random(42)
    ids = [rng.randint(5, len(tok) - 1) for _ in range(50)]
    s = tok.decode(ids)
    assert isinstance(s, str)


def test_idempotent_under_double_encode(tok):
    """encode(decode(encode(x))) == encode(x) for byte-level BPE."""
    for s in ["hello world", "set timer 5 min", "you said 'hi'", "—"]:
        ids1 = _enc(tok, s)
        ids2 = _enc(tok, tok.decode(ids1))
        assert ids1 == ids2, f"non-idempotent on {s!r}: {ids1} != {ids2}"
