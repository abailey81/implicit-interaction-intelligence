"""Hypothesis property tests for :class:`SimpleTokenizer`.

Invariants:
    * Out-of-vocabulary tokens always encode to ``UNK_ID`` (never crash,
      never produce an out-of-range id).
    * ``encode`` / ``decode`` is a round-trip for tokens that are **in
      vocab**, modulo punctuation-splitting and whitespace normalisation.
    * Encoded ids are always within ``[0, vocab_actual_size)``.
    * ``encode`` always prefixes BOS and suffixes EOS when
      ``add_special=True``.
    * ``encode`` with ``padding=True`` always produces a list of length
      exactly ``max_length``.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings, strategies as st

from i3.slm.tokenizer import SimpleTokenizer


# A small, known corpus used to build a reproducible vocabulary.
_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world how are you today",
    "this is a simple test of the tokenizer",
    "the model learns from text and numbers like 42",
    "we test encode decode round trip behaviours",
]

# A set of known in-vocab tokens that definitely round-trip cleanly.
_IN_VOCAB_TOKENS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "hello", "world", "how", "are", "you", "today",
    "this", "is", "a", "simple", "test", "of", "tokenizer",
    "model", "learns", "from", "text", "numbers",
]


@pytest.fixture(scope="module")
def tokenizer() -> SimpleTokenizer:
    tok = SimpleTokenizer(vocab_size=128)
    tok.build_vocab(_CORPUS)
    return tok


_WORD_STRATEGY = st.text(
    alphabet=st.characters(
        min_codepoint=97, max_codepoint=122,  # lowercase ASCII
    ),
    min_size=1, max_size=20,
)


class TestTokenizerOOV:
    @given(text=_WORD_STRATEGY)
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @example(text="qqqqqqqzzzzzzz")  # near-guaranteed OOV
    def test_oov_maps_to_unk(
        self, tokenizer: SimpleTokenizer, text: str
    ) -> None:
        """A token that is not in vocab encodes to ``UNK_ID``."""
        if text in tokenizer.token_to_id:
            # In-vocab by accident — skip, the invariant is about OOV.
            return
        ids = tokenizer.encode(text, add_special=False)
        assert all(0 <= i < tokenizer.vocab_size for i in ids)
        # If the preprocessor produced at least one token, that token is
        # OOV and must map to UNK.  All-punctuation / empty input yields
        # an empty id list, which is a separately-tested no-op.
        if ids:
            assert SimpleTokenizer.UNK_ID in ids


class TestTokenizerIdRange:
    @given(text=st.text(min_size=0, max_size=200))
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_all_ids_in_range(
        self, tokenizer: SimpleTokenizer, text: str
    ) -> None:
        """Every produced id lies in ``[0, vocab_size)`` and is a valid
        lookup key in ``id_to_token``."""
        ids = tokenizer.encode(text, add_special=True)
        for i in ids:
            assert 0 <= i < tokenizer.vocab_size, i
            # Every emitted id must decode — OOV encodes to UNK_ID (4),
            # which is a special-token id < vocab_actual_size.
            assert i in tokenizer.id_to_token, i


class TestTokenizerRoundTrip:
    @given(
        tokens=st.lists(
            st.sampled_from(_IN_VOCAB_TOKENS),
            min_size=1, max_size=10,
        ),
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @example(tokens=["the", "quick", "brown", "fox"])
    def test_in_vocab_round_trip(
        self, tokenizer: SimpleTokenizer, tokens: list[str]
    ) -> None:
        """encode → decode preserves in-vocab tokens (skipping specials)."""
        text = " ".join(tokens)
        ids = tokenizer.encode(text, add_special=True)
        decoded = tokenizer.decode(ids, skip_special=True)
        assert decoded.split() == tokens


class TestTokenizerSpecials:
    @given(text=st.text(min_size=1, max_size=64))
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_bos_eos_wrap(
        self, tokenizer: SimpleTokenizer, text: str
    ) -> None:
        """When ``add_special=True`` the first id is BOS and last is EOS."""
        ids = tokenizer.encode(text, add_special=True)
        if not ids:
            return
        assert ids[0] == SimpleTokenizer.BOS_ID
        assert ids[-1] == SimpleTokenizer.EOS_ID


class TestTokenizerPadding:
    @given(
        text=st.text(min_size=0, max_size=32),
        max_length=st.integers(min_value=2, max_value=64),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_padded_length_exact(
        self,
        tokenizer: SimpleTokenizer,
        text: str,
        max_length: int,
    ) -> None:
        """With ``padding=True`` output length equals ``max_length``."""
        ids = tokenizer.encode(
            text,
            add_special=True,
            max_length=max_length,
            padding=True,
        )
        assert len(ids) == max_length


class TestTokenizerDecodeRobustness:
    @given(
        ids=st.lists(
            st.integers(min_value=-10_000, max_value=10_000),
            min_size=0, max_size=50,
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_decode_never_raises(
        self, tokenizer: SimpleTokenizer, ids: list[int]
    ) -> None:
        """``decode`` tolerates arbitrary integer ids (renders as [UNK])."""
        out = tokenizer.decode(ids, skip_special=True)
        assert isinstance(out, str)
