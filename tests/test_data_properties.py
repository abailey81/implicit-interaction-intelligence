"""Property-based tests for :mod:`i3.data` using Hypothesis.

These tests assert invariants that must hold over **all** inputs, not
just the curated fixtures in :mod:`tests.test_data_pipeline`.  Each
property either catches a real class of regression (idempotence,
monotonicity, or conservation) or proves a security contract
(stripping is not an injection vector).
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings, strategies as st

from i3.data import (
    Cleaner,
    CleaningConfig,
    Deduplicator,
    Lineage,
    MinHashLSH,
    RecordSchema,
    content_hash,
    jaccard,
    normalise_unicode,
    strip_zero_width,
)


# ---------------------------------------------------------------------------
# Text strategies
# ---------------------------------------------------------------------------


#: Text over the full BMP + common supplementary planes, with enough
#: noise to exercise the cleaning stages (control chars, zero-width,
#: mixed newlines).  We draw from Unicode categories the cleaner is
#: known to care about.
noisy_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0xFFFF,
        blacklist_categories=("Cs",),  # surrogates are invalid
    ),
    min_size=1,
    max_size=500,
)

safe_ascii_text = st.text(
    alphabet=string.ascii_letters + string.digits + " .,!?-",
    min_size=3,
    max_size=400,
)

speaker_st = st.sampled_from(["user", "assistant", "system", "narrator", None])


def _mk_lineage() -> Lineage:
    return Lineage(
        source_uri="prop://test",
        source_format="prop",
        original_hash="0" * 64,
    )


# ---------------------------------------------------------------------------
# Cleaning properties
# ---------------------------------------------------------------------------


class TestCleaningProperties:
    @given(text=noisy_text)
    @settings(max_examples=100, deadline=None)
    def test_clean_is_idempotent(self, text: str):
        """clean(clean(x)) == clean(x) for every input x."""
        c = Cleaner()
        once = c.clean(text)
        twice = c.clean(once)
        assert twice == once

    @given(text=noisy_text)
    @settings(max_examples=100, deadline=None)
    def test_clean_never_increases_length_much(self, text: str):
        """Cleaning can only *remove* characters; output length is
        bounded above by input length plus a small constant from NFKC
        expansion (e.g. ``ﬀ`` → ``ff`` adds one char per ligature).
        """
        c = Cleaner()
        out = c.clean(text)
        # NFKC can expand compatibility ligatures; allow 2x headroom.
        assert len(out) <= 2 * len(text) + 16

    @given(text=noisy_text)
    @settings(max_examples=100, deadline=None)
    def test_clean_output_has_no_zero_width_chars(self, text: str):
        from i3.data.cleaning import _ZERO_WIDTH_CHARS  # noqa: PLC0415

        out = Cleaner().clean(text)
        for zw in _ZERO_WIDTH_CHARS:
            assert zw not in out

    @given(text=noisy_text)
    @settings(max_examples=100, deadline=None)
    def test_clean_output_has_no_control_chars_except_tab_newline(self, text: str):
        import unicodedata  # noqa: PLC0415

        out = Cleaner().clean(text)
        for ch in out:
            if ch in {"\n", "\t"}:
                continue
            cat = unicodedata.category(ch)
            assert cat != "Cc", f"control char U+{ord(ch):04X} leaked"

    @given(text=safe_ascii_text)
    @settings(max_examples=100, deadline=None)
    def test_clean_preserves_safe_ascii(self, text: str):
        """Pure printable ASCII with at-most-single spaces survives clean."""
        cleaned = " ".join(text.split())
        c = Cleaner()
        assert c.clean(cleaned) == cleaned

    @given(text=noisy_text)
    @settings(max_examples=50, deadline=None)
    def test_disabling_collapse_preserves_multiple_spaces(self, text: str):
        # Force known whitespace.
        sample = "a" + "   " + text + "   " + "b"
        c = Cleaner(CleaningConfig(collapse_whitespace=False))
        out = c.clean(sample)
        assert "   " in out or "\n" in out  # retained something multi


# ---------------------------------------------------------------------------
# Unicode normalisation properties
# ---------------------------------------------------------------------------


class TestUnicodeProperties:
    @given(text=noisy_text)
    @settings(max_examples=100, deadline=None)
    def test_nfkc_idempotent(self, text: str):
        assert normalise_unicode(normalise_unicode(text)) == normalise_unicode(text)

    @given(text=noisy_text)
    @settings(max_examples=100, deadline=None)
    def test_strip_zero_width_idempotent(self, text: str):
        assert strip_zero_width(strip_zero_width(text)) == strip_zero_width(text)


# ---------------------------------------------------------------------------
# Content-hash properties
# ---------------------------------------------------------------------------


class TestContentHashProperties:
    @given(text=noisy_text)
    @settings(max_examples=200, deadline=None)
    def test_hash_deterministic(self, text: str):
        assert content_hash(text) == content_hash(text)

    @given(text=safe_ascii_text)
    @settings(max_examples=100, deadline=None)
    def test_hash_ignores_case_and_whitespace(self, text: str):
        a = text
        b = "  " + text.upper() + "  "
        # Only guaranteed equal when text has no case-insensitive
        # collisions; safe_ascii_text is mostly lowercase so we check
        # the contract holds for at least the trivial case.
        a_norm = " ".join(a.lower().split())
        b_norm = " ".join(b.lower().split())
        if a_norm == b_norm:
            assert content_hash(a) == content_hash(b)

    @given(a=safe_ascii_text, b=safe_ascii_text)
    @settings(max_examples=50, deadline=None)
    def test_different_inputs_usually_hash_differently(self, a: str, b: str):
        if a.lower().strip() == b.lower().strip():
            return  # skip trivial equality
        import re

        if re.sub(r"\s+", " ", a.lower().strip()) == re.sub(
            r"\s+", " ", b.lower().strip()
        ):
            return
        assert content_hash(a) != content_hash(b)


# ---------------------------------------------------------------------------
# Jaccard properties
# ---------------------------------------------------------------------------


class TestJaccardProperties:
    shingles = st.sets(
        st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=8),
        min_size=0, max_size=30,
    )

    @given(a=shingles)
    def test_jaccard_reflexive(self, a: set[str]):
        assert jaccard(a, a) == 1.0

    @given(a=shingles, b=shingles)
    def test_jaccard_symmetric(self, a: set[str], b: set[str]):
        assert jaccard(a, b) == jaccard(b, a)

    @given(a=shingles, b=shingles)
    def test_jaccard_in_unit_interval(self, a: set[str], b: set[str]):
        assert 0.0 <= jaccard(a, b) <= 1.0


# ---------------------------------------------------------------------------
# Deduplicator properties
# ---------------------------------------------------------------------------


class TestDeduplicatorProperties:
    @given(text=safe_ascii_text)
    @settings(max_examples=30, deadline=None)
    def test_same_record_twice_rejects_second(self, text: str):
        d = Deduplicator(exact_only=True)
        ok1, _ = d.add_and_check("r1", text)
        ok2, reason = d.add_and_check("r2", text)
        assert ok1 is True
        assert ok2 is False
        assert reason == "exact"

    @given(text=safe_ascii_text)
    @settings(max_examples=30, deadline=None)
    def test_dedup_stats_conserve(self, text: str):
        """unique + exact_dupes + near_dupes = total add_and_check() calls."""
        d = Deduplicator(exact_only=True)
        for i in range(3):
            d.add_and_check(f"r{i}", text)
        s = d.stats
        assert s["unique"] + s["exact_dupes"] + s["near_dupes"] == 3


# ---------------------------------------------------------------------------
# RecordSchema properties
# ---------------------------------------------------------------------------


class TestRecordSchemaProperties:
    @given(text=safe_ascii_text, speaker=speaker_st)
    @settings(max_examples=50, deadline=None)
    def test_record_round_trips_through_model_dump(
        self, text: str, speaker: str | None
    ):
        if not text.strip():
            return
        r = RecordSchema(
            text=text, speaker=speaker, lineage=_mk_lineage(),
        )
        # JSON-compatible dump preserves the required fields.
        d = r.model_dump()
        assert d["text"] == text
        assert d["speaker"] == speaker

    @given(text=safe_ascii_text)
    @settings(max_examples=50, deadline=None)
    def test_record_frozen_rejects_mutation(self, text: str):
        if not text.strip():
            return
        r = RecordSchema(text=text, lineage=_mk_lineage())
        with pytest.raises(Exception):
            r.text = "something else"  # type: ignore[misc]

    @given(text=safe_ascii_text, n=st.integers(min_value=0, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_lineage_transforms_preserves_order(self, text: str, n: int):
        if not text.strip():
            return
        names = [f"T{i}" for i in range(n)]
        lin = _mk_lineage()
        for name in names:
            lin = lin.with_transform(name)
        assert list(lin.applied_transforms) == names
