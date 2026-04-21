"""Hypothesis property tests for :class:`PrivacySanitizer`.

Invariants:
    * **Idempotency** — ``sanitize(sanitize(text))`` equals the first
      sanitised result.  Redacting a redacted string does not introduce
      new placeholders.
    * **Non-PII preservation** — text that contains no PII comes back
      unchanged.
    * **PII always masked** — known PII patterns (emails, SSNs, credit
      cards, phone numbers) are always replaced.
    * **No crash on arbitrary strings** — the sanitiser never raises on
      any unicode input.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, example, given, settings, strategies as st

from i3.privacy.sanitizer import PrivacySanitizer


# Characters that absolutely cannot match any PII pattern we ship.  We
# keep strategy text small and lowercase-alpha so noise cannot accidentally
# form an email / phone / SSN.
_SAFE_ALPHA = st.characters(
    min_codepoint=97,  # "a"
    max_codepoint=122,  # "z"
)

_SAFE_TEXT = st.text(alphabet=_SAFE_ALPHA, min_size=0, max_size=200)

_ANY_TEXT = st.text(min_size=0, max_size=200)


@pytest.fixture(scope="module")
def sanitizer() -> PrivacySanitizer:
    return PrivacySanitizer(enabled=True)


# ─────────────────────────────────────────────────────────────────────────
#  Idempotency
# ─────────────────────────────────────────────────────────────────────────


class TestSanitizerIdempotency:
    @given(text=_ANY_TEXT)
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @example(text="contact me at alice@example.com any time")
    @example(text="SSN 123-45-6789 and phone 555-123-4567")
    def test_sanitize_is_idempotent(
        self, sanitizer: PrivacySanitizer, text: str
    ) -> None:
        """Applying sanitize twice yields the same text as applying once."""
        first = sanitizer.sanitize(text).sanitized_text
        second = sanitizer.sanitize(first).sanitized_text
        assert first == second


# ─────────────────────────────────────────────────────────────────────────
#  Non-PII preservation
# ─────────────────────────────────────────────────────────────────────────


class TestSanitizerPreservesNonPII:
    @given(text=_SAFE_TEXT)
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_no_pii_means_no_change(
        self, sanitizer: PrivacySanitizer, text: str
    ) -> None:
        """Lowercase-alpha text contains no PII — it must be unchanged."""
        result = sanitizer.sanitize(text)
        assert result.sanitized_text == text
        assert result.pii_detected is False
        assert result.replacements_made == 0

    @given(text=st.sampled_from([
        "hello world",
        "this is a test message",
        "the quick brown fox",
        "no personal data here",
        "",
    ]))
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_known_clean_strings_unchanged(
        self, sanitizer: PrivacySanitizer, text: str
    ) -> None:
        assert sanitizer.sanitize(text).sanitized_text == text


# ─────────────────────────────────────────────────────────────────────────
#  PII masking
# ─────────────────────────────────────────────────────────────────────────


class TestSanitizerMasksPII:
    @given(
        local=st.text(
            alphabet=_SAFE_ALPHA, min_size=3, max_size=10,
        ),
        domain=st.sampled_from(["example.com", "foo.org", "bar.net"]),
        wrapper_prefix=_SAFE_TEXT,
        wrapper_suffix=_SAFE_TEXT,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_email_always_masked(
        self,
        sanitizer: PrivacySanitizer,
        local: str,
        domain: str,
        wrapper_prefix: str,
        wrapper_suffix: str,
    ) -> None:
        raw = f"{wrapper_prefix} {local}@{domain} {wrapper_suffix}".strip()
        result = sanitizer.sanitize(raw)
        assert "[EMAIL]" in result.sanitized_text
        assert f"{local}@{domain}" not in result.sanitized_text
        assert result.pii_detected is True

    def test_ssn_always_masked(self, sanitizer: PrivacySanitizer) -> None:
        result = sanitizer.sanitize("user ssn 123-45-6789 filed")
        assert "[SSN]" in result.sanitized_text
        assert "123-45-6789" not in result.sanitized_text

    def test_credit_card_always_masked(
        self, sanitizer: PrivacySanitizer
    ) -> None:
        result = sanitizer.sanitize("card 4111 1111 1111 1111 charged")
        assert "[CREDIT_CARD]" in result.sanitized_text


# ─────────────────────────────────────────────────────────────────────────
#  No-crash property on arbitrary unicode
# ─────────────────────────────────────────────────────────────────────────


class TestSanitizerNoCrash:
    @given(text=_ANY_TEXT)
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_never_raises(
        self, sanitizer: PrivacySanitizer, text: str
    ) -> None:
        """Arbitrary unicode input never crashes the sanitiser."""
        result = sanitizer.sanitize(text)
        assert isinstance(result.sanitized_text, str)
        assert isinstance(result.pii_detected, bool)
        assert result.replacements_made >= 0

    @given(text=_ANY_TEXT)
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_contains_pii_never_raises(
        self, sanitizer: PrivacySanitizer, text: str
    ) -> None:
        assert isinstance(sanitizer.contains_pii(text), bool)
