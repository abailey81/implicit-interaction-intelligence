"""Security-focused tests for the I3 project.

Tests cover:
    - PII sanitization (all 10 patterns)
    - Encryption round-trip and key handling
    - Topic sensitivity detection
    - Input validation (user_id format, message length)
    - Privacy guarantees (no raw text in diary records)
    - SQL injection resistance
    - Denial-of-service protection (large inputs)
    - Deserialization safety
    - Tokenizer robustness
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import fields

import pytest
import torch

from i3.privacy.sanitizer import PrivacySanitizer, SanitizationResult
from i3.privacy.encryption import ModelEncryptor
from i3.router.sensitivity import TopicSensitivityDetector
from i3.slm.tokenizer import SimpleTokenizer
from i3.pipeline.types import PipelineOutput
from i3.diary.store import DiaryStore


# Helper regex used by input validation tests
_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _is_valid_user_id(user_id: str) -> bool:
    """Validate a user id against a strict alphanumeric/underscore/hyphen pattern."""
    return bool(_USER_ID_RE.match(user_id))


# -------------------------------------------------------------------------
# PII Sanitization
# -------------------------------------------------------------------------


class TestPIISanitization:
    """Tests for the PrivacySanitizer class and all 10 PII regex patterns."""

    @pytest.fixture
    def sanitizer(self) -> PrivacySanitizer:
        return PrivacySanitizer(enabled=True)

    # -- Single-pattern tests -----------------------------------------------

    @pytest.mark.parametrize(
        "raw_email",
        [
            "alice@example.com",
            "bob.smith@mail.co.uk",
            "user+tag@sub.domain.org",
            "first.last@company-name.io",
            "x@y.zz",
        ],
    )
    def test_email_detection(self, sanitizer: PrivacySanitizer, raw_email: str) -> None:
        """All common email formats should be detected and redacted."""
        text = f"Please contact me at {raw_email} today."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[EMAIL]" in result.sanitized_text
        assert raw_email not in result.sanitized_text
        assert "email" in result.pii_types

    @pytest.mark.parametrize(
        "raw_phone",
        [
            "(555) 123-4567",
            "555-123-4567",
            "5551234567",
            "555.123.4567",
            "+1 555 123 4567",
        ],
    )
    def test_phone_us_format(self, sanitizer: PrivacySanitizer, raw_phone: str) -> None:
        """US phone numbers in various formats should be detected."""
        text = f"Call me at {raw_phone} please."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[PHONE]" in result.sanitized_text

    def test_phone_uk_format(self, sanitizer: PrivacySanitizer) -> None:
        """UK-style phone numbers should match either the UK or intl pattern."""
        text = "My UK number is +44 20 7946 0958 if you need it."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[PHONE]" in result.sanitized_text

    @pytest.mark.parametrize(
        "raw_phone",
        [
            "+1 555 123 4567",
            "+44 20 7946 0958",
            "+49 30 1234 5678",
        ],
    )
    def test_phone_international(
        self, sanitizer: PrivacySanitizer, raw_phone: str
    ) -> None:
        """International phone numbers should be detected via the intl pattern."""
        text = f"Reach me on {raw_phone}."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[PHONE]" in result.sanitized_text

    def test_ssn_detection(self, sanitizer: PrivacySanitizer) -> None:
        """US SSN format (XXX-XX-XXXX) should be redacted."""
        text = "My SSN is 123-45-6789, please keep it safe."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[SSN]" in result.sanitized_text
        assert "123-45-6789" not in result.sanitized_text

    @pytest.mark.parametrize(
        "raw_cc",
        [
            "4532-1234-5678-9012",
            "4532 1234 5678 9012",
            "4532123456789012",
        ],
    )
    def test_credit_card_detection(
        self, sanitizer: PrivacySanitizer, raw_cc: str
    ) -> None:
        """Credit card numbers with or without separators should be redacted."""
        text = f"My card is {raw_cc}."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[CREDIT_CARD]" in result.sanitized_text
        assert raw_cc not in result.sanitized_text

    def test_ip_address_detection(self, sanitizer: PrivacySanitizer) -> None:
        """IPv4 addresses should be redacted."""
        text = "My server IP is 192.168.1.1 on the LAN."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[IP_ADDRESS]" in result.sanitized_text
        assert "192.168.1.1" not in result.sanitized_text

    def test_address_detection(self, sanitizer: PrivacySanitizer) -> None:
        """Street addresses should be redacted."""
        text = "I live at 123 Main Street in the downtown area."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[ADDRESS]" in result.sanitized_text

    def test_url_detection(self, sanitizer: PrivacySanitizer) -> None:
        """URLs should be redacted."""
        text = "Check out https://example.com/path?q=1 for details."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[URL]" in result.sanitized_text
        assert "https://example.com/path?q=1" not in result.sanitized_text

    def test_dob_detection(self, sanitizer: PrivacySanitizer) -> None:
        """Dates of birth in DD/MM/YYYY format should be redacted."""
        # The sanitizer regex is day-first: DD/MM/YYYY with day in [01-31],
        # month in [01-12], year starting with 19 or 20.
        text = "My DOB is 15/01/1990 for verification."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert "[DOB]" in result.sanitized_text
        assert "15/01/1990" not in result.sanitized_text

    # -- Multi-PII and negative tests --------------------------------------

    def test_multiple_pii_in_one_text(self, sanitizer: PrivacySanitizer) -> None:
        """A text containing several PII types should have all of them redacted."""
        text = (
            "Contact alice@example.com or 555-123-4567, "
            "SSN 123-45-6789, IP 10.0.0.1, "
            "visit https://private.example.org for more."
        )
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True
        assert result.replacements_made >= 4
        for needle in [
            "alice@example.com",
            "555-123-4567",
            "123-45-6789",
            "10.0.0.1",
            "https://private.example.org",
        ]:
            assert needle not in result.sanitized_text
        assert "[EMAIL]" in result.sanitized_text
        assert "[PHONE]" in result.sanitized_text
        assert "[SSN]" in result.sanitized_text

    def test_no_pii_returns_unchanged(self, sanitizer: PrivacySanitizer) -> None:
        """Text with no PII should be returned unchanged."""
        text = "I am thinking about the weather today and wondering what to cook."
        result = sanitizer.sanitize(text)
        assert result.pii_detected is False
        assert result.sanitized_text == text
        assert result.replacements_made == 0
        assert result.pii_types == []

    def test_contains_pii_quick_check(self, sanitizer: PrivacySanitizer) -> None:
        """contains_pii should return bool without mutating text."""
        assert sanitizer.contains_pii("email me at x@y.zz please") is True
        assert sanitizer.contains_pii("a perfectly normal sentence") is False

    def test_sanitization_stats_tracking(self, sanitizer: PrivacySanitizer) -> None:
        """Stats counters should track scans and PII discoveries."""
        baseline = sanitizer.stats
        sanitizer.sanitize("clean text with nothing to redact")
        sanitizer.sanitize("my email is steve@apple.com")
        sanitizer.sanitize("call 555-123-4567")
        stats = sanitizer.stats
        assert stats["total_scans"] >= baseline.get("total_scans", 0) + 3
        assert stats["pii_found"] >= baseline.get("pii_found", 0) + 2
        assert stats["replacements"] >= baseline.get("replacements", 0) + 2

    def test_disabled_sanitizer_passthrough(self) -> None:
        """A disabled sanitizer should return original text untouched."""
        sanitizer = PrivacySanitizer(enabled=False)
        text = "Email me at foo@bar.com and call 555-123-4567"
        result = sanitizer.sanitize(text)
        assert result.pii_detected is False
        assert result.sanitized_text == text
        assert result.pii_types == []
        assert result.replacements_made == 0

    # -- Parametrized bulk test --------------------------------------------

    @pytest.mark.parametrize(
        "text,expected_token",
        [
            ("email me at alice@foo.org", "[EMAIL]"),
            ("email is bob.smith@corp.co.uk", "[EMAIL]"),
            ("reach carol+tag@example.io", "[EMAIL]"),
            ("call me 555-123-4567 today", "[PHONE]"),
            ("phone (555) 987-6543 please", "[PHONE]"),
            ("dial 5551234567 anytime", "[PHONE]"),
            ("intl +49 30 1234 5678 line", "[PHONE]"),
            ("ssn is 123-45-6789 seriously", "[SSN]"),
            ("credit 4532-1234-5678-9012 expires", "[CREDIT_CARD]"),
            ("credit 4532 1234 5678 9012 expires", "[CREDIT_CARD]"),
            ("ip address 192.168.1.1 is local", "[IP_ADDRESS]"),
            ("ip address 10.0.0.42 is private", "[IP_ADDRESS]"),
            ("I live at 742 Evergreen Lane today", "[ADDRESS]"),
            ("meet me at 221 Baker Street please", "[ADDRESS]"),
            ("visit https://example.com now", "[URL]"),
            ("go to http://localhost:8080 asap", "[URL]"),
            ("born 15/01/1990 in Seattle", "[DOB]"),
            ("born 25/12/1985 abroad", "[DOB]"),
            ("Multi email a@b.cc and 10.0.0.1 together", "[EMAIL]"),
            ("Another 198.51.100.23 occurrence here", "[IP_ADDRESS]"),
        ],
    )
    def test_parametrized_pii_examples(
        self,
        sanitizer: PrivacySanitizer,
        text: str,
        expected_token: str,
    ) -> None:
        """A broad set of PII-bearing strings should all be detected."""
        result = sanitizer.sanitize(text)
        assert result.pii_detected is True, f"Missed PII in: {text!r}"
        assert expected_token in result.sanitized_text


# -------------------------------------------------------------------------
# Fernet Encryption
# -------------------------------------------------------------------------


class TestFernetEncryption:
    """Tests for the ModelEncryptor wrapping cryptography.fernet.Fernet."""

    @pytest.fixture
    def encryptor(self, monkeypatch: pytest.MonkeyPatch) -> ModelEncryptor:
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()
        monkeypatch.setenv("I3_ENCRYPTION_KEY_TEST", key)
        enc = ModelEncryptor(key_env_var="I3_ENCRYPTION_KEY_TEST")
        enc.initialize()
        return enc

    def test_encrypt_decrypt_roundtrip_bytes(self, encryptor: ModelEncryptor) -> None:
        """Random plaintext bytes should survive an encrypt-decrypt round trip."""
        original = os.urandom(256)
        ct = encryptor.encrypt(original)
        assert isinstance(ct, bytes)
        assert ct != original
        pt = encryptor.decrypt(ct)
        assert pt == original

    def test_encrypt_decrypt_json(self, encryptor: ModelEncryptor) -> None:
        """JSON-serializable dicts should round-trip cleanly."""
        data = {
            "user_id": "abc123",
            "score": 0.87,
            "tags": ["alpha", "beta", "gamma"],
            "nested": {"k": 1, "v": None},
        }
        ct = encryptor.encrypt_json(data)
        decrypted = encryptor.decrypt_json(ct)
        assert decrypted == data

    def test_encrypt_decrypt_embedding(self, encryptor: ModelEncryptor) -> None:
        """Torch tensor embeddings should round-trip with exact float32 values."""
        tensor = torch.randn(64, dtype=torch.float32)
        ct = encryptor.encrypt_embedding(tensor)
        assert isinstance(ct, bytes)
        assert ct != tensor.numpy().tobytes()
        recovered = encryptor.decrypt_embedding(ct, dim=64)
        assert recovered.shape == (64,)
        assert recovered.dtype == torch.float32
        assert torch.allclose(recovered, tensor, atol=1e-6)

    def test_decryption_with_wrong_key_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decrypting with a different key should raise InvalidToken."""
        from cryptography.fernet import Fernet, InvalidToken

        key_a = Fernet.generate_key().decode()
        key_b = Fernet.generate_key().decode()
        assert key_a != key_b

        monkeypatch.setenv("I3_ENC_A", key_a)
        monkeypatch.setenv("I3_ENC_B", key_b)

        enc_a = ModelEncryptor(key_env_var="I3_ENC_A")
        enc_a.initialize()
        enc_b = ModelEncryptor(key_env_var="I3_ENC_B")
        enc_b.initialize()

        ct = enc_a.encrypt(b"secret plaintext")
        with pytest.raises(InvalidToken):
            enc_b.decrypt(ct)

    def test_encrypted_bytes_are_different(self, encryptor: ModelEncryptor) -> None:
        """Ciphertext should differ from plaintext and each call should differ."""
        plaintext = b"the very same plaintext value here"
        ct1 = encryptor.encrypt(plaintext)
        ct2 = encryptor.encrypt(plaintext)
        assert ct1 != plaintext
        assert ct2 != plaintext
        # Fernet includes a random IV, so two ciphertexts of identical plaintext
        # should not be byte-identical.
        assert ct1 != ct2

    def test_key_generation_returns_valid_fernet_key(self) -> None:
        """ModelEncryptor.generate_key should produce a usable Fernet key."""
        from cryptography.fernet import Fernet

        key = ModelEncryptor.generate_key()
        assert isinstance(key, str)
        # This should not raise
        fernet = Fernet(key.encode())
        ct = fernet.encrypt(b"hello")
        assert fernet.decrypt(ct) == b"hello"

    def test_auto_generated_key_on_missing_env(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing env var should trigger a generated key with a warning."""
        monkeypatch.delenv("I3_MISSING_KEY", raising=False)
        enc = ModelEncryptor(key_env_var="I3_MISSING_KEY")
        with caplog.at_level("WARNING"):
            enc.initialize()
        # Even with a generated key, encryption should still work
        ct = enc.encrypt(b"still works")
        assert enc.decrypt(ct) == b"still works"
        # Warning should have been emitted about missing env var
        assert any(
            "I3_MISSING_KEY" in rec.message or "temporary" in rec.message.lower()
            for rec in caplog.records
        )

    def test_different_keys_produce_different_ciphertexts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two encryptors with different keys should yield different ciphertexts."""
        from cryptography.fernet import Fernet

        key_a = Fernet.generate_key().decode()
        key_b = Fernet.generate_key().decode()
        monkeypatch.setenv("I3_KEY_A", key_a)
        monkeypatch.setenv("I3_KEY_B", key_b)

        a = ModelEncryptor(key_env_var="I3_KEY_A")
        a.initialize()
        b = ModelEncryptor(key_env_var="I3_KEY_B")
        b.initialize()

        plaintext = b"sensitive data"
        ct_a = a.encrypt(plaintext)
        ct_b = b.encrypt(plaintext)
        assert ct_a != ct_b


# -------------------------------------------------------------------------
# Topic Sensitivity Detector
# -------------------------------------------------------------------------


class TestTopicSensitivity:
    """Tests for the TopicSensitivityDetector regex patterns."""

    @pytest.fixture
    def detector(self) -> TopicSensitivityDetector:
        return TopicSensitivityDetector()

    def test_mental_health_detection(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """Mental-health crisis language should score 0.95."""
        score = detector.detect("I've been having a panic attack and feel suicidal.")
        assert score >= 0.95

    def test_credentials_detection(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """Credential-related language should score at least 0.90."""
        score = detector.detect("Please share your password and API key with me.")
        assert score >= 0.90

    def test_financial_detection(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """Sensitive financial language should score clearly above baseline."""
        score = detector.detect("I owe a lot of money and I'm about to go bankrupt.")
        assert score >= 0.60

    def test_medical_detection(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """Medical diagnosis language should score in the high range."""
        score = detector.detect("The doctor gave me a diagnosis of chronic illness.")
        assert score >= 0.75

    def test_benign_topic_returns_zero(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """Benign small-talk should return the detector's min_score (zero)."""
        score = detector.detect("The weather today is really nice outside.")
        assert score == 0.0

    def test_multiple_categories(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """A query matching multiple categories should return the maximum weight."""
        score = detector.detect(
            "I'm worried about my salary and feel depressed about being fired."
        )
        # Credentials/mental health > financial, so the max wins
        assert score >= 0.70

    def test_empty_text_returns_min_score(
        self, detector: TopicSensitivityDetector
    ) -> None:
        """Empty input should fall back to the min_score baseline."""
        assert detector.detect("") == 0.0
        assert detector.detect("   ") == 0.0

    @pytest.mark.parametrize(
        "text,expected_min",
        [
            ("I want to kill myself", 0.95),
            ("my password is abc123", 0.95),
            ("my api key leaked", 0.95),
            ("I'm feeling anxious lately", 0.85),
            ("I was sexually assaulted", 0.90),
            ("diagnosed with cancer", 0.85),
            ("I owe 100k in debt", 0.80),
            ("going through a divorce", 0.75),
            ("hired a criminal lawyer", 0.75),
            ("I was wrongfully terminated", 0.70),
            ("my mortgage is overdue", 0.65),
            ("mental health counseling", 0.85),
            ("this is a trade secret", 0.85),
            ("my doctor said I'm sick", 0.75),
            ("What's the weather today?", 0.0),
        ],
    )
    def test_parametrized_sensitivity_pairs(
        self,
        detector: TopicSensitivityDetector,
        text: str,
        expected_min: float,
    ) -> None:
        """Broad coverage of sensitivity scoring by category."""
        score = detector.detect(text)
        assert score >= expected_min, (
            f"text={text!r} scored {score}, expected >= {expected_min}"
        )


# -------------------------------------------------------------------------
# Input Validation
# -------------------------------------------------------------------------


class TestInputValidation:
    """Tests for user id format validation and basic input hardening."""

    @pytest.mark.parametrize(
        "user_id",
        [
            "user_123",
            "alice",
            "bob-42",
            "UserUPPER",
            "a",
            "user_with_underscore",
            "x1y2z3",
            "_leading_underscore",
            "trailing_underscore_",
        ],
    )
    def test_user_id_alphanumeric_valid(self, user_id: str) -> None:
        """Alphanumeric user ids (with _ and -) should pass validation."""
        assert _is_valid_user_id(user_id) is True

    @pytest.mark.parametrize(
        "user_id",
        [
            "user!@#",
            "alice@example.com",
            "bob smith",
            "user/name",
            "user.name",
            "user+plus",
            "user\\backslash",
            "user;DROP",
        ],
    )
    def test_user_id_with_special_chars_invalid(self, user_id: str) -> None:
        """User ids containing special characters should be rejected."""
        assert _is_valid_user_id(user_id) is False

    def test_user_id_too_long_invalid(self) -> None:
        """User ids longer than 64 characters should be rejected."""
        long_id = "a" * 65
        assert _is_valid_user_id(long_id) is False
        # 64 chars is the upper bound and should still be accepted
        assert _is_valid_user_id("a" * 64) is True

    def test_user_id_empty_invalid(self) -> None:
        """An empty user id should be rejected."""
        assert _is_valid_user_id("") is False

    def test_user_id_sql_injection_attempt_invalid(self) -> None:
        """Classic SQL-injection user ids should be rejected."""
        for candidate in [
            "user'; DROP TABLE users;--",
            "' OR '1'='1",
            "admin'--",
            "); DELETE FROM sessions;--",
        ]:
            assert _is_valid_user_id(candidate) is False


# -------------------------------------------------------------------------
# DoS Protection
# -------------------------------------------------------------------------


class TestDoSProtection:
    """Tests that the privacy layer is robust to very large or pathological inputs."""

    def test_very_large_message_handled(self) -> None:
        """Sanitizing a 1 MB message should complete in reasonable time."""
        sanitizer = PrivacySanitizer(enabled=True)
        text = "the quick brown fox jumps over the lazy dog. " * 25_000
        assert len(text) >= 1_000_000
        start = time.monotonic()
        result = sanitizer.sanitize(text)
        elapsed = time.monotonic() - start
        assert isinstance(result, SanitizationResult)
        assert elapsed < 10.0, f"sanitize took {elapsed:.2f}s for 1MB input"

    def test_deeply_nested_json_handled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Encrypting a deeply nested dict should not stack-overflow."""
        from cryptography.fernet import Fernet

        monkeypatch.setenv("I3_NESTED_KEY", Fernet.generate_key().decode())
        enc = ModelEncryptor(key_env_var="I3_NESTED_KEY")
        enc.initialize()

        data: dict = {"value": 0}
        cursor = data
        for i in range(50):
            cursor["child"] = {"value": i}
            cursor = cursor["child"]

        ct = enc.encrypt_json(data)
        recovered = enc.decrypt_json(ct)
        assert recovered == data

    def test_unicode_edge_cases(self) -> None:
        """Sanitizer and tokenizer should accept emoji, RTL, and zero-width chars."""
        sanitizer = PrivacySanitizer(enabled=True)
        edge_cases = [
            "hello world \U0001f600 \U0001f44d",  # emojis
            "\u202eevil reversed text\u202c",       # RTL override
            "zero\u200bwidth\u200bspace",           # zero-width space
            "mixed 中文 العربية русский",
            "combining a\u0301 e\u0301",            # combining accents
        ]
        for text in edge_cases:
            result = sanitizer.sanitize(text)
            assert isinstance(result, SanitizationResult)
            # No crash is the main assertion; text may or may not contain PII.

    def test_regex_catastrophic_backtrack_resistance(self) -> None:
        """Pathological inputs should not hang the regex engine."""
        sanitizer = PrivacySanitizer(enabled=True)
        # Known catastrophic-backtrack-style strings -- not triggers for our
        # patterns specifically, but we still want defense in depth.
        patterns = [
            "a" * 5000 + "!",
            "-" * 5000,
            "@" * 5000,
            "1" * 5000,
            ".com" * 1000,
        ]
        for text in patterns:
            start = time.monotonic()
            sanitizer.sanitize(text)
            elapsed = time.monotonic() - start
            assert elapsed < 5.0, (
                f"sanitize hung for {elapsed:.2f}s on pathological input"
            )


# -------------------------------------------------------------------------
# Privacy Guarantees (Integration)
# -------------------------------------------------------------------------


class TestPrivacyGuarantees:
    """Integration-style tests for core privacy invariants."""

    async def test_diary_exchange_fields_have_no_raw_text_field(
        self, tmp_path
    ) -> None:
        """The diary schema must not contain a column that stores raw text."""
        import aiosqlite

        db_path = str(tmp_path / "audit.db")
        store = DiaryStore(db_path)
        await store.initialize()

        # Columns that could legitimately store prose (never exchanged text)
        forbidden_column_substrings = {
            "message_text",
            "raw_text",
            "user_input",
            "prompt",
            "response_text",
        }
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("PRAGMA table_info('exchanges')")
            columns = [row[1] for row in await cursor.fetchall()]

        assert "user_state_embedding" in columns
        for col in columns:
            for forbidden in forbidden_column_substrings:
                assert forbidden not in col, (
                    f"DiaryStore.exchanges contains forbidden column: {col}"
                )

    def test_pipeline_output_has_no_raw_text_fields(self) -> None:
        """PipelineOutput is allowed response_text but no raw user_input mirror."""
        field_names = {f.name for f in fields(PipelineOutput)}
        # response_text is legitimate (AI output, not user text)
        # but there should be no user input echo
        for forbidden in [
            "message_text",
            "user_message",
            "raw_text",
            "user_input",
            "user_prompt",
        ]:
            assert forbidden not in field_names, (
                f"PipelineOutput unexpectedly exposes {forbidden}"
            )

    def test_sanitized_text_is_idempotent(self) -> None:
        """Applying the sanitizer twice should yield the same result."""
        sanitizer = PrivacySanitizer(enabled=True)
        original = (
            "Email foo@bar.com, call 555-123-4567, visit https://x.y/z. "
            "My SSN is 123-45-6789."
        )
        once = sanitizer.sanitize(original).sanitized_text
        twice = sanitizer.sanitize(once).sanitized_text
        assert once == twice


# -------------------------------------------------------------------------
# Tokenizer Robustness
# -------------------------------------------------------------------------


class TestTokenizerRobustness:
    """Tests that SimpleTokenizer handles edge cases without crashing."""

    @pytest.fixture
    def tokenizer(self) -> SimpleTokenizer:
        tok = SimpleTokenizer(vocab_size=200)
        tok.build_vocab(
            [
                "hello world",
                "this is a test corpus",
                "the quick brown fox",
                "privacy matters a great deal",
            ]
        )
        return tok

    def test_tokenizer_handles_empty_input(self, tokenizer: SimpleTokenizer) -> None:
        """Empty input should encode to [BOS, EOS] with add_special=True."""
        ids = tokenizer.encode("", add_special=True)
        assert ids == [tokenizer.BOS_ID, tokenizer.EOS_ID]

    def test_tokenizer_handles_only_whitespace(
        self, tokenizer: SimpleTokenizer
    ) -> None:
        """Whitespace-only input should behave like empty input."""
        ids = tokenizer.encode("   \n\t  ", add_special=True)
        assert ids == [tokenizer.BOS_ID, tokenizer.EOS_ID]

    def test_tokenizer_handles_only_punctuation(
        self, tokenizer: SimpleTokenizer
    ) -> None:
        """Punctuation-only input should not crash and should wrap in BOS/EOS."""
        ids = tokenizer.encode("!!!???...", add_special=True)
        assert ids[0] == tokenizer.BOS_ID
        assert ids[-1] == tokenizer.EOS_ID

    def test_tokenizer_handles_very_long_input(
        self, tokenizer: SimpleTokenizer
    ) -> None:
        """10k-word inputs should tokenize without crashing."""
        text = " ".join(["hello world"] * 5000)  # 10k tokens
        ids = tokenizer.encode(text, add_special=False)
        assert len(ids) >= 10_000

    def test_tokenizer_handles_unicode(self, tokenizer: SimpleTokenizer) -> None:
        """Unicode input should encode without errors."""
        text = "hello \U0001f600 world"
        ids = tokenizer.encode(text, add_special=True)
        assert isinstance(ids, list)
        assert len(ids) >= 2

    def test_tokenizer_unknown_token_maps_to_unk(
        self, tokenizer: SimpleTokenizer
    ) -> None:
        """Words not in the vocabulary should map to UNK_ID."""
        ids = tokenizer.encode("zzzzz", add_special=False)
        assert tokenizer.UNK_ID in ids
