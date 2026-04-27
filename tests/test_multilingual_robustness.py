"""Iter 57 — multilingual + adversarial robustness coverage.

Exercises the *upstream* layers (BPE tokenizer, intent regex gate, PII
sanitiser, sensitivity classifier) against non-Latin scripts, RTL,
emoji, and code-switching input.  These are the surfaces that have to
not crash on a multilingual user — the SLM itself can still respond
in English; the test confirms the pipeline doesn't choke at the gate.

Tests are CPU-only and < 1 s in aggregate.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. BPE tokenizer round-trip on multilingual text
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bpe():
    from pathlib import Path
    from i3.slm.bpe_tokenizer import BPETokenizer
    candidates = [
        Path("checkpoints/slm_v2/tokenizer_bpe.json"),
        Path("checkpoints/slm/tokenizer_bpe.json"),
    ]
    for p in candidates:
        if p.exists():
            return BPETokenizer.load(p)
    pytest.skip("BPE tokenizer file not present (run training first)")


@pytest.mark.parametrize("text", [
    "hello world",
    "Bonjour, comment ça va aujourd'hui?",
    "Hola, ¿cómo estás?",
    "こんにちは、お元気ですか",
    "你好，今天怎么样",
    "안녕하세요",
    "مرحبا كيف حالك",
    "שלום מה שלומך",
    "Привет, как дела?",
    "Γειά σας, πώς είστε;",
    "नमस्ते, आप कैसे हैं?",
    "שלום, what's up?",  # code switching
    "I love mondays 😂🙄💯",
    "set timer for ⏱ 5 min",
])
def test_bpe_roundtrip_multilingual(bpe, text):
    """Encoding then decoding must produce a semantically equivalent
    string.  Byte-level BPE is lossless on bytes by construction; we
    confirm the decode of the encode equals the original.
    """
    ids = bpe.encode(text, add_bos=False, add_eos=False)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    decoded = bpe.decode(ids)
    assert decoded == text, f"BPE round-trip failed: {text!r} -> {decoded!r}"


def test_bpe_does_not_explode_on_zalgo(bpe):
    """Combining-character soup (Zalgo text) must not crash the
    tokenizer or produce > 10× expansion."""
    zalgo = "h̸̢̛̛̦̦̤̭e̴̛͚l̴̢͕l̵͖͆o̸̦̕"
    ids = bpe.encode(zalgo, add_bos=False, add_eos=False)
    # Byte-level BPE will split combining marks; check it didn't blow up
    assert isinstance(ids, list)
    assert len(ids) < len(zalgo.encode("utf-8")) * 2 + 50


# ---------------------------------------------------------------------------
# 2. Intent regex gate on multilingual command-shaped text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # English commands should still match
    "set timer for 5 minutes",
    "play jazz",
    # Non-English chat should NOT match (the gate is English-only by design;
    # multilingual chat falls through to the SLM)
    "comment ça va aujourd'hui",
    "你好，今天怎么样",
    "اهلا كيف حالك",
])
def test_intent_gate_handles_multilingual(text):
    from i3.pipeline.engine import Pipeline
    p = Pipeline.__new__(Pipeline)
    # Should not raise on non-ASCII input.
    result = p._looks_like_command(text)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 3. PII sanitiser doesn't false-positive on non-Latin tokens
# ---------------------------------------------------------------------------

def test_pii_sanitiser_no_false_positives_multilingual():
    from i3.privacy.sanitizer import PrivacySanitizer
    s = PrivacySanitizer()
    benign = [
        "你好世界",
        "مرحبا بالعالم",
        "Привет мир",
        "Здравствуй, мир",
        "नमस्ते दुनिया",
        "I love mondays 😂",
        "Bonjour Paris",
    ]
    for t in benign:
        r = s.sanitize(t)
        assert r.pii_detected is False, \
            f"false-positive PII on benign multilingual: {t!r} -> {r}"


# ---------------------------------------------------------------------------
# 4. Sensitivity classifier doesn't crash on non-Latin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "你好",
    "مرحبا",
    "Привет",
    "안녕하세요",
    "Bonjour",
    "Hola",
])
def test_sensitivity_handles_multilingual(text):
    from i3.router.sensitivity import TopicSensitivityDetector
    d = TopicSensitivityDetector()
    # No crash; score is the safe-default min_score for benign greetings.
    score = d.detect(text)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
