"""Iter 11 — KeystrokeAuthenticator embedding-shape robustness.

The keystroke authenticator (Identity Lock) consumes a 64-dim user-
state embedding from the TCN encoder.  Before iter 11, its
``_coerce_embedding`` flattened multi-dim inputs but did not canonicalise
the flat dim — so a mismatched-shape embedding would silently produce
cosine_sim=0 inside ``_score_match`` (via the ``_safe_cosine`` shape-
mismatch guard), making the user appear unrecognisable for that turn.

After iter 11: every embedding canonicalises to 64-dim before being
stored or compared.  Shape transitions during a session don't break
the match score.
"""

from __future__ import annotations

import math

import torch

from i3.biometric.keystroke_auth import (
    BiometricMatch,
    KeystrokeAuthenticator,
)


def _kw(**overrides) -> dict:
    base = dict(
        iki_mean=100.0,
        iki_std=15.0,
        composition_time_ms=1200.0,
        edit_count=0,
    )
    base.update(overrides)
    return base


def test_undersized_embedding_pads_to_64() -> None:
    """A 32-dim embedding gets zero-padded to 64-dim before storage,
    so subsequent 64-dim observations don't shape-mismatch the
    template's cosine."""
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "u_short"

    # Register with a short embedding.
    short = torch.ones(32) * 0.5
    for _ in range(3):
        m = auth.observe(user, embedding=short, **_kw())
        assert isinstance(m, BiometricMatch)

    # Now observe a full-size embedding: cosine should not zero out.
    full = torch.ones(64) * 0.5
    m = auth.observe(user, embedding=full, **_kw())
    # Internal similarity score must be finite and (importantly)
    # non-zero — the template canonicalises matching against the
    # full embedding instead of returning 0 from a shape mismatch.
    assert math.isfinite(m.similarity)
    assert m.similarity > 0.0, (
        f"similarity should be > 0 with canonicalised 64-dim embedding; "
        f"got {m.similarity}"
    )


def test_oversized_embedding_truncates_to_64() -> None:
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "u_long"

    big = torch.ones(128) * 0.4
    for _ in range(3):
        m = auth.observe(user, embedding=big, **_kw())

    m = auth.observe(user, embedding=big, **_kw())
    assert math.isfinite(m.similarity)


def test_multidim_embedding_is_flattened() -> None:
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "u_2d"

    multi = torch.ones((4, 16)) * 0.3  # 64 elements total
    for _ in range(3):
        m = auth.observe(user, embedding=multi, **_kw())

    m = auth.observe(user, embedding=multi, **_kw())
    assert math.isfinite(m.similarity)


def test_mixed_shapes_dont_silently_zero_similarity() -> None:
    """Before iter 11: a session that mixed 32-dim and 64-dim
    embeddings would produce silently-zero similarity on the
    mismatched-shape turns.  After iter 11: every observation
    canonicalises, so the similarity reflects the actual match."""
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "u_mix"

    # Register with full-size.
    full = torch.ones(64) * 0.6
    for _ in range(3):
        auth.observe(user, embedding=full, **_kw())

    # Now hit it with a short embedding similar in upper 32 dims.
    short = torch.ones(32) * 0.6
    m = auth.observe(user, embedding=short, **_kw())
    # Canonicalised short_padded = [0.6, ..., 0.6, 0, ..., 0] vs full
    # = [0.6, ..., 0.6].  The cosine should be substantial (the upper
    # 32 dims match perfectly) — not zero.  Before iter 11, _safe_cosine
    # returned 0.0 outright on shape mismatch, so similarity dropped to
    # the negative tail of the z-score penalties.
    assert m.similarity > 0.0, (
        f"mixed-shape session should still produce a positive similarity "
        f"after iter 11 canonicalisation; got {m.similarity}"
    )


def test_none_embedding_does_not_raise() -> None:
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "u_none"

    m = auth.observe(user, embedding=None, **_kw())
    assert isinstance(m, BiometricMatch)
    assert math.isfinite(m.similarity)


def test_nan_embedding_zeroed() -> None:
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "u_nan"

    bad = torch.tensor([float("nan")] * 64)
    m = auth.observe(user, embedding=bad, **_kw())
    assert math.isfinite(m.similarity)
