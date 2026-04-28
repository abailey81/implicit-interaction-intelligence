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


# ---------------------------------------------------------------------------
# Iter 12 — cross-user isolation
# ---------------------------------------------------------------------------


def test_scalar_zero_dim_embedding_does_not_crash() -> None:
    """Iter 25: a 0-dim scalar tensor flattens to a 1-element tensor
    and zero-pads to canonical 64-dim instead of crashing torch.cat."""
    auth = KeystrokeAuthenticator(enrolment_target=3)
    scalar = torch.tensor(0.5)
    m = auth.observe("u_scalar", embedding=scalar, **_kw())
    assert math.isfinite(m.similarity)


def test_separate_users_have_independent_templates() -> None:
    """A single shared KeystrokeAuthenticator instance must keep
    per-user templates fully isolated — registering user A doesn't
    affect user B's classification, and vice versa."""
    auth = KeystrokeAuthenticator(enrolment_target=3)

    # User A: a specific embedding pattern.
    emb_a = torch.zeros(64)
    emb_a[:32] = 1.0  # first half all ones
    # User B: a clearly different pattern.
    emb_b = torch.zeros(64)
    emb_b[32:] = 1.0  # second half all ones

    # Register both users.
    for _ in range(3):
        auth.observe("alice", embedding=emb_a, **_kw(iki_mean=100.0))
        auth.observe("bob", embedding=emb_b, **_kw(iki_mean=140.0))

    # Now an observation from user A using emb_a should be high-
    # similarity for alice's template.
    m_alice = auth.observe("alice", embedding=emb_a, **_kw(iki_mean=100.0))
    # And the same emb_a sent for "bob" should NOT be high-similarity
    # against bob's template (because bob registered with a different
    # vector).
    m_bob = auth.observe("bob", embedding=emb_a, **_kw(iki_mean=140.0))

    assert m_alice.similarity > m_bob.similarity, (
        f"alice's match for her own embedding ({m_alice.similarity}) "
        f"should exceed bob's match for the same embedding "
        f"({m_bob.similarity})"
    )


def test_user_template_persists_after_other_user_observations() -> None:
    """Observations from other users mustn't drift any given user's
    template — each template only updates when its own user observes."""
    auth = KeystrokeAuthenticator(enrolment_target=3)

    # Establish alice with a specific embedding.
    alice_emb = torch.zeros(64)
    alice_emb[:32] = 0.5
    for _ in range(3):
        auth.observe("alice", embedding=alice_emb, **_kw(iki_mean=100.0))

    m_before = auth.observe("alice", embedding=alice_emb, **_kw(iki_mean=100.0))

    # Now bob makes 10 observations with very different embeddings.
    bob_emb = torch.zeros(64)
    bob_emb[32:] = 0.5
    for _ in range(10):
        auth.observe("bob", embedding=bob_emb, **_kw(iki_mean=200.0))

    # Alice's similarity for her own embedding should be unchanged.
    m_after = auth.observe("alice", embedding=alice_emb, **_kw(iki_mean=100.0))
    # Allow for the EWMA of alice's own subsequent observations to
    # nudge the score slightly, but never by more than 0.1.
    assert abs(m_after.similarity - m_before.similarity) < 0.1, (
        f"alice's similarity drifted after bob's observations: "
        f"before={m_before.similarity} after={m_after.similarity}"
    )


def test_force_register_isolates_per_user() -> None:
    """force_register on one user doesn't reset any other user's state."""
    auth = KeystrokeAuthenticator(enrolment_target=3)

    # Register alice fully.
    alice_emb = torch.zeros(64)
    alice_emb[:32] = 1.0
    for _ in range(3):
        auth.observe("alice", embedding=alice_emb, **_kw())

    # Register bob fully (different embedding).
    bob_emb = torch.zeros(64)
    bob_emb[32:] = 1.0
    for _ in range(3):
        auth.observe("bob", embedding=bob_emb, **_kw())

    # Force-register charlie via observe + force_register from recent
    # observations.  charlie_emb is yet another distinct pattern.
    charlie_emb = torch.zeros(64)
    charlie_emb[16:48] = 1.0
    for _ in range(2):
        auth.observe(
            "charlie",
            embedding=charlie_emb,
            iki_mean=80.0,
            iki_std=10.0,
            composition_time_ms=900.0,
            edit_count=0,
        )
    auth.force_register("charlie")

    # Alice and bob's status should remain registered + intact.
    s_alice = auth.status("alice")
    s_bob = auth.status("bob")
    s_charlie = auth.status("charlie")
    assert s_alice.state in {"registered", "verifying"}
    assert s_bob.state in {"registered", "verifying"}
    assert s_charlie.state in {"registered", "verifying"}


def test_reset_for_user_only_affects_that_user() -> None:
    auth = KeystrokeAuthenticator(enrolment_target=3)

    a = torch.zeros(64); a[:32] = 1.0
    b = torch.zeros(64); b[32:] = 1.0
    for _ in range(3):
        auth.observe("alice", embedding=a, **_kw())
        auth.observe("bob", embedding=b, **_kw())

    auth.reset_for_user("alice")

    # Alice is back to unregistered; bob is still registered.
    s_alice = auth.status("alice")
    s_bob = auth.status("bob")
    assert s_alice.state == "unregistered"
    assert s_bob.state in {"registered", "verifying"}


# ---------------------------------------------------------------------------
# Iter 49 — composition-cadence variance floor
# ---------------------------------------------------------------------------


def test_owner_typing_long_message_does_not_false_positive() -> None:
    """Iter 49: the owner registers on short messages (3 s composition)
    and then types a long message (12 s composition).  Pre-iter-49 the
    composition-cadence z-score divisor was ``template_comp_mean * 0.3``
    — so a 4× delta from a 3000 ms template clipped to 5σ and the
    Identity Lock falsely flagged the owner as a mismatch.

    After iter 49 the divisor is ``max(template_comp_mean * 0.5,
    2000)``, so the same 4× delta produces ``z_comp ≈ 5.0`` instead of
    a clipped 5σ, and the similarity stays above mismatch territory."""
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "owner_short_then_long"

    emb = torch.ones(64) * 0.5

    # Register on three short, fast messages — typical chat.
    for _ in range(3):
        auth.observe(user, embedding=emb,
                     iki_mean=100.0, iki_std=15.0,
                     composition_time_ms=3000.0, edit_count=0)

    # Now the owner types a long, thoughtful 12-second message — same
    # rhythm, just longer text.  The template's comp_mean is 3000 ms;
    # the new comp is 12000 ms — a 4x delta.
    m = auth.observe(user, embedding=emb,
                     iki_mean=100.0, iki_std=15.0,
                     composition_time_ms=12000.0, edit_count=0)

    # The match similarity must remain above 0 (i.e. the cosine + IKI
    # contributions still dominate the comp penalty).  Pre-iter-49 the
    # comp penalty alone was 0.10 * 5 = 0.5, dragging similarity into
    # the mismatch band even with a perfect cosine.
    assert m.similarity > 0.0, (
        f"owner typing a longer message should not produce negative "
        f"similarity; got {m.similarity:.3f}"
    )


def test_clearly_different_typist_still_detected() -> None:
    """Iter 49 widened the comp-variance floor; this test pins that
    truly-impostor users with massive comp-time deltas are still
    flagged.  A 30000 ms message against a 1500 ms template is a 20×
    delta and must still produce z_comp ≥ 5σ (clipped)."""
    auth = KeystrokeAuthenticator(enrolment_target=3)
    user = "owner_then_impostor"

    emb_owner = torch.zeros(64); emb_owner[:32] = 1.0

    # Register the owner on quick 1.5 s messages.
    for _ in range(3):
        auth.observe(user, embedding=emb_owner,
                     iki_mean=90.0, iki_std=10.0,
                     composition_time_ms=1500.0, edit_count=0)

    # An impostor with a wholly different embedding AND 20× comp time.
    emb_impostor = torch.zeros(64); emb_impostor[32:] = 1.0
    m = auth.observe(user, embedding=emb_impostor,
                     iki_mean=300.0, iki_std=80.0,
                     composition_time_ms=30000.0, edit_count=15)

    # Similarity must be pushed below the recognition threshold even
    # with the iter-49 wider comp variance — embedding cosine + iki
    # penalties + edit penalty stack with the comp penalty.
    assert m.similarity < 0.5, (
        f"clearly impostor typing should drop similarity well below 0.5; "
        f"got {m.similarity:.3f}"
    )
