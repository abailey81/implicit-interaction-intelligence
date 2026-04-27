"""Iter 80 — ModelEncryptor Fernet round-trip + key-rotation tests.

The diary store + user_facts table use ModelEncryptor for at-rest
encryption.  These tests pin the encrypt/decrypt round-trip across
bytes / embeddings / JSON, plus the key-rotation behaviour (a fresh
encryptor with a different key cannot decrypt the old ciphertext).
"""
from __future__ import annotations

import os

import pytest


def _make_encryptor(key: str | None):
    """Construct a ModelEncryptor with *key* (or the ephemeral path
    when key is None) and initialise it."""
    from i3.privacy.encryption import ModelEncryptor
    if key is not None:
        os.environ["I3_TEST_KEY"] = key
        enc = ModelEncryptor(key_env_var="I3_TEST_KEY")
    else:
        os.environ.pop("I3_TEST_KEY", None)
        enc = ModelEncryptor(key_env_var="I3_TEST_KEY")
    enc.initialize()
    return enc


def _fresh_key() -> str:
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


# ---------------------------------------------------------------------------
# Bytes round-trip
# ---------------------------------------------------------------------------

def test_bytes_round_trip():
    e = _make_encryptor(_fresh_key())
    plaintext = b"hello, world"
    cipher = e.encrypt(plaintext)
    assert isinstance(cipher, bytes)
    assert cipher != plaintext
    assert e.decrypt(cipher) == plaintext


def test_empty_bytes_round_trip():
    e = _make_encryptor(_fresh_key())
    assert e.decrypt(e.encrypt(b"")) == b""


def test_long_bytes_round_trip():
    e = _make_encryptor(_fresh_key())
    pt = b"x" * 100_000
    assert e.decrypt(e.encrypt(pt)) == pt


def test_invalid_input_type_raises():
    e = _make_encryptor(_fresh_key())
    with pytest.raises(TypeError):
        e.encrypt("a string, not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Embedding round-trip
# ---------------------------------------------------------------------------

def test_embedding_round_trip():
    import torch
    e = _make_encryptor(_fresh_key())
    emb = torch.randn(64)
    cipher = e.encrypt_embedding(emb)
    decoded = e.decrypt_embedding(cipher, dim=64)
    assert torch.allclose(emb, decoded, atol=1e-5)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_json_round_trip():
    e = _make_encryptor(_fresh_key())
    payload = {"k": "v", "nested": [1, 2, 3], "flag": True}
    assert e.decrypt_json(e.encrypt_json(payload)) == payload


# ---------------------------------------------------------------------------
# Key isolation / rotation
# ---------------------------------------------------------------------------

def test_different_keys_cannot_decrypt_each_others_ciphertext():
    e1 = _make_encryptor(_fresh_key())
    e2 = _make_encryptor(_fresh_key())
    cipher = e1.encrypt(b"secret")
    with pytest.raises(Exception):
        e2.decrypt(cipher)


def test_invalid_fernet_key_raises_at_init():
    from i3.privacy.encryption import ModelEncryptor
    os.environ["I3_TEST_KEY"] = "this is definitely not a valid fernet key"
    enc = ModelEncryptor(key_env_var="I3_TEST_KEY")
    with pytest.raises(ValueError):
        enc.initialize()


def test_no_key_falls_back_to_ephemeral():
    """When the env var isn't set, the encryptor must still come up
    (with an ephemeral key) — the warning is logged but no exception."""
    e = _make_encryptor(None)
    # Round-trip works within the same process / encryptor instance.
    assert e.decrypt(e.encrypt(b"hi")) == b"hi"
