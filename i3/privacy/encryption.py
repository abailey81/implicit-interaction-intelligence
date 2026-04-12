"""Fernet symmetric encryption for user model data at rest.

This module ensures that sensitive user-model artifacts (embeddings, serialized
profiles) are encrypted before being written to disk. Scalar metrics, topic
keywords, and configuration are NOT encrypted -- only data that could be used
to reconstruct or fingerprint a user's interaction patterns.

Key management:
    - The encryption key is loaded from the I3_ENCRYPTION_KEY environment variable.
    - If no key is set, a temporary key is generated and a warning is logged.
    - The key is a URL-safe base64-encoded 32-byte Fernet key.
"""

import os
import json
import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class ModelEncryptor:
    """Encrypts user model data at rest using Fernet symmetric encryption.

    Key management:
    - Key is loaded from environment variable (I3_ENCRYPTION_KEY)
    - If no key exists, generates one and logs a warning
    - Key is base64-encoded 32-byte value

    What gets encrypted:
    - User state embeddings (64-dim float vectors)
    - Baseline embeddings
    - Any serialized user profile data

    What does NOT get encrypted:
    - Scalar metrics (engagement, relationship_strength)
    - Topic keywords
    - Configuration
    """

    def __init__(self, key_env_var: str = "I3_ENCRYPTION_KEY"):
        self._key_env = key_env_var
        self._fernet: Optional[Fernet] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize encryption with key from environment.

        Behavior:
            - If the key environment variable is set, use that key.
            - Otherwise, generate an ephemeral key for the current
              process ONLY and log a prominent warning.  The generated
              key itself is NEVER logged — doing so would defeat the
              purpose of encryption at rest.  Operators should always
              set the environment variable in production.
        """
        key = os.environ.get(self._key_env)
        if key:
            try:
                self._fernet = Fernet(
                    key.encode() if isinstance(key, str) else key
                )
            except Exception as exc:
                # Do not echo the key (even malformed) back into logs.
                raise ValueError(
                    f"Invalid Fernet key in ${self._key_env}: "
                    "expected 32 url-safe base64-encoded bytes"
                ) from exc
            logger.info(
                "Encryption initialized with key from %s", self._key_env
            )
        else:
            # Generate an ephemeral key.  The key is NOT logged.
            self._fernet = Fernet(Fernet.generate_key())
            logger.warning(
                "No encryption key found in %s. "
                "A temporary in-memory key has been generated. "
                "All data encrypted with this key will be LOST when the "
                "process exits.  Set %s in production.",
                self._key_env,
                self._key_env,
            )
        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Lazy initialization guard."""
        if not self._initialized:
            self.initialize()

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt raw bytes data.

        Args:
            data: Plaintext bytes to encrypt.

        Returns:
            Fernet-encrypted bytes (URL-safe base64 encoded).

        Raises:
            TypeError: If data is not bytes.
        """
        self._ensure_initialized()
        if not isinstance(data, bytes):
            raise TypeError(f"encrypt() requires bytes, got {type(data).__name__}")
        return self._fernet.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt Fernet-encrypted bytes data.

        Args:
            encrypted_data: Fernet-encrypted bytes to decrypt.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            cryptography.fernet.InvalidToken: If the token is invalid or
                was encrypted with a different key.
        """
        self._ensure_initialized()
        return self._fernet.decrypt(encrypted_data)

    def encrypt_embedding(self, embedding: "torch.Tensor") -> bytes:
        """Encrypt a torch tensor embedding for storage.

        The tensor is detached from any computation graph, moved to CPU,
        converted to a numpy float32 array, serialized to raw bytes, then
        encrypted.

        Args:
            embedding: A 1-D torch.Tensor (typically 64-dim float32).

        Returns:
            Fernet-encrypted bytes.
        """
        import numpy as np
        raw_bytes = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
        return self.encrypt(raw_bytes)

    def decrypt_embedding(self, encrypted: bytes, dim: int = 64) -> "torch.Tensor":
        """Decrypt stored bytes back to a torch tensor.

        Args:
            encrypted: Fernet-encrypted bytes (from encrypt_embedding).
            dim: Dimensionality of the embedding vector (default 64).

        Returns:
            A 1-D torch.Tensor of shape (dim,) with dtype float32.
        """
        import torch
        import numpy as np
        raw_bytes = self.decrypt(encrypted)
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(dim)
        return torch.from_numpy(arr.copy())

    def encrypt_json(self, data: dict) -> bytes:
        """Encrypt a JSON-serializable dict.

        Args:
            data: Dictionary to encrypt. Must be JSON-serializable.

        Returns:
            Fernet-encrypted bytes containing the JSON payload.
        """
        json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
        return self.encrypt(json_bytes)

    def decrypt_json(self, encrypted: bytes) -> dict:
        """Decrypt Fernet-encrypted bytes back to a dict.

        Args:
            encrypted: Fernet-encrypted bytes (from encrypt_json).

        Returns:
            The original dictionary.
        """
        json_bytes = self.decrypt(encrypted)
        return json.loads(json_bytes.decode("utf-8"))

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key.

        Returns:
            A URL-safe base64-encoded 32-byte key as a string.
            Suitable for setting as the I3_ENCRYPTION_KEY environment variable.
        """
        return Fernet.generate_key().decode()

    def rotate_to(self, new_key: str) -> "MultiFernet":
        """Create a :class:`MultiFernet` that decrypts with the old key
        and encrypts with a new key.

        Use this to migrate a database to a new key without downtime::

            encryptor = ModelEncryptor()
            encryptor.initialize()
            mf = encryptor.rotate_to(new_key)
            for row in db.rows():
                row.blob = mf.rotate(row.blob)

        The rotation preserves the ability to decrypt existing data
        while ensuring that all *new* writes use the new key.

        Args:
            new_key: The new Fernet key (base64-encoded 32 bytes).

        Returns:
            A :class:`MultiFernet` instance configured with
            [new, old] Fernet keys.  ``mf.rotate(token)`` re-encrypts
            any token originally encrypted under the old key.
        """
        from cryptography.fernet import MultiFernet
        self._ensure_initialized()
        new_fernet = Fernet(
            new_key.encode() if isinstance(new_key, str) else new_key
        )
        assert self._fernet is not None
        mf = MultiFernet([new_fernet, self._fernet])
        logger.info("Created MultiFernet for key rotation.")
        return mf
