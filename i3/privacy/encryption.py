"""Fernet symmetric encryption for user model data at rest.

This module ensures that sensitive user-model artifacts (embeddings, serialized
profiles) are encrypted before being written to disk. Scalar metrics, topic
keywords, and configuration are NOT encrypted -- only data that could be used
to reconstruct or fingerprint a user's interaction patterns.

Key management:
    - The encryption key is loaded from the I3_ENCRYPTION_KEY environment variable.
    - If no key is set, a temporary key is generated and a warning is logged.
    - The key is a URL-safe base64-encoded 32-byte Fernet key.
    - The key value is NEVER logged or written to disk by this module.

Thread safety:
    - ``cryptography.fernet.Fernet`` and ``MultiFernet`` instances are
      thread-safe; they hold no mutable state after construction.  A single
      :class:`ModelEncryptor` instance can therefore be shared safely across
      threads / asyncio tasks once :meth:`initialize` has completed.

Cryptographic guarantees (provided by the ``cryptography`` library):
    - Authenticated encryption (AES-128-CBC + HMAC-SHA256).
    - Constant-time HMAC verification, mitigating timing attacks during
      decryption.  No custom MAC comparison is performed in this module.

Future work (intentionally not implemented):
    - Pluggable key sourcing (KMS / Secret Manager / file with strict perms).
      Today only the environment variable backend is supported.
"""

import json
import logging
import os
from typing import TYPE_CHECKING

from cryptography.fernet import Fernet, MultiFernet

if TYPE_CHECKING:
    import torch

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
        self._fernet: Fernet | None = None
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
        assert self._fernet is not None
        token: bytes = self._fernet.encrypt(data)
        return token

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
        assert self._fernet is not None
        plaintext: bytes = self._fernet.decrypt(encrypted_data)
        return plaintext

    def encrypt_embedding(self, embedding: "torch.Tensor") -> bytes:
        """Encrypt a torch tensor embedding for storage.

        The tensor is detached from any computation graph, moved to CPU,
        converted to a numpy float32 array, serialized to raw bytes, then
        encrypted.

        Args:
            embedding: A 1-D torch.Tensor (typically 64-dim float32).

        Returns:
            Fernet-encrypted bytes.

        Raises:
            TypeError: If ``embedding`` is not a torch.Tensor.
            ValueError: If ``embedding`` is not 1-D.  We deliberately refuse
                multi-dimensional inputs so the round trip cannot silently
                lose shape information through ``decrypt_embedding(dim=...)``.
        """
        # SEC: validate input shape so the round trip is unambiguous.  A 2-D
        # tensor would otherwise be flattened to bytes here and then come back
        # mis-shaped from decrypt_embedding(dim=...).
        import numpy as np
        try:
            import torch as _torch
        except ImportError:  # pragma: no cover - torch is a hard dep elsewhere
            _torch = None
        if _torch is not None and not isinstance(embedding, _torch.Tensor):
            raise TypeError(
                "encrypt_embedding() requires a torch.Tensor, got "
                f"{type(embedding).__name__}"
            )
        if embedding.dim() != 1:
            raise ValueError(
                "encrypt_embedding() requires a 1-D tensor, got "
                f"shape={tuple(embedding.shape)}"
            )
        raw_bytes = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
        return self.encrypt(raw_bytes)

    def decrypt_embedding(self, encrypted: bytes, dim: int = 64) -> "torch.Tensor":
        """Decrypt stored bytes back to a torch tensor.

        Args:
            encrypted: Fernet-encrypted bytes (from encrypt_embedding).
            dim: Dimensionality of the embedding vector (default 64).
                Callers MUST pass the same ``dim`` that was used at encrypt
                time.  No metadata is stored alongside the ciphertext, so a
                wrong ``dim`` is detected by a byte-length mismatch (see
                below).

        Returns:
            A 1-D torch.Tensor of shape (dim,) with dtype float32.

        Raises:
            cryptography.fernet.InvalidToken: If ciphertext is invalid.
            ValueError: If ``dim`` does not match the actual byte length of
                the decrypted payload (i.e. the caller passed the wrong
                dimensionality).  The error message contains the expected and
                actual element counts but no plaintext content.
        """
        import numpy as np
        import torch as _torch
        raw_bytes = self.decrypt(encrypted)

        # SEC: explicit dim validation.  Without this, np.frombuffer().reshape()
        # would raise an opaque numpy ValueError ("cannot reshape array of size
        # N into shape (M,)") that does not point at the dim parameter.  We
        # leak only sizes (in float32 elements), never plaintext content.
        bytes_per_element = np.dtype(np.float32).itemsize  # 4
        if len(raw_bytes) % bytes_per_element != 0:
            raise ValueError(
                "decrypt_embedding(): ciphertext does not decode to a whole "
                f"number of float32 elements ({len(raw_bytes)} bytes); the "
                "stored payload is not a float32 embedding."
            )
        actual_elements = len(raw_bytes) // bytes_per_element
        if actual_elements != dim:
            raise ValueError(
                "decrypt_embedding(): dim parameter does not match stored "
                f"embedding (expected dim={dim}, "
                f"actual dim={actual_elements}). "
                "Pass the same dim that was used at encrypt time."
            )

        # SEC: .copy() converts the read-only numpy buffer returned by
        # frombuffer() into a writable one.  Without it, torch.from_numpy()
        # would yield a tensor whose .data buffer is read-only and any
        # in-place op would crash with an opaque RuntimeError.
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(dim)
        return _torch.from_numpy(arr.copy())

    def encrypt_json(self, data: dict) -> bytes:
        """Encrypt a JSON-serializable dict.

        Args:
            data: Dictionary to encrypt. Must be JSON-serializable.  Keys
                must be strings (other types raise ``TypeError`` from
                json.dumps; this matches Python's default JSON behaviour).

        Returns:
            Fernet-encrypted bytes containing the JSON payload.

        Raises:
            TypeError: If ``data`` is not a dict, or contains values that
                are not JSON-serializable.  The error message does NOT
                include the offending value to avoid leaking secrets through
                exception strings.
        """
        # SEC: enforce dict at the boundary.  json.dumps would happily encode
        # a list / str / None and decrypt_json() would then return a non-dict,
        # silently breaking callers that expect a mapping.
        if not isinstance(data, dict):
            raise TypeError(
                f"encrypt_json() requires a dict, got {type(data).__name__}"
            )
        try:
            # SEC: sort_keys=True makes the JSON serialization deterministic
            # for any given dict, which is important for reproducible diary
            # writes and for any caller that compares ciphertext-derived
            # hashes.  separators=(",", ":") drops insignificant whitespace.
            json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        except TypeError as exc:
            # SEC: deliberately drop the offending value from the message;
            # the original TypeError chain is preserved via __cause__ for
            # local debugging but the public message stays content-free.
            raise TypeError(
                "encrypt_json(): dict contains a value that is not "
                "JSON-serializable"
            ) from exc
        return self.encrypt(json_str.encode("utf-8"))

    def decrypt_json(self, encrypted: bytes) -> dict:
        """Decrypt Fernet-encrypted bytes back to a dict.

        Args:
            encrypted: Fernet-encrypted bytes (from encrypt_json).

        Returns:
            The original dictionary.

        Raises:
            cryptography.fernet.InvalidToken: If the token is invalid or
                was encrypted with a different key.
            ValueError: If the decrypted payload is valid JSON but is not a
                dict at the top level (defence-in-depth: Fernet
                authentication should already prevent tampering).
        """
        json_bytes = self.decrypt(encrypted)
        # SEC: json.loads is called WITHOUT object_hook / object_pairs_hook,
        # so it cannot trigger arbitrary Python code execution.  We do not
        # use any custom decoder.
        decoded = json.loads(json_bytes.decode("utf-8"))
        if not isinstance(decoded, dict):
            # SEC: enforce the documented return type.  This guards against
            # the (extremely unlikely, since Fernet authenticates) case of a
            # malformed payload that happens to decode as a JSON array, str,
            # number, etc.
            raise ValueError(
                "decrypt_json(): decoded payload is not a JSON object"
            )
        return decoded

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key.

        Returns:
            A URL-safe base64-encoded 32-byte key as a string.
            Suitable for setting as the I3_ENCRYPTION_KEY environment variable.
        """
        key: str = Fernet.generate_key().decode()
        return key

    def rotate_to(self, new_key: str) -> "MultiFernet":
        """Create a :class:`MultiFernet` for zero-downtime key rotation.

        The returned ``MultiFernet`` is constructed with the keys in
        ``[new, old]`` order.  Per the ``cryptography`` library contract:

        * ``mf.encrypt(plaintext)``  -- always uses the **first** key
          (the new one), so all *new* writes are immediately on the new
          key.
        * ``mf.decrypt(token)``      -- tries the keys in order, so
          existing tokens encrypted under the old key still decrypt
          successfully.
        * ``mf.rotate(token)``       -- decrypts with whichever key works
          and re-encrypts under the first (new) key.

        Recommended migration workflow::

            # 1. Generate the new key OUT-OF-BAND and store it in the
            #    secret manager.  Do NOT log it.
            new_key = ModelEncryptor.generate_key()

            # 2. Create the MultiFernet for the rotation window.
            encryptor = ModelEncryptor()
            encryptor.initialize()           # loads OLD key from env
            mf = encryptor.rotate_to(new_key)

            # 3. Sweep the database, re-encrypting every row.
            for row in db.rows():
                row.blob = mf.rotate(row.blob)
                db.save(row)

            # 4. Once the sweep is complete, swap I3_ENCRYPTION_KEY in the
            #    secret manager to the new key, restart the process, and
            #    DROP the old key.  After this point the old key has no
            #    further use and should be destroyed.

        Args:
            new_key: The new Fernet key (URL-safe base64-encoded 32 bytes).
                Accepted as ``str`` or ``bytes``.

        Returns:
            A :class:`MultiFernet` instance configured with
            ``[new, old]`` Fernet keys.

        Raises:
            ValueError: If ``new_key`` is not a valid Fernet key.  The
                offending key value is NOT included in the error message.
        """
        self._ensure_initialized()
        try:
            new_fernet = Fernet(
                new_key.encode() if isinstance(new_key, str) else new_key
            )
        except Exception as exc:
            # SEC: never echo the candidate key (even malformed) into logs
            # or exception messages.
            raise ValueError(
                "rotate_to(): new_key is not a valid Fernet key "
                "(expected 32 url-safe base64-encoded bytes)"
            ) from exc
        assert self._fernet is not None
        mf = MultiFernet([new_fernet, self._fernet])
        logger.info("Created MultiFernet for key rotation.")
        return mf
