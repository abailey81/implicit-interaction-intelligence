"""Mock of the HarmonyOS Distributed Data Management (DDM) sync client.

Implements the ``I3UserStateSync`` contract from
``docs/huawei/harmony_hmaf_integration.md §4.3``: a ~680-byte payload carrying
a user's long-term profile (means + variances over 32 features) plus the
session and long-term embeddings.

The implementation is **pure-Python, in-memory**.  Production should target
HarmonyOS's distributed KV store (``@ohos.data.distributedKVStore``) or the
higher-level ``@ohos.data.distributedDataObject`` surface; the code here is
the shape, not the substance.

Privacy
-------
Every payload is **Fernet-encrypted** at rest with the existing
:class:`~i3.privacy.encryption.ModelEncryptor`.  A SHA-256 integrity tag is
appended so tampering is detectable on pull — Fernet already provides
authenticated encryption, but an explicit tag matters for callers that
decrypt on a device they don't fully trust (e.g. a shared Smart Hanhan).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from i3.adaptation.types import AdaptationVector
from i3.crossdevice.device_registry import DeviceInfo, DeviceRegistry
from i3.privacy.encryption import ModelEncryptor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: int = 1
_FEATURE_COUNT: int = 32
_EMBEDDING_DIM: int = 64


@dataclass
class AdaptationVectorPayload:
    """Serialisable projection of an :class:`AdaptationVector` for the wire.

    Stored as eight floats — the same canonical layout used by
    :meth:`AdaptationVector.to_tensor`.
    """

    cognitive_load: float
    formality: float
    verbosity: float
    emotionality: float
    directness: float
    emotional_tone: float
    accessibility: float
    reserved: float = 0.0

    @classmethod
    def from_adaptation(cls, adaptation: AdaptationVector) -> AdaptationVectorPayload:
        """Flatten an :class:`AdaptationVector` into the payload form."""
        arr = adaptation.to_tensor().tolist()
        return cls(
            cognitive_load=float(arr[0]),
            formality=float(arr[1]),
            verbosity=float(arr[2]),
            emotionality=float(arr[3]),
            directness=float(arr[4]),
            emotional_tone=float(arr[5]),
            accessibility=float(arr[6]),
            reserved=float(arr[7]),
        )

    def to_adaptation(self) -> AdaptationVector:
        """Rehydrate the payload into an :class:`AdaptationVector`."""
        t = torch.tensor(
            [
                self.cognitive_load,
                self.formality,
                self.verbosity,
                self.emotionality,
                self.directness,
                self.emotional_tone,
                self.accessibility,
                self.reserved,
            ],
            dtype=torch.float32,
        )
        return AdaptationVector.from_tensor(t)


@dataclass
class I3UserStateSync:
    """The user-state payload exchanged across the distributed databus.

    Matches the ``struct I3UserStateSync`` shape in
    ``docs/huawei/harmony_hmaf_integration.md §4.3``.  Total plaintext size
    ≈ 680 bytes — cheap to sync every minute on any plausible Huawei device
    network.
    """

    schema_version: int
    owner_device_id: int
    monotonic_seq: int
    user_id: str
    mean: np.ndarray  # shape (32,) float32
    var: np.ndarray   # shape (32,) float32
    embedding_lt: np.ndarray   # shape (64,) float32
    embedding_sess: np.ndarray  # shape (64,) float32
    adaptation: AdaptationVectorPayload
    ttl_s: int = 300
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_bytes(self) -> bytes:
        """Pack the record into a compact binary blob."""
        if self.mean.shape != (_FEATURE_COUNT,) or self.var.shape != (_FEATURE_COUNT,):
            raise ValueError(
                f"mean/var must have shape ({_FEATURE_COUNT},), got "
                f"{self.mean.shape}/{self.var.shape}"
            )
        if self.embedding_lt.shape != (_EMBEDDING_DIM,) or self.embedding_sess.shape != (_EMBEDDING_DIM,):
            raise ValueError(
                f"embeddings must have shape ({_EMBEDDING_DIM},)"
            )
        uid_bytes = self.user_id.encode("utf-8")
        header = struct.pack(
            "<HQQBI",
            self.schema_version,
            self.owner_device_id,
            self.monotonic_seq,
            _FEATURE_COUNT,
            len(uid_bytes),
        )
        mean_b = self.mean.astype(np.float32).tobytes()
        var_b = self.var.astype(np.float32).tobytes()
        emb_lt_b = self.embedding_lt.astype(np.float32).tobytes()
        emb_ss_b = self.embedding_sess.astype(np.float32).tobytes()
        adapt_b = struct.pack("<8f", *dataclasses.astuple(self.adaptation))
        meta_b = struct.pack("<id", self.ttl_s, self.created_at)
        return (
            header
            + uid_bytes
            + mean_b
            + var_b
            + emb_lt_b
            + emb_ss_b
            + adapt_b
            + meta_b
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> I3UserStateSync:
        """Unpack a binary blob produced by :meth:`to_bytes`."""
        header_size = struct.calcsize("<HQQBI")
        schema, owner, seq, fc, uid_len = struct.unpack_from("<HQQBI", data, 0)
        if fc != _FEATURE_COUNT:
            raise ValueError(f"unexpected feature count in payload: {fc}")
        off = header_size
        uid = data[off : off + uid_len].decode("utf-8")
        off += uid_len
        mean = np.frombuffer(
            data, dtype=np.float32, count=_FEATURE_COUNT, offset=off
        ).copy()
        off += _FEATURE_COUNT * 4
        var = np.frombuffer(
            data, dtype=np.float32, count=_FEATURE_COUNT, offset=off
        ).copy()
        off += _FEATURE_COUNT * 4
        emb_lt = np.frombuffer(
            data, dtype=np.float32, count=_EMBEDDING_DIM, offset=off
        ).copy()
        off += _EMBEDDING_DIM * 4
        emb_ss = np.frombuffer(
            data, dtype=np.float32, count=_EMBEDDING_DIM, offset=off
        ).copy()
        off += _EMBEDDING_DIM * 4
        adapt_vals = struct.unpack_from("<8f", data, off)
        off += 8 * 4
        ttl_s, created_at = struct.unpack_from("<id", data, off)
        return cls(
            schema_version=schema,
            owner_device_id=owner,
            monotonic_seq=seq,
            user_id=uid,
            mean=mean,
            var=var,
            embedding_lt=emb_lt,
            embedding_sess=emb_ss,
            adaptation=AdaptationVectorPayload(*adapt_vals),
            ttl_s=ttl_s,
            created_at=created_at,
        )


# ---------------------------------------------------------------------------
# Mock DDM client
# ---------------------------------------------------------------------------

@dataclass
class _StoredRecord:
    """Internal storage record — envelope + decryption metadata."""

    encrypted_payload: bytes
    integrity_tag: bytes
    expires_at: float


class DDMSyncClient:
    """In-memory mock of the HarmonyOS DDM sync client.

    The client supports the four methods specified in
    ``docs/huawei/harmony_hmaf_integration.md``: ``push``, ``pull``,
    ``subscribe``, and ``list_peers``.  Payloads are Fernet-encrypted and
    accompanied by a SHA-256 integrity tag.

    Args:
        owner_device_id: The monotonic 64-bit id of the *calling* device.
        encryptor: A :class:`ModelEncryptor` used to wrap payloads.  In
            production this would be the HMAF-derived per-user key.
        registry: The shared :class:`DeviceRegistry`.
    """

    def __init__(
        self,
        owner_device_id: int,
        encryptor: ModelEncryptor,
        registry: DeviceRegistry | None = None,
    ) -> None:
        self.owner_device_id = owner_device_id
        self._encryptor = encryptor
        self._registry = registry or DeviceRegistry()

        self._store: dict[str, _StoredRecord] = {}
        self._subscribers: dict[str, list[Callable[[I3UserStateSync], None]]] = {}
        self._seq = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def push(
        self,
        user_id: str,
        embedding: np.ndarray,
        adaptation: AdaptationVector,
        ttl_s: int = 300,
        *,
        mean: np.ndarray | None = None,
        var: np.ndarray | None = None,
        embedding_sess: np.ndarray | None = None,
    ) -> None:
        """Publish a user-state record to the mock databus.

        Args:
            user_id: Opaque user identifier (not PII; typically a salted hash).
            embedding: Long-term 64-dim embedding.
            adaptation: Current :class:`AdaptationVector`.
            ttl_s: Record lifetime.  Defaults to 300 seconds.
            mean: Optional 32-dim long-term feature mean; zeros if omitted.
            var: Optional 32-dim long-term feature variance; zeros if omitted.
            embedding_sess: Optional 64-dim session embedding; zeros if omitted.
        """
        if embedding.shape != (_EMBEDDING_DIM,):
            raise ValueError(
                f"embedding must have shape ({_EMBEDDING_DIM},), "
                f"got {embedding.shape}"
            )

        mean_arr = np.zeros(_FEATURE_COUNT, dtype=np.float32) if mean is None else mean.astype(np.float32)
        var_arr = np.zeros(_FEATURE_COUNT, dtype=np.float32) if var is None else var.astype(np.float32)
        emb_ss = (
            np.zeros(_EMBEDDING_DIM, dtype=np.float32)
            if embedding_sess is None
            else embedding_sess.astype(np.float32)
        )

        with self._lock:
            self._seq += 1
            payload = I3UserStateSync(
                schema_version=_SCHEMA_VERSION,
                owner_device_id=self.owner_device_id,
                monotonic_seq=self._seq,
                user_id=user_id,
                mean=mean_arr,
                var=var_arr,
                embedding_lt=embedding.astype(np.float32),
                embedding_sess=emb_ss,
                adaptation=AdaptationVectorPayload.from_adaptation(adaptation),
                ttl_s=ttl_s,
            )
            raw = payload.to_bytes()
            integrity = hashlib.sha256(raw).digest()
            encrypted = self._encryptor.encrypt(raw)
            self._store[user_id] = _StoredRecord(
                encrypted_payload=encrypted,
                integrity_tag=integrity,
                expires_at=time.time() + ttl_s,
            )
            subscribers = list(self._subscribers.get(user_id, []))

        logger.debug(
            "DDMSyncClient.push user=%s seq=%d ttl=%ds",
            user_id,
            self._seq,
            ttl_s,
        )
        for cb in subscribers:
            try:
                cb(payload)
            except Exception as exc:  # pragma: no cover - demo path
                logger.warning("Subscriber callback raised: %s", exc)

    def pull(self, user_id: str) -> I3UserStateSync | None:
        """Retrieve the latest published state for *user_id*, if any.

        Returns ``None`` when no record exists or the record has expired.
        Raises :class:`ValueError` if the integrity tag does not match — a
        signal of tampering (or a bug in the mock).
        """
        with self._lock:
            rec = self._store.get(user_id)
            if rec is None:
                return None
            if rec.expires_at < time.time():
                del self._store[user_id]
                return None
            raw = self._encryptor.decrypt(rec.encrypted_payload)

        if hashlib.sha256(raw).digest() != rec.integrity_tag:
            raise ValueError(
                "DDMSyncClient.pull: integrity tag mismatch — payload tampered"
            )
        return I3UserStateSync.from_bytes(raw)

    def subscribe(
        self,
        user_id: str,
        callback: Callable[[I3UserStateSync], None],
    ) -> Callable[[], None]:
        """Register *callback* for push events on *user_id*.

        Returns an ``unsubscribe`` callable.
        """
        with self._lock:
            self._subscribers.setdefault(user_id, []).append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                if user_id in self._subscribers:
                    self._subscribers[user_id] = [
                        c for c in self._subscribers[user_id] if c is not callback
                    ]

        return _unsubscribe

    def list_peers(self) -> list[DeviceInfo]:
        """Return the current set of paired devices from the registry."""
        return self._registry.list_devices()
