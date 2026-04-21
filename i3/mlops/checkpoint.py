"""Checkpoint integrity for I3: SHA-256, signature stub, safe loading.

This module provides a thin, defensive layer around ``torch.save`` /
``torch.load`` that prevents three classes of failure:

1. **Corruption on disk** -- verified by a SHA-256 sidecar written next
   to the checkpoint.
2. **Tampered checkpoints in transit** -- optional cosign-style detached
   signature sidecar (``<path>.sig``).  Signature verification is a stub
   today; production deployments should wire a real KMS-backed verifier
   by overriding :func:`_verify_signature`.
3. **Pickle code execution** -- :func:`load_verified` always calls
   ``torch.load`` with ``weights_only=True`` by default (PyTorch >= 2.6).

Example::

    from i3.mlops.checkpoint import save_with_hash, load_verified

    save_with_hash(
        model,
        "checkpoints/encoder/best.pt",
        metadata={"epoch": 42, "val_loss": 0.12},
    )

    state = load_verified("checkpoints/encoder/best.pt")
    model.load_state_dict(state)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ChecksumError(RuntimeError):
    """Raised when a checkpoint's SHA-256 does not match its sidecar."""


class SignatureError(RuntimeError):
    """Raised when a checkpoint's detached signature fails verification."""


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #


_READ_CHUNK = 1 << 20  # 1 MiB


def compute_sha256(path: str | Path) -> str:
    """Stream a file through SHA-256 and return the hex digest.

    Args:
        path: Path to the file on disk.

    Returns:
        Lower-case hex digest string (64 chars).
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_hash_sidecar(path: Path, digest: str) -> Path:
    """Write ``<digest>  <basename>`` to ``<path>.sha256``.

    Args:
        path: Primary checkpoint path.
        digest: Hex digest to record.

    Returns:
        Path to the written sidecar.
    """
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return sidecar


def _read_hash_sidecar(path: Path) -> Optional[str]:
    """Return the digest recorded in ``<path>.sha256`` or ``None``.

    Args:
        path: Primary checkpoint path.

    Returns:
        Hex digest string, or ``None`` if the sidecar does not exist or
        cannot be parsed.
    """
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return None
    try:
        line = sidecar.read_text(encoding="utf-8").strip().split()
        if not line:
            return None
        return line[0].lower()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unreadable sidecar %s: %s", sidecar, exc)
        return None


# --------------------------------------------------------------------------- #
# Signature stub
# --------------------------------------------------------------------------- #


def _verify_signature(path: Path, signature_path: Path) -> bool:
    """Verify a detached cosign-style signature.

    This is a stub: production deployments should replace the body with
    a real verifier (e.g. sigstore-python, KMS).  For now, the presence
    of a non-empty signature file is treated as a soft "trust on first
    use" signal with a warning in the log.

    Args:
        path: Checkpoint file.
        signature_path: Detached signature file.

    Returns:
        ``True`` when the signature is considered valid.
    """
    if not signature_path.exists():
        return True  # No signature requested.
    try:
        data = signature_path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        raise SignatureError(f"signature unreadable: {exc}") from exc
    if not data:
        raise SignatureError(f"signature file empty: {signature_path}")
    logger.warning(
        "Signature sidecar %s present but verification is a stub; "
        "replace i3.mlops.checkpoint._verify_signature in production.",
        signature_path,
    )
    return True


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


def save_with_hash(
    model: Any,
    path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, str]:
    """Save a model state_dict together with SHA-256 and metadata sidecars.

    The primary artefact at ``<path>`` contains the model's state_dict.
    Two sidecars are written alongside:

    * ``<path>.sha256`` - ``<digest>  <basename>`` textual checksum.
    * ``<path>.json``   - JSON metadata (caller-supplied + defaults such
      as timestamp, digest, torch version, python version).

    Args:
        model: A ``torch.nn.Module`` or a pre-computed ``state_dict``.
        path: Destination path for the checkpoint.
        metadata: Optional mapping appended to the JSON metadata file.

    Returns:
        Dictionary with keys ``"path"``, ``"sha256"``, ``"metadata"``
        pointing at the three files written.
    """
    # Local import keeps torch optional at module import time.
    import torch  # type: ignore[import-not-found]

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "state_dict"):
        state = model.state_dict()
    else:
        state = model  # assume caller passed a mapping

    torch.save(state, target)

    digest = compute_sha256(target)
    sha_sidecar = _write_hash_sidecar(target, digest)

    meta_sidecar = target.with_suffix(target.suffix + ".json")
    meta_payload: dict[str, Any] = {
        "path": target.name,
        "sha256": digest,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
    }
    try:
        meta_payload["torch"] = str(torch.__version__)
    except Exception:  # noqa: BLE001
        meta_payload["torch"] = "unknown"
    if metadata:
        # Caller-supplied values override defaults.
        meta_payload.update({str(k): v for k, v in metadata.items()})
    meta_sidecar.write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    logger.info(
        "save_with_hash wrote %s (sha256=%s) + metadata", target, digest
    )
    return {
        "path": str(target),
        "sha256": str(sha_sidecar),
        "metadata": str(meta_sidecar),
    }


def load_verified(
    path: str | Path,
    expected_hash: Optional[str] = None,
    weights_only: bool = True,
    map_location: Any = "cpu",
    verify_signature: bool = True,
) -> Any:
    """Load a checkpoint after verifying its integrity.

    Verification order:

    1. If ``expected_hash`` is provided it takes precedence over the
       sidecar.  The on-disk SHA-256 must match exactly or a
       :class:`ChecksumError` is raised.
    2. Otherwise, if ``<path>.sha256`` exists, its digest is compared
       against the freshly-computed digest.
    3. If neither is present, a warning is logged and the checkpoint is
       loaded without integrity verification.
    4. If ``<path>.sig`` exists and ``verify_signature`` is true,
       :func:`_verify_signature` is invoked (stub today).
    5. Finally ``torch.load`` is called with ``weights_only=True`` by
       default to prevent pickle code execution.

    Args:
        path: Path to the checkpoint file.
        expected_hash: Optional SHA-256 hex digest to compare against.
        weights_only: Passed straight through to ``torch.load``.  The
            default is ``True`` for safety.
        map_location: Passed through to ``torch.load``.
        verify_signature: If ``True``, verify ``<path>.sig`` when present.

    Returns:
        The loaded object (typically a state_dict).

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ChecksumError: When the recorded and computed digests differ.
        SignatureError: When signature verification fails.
    """
    import torch  # type: ignore[import-not-found]

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"checkpoint not found: {target}")

    computed = compute_sha256(target)

    reference: Optional[str] = None
    source: str = "none"
    if expected_hash is not None:
        reference = expected_hash.lower()
        source = "argument"
    else:
        sidecar_digest = _read_hash_sidecar(target)
        if sidecar_digest is not None:
            reference = sidecar_digest
            source = "sidecar"

    if reference is None:
        logger.warning(
            "load_verified: no reference hash for %s -- loading without "
            "integrity verification.",
            target,
        )
    elif reference != computed:
        raise ChecksumError(
            f"SHA-256 mismatch for {target}: expected {reference} "
            f"({source}) but file hashed to {computed}"
        )
    else:
        logger.debug(
            "Integrity OK for %s (source=%s, digest=%s)",
            target,
            source,
            computed,
        )

    if verify_signature:
        sig_path = target.with_suffix(target.suffix + ".sig")
        _verify_signature(target, sig_path)

    return torch.load(
        target,
        map_location=map_location,
        weights_only=weights_only,
    )


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #


def verify_file(path: str | Path, expected_hash: Optional[str] = None) -> str:
    """Hash a file and compare against ``expected_hash`` or the sidecar.

    Args:
        path: File to verify.
        expected_hash: Optional hex digest override.

    Returns:
        The computed SHA-256 hex digest.

    Raises:
        ChecksumError: When digests differ.
    """
    target = Path(path)
    computed = compute_sha256(target)
    reference = expected_hash or _read_hash_sidecar(target)
    if reference is not None and reference.lower() != computed:
        raise ChecksumError(
            f"SHA-256 mismatch for {target}: expected {reference}, got {computed}"
        )
    return computed


__all__ = [
    "ChecksumError",
    "SignatureError",
    "compute_sha256",
    "load_verified",
    "save_with_hash",
    "verify_file",
]


# Explicitly document that hardware or OS noise is not part of the digest.
_UMASK_HINT = os.umask(0o022)
os.umask(_UMASK_HINT)
