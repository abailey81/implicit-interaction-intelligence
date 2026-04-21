"""OpenSSF Model Signing v1.0 integration (sigstore-backed).

Wraps the ``model-signing`` PyPI package
(https://pypi.org/project/model-signing/) to sign and verify I3 model
artefacts according to the OpenSSF Model Signing v1.0 specification.
The specification, announced in April 2025 by the OpenSSF AI/ML
Working Group, provides a standard manifest format and verification
flow for ML models, anchored by default to the public Sigstore
transparency log (https://www.sigstore.dev/).

References:

* OpenSSF blog — "Announcing Model Signing v1.0" (April 2025):
  https://openssf.org/blog/2025/04/
* sigstore blog — Model signing with sigstore:
  https://blog.sigstore.dev/

Three signing backends are supported:

* ``"sigstore"`` — keyless signing via OIDC, the default. Produces
  a ``.sig`` manifest that can be verified by anyone with internet
  access and a trusted OIDC issuer.
* ``"pki"`` — X.509 certificate-based signing for air-gapped or
  enterprise PKI deployments.
* ``"bare_key"`` — raw public/private key pair, typically for
  offline development and test fixtures.

Soft-imports ``model_signing``. If the package is missing every
operation raises :class:`ModuleNotFoundError` with a clear install
command.

Usage::

    from pathlib import Path
    from i3.mlops.model_signing import ModelSigner

    signer = ModelSigner()
    signer.sign(Path("checkpoints/slm/best.pt"), method="sigstore")
    ok = signer.verify(
        Path("checkpoints/slm/best.pt"),
        Path("checkpoints/slm/best.pt.sig"),
        identity="ci@example.com",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import model_signing as _model_signing_module  # type: ignore[import-not-found]

    _MODEL_SIGNING_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _model_signing_module = None  # type: ignore[assignment]
    _MODEL_SIGNING_AVAILABLE = False
    logger.info(
        "model-signing package not installed; ModelSigner will be unavailable. "
        "Install with: pip install model-signing>=1.0"
    )


SigningMethod = Literal["sigstore", "pki", "bare_key"]

_INSTALL_HINT: str = (
    "The 'model-signing' package is required for ModelSigner operations. "
    "Install it with:\n\n    pip install 'model-signing>=1.0'\n\n"
    "See https://pypi.org/project/model-signing/ for backend setup."
)

_DEFAULT_SIGSTORE_ISSUER: str = "https://accounts.google.com"


def _require_model_signing() -> ModuleType:
    """Return the imported ``model_signing`` module or raise."""
    if not _MODEL_SIGNING_AVAILABLE or _model_signing_module is None:
        raise ModuleNotFoundError(_INSTALL_HINT)
    return _model_signing_module


class ModelSigner:
    """Sign and verify model artefacts per OpenSSF Model Signing v1.0.

    The class is a thin adapter around the ``model_signing`` Python
    API. It centralises backend selection so the rest of the I3
    codebase does not need to know how to configure Sigstore, PKI or
    bare-key signing.

    Attributes:
        default_method: The signing backend used when ``method`` is
            not supplied at call time.
        sigstore_issuer: OIDC issuer URL for Sigstore keyless signing.
    """

    def __init__(
        self,
        *,
        default_method: SigningMethod = "sigstore",
        sigstore_issuer: str = _DEFAULT_SIGSTORE_ISSUER,
    ) -> None:
        """Initialise the signer.

        Args:
            default_method: Default signing backend.
            sigstore_issuer: OIDC issuer URL used for keyless signing.
        """
        self.default_method: SigningMethod = default_method
        self.sigstore_issuer: str = sigstore_issuer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sign_config(
        self,
        method: SigningMethod,
        *,
        identity: Optional[str] = None,
        private_key: Optional[Path] = None,
        certificate: Optional[Path] = None,
    ) -> Any:
        """Construct a ``model_signing.sign.Config`` for *method*.

        Args:
            method: One of ``"sigstore"``, ``"pki"``, ``"bare_key"``.
            identity: OIDC identity (sigstore only).
            private_key: Path to the private key file (bare_key / pki).
            certificate: Path to the X.509 certificate (pki).

        Returns:
            A configured ``model_signing.sign.Config`` instance.

        Raises:
            RuntimeError: If the selected method is unrecognised or
                the required parameters are missing.
            ModuleNotFoundError: If ``model_signing`` is absent.
        """
        ms = _require_model_signing()
        sign_mod = getattr(ms, "sign", None)
        if sign_mod is None:
            raise RuntimeError(
                "model_signing.sign submodule is missing; upgrade the package."
            )
        config_cls = getattr(sign_mod, "Config", None)
        if config_cls is None:
            raise RuntimeError(
                "model_signing.sign.Config is missing; upgrade the package."
            )
        cfg = config_cls()

        if method == "sigstore":
            fn = getattr(cfg, "use_sigstore_signer", None)
            if fn is None:
                raise RuntimeError(
                    "model_signing.sign.Config lacks use_sigstore_signer()."
                )
            kwargs: dict[str, Any] = {"oidc_issuer": self.sigstore_issuer}
            if identity is not None:
                kwargs["identity"] = identity
            try:
                cfg = fn(**kwargs)
            except TypeError:
                # Older API versions may not accept identity; fall back.
                cfg = fn()
        elif method == "pki":
            fn = getattr(cfg, "use_pki_signer", None)
            if fn is None:
                raise RuntimeError(
                    "model_signing.sign.Config lacks use_pki_signer()."
                )
            if private_key is None or certificate is None:
                raise RuntimeError(
                    "PKI signing requires both 'private_key' and 'certificate'."
                )
            cfg = fn(
                private_key=str(private_key),
                certificate_chain=str(certificate),
            )
        elif method == "bare_key":
            fn = getattr(cfg, "use_private_key_signer", None) or getattr(
                cfg, "use_bare_key_signer", None
            )
            if fn is None:
                raise RuntimeError(
                    "model_signing.sign.Config lacks a bare-key signer method."
                )
            if private_key is None:
                raise RuntimeError(
                    "Bare-key signing requires 'private_key'."
                )
            cfg = fn(private_key=str(private_key))
        else:  # pragma: no cover - typed Literal guard
            raise RuntimeError(
                f"Unknown signing method: {method!r}. "
                f"Expected one of: sigstore, pki, bare_key."
            )
        return cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sign(
        self,
        path: Path,
        method: Optional[SigningMethod] = None,
        *,
        out_path: Optional[Path] = None,
        identity: Optional[str] = None,
        private_key: Optional[Path] = None,
        certificate: Optional[Path] = None,
    ) -> Path:
        """Sign *path* and write the OpenSSF manifest to *out_path*.

        Args:
            path: Path to the model artefact (file or directory).
            method: Signing backend. Defaults to :attr:`default_method`.
            out_path: Where to write the ``.sig`` manifest. Defaults to
                ``path.with_suffix(path.suffix + '.sig')``.
            identity: OIDC identity for Sigstore signing.
            private_key: Private key file for PKI / bare-key signing.
            certificate: X.509 certificate for PKI signing.

        Returns:
            The path to the written signature manifest.

        Raises:
            ModuleNotFoundError: If ``model_signing`` is not installed.
            FileNotFoundError: If *path* does not exist.
            RuntimeError: If the backend fails or is misconfigured.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artefact not found: {path}")
        actual_method: SigningMethod = method or self.default_method
        actual_out = (
            Path(out_path)
            if out_path is not None
            else path.with_suffix(path.suffix + ".sig")
        )
        cfg = self._build_sign_config(
            actual_method,
            identity=identity,
            private_key=private_key,
            certificate=certificate,
        )
        try:
            sign_fn = getattr(cfg, "sign", None)
            if not callable(sign_fn):
                raise RuntimeError(
                    "model_signing Config instance has no sign() method."
                )
            try:
                sign_fn(model_path=str(path), signature_path=str(actual_out))
            except TypeError:
                # Some SDK versions use (model, signature).
                sign_fn(str(path), str(actual_out))
        except ModuleNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Model signing failed ({type(exc).__name__}): {exc}"
            ) from exc
        logger.info(
            "Signed %s with method=%s -> %s", path, actual_method, actual_out
        )
        return actual_out

    def verify(
        self,
        path: Path,
        signature_path: Path,
        *,
        identity: Optional[str] = None,
        issuer: str = _DEFAULT_SIGSTORE_ISSUER,
        method: Optional[SigningMethod] = None,
        public_key: Optional[Path] = None,
    ) -> bool:
        """Verify *path* against *signature_path*.

        Args:
            path: Path to the model artefact.
            signature_path: Path to the ``.sig`` manifest produced by
                :meth:`sign`.
            identity: Expected OIDC identity for Sigstore verification.
            issuer: Expected OIDC issuer (sigstore only).
            method: Verification backend. Defaults to
                :attr:`default_method`.
            public_key: Optional public key for bare-key verification.

        Returns:
            ``True`` if verification succeeds, ``False`` otherwise.

        Raises:
            ModuleNotFoundError: If ``model_signing`` is not installed.
            FileNotFoundError: If either path does not exist.
        """
        path = Path(path)
        signature_path = Path(signature_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artefact not found: {path}")
        if not signature_path.exists():
            raise FileNotFoundError(
                f"Signature manifest not found: {signature_path}"
            )
        ms = _require_model_signing()
        verify_mod = getattr(ms, "verify", None)
        if verify_mod is None:
            raise RuntimeError(
                "model_signing.verify submodule is missing; upgrade the package."
            )
        config_cls = getattr(verify_mod, "Config", None)
        if config_cls is None:
            raise RuntimeError(
                "model_signing.verify.Config is missing; upgrade the package."
            )
        cfg = config_cls()
        actual_method: SigningMethod = method or self.default_method
        try:
            if actual_method == "sigstore":
                fn = getattr(cfg, "use_sigstore_verifier", None)
                if fn is None:
                    raise RuntimeError(
                        "model_signing.verify.Config lacks "
                        "use_sigstore_verifier()."
                    )
                kwargs: dict[str, Any] = {"oidc_issuer": issuer}
                if identity is not None:
                    kwargs["identity"] = identity
                try:
                    cfg = fn(**kwargs)
                except TypeError:
                    cfg = fn()
            elif actual_method == "pki":
                fn = getattr(cfg, "use_pki_verifier", None)
                if fn is None:
                    raise RuntimeError(
                        "model_signing.verify.Config lacks use_pki_verifier()."
                    )
                cfg = fn()
            else:  # bare_key
                fn = getattr(cfg, "use_public_key_verifier", None) or getattr(
                    cfg, "use_bare_key_verifier", None
                )
                if fn is None:
                    raise RuntimeError(
                        "model_signing.verify.Config lacks a bare-key verifier."
                    )
                if public_key is None:
                    raise RuntimeError(
                        "Bare-key verification requires 'public_key'."
                    )
                cfg = fn(public_key=str(public_key))

            verify_fn = getattr(cfg, "verify", None)
            if not callable(verify_fn):
                raise RuntimeError(
                    "model_signing Config instance has no verify() method."
                )
            try:
                verify_fn(
                    model_path=str(path),
                    signature_path=str(signature_path),
                )
            except TypeError:
                verify_fn(str(path), str(signature_path))
            logger.info(
                "Verified %s against %s (method=%s)",
                path,
                signature_path,
                actual_method,
            )
            return True
        except ModuleNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Model verification failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return False


__all__ = ["ModelSigner", "SigningMethod"]
