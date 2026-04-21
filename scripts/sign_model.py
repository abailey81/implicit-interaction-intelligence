#!/usr/bin/env python
"""CLI wrapper for :class:`i3.mlops.model_signing.ModelSigner`.

Examples::

    # Keyless sigstore signing of the SLM checkpoint
    python scripts/sign_model.py sign \\
        --model checkpoints/slm/best.pt \\
        --method sigstore

    # Verification with expected identity
    python scripts/sign_model.py verify \\
        --model checkpoints/slm/best.pt \\
        --signature checkpoints/slm/best.pt.sig \\
        --identity ci@example.com

    # Bare-key signing for offline test fixtures
    python scripts/sign_model.py sign \\
        --model checkpoints/slm/best.pt \\
        --method bare_key \\
        --private-key keys/dev.key

Requires the optional ``model-signing>=1.0`` package:

    pip install 'model-signing>=1.0'
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from i3.mlops.model_signing import ModelSigner, SigningMethod

logger = logging.getLogger("i3.scripts.sign_model")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree for sign / verify subcommands."""
    parser = argparse.ArgumentParser(
        prog="sign_model",
        description=(
            "Sign and verify I3 model artefacts using OpenSSF Model "
            "Signing v1.0 (sigstore / PKI / bare-key)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sign = sub.add_parser("sign", help="Sign a model artefact.")
    p_sign.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model file or directory to sign.",
    )
    p_sign.add_argument(
        "--method",
        choices=("sigstore", "pki", "bare_key"),
        default="sigstore",
        help="Signing backend (default: sigstore).",
    )
    p_sign.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the .sig manifest (default: <model>.sig).",
    )
    p_sign.add_argument(
        "--identity",
        default=None,
        help="OIDC identity for sigstore (email or SPIFFE URI).",
    )
    p_sign.add_argument(
        "--identity-issuer",
        default="https://accounts.google.com",
        help="OIDC issuer URL for sigstore (default: Google).",
    )
    p_sign.add_argument(
        "--private-key",
        type=Path,
        default=None,
        help="Private key path for PKI or bare-key signing.",
    )
    p_sign.add_argument(
        "--certificate",
        type=Path,
        default=None,
        help="X.509 certificate path for PKI signing.",
    )

    p_verify = sub.add_parser("verify", help="Verify a model artefact.")
    p_verify.add_argument(
        "--model", type=Path, required=True, help="Path to the model artefact."
    )
    p_verify.add_argument(
        "--signature",
        type=Path,
        required=True,
        help="Path to the .sig manifest.",
    )
    p_verify.add_argument(
        "--identity",
        default=None,
        help="Expected OIDC identity (sigstore only).",
    )
    p_verify.add_argument(
        "--identity-issuer",
        default="https://accounts.google.com",
        help="Expected OIDC issuer URL (default: Google).",
    )
    p_verify.add_argument(
        "--method",
        choices=("sigstore", "pki", "bare_key"),
        default="sigstore",
        help="Verification backend (default: sigstore).",
    )
    p_verify.add_argument(
        "--public-key",
        type=Path,
        default=None,
        help="Public key path for bare-key verification.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a shell-style exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    signer = ModelSigner(sigstore_issuer=args.identity_issuer)

    if args.command == "sign":
        method: SigningMethod = args.method
        out = signer.sign(
            args.model,
            method=method,
            out_path=args.out,
            identity=args.identity,
            private_key=args.private_key,
            certificate=args.certificate,
        )
        print(f"Signed -> {out}")
        return 0

    if args.command == "verify":
        method_v: SigningMethod = args.method
        ok = signer.verify(
            args.model,
            args.signature,
            identity=args.identity,
            issuer=args.identity_issuer,
            method=method_v,
            public_key=args.public_key,
        )
        if ok:
            print("Signature VALID")
            return 0
        print("Signature INVALID", file=sys.stderr)
        return 1

    parser.error("Unknown command.")
    return 2  # pragma: no cover - argparse raises before we get here


if __name__ == "__main__":
    raise SystemExit(main())
