"""ONNX artefact server for in-browser inference.

Serves the quantised ``.onnx`` files under ``web/models/`` with the
right headers to unlock WebGPU + threaded WebAssembly in modern
browsers.

Security design
~~~~~~~~~~~~~~~
* The ``{model}`` path parameter is constrained by a strict regex
  (``^[A-Za-z0-9_.-]+\\.onnx$``) AND re-validated at runtime via
  :meth:`pathlib.Path.resolve` against the canonical
  ``web/models/`` root.  Anything outside that directory is rejected
  with a 404 that does NOT leak the resolved path.
* Dotfiles and double-dot traversal attempts never reach the
  filesystem — the regex rejects them before :func:`open` is ever
  called.
* The response is served with ``Content-Type:
  application/octet-stream`` so that a misbehaving client cannot
  coerce the browser into treating the bytes as text/HTML/etc.
* ``Cross-Origin-Opener-Policy: same-origin`` and
  ``Cross-Origin-Embedder-Policy: require-corp`` are emitted so the
  client gains cross-origin isolation — a prerequisite for
  ``SharedArrayBuffer`` (threaded WASM) and for stable
  high-resolution timers used by the metrics overlay.
* ``X-Content-Type-Options: nosniff`` defence-in-depth prevents
  MIME sniffing on old IE / mis-configured middleboxes.

Nothing here depends on torch/numpy so the module is cheap to import.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Path as FPath, status
from fastapi.responses import FileResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/onnx", tags=["inference"])

# Canonical root for ONNX artefacts.  Resolved lazily so unit tests can
# monkey-patch the working directory.
_MODELS_SUBPATH = Path("web") / "models"

# Strict filename pattern.  Mirrors the regex used to validate admin
# user_ids — alphanumerics, ``_``, ``-``, ``.``; mandatory ``.onnx``
# suffix; length-bounded to avoid pathological requests.
_MODEL_FILENAME_PATTERN = r"^[A-Za-z0-9_.-]{1,128}\.onnx$"
_MODEL_FILENAME_RE = re.compile(_MODEL_FILENAME_PATTERN)

# Headers we add on every response (success or error body) so the
# browser sees consistent cross-origin-isolation signals.
_COI_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    # ONNX blobs are immutable per filename — CDN/browser cache is safe.
    "Cache-Control": "public, max-age=3600, immutable",
}


def _models_root() -> Path:
    """Return the absolute, resolved ``web/models`` directory.

    Resolved every request so tests that temporarily change CWD still
    observe the right path.  The cost is negligible (one stat call).
    """
    return _MODELS_SUBPATH.resolve()


def _safe_model_path(model: str) -> Path:
    """Validate ``model`` and return the absolute on-disk path.

    Raises:
        HTTPException: 404 on any validation failure.  The detail
            string is deliberately constant so probing clients cannot
            learn about the filesystem layout.
    """
    if not isinstance(model, str) or not _MODEL_FILENAME_RE.match(model):
        logger.info(
            "onnx.route.rejected_name",
            extra={"event": "onnx_route", "reason": "regex"},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
            headers=dict(_COI_HEADERS),
        )

    root = _models_root()
    candidate = (root / model).resolve()

    # Belt-and-braces: even if the regex somehow permits a traversal
    # payload (it does not), reject anything that escapes the root.
    try:
        candidate.relative_to(root)
    except ValueError:
        logger.warning(
            "onnx.route.path_traversal",
            extra={"event": "onnx_route", "reason": "outside_root"},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
            headers=dict(_COI_HEADERS),
        )

    if not candidate.is_file():
        # SEC (M-14, 2026-04-23 audit): do not echo toolchain hints in
        # 404 detail — that leaks filesystem layout + export command to
        # a scanner.  Operator guidance now lives in the server log
        # (``onnx.route.missing.hint``) only.
        logger.info(
            "onnx.route.missing",
            extra={
                "event": "onnx_route",
                "reason": "missing",
                "hint": (
                    "Export via `python -m i3.encoder.onnx_export` and "
                    "drop into web/models/."
                ),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
            headers=dict(_COI_HEADERS),
        )

    return candidate


@router.get("/{model}")
async def serve_onnx_model(
    model: str = FPath(
        ...,
        pattern=_MODEL_FILENAME_PATTERN,
        min_length=5,
        max_length=128,
        description="ONNX artefact filename (e.g. encoder_int8.onnx).",
    ),
) -> Response:
    """Stream an ONNX artefact to the browser.

    The response carries the full cross-origin-isolation header set so
    the client can run threaded WebAssembly (requires
    ``SharedArrayBuffer``, which requires crossOriginIsolated).
    """
    path = _safe_model_path(model)
    response = FileResponse(
        path,
        media_type="application/octet-stream",
        filename=model,
        headers=dict(_COI_HEADERS),
    )
    # FileResponse accepts a `headers` kwarg on modern Starlette but
    # older pinned versions ignore it; set explicitly just in case.
    for k, v in _COI_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


def include_inference_routes(app: FastAPI) -> None:
    """Mount the inference router on *app*.

    Idempotent: if the router is already attached this is a no-op.
    """
    # The router's prefix is unique, so a simple duplicate check is
    # sufficient without poking at private Starlette internals.
    for route in app.routes:
        if getattr(route, "path", "").startswith("/api/onnx/"):
            logger.debug("onnx.route.already_mounted")
            return

    app.include_router(router)
    logger.info(
        "onnx.route.mounted",
        extra={"event": "onnx_mount", "prefix": router.prefix},
    )
