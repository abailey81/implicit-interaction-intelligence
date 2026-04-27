"""REST endpoints for the vision-gaze fine-tuning showcase.

Three endpoints back the third multimodal flagship surface:

* ``POST /api/gaze/calibrate``    — fine-tune the user's head on
  ~30 calibration frames (4 classes × 5–10 frames per class).
* ``POST /api/gaze/infer``        — run one frame through the
  classifier and return the :class:`GazeFeatures` snapshot.
* ``GET  /api/gaze/{user_id}/status`` — return calibration state +
  on-disk-checkpoint presence + label set.

The browser-side companion is :mod:`web/js/gaze_capture.js`.

Privacy contract
----------------
* The browser ships a flat 64×48 grayscale fingerprint as a list of
  3072 uint8 ints (or its base64 form).  No raw frames cross the wire.
* The server-side fine-tuned head is keyed on
  ``sha256(biometric_template)`` so the per-user model is not
  identifiable from the user_id alone — same scheme as the LoRA
  personalisation route.
* The fingerprint is held only for the single forward pass and then
  discarded; nothing persists beyond the small (≤300 KB) head
  checkpoint at ``checkpoints/gaze/<user_key>.pt``.

Cites Howard et al. 2019 ("Searching for MobileNetV3") for the
backbone; the head fine-tuning recipe is the classical frozen-backbone
+ new-head transfer-learning of Yosinski et al. 2014 / Donahue et
al. 2014.  See :mod:`i3.multimodal.gaze_classifier` for the full
design rationale.
"""

from __future__ import annotations

import base64
import logging
import re

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from i3.multimodal.gaze_classifier import (
    FINGERPRINT_HEIGHT,
    FINGERPRINT_PIXELS,
    FINGERPRINT_WIDTH,
    GAZE_LABELS,
)

logger = logging.getLogger(__name__)


_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_FRAMES_PER_CLASS = 30
_MAX_TOTAL_FRAMES = 4 * _MAX_FRAMES_PER_CLASS
_MAX_FINGERPRINT_BYTES = FINGERPRINT_PIXELS  # 3072

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class GazeCalibrateRequest(BaseModel):
    """Calibration payload from the browser.

    ``calibration_frames`` is a dict mapping each gaze label
    (``at_screen``, ``away_left``, ``away_right``, ``away_other``) to
    a list of fingerprints.  Each fingerprint is either:

    * a ``list[int]`` of length ``FINGERPRINT_PIXELS`` (3072 uint8
      values, little-endian row-major), or
    * a base64-encoded byte string of length ``FINGERPRINT_PIXELS``.

    The list-of-int form is what the JS extractor sends; we accept the
    base64 form as a convenience for tests / Python probes.
    """

    user_id: str = Field(..., min_length=1, max_length=64)
    calibration_frames: dict
    epochs: int = Field(default=50, ge=1, le=200)
    lr: float = Field(default=1e-3, gt=0.0, lt=1.0)


class GazeInferRequest(BaseModel):
    """Single-frame inference payload."""

    user_id: str = Field(..., min_length=1, max_length=64)
    fingerprint: list[int] | str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_user_id(user_id: str) -> None:
    """Raise HTTP 400 if *user_id* fails the standard regex."""
    if not _USER_ID_RE.match(user_id or ""):
        raise HTTPException(status_code=400, detail="invalid user_id")


def _get_pipeline(request: Request):
    """Return the live Pipeline or raise 503 if not yet ready."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")
    return pipeline


def _decode_fingerprint(raw) -> np.ndarray:
    """Convert one fingerprint payload into a ``[H, W]`` uint8 array.

    Accepts a list of 3072 ints, a base64 string, or a flat ndarray.
    Raises :class:`ValueError` on anything else (the route handler
    converts that to HTTP 400).
    """
    if isinstance(raw, list):
        if len(raw) != FINGERPRINT_PIXELS:
            raise ValueError(
                f"fingerprint length {len(raw)} != expected {FINGERPRINT_PIXELS}"
            )
        arr = np.asarray(raw, dtype=np.float32)
    elif isinstance(raw, str):
        try:
            decoded = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise ValueError(f"invalid base64 fingerprint: {exc}") from exc
        if len(decoded) != FINGERPRINT_PIXELS:
            raise ValueError(
                f"decoded fingerprint length {len(decoded)} "
                f"!= expected {FINGERPRINT_PIXELS}"
            )
        arr = np.frombuffer(decoded, dtype=np.uint8).astype(np.float32)
    elif isinstance(raw, np.ndarray):
        flat = raw.flatten()
        if flat.size != FINGERPRINT_PIXELS:
            raise ValueError(
                f"fingerprint size {flat.size} != {FINGERPRINT_PIXELS}"
            )
        arr = flat.astype(np.float32)
    else:
        raise ValueError(f"unsupported fingerprint type: {type(raw).__name__}")

    # Clamp to [0, 255] in case the JS sent floats; reshape to [H, W].
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr.reshape(FINGERPRINT_HEIGHT, FINGERPRINT_WIDTH)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/api/gaze/calibrate")
async def calibrate_gaze(request: Request, body: GazeCalibrateRequest):
    """Fine-tune the user's gaze head on the supplied calibration frames.

    Request body shape:

        {
          "user_id": "demo_user",
          "calibration_frames": {
            "at_screen": [<fingerprint>, ...],
            "away_left": [<fingerprint>, ...],
            "away_right": [<fingerprint>, ...],
            "away_other": [<fingerprint>, ...]
          },
          "epochs": 50,
          "lr": 0.001
        }

    Returns ``{success, final_loss, val_accuracy, n_frames_used,
    user_key, epochs}`` on success.  The fine-tuned head is persisted
    to ``checkpoints/gaze/<user_key>.pt`` (≤300 KB).
    """
    _validate_user_id(body.user_id)
    pipeline = _get_pipeline(request)

    if not isinstance(body.calibration_frames, dict):
        raise HTTPException(status_code=400, detail="calibration_frames_must_be_dict")

    decoded: dict[str, list[np.ndarray]] = {}
    total = 0
    for lbl, frames in body.calibration_frames.items():
        if lbl not in GAZE_LABELS:
            # Reject unknown labels rather than silently dropping them
            # so a client bug fails loud.
            raise HTTPException(
                status_code=400, detail=f"unknown gaze label: {lbl!r}",
            )
        if not isinstance(frames, list):
            raise HTTPException(
                status_code=400,
                detail=f"frames for {lbl!r} must be a list",
            )
        if len(frames) > _MAX_FRAMES_PER_CLASS:
            raise HTTPException(
                status_code=413,
                detail=f"too many frames for {lbl!r} "
                f"({len(frames)} > {_MAX_FRAMES_PER_CLASS})",
            )
        decoded_list: list[np.ndarray] = []
        for raw in frames:
            try:
                decoded_list.append(_decode_fingerprint(raw))
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"bad fingerprint for {lbl!r}: {exc}",
                ) from exc
        decoded[lbl] = decoded_list
        total += len(decoded_list)

    if total == 0:
        raise HTTPException(
            status_code=400, detail="calibration_frames was empty",
        )
    if total > _MAX_TOTAL_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"too many total frames ({total} > {_MAX_TOTAL_FRAMES})",
        )

    try:
        result = pipeline.calibrate_gaze_classifier(
            body.user_id, decoded,
            epochs=body.epochs, lr=body.lr,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "calibrate_gaze failed for user_id=%s", body.user_id,
        )
        raise HTTPException(status_code=500, detail="calibration_failed")

    return JSONResponse({
        "success": bool(result.get("success", True)),
        "final_loss": float(result.get("final_loss", 0.0)),
        "val_accuracy": float(result.get("val_accuracy", 0.0)),
        "n_frames_used": int(result.get("n_frames_used", total)),
        "epochs": int(result.get("epochs", body.epochs)),
        # SEC: Only the first 16 chars of the user_key are exposed —
        # enough for debugging which checkpoint was written, but not
        # enough to invert back to a biometric template.
        "user_key_prefix": str(result.get("user_key", ""))[:16],
    })


@router.post("/api/gaze/infer")
async def infer_gaze(request: Request, body: GazeInferRequest):
    """Run a single fingerprint through the user's gaze classifier.

    Returns the :class:`~i3.multimodal.GazeFeatures` snapshot as a
    JSON-safe dict.  When the classifier is uncalibrated (fresh user)
    the returned label / confidence reflect the un-fine-tuned head's
    output — typically near-uniform across classes.
    """
    _validate_user_id(body.user_id)
    pipeline = _get_pipeline(request)
    if body.fingerprint is None:
        raise HTTPException(status_code=400, detail="fingerprint_required")
    try:
        arr = _decode_fingerprint(body.fingerprint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        snap = pipeline.infer_gaze(body.user_id, arr)
    except Exception:
        logger.exception(
            "infer_gaze failed for user_id=%s", body.user_id,
        )
        raise HTTPException(status_code=500, detail="infer_failed")
    return JSONResponse(snap)


@router.get("/api/gaze/{user_id}/status")
async def get_gaze_status(
    request: Request,
    user_id: str = Path(..., min_length=1, max_length=64),
) -> JSONResponse:
    """Return the per-user gaze classifier status.

    Always returns a valid dict; the ``calibrated`` field tells the
    caller whether the user has completed the calibration flow.
    """
    _validate_user_id(user_id)
    pipeline = _get_pipeline(request)
    try:
        payload = pipeline.get_gaze_status(user_id)
    except Exception:
        logger.exception(
            "get_gaze_status failed for user_id=%s", user_id,
        )
        raise HTTPException(status_code=500, detail="status_failed")
    # Truncate the user_key for the wire — the full key is used only
    # internally to address the on-disk checkpoint.
    if isinstance(payload, dict) and payload.get("user_key"):
        payload["user_key_prefix"] = str(payload["user_key"])[:16]
        payload.pop("user_key", None)
    return JSONResponse(payload)


def include_gaze_routes(app: FastAPI) -> None:
    """Mount the gaze router on *app*."""
    app.include_router(router)


__all__ = ["include_gaze_routes", "router"]
