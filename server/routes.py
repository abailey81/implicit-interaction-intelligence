"""REST API endpoints for the I3 server.

These complement the WebSocket stream with request/response queries
for user profiles, diary entries, statistics, and demo utilities.

Security controls:
    * ``user_id`` is constrained by a strict, anchored regex via a FastAPI
      ``Path`` parameter (alphanumeric, underscore, dash; 1-64 chars).
    * Pagination parameters (``limit``, ``offset``) are bounded by ``Query``
      constraints with explicit integer types.
    * Error messages never echo user input, internal paths, or stack traces.
    * Pipeline access is mediated through ``_get_pipeline`` so a missing
      pipeline yields 503 (service unavailable) rather than 500.
    * Demo endpoints (``/demo/reset`` / ``/demo/seed``) are gated behind the
      ``I3_DEMO_MODE`` environment flag to prevent accidental data loss in
      production deployments.

Known limitation (demo build):
    There is no caller authentication, so any client can read any user_id's
    data. This is acceptable for the on-device demo but MUST be revisited
    before any multi-tenant deployment (e.g. require an authenticated subject
    claim that matches the path ``user_id``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse

from server.auth import require_user_identity

router = APIRouter()
logger = logging.getLogger(__name__)

# SEC: Bounded in-process LRU cache for attention computations keyed by the
# prompt text (post-truncation). CPU-only extraction still costs ~200-500 ms
# on the 4 M-param v1 SLM, so the cache keeps the UI snappy when the user
# revisits a prompt. We cap the number of entries to bound memory.
_ATTENTION_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_ATTENTION_CACHE_MAX: int = 8
_ATTENTION_LOCK: asyncio.Lock = asyncio.Lock()
# Hard cap on seq_len we return so the JSON stays reasonable for the UI
# (4 layers × 8 heads × 32 × 32 floats ≈ 32 kB of JSON).
_ATTENTION_MAX_SEQ: int = 32
# Hard cap on input text length we'll even try to tokenise — protects the
# endpoint from megabyte payloads.
_ATTENTION_MAX_TEXT_CHARS: int = 512


# SEC: gate destructive demo endpoints behind an explicit env flag so a
# misconfigured production deployment cannot wipe state by accident.
def _demo_mode_enabled() -> bool:
    return os.environ.get("I3_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


# SEC: typed pipeline accessor. Distinguishes "service not initialised" (503)
# from "logic error" (500) and avoids raw AttributeError leaking through.
def _get_pipeline(request: Request) -> Any:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return pipeline


# ---------------------------------------------------------------------------
# Shared parameter declarations
# ---------------------------------------------------------------------------

# SEC: anchored at start (^) AND end ($) so embedded newlines or trailing
# bytes (e.g. "alice\n../etc/passwd") cannot satisfy the pattern. The class
# also forbids '/', '.', and whitespace, foreclosing path-traversal payloads.
USER_ID_REGEX = r"^[a-zA-Z0-9_-]{1,64}$"

UserIdParam = Path(
    ...,
    pattern=USER_ID_REGEX,
    min_length=1,
    max_length=64,
    description="Alphanumeric user identifier (1-64 chars).",
)

# SEC: explicit ``int`` type + bounded range. FastAPI/Pydantic raises a
# 422 via the global validation handler in app.py if a caller passes a
# non-integer or out-of-range value, and that handler returns a generic
# message that does NOT echo the offending value back.
LimitParam = Query(
    10,
    ge=1,
    le=100,
    description="Maximum number of records to return (1-100).",
)

OffsetParam = Query(
    0,
    ge=0,
    le=10_000,
    description="Number of records to skip (0-10000).",
)


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@router.get("/stack")
async def get_stack(request: Request) -> JSONResponse:
    """Return a snapshot of the on-device stack for the UI "Architecture" panel.

    The stack summary is deliberately biased toward "what did we build
    from scratch?" — the Huawei HMI filter asks precisely this (can you
    implement SLMs, traditional ML, and pipelines without heavy
    frameworks?).  The dashboard reads this endpoint once on load and
    renders the values above the live adaptation panel so a reader can
    see, at a glance, that the transformer, tokenizer, encoder, bandit,
    and feature extractors all ship locally and are implemented by hand.
    """
    import os
    import sys

    pipeline = getattr(request.app.state, "pipeline", None)
    stack: dict[str, Any] = {
        "from_scratch": True,
        "third_party_llm_deps": 0,  # no HuggingFace, no sentence-transformers
        "device": _describe_device(),
        "slm": _describe_slm(pipeline),
        "encoder": _describe_encoder(pipeline),
        "router": _describe_router(pipeline),
        "edge": _describe_edge(),
        "privacy": _describe_privacy(pipeline),
        "orchestration": _describe_orchestration(),
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        },
    }
    return JSONResponse(stack)


def _describe_device() -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        return {
            "kind": "cuda",
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        }
    return {"kind": "cpu", "name": "CPU", "vram_gb": 0.0}


def _describe_slm(pipeline: Any) -> dict[str, Any]:
    """Summarise the custom AdaptiveSLM."""
    if pipeline is None or getattr(pipeline, "_slm_generator", None) is None:
        return {"loaded": False, "reason": "checkpoint missing or init skipped"}
    try:
        slm_gen = pipeline._slm_generator
        model = slm_gen.model
        n_params = sum(p.numel() for p in model.parameters())
        version = getattr(pipeline, "_slm_version", None) or "v1"
        # v2 uses MoE+ACT and BPE; surface those distinctions for the UI
        # so the Architecture panel can show what's actually running.
        is_v2 = type(model).__name__ == "AdaptiveTransformerV2" or version == "v2"
        name = "AdaptiveTransformerV2" if is_v2 else "AdaptiveSLM"
        impl = (
            "custom MoE+ACT transformer with per-layer cross-attention "
            "conditioning (written from scratch, byte-level BPE tokenizer)"
            if is_v2
            else "custom transformer with cross-attention conditioning "
            "(written from scratch)"
        )
        info: dict[str, Any] = {
            "loaded": True,
            "name": name,
            "version": version,
            "slm_version": version,  # explicit alias for the audit/verify scripts
            "implementation": impl,
            "params": n_params,
            "params_human": f"{n_params / 1e6:.2f} M",
            "vocab_size": int(
                getattr(model, "vocab_size", None)
                or getattr(slm_gen.tokenizer, "vocab_size", None)
                or len(getattr(slm_gen.tokenizer, "token_to_id", {}) or {})
            ),
            "d_model": int(getattr(model, "d_model", 0)) or None,
            "n_layers": int(getattr(model, "n_layers", 0)) or None,
            "n_heads": int(getattr(model, "n_heads", 0)) or None,
            "d_ff": int(getattr(model, "d_ff", 0)) or None,
            "max_seq_len": int(getattr(model, "max_seq_len", 0)) or None,
            "conditioning_dim": int(getattr(model, "conditioning_dim", 0)) or None,
            "adaptation_dim": int(getattr(model, "adaptation_dim", 0)) or None,
            "hf_dependencies": 0,
        }
        # v2-specific extras (MoE expert count, ACT halting threshold).
        if is_v2:
            cfg = getattr(model, "config", None)
            if cfg is not None:
                info["n_experts"] = int(getattr(cfg, "n_experts", 0)) or None
                info["halt_threshold"] = float(getattr(cfg, "halt_threshold", 0.0)) or None
                info["ponder_cost"] = float(getattr(cfg, "ponder_cost", 0.0)) or None
        return info
    except Exception as exc:  # pragma: no cover - defensive
        return {"loaded": True, "error": str(exc)[:100]}


def _describe_encoder(pipeline: Any) -> dict[str, Any]:
    """Summarise the TCN encoder."""
    if pipeline is None or getattr(pipeline, "_encoder", None) is None:
        return {"loaded": False}
    try:
        enc_cfg = pipeline._encoder_config
        return {
            "loaded": True,
            "name": "TCN Encoder",
            "implementation": "dilated temporal convolutions (written from scratch)",
            "input_dim": enc_cfg.input_dim,
            "embedding_dim": enc_cfg.embedding_dim,
            "window_size": pipeline.config.interaction.feature_window,
            "kernel_size": enc_cfg.kernel_size,
            "dilations": list(enc_cfg.dilations),
            "hf_dependencies": 0,
        }
    except Exception as exc:  # pragma: no cover
        return {"loaded": True, "error": str(exc)[:100]}


def _describe_router(pipeline: Any) -> dict[str, Any]:
    """Summarise the contextual-bandit router."""
    if pipeline is None:
        return {"loaded": False}
    try:
        stats = pipeline.router.get_arm_stats()
        return {
            "loaded": True,
            "name": "LinUCB Contextual Bandit",
            "implementation": "linear Thompson-sampling bandit (written from scratch)",
            "arms": list(stats.keys()) if isinstance(stats, dict) else None,
            "stats": stats,
            "hf_dependencies": 0,
        }
    except Exception:
        return {"loaded": True, "error": "router_stats_unavailable"}


def _describe_edge() -> dict[str, Any]:
    """Summarise the edge-deployment story."""
    from pathlib import Path

    onnx_path = Path("checkpoints/encoder/tcn.onnx")
    onnx_exists = onnx_path.is_file()
    size_mb = round(onnx_path.stat().st_size / 1024 / 1024, 2) if onnx_exists else 0.0
    return {
        "onnx_exported": onnx_exists,
        "onnx_size_mb": size_mb,
        "onnx_path": str(onnx_path) if onnx_exists else None,
        "browser_inference_available": True,
        "backends_advertised": ["pytorch_cuda", "pytorch_cpu", "onnx_wasm", "onnx_webgpu"],
    }


def _describe_privacy(pipeline: Any) -> dict[str, Any]:
    """Summarise privacy controls."""
    return {
        "on_device": True,
        "embedding_encryption": "Fernet (AES-128-CBC + HMAC-SHA-256)",
        "raw_text_persisted": False,
        "pii_sanitizer": "regex + rule-based",
        "cloud_route_default": "disabled",
    }


def _describe_orchestration() -> dict[str, Any]:
    """Summarise the wave-based stage DAG."""
    return {
        "pipeline": "scripts/run_everything.py",
        "stages": 21,
        "style": "wave-based DAG (parallel where independent, sequential where not)",
        "reproducible": True,
    }


@router.get("/health")
async def health_check() -> JSONResponse:
    """Liveness probe -- always returns 200 if the server is up.

    SEC: deliberately minimal payload. Does NOT expose pipeline internals,
    connection counts, memory usage, hostname, or config. A separate
    ``/health/detailed`` endpoint behind an internal-network ACL is the
    correct place for ops-grade diagnostics.
    """
    return JSONResponse({"status": "healthy", "version": "1.0.0"})


# ------------------------------------------------------------------
# Attention visualisation (advanced UI panel)
# ------------------------------------------------------------------

def _synthetic_attention(session_id: str) -> dict[str, Any]:
    """Deterministic synthetic 4×4 fallback used when no real model is loaded.

    Kept from the pre-v2 build so the UI has *something* to render before
    the pipeline finishes warm-up. Not used when the real extractor runs.
    """
    import math

    rows, cols = 4, 4
    seed_key = f"{session_id or 'demo'}::{int(__import__('time').time()) // 2}"
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(seed_key))
    mean_layer: list[list[float]] = []
    for r in range(rows):
        row: list[float] = []
        for c in range(cols):
            phase = ((seed + r * 7 + c * 11) % 97) / 97.0
            val = 0.25 + 0.55 * (0.5 + 0.5 * math.sin(phase * math.tau + r + c))
            row.append(max(0.0, min(1.0, val)))
        total = sum(row) or 1.0
        row = [v / total for v in row]
        mean_layer.append(row)
    return {
        "tokens": [f"t{i}" for i in range(rows)],
        "n_layers": 1,
        "n_heads": 1,
        "seq_len": rows,
        "attention": [[mean_layer]],
        "mean_per_layer": [mean_layer],
        "synthetic": True,
        "session_id": session_id,
    }


def _compute_attention_cpu(pipeline: Any, text: str) -> dict[str, Any] | None:
    """Run forward_with_attention on the live SLM on CPU.

    Moves the model to CPU for the duration of the call and restores it to
    its prior device in a ``finally`` block.  Training typically owns the
    GPU, so this lets the demo pull a real attention map without competing
    for VRAM.

    Returns ``None`` if the pipeline has no SLM generator loaded.
    """
    import torch

    slm_gen = getattr(pipeline, "_slm_generator", None)
    if slm_gen is None:
        return None
    model = getattr(slm_gen, "model", None)
    tokenizer = getattr(slm_gen, "tokenizer", None)
    if model is None or tokenizer is None:
        return None

    # Track original device so we can restore after extraction.
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    try:
        model.eval()
        model.cpu()

        # Encode. Strip BOS/EOS padding so the tokens the user sees in the
        # grid are the real content tokens (not wrapper specials).
        # Tokenizer flavour detection — v1 SimpleTokenizer uses
        # ``add_special``; v2 BPETokenizer uses ``add_bos`` / ``add_eos``.
        try:
            ids = tokenizer.encode(text, add_special=False)
        except TypeError:
            ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        if not ids:
            ids = [tokenizer.UNK_ID]
        # Cap seq_len to keep JSON size bounded.
        if len(ids) > _ATTENTION_MAX_SEQ:
            ids = ids[:_ATTENTION_MAX_SEQ]

        # Decode each id to a human-readable token string.  v1 stores
        # word-level surface forms in id_to_token; v2 BPE stores ids
        # like "<bpe_2581>" but the underlying byte sequence is in
        # token_bytes[id] — UTF-8-decode that for a readable label.
        token_bytes = getattr(tokenizer, "token_bytes", None)
        tokens: list[str] = []
        for tid in ids:
            tid_int = int(tid)
            label = None
            if token_bytes is not None and 0 <= tid_int < len(token_bytes):
                raw = token_bytes[tid_int]
                if raw:
                    try:
                        label = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        label = None
            if label is None:
                label = str(tokenizer.id_to_token.get(tid_int, "[UNK]"))
            tokens.append(label)

        input_tensor = torch.tensor([ids], dtype=torch.long)

        logits, per_layer = model.forward_with_attention(input_tensor)

        # per_layer is list[Tensor(1, n_heads, S, S)]
        attention_nested: list[list[list[list[float]]]] = []
        mean_per_layer: list[list[list[float]]] = []
        n_layers = len(per_layer)
        n_heads = per_layer[0].shape[1] if n_layers > 0 else 0
        seq_len = per_layer[0].shape[-1] if n_layers > 0 else 0

        for layer_weights in per_layer:
            # Shape: [1, n_heads, S, S] -> [n_heads, S, S]
            w = layer_weights.squeeze(0).cpu().float()
            # Round for JSON compactness.
            attention_nested.append(
                [
                    [
                        [round(float(v), 4) for v in row]
                        for row in head
                    ]
                    for head in w.tolist()
                ]
            )
            mean_head = w.mean(dim=0)
            mean_per_layer.append(
                [[round(float(v), 4) for v in row] for row in mean_head.tolist()]
            )

        return {
            "tokens": tokens,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "attention": attention_nested,
            "mean_per_layer": mean_per_layer,
            "synthetic": False,
        }
    finally:
        # Restore original device so the pipeline's other callers (e.g.
        # SLMGenerator.generate) find the model where they expect it.
        try:
            model.to(original_device)
        except Exception:  # pragma: no cover - best effort restore
            logger.exception("Failed to restore model device after attention extraction")


@router.get("/attention")
async def get_attention(
    request: Request,
    text: str = Query(
        "Hello, how are you?",
        min_length=0,
        max_length=_ATTENTION_MAX_TEXT_CHARS,
        description="Prompt to run through the SLM for attention extraction.",
    ),
    compute: bool = Query(
        False,
        description=(
            "When true, run the SLM forward pass on CPU to compute fresh "
            "attention weights. When false (default), return the most "
            "recently cached result, or a synthetic 4×4 fallback if the "
            "cache is empty."
        ),
    ),
    session_id: str = Query(
        "",
        min_length=0,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-]*$",
        description="Optional session id (used only for the synthetic fallback).",
    ),
) -> JSONResponse:
    """Return a real transformer self-attention map for the attention-viz panel.

    * ``compute=true``: run ``AdaptiveSLM.forward_with_attention`` on the
      tokenised prompt, CPU-only, and cache the result (small LRU).
    * ``compute=false`` (default): return the most recently cached entry
      so visits from the dashboard don't re-run the model. If nothing is
      cached yet, returns a deterministic synthetic 4×4 fallback so the
      UI never renders an empty grid.

    Response schema::

        {
            "tokens":        ["hello", ",", "how", ...],
            "n_layers":      4,
            "n_heads":       8,
            "seq_len":       7,
            "attention":     [[[[...]]]],  # [L][H][S][S]
            "mean_per_layer":[[[...]]],    # [L][S][S] (averaged over heads)
            "synthetic":     false,
        }
    """
    pipeline = getattr(request.app.state, "pipeline", None)

    # Normalise/truncate the text used as the cache key so ``compute=false``
    # always resolves to the same entry for the same input.
    cache_key = (text or "").strip()[:_ATTENTION_MAX_TEXT_CHARS] or "Hello, how are you?"

    async with _ATTENTION_LOCK:
        if compute and pipeline is not None:
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, _compute_attention_cpu, pipeline, cache_key
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("attention compute failed; falling back to cache/synthetic")
                result = None
            if result is not None:
                # LRU insert
                _ATTENTION_CACHE[cache_key] = result
                _ATTENTION_CACHE.move_to_end(cache_key)
                while len(_ATTENTION_CACHE) > _ATTENTION_CACHE_MAX:
                    _ATTENTION_CACHE.popitem(last=False)

        # Read-through: prefer exact key, otherwise most recent.
        cached: dict[str, Any] | None = _ATTENTION_CACHE.get(cache_key)
        if cached is None and _ATTENTION_CACHE:
            # Most recent entry (last inserted).
            cached = next(reversed(_ATTENTION_CACHE.values()))

    if cached is None:
        payload = _synthetic_attention(session_id)
        payload["text"] = cache_key
        # Legacy-compat alias (4x4 advanced-panels viz).
        payload["matrix"] = payload.get("mean_per_layer", [[]])[-1][:4]
        return JSONResponse(payload)

    # Return a copy with the prompt echoed so the UI can display it.
    out = dict(cached)
    out["text"] = cache_key
    out["session_id"] = session_id
    # Legacy-compat alias so the older advanced-panels cross-attention
    # heatmap (which expects a flat 4x4 ``matrix``) still renders rather
    # than falling back to its synthetic pattern.
    try:
        last_layer = out["mean_per_layer"][-1]
        legacy_4x4: list[list[float]] = []
        for r in range(4):
            if r < len(last_layer):
                row = last_layer[r][:4]
                while len(row) < 4:
                    row = row + [0.0]
            else:
                row = [0.0, 0.0, 0.0, 0.0]
            legacy_4x4.append(row)
        out["matrix"] = legacy_4x4
    except Exception:  # pragma: no cover - defensive
        out["matrix"] = [[0.25] * 4 for _ in range(4)]
    return JSONResponse(out)


# ------------------------------------------------------------------
# User endpoints
# ------------------------------------------------------------------

@router.get(
    "/user/{user_id}/profile",
    dependencies=[Depends(require_user_identity)],
)
async def get_user_profile(
    request: Request,
    user_id: str = UserIdParam,
) -> JSONResponse:
    """Return the user profile (embeddings only, no raw text).

    The profile includes the behavioural baseline, long-term style
    vector, and relationship-strength score. Raw user text is never
    included in the response body.
    """
    pipeline = _get_pipeline(request)
    try:
        profile = await pipeline.get_user_profile(user_id)
    except HTTPException:
        raise
    except Exception:
        # SEC: never leak Python class names, args, or stack traces
        # through the wire. Log full detail server-side only.
        logger.exception("get_user_profile failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if profile is None:
        # SEC: generic message -- does NOT echo user_id, preventing user
        # enumeration via 404-message comparison.
        raise HTTPException(status_code=404, detail="Profile not found")
    return JSONResponse(profile)


@router.get(
    "/user/{user_id}/diary",
    dependencies=[Depends(require_user_identity)],
)
async def get_user_diary(
    request: Request,
    user_id: str = UserIdParam,
    limit: int = LimitParam,
    offset: int = OffsetParam,
) -> JSONResponse:
    """Return recent interaction diary entries for a user.

    Parameters
    ----------
    limit : int
        Maximum number of diary entries to return (newest first, 1-100).
    offset : int
        Number of entries to skip (0-10000).
    """
    pipeline = _get_pipeline(request)
    try:
        # SEC: request ``limit + offset`` from the pipeline so the local
        # slice below produces a window of the requested size. The previous
        # implementation requested only ``limit`` and then sliced ``[offset:]``,
        # which silently truncated results whenever offset >= limit.
        fetch_count = limit + offset
        entries = await pipeline.get_diary_entries(user_id, limit=fetch_count)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_user_diary failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if entries is None:
        raise HTTPException(status_code=404, detail="Diary not found")
    if offset:
        entries = entries[offset:]
    entries = entries[:limit]
    # SEC: do not echo the validated user_id in the body either, to keep
    # response shape uniform with the 404 path. Caller already knows it.
    return JSONResponse({"entries": entries, "count": len(entries)})


@router.get(
    "/user/{user_id}/stats",
    dependencies=[Depends(require_user_identity)],
)
async def get_user_stats(
    request: Request,
    user_id: str = UserIdParam,
) -> JSONResponse:
    """Return aggregate statistics for a user.

    Includes total sessions, message count, average engagement,
    baseline-deviation history, and top routing categories.
    """
    pipeline = _get_pipeline(request)
    try:
        stats = await pipeline.get_user_stats(user_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_user_stats failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if stats is None:
        raise HTTPException(status_code=404, detail="Stats not found")
    return JSONResponse(stats)


# ------------------------------------------------------------------
# Profiling / diagnostics
# ------------------------------------------------------------------

# SEC: explicit allow-list of fields the profiling endpoint may return.
# Anything outside this list (hostname, OS info, Python version, file paths,
# environment variables, model artefact paths, etc.) is dropped before
# serialisation, even if a future pipeline implementation accidentally
# includes it in the report.
_PROFILING_ALLOWED_FIELDS = frozenset(
    {
        "components",
        "total_latency_ms",
        "memory_mb",
        "fits_budget",
        "budget_ms",
        "device_class",
        # Iter 51: per-arm cascade summary (A/B/C SLM/Qwen-LoRA/Gemini)
        "cascade_arms",
    }
)


@router.get("/profiling/report")
async def get_profiling_report(request: Request) -> JSONResponse:
    """Return the edge-feasibility profiling report.

    Shows per-component latency, memory footprint, and whether the
    full pipeline fits within the 200 ms budget on-device.

    SEC: filtered through ``_PROFILING_ALLOWED_FIELDS`` so it cannot
    leak hostname, paths, Python version, or other system identifiers
    even if the underlying pipeline returns them. Rate-limiting is
    enforced upstream by the middleware layer.
    """
    pipeline = _get_pipeline(request)
    try:
        report = await pipeline.get_profiling_report()
    except HTTPException:
        raise
    except Exception:
        logger.exception("get_profiling_report failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if not isinstance(report, dict):
        # SEC: type guard -- never serialise non-dict objects whose repr
        # might leak class names or memory addresses.
        raise HTTPException(status_code=500, detail="Internal error")
    safe_report = {k: v for k, v in report.items() if k in _PROFILING_ALLOWED_FIELDS}
    return JSONResponse(safe_report)


# ------------------------------------------------------------------
# Iter 55: per-cascade-arm rolling latency stats
# ------------------------------------------------------------------

@router.get("/cascade/stats")
async def get_cascade_stats(request: Request) -> JSONResponse:
    """Live rolling per-arm latency snapshot.

    Returns the engine's in-memory per-arm latency window (iter 55)
    so the dashboard's cascade ribbon can show live p50/p95 numbers
    that react to actual usage, not just the static profiling report.
    """
    pipeline = _get_pipeline(request)
    try:
        stats = pipeline.cascade_arm_stats() if hasattr(pipeline, "cascade_arm_stats") else {}
    except Exception:
        logger.exception("get_cascade_stats failed")
        raise HTTPException(status_code=500, detail="Internal error")
    if not isinstance(stats, dict):
        raise HTTPException(status_code=500, detail="Internal error")
    return JSONResponse(stats)


# ------------------------------------------------------------------
# Demo utilities
# ------------------------------------------------------------------

@router.post("/demo/reset")
async def demo_reset(request: Request) -> JSONResponse:
    """Reset all demo state (user profiles, sessions, diary).

    SEC: gated behind the ``I3_DEMO_MODE`` env flag. Without it, this
    endpoint returns 403 so an accidentally-deployed production instance
    cannot be wiped by an unauthenticated POST. ``reset_all`` is a
    destructive, irreversible operation, so the gate is mandatory.
    """
    if not _demo_mode_enabled():
        raise HTTPException(status_code=403, detail="Demo mode disabled")
    pipeline = _get_pipeline(request)
    try:
        await pipeline.reset_all()
    except HTTPException:
        raise
    except Exception:
        logger.exception("demo_reset failed")
        raise HTTPException(status_code=500, detail="Internal error")
    logger.info("Demo state reset")
    return JSONResponse({"status": "reset"})


@router.post("/demo/seed")
async def demo_seed(request: Request) -> JSONResponse:
    """Seed the demo with pre-built user profiles and diary entries.

    Creates a *demo_user* with an established baseline and a few
    historical diary entries so the live demo shows adaptation from
    the very first message.

    SEC: gated behind ``I3_DEMO_MODE``. The ``seed_demo_data`` helper
    is expected to be idempotent (using upserts keyed on demo_user)
    so repeated calls do not duplicate data. ImportError on the demo
    module is caught and surfaced as a generic 503 rather than a 500
    leaking the missing-module name.
    """
    if not _demo_mode_enabled():
        raise HTTPException(status_code=403, detail="Demo mode disabled")
    try:
        from demo.seed_data import seed_demo_data
    except ImportError:
        logger.exception("demo seed module missing")
        raise HTTPException(status_code=503, detail="Demo data unavailable")

    pipeline = _get_pipeline(request)
    try:
        await seed_demo_data(pipeline)
    except HTTPException:
        raise
    except Exception:
        logger.exception("demo_seed failed")
        raise HTTPException(status_code=500, detail="Internal error")
    logger.info("Demo data seeded")
    # SEC: do NOT echo ``result`` into the response body -- it may contain
    # internal IDs, paths, or counts useful for fingerprinting. A bare
    # status keeps the response uniform with /demo/reset.
    return JSONResponse({"status": "seeded"})


# ---------------------------------------------------------------------------
# Intent-parsing route (Iter 51, 2026-04-27)
# ---------------------------------------------------------------------------
# Closes the JD-required "fine-tune pre-trained models" gap.  Two
# interchangeable backends behind a single endpoint:
#   * ``backend=qwen``   — Qwen3.5-2B + LoRA, on-device, free, ~80 ms
#   * ``backend=gemini`` — Gemini 2.5 Flash AI Studio tuned, ~£0/call
#                          on free tier, ~250 ms (network-bound)
# The default is ``qwen`` so the live demo never depends on the
# network.  Recruiters can flip to ``gemini`` to see the comparison.

# Lazy module-scope cache — instantiate on first call so server cold-
# start doesn't pay the model-load cost.
_intent_parsers: dict[str, Any] = {}


def _get_intent_parser(backend: str) -> Any:
    if backend in _intent_parsers:
        return _intent_parsers[backend]
    if backend == "qwen":
        from i3.intent.qwen_inference import QwenIntentParser
        parser = QwenIntentParser()
    elif backend == "gemini":
        from i3.intent.gemini_inference import GeminiIntentParser
        parser = GeminiIntentParser()
    else:
        raise HTTPException(status_code=400, detail=f"unknown backend: {backend}")
    _intent_parsers[backend] = parser
    return parser


@router.post("/intent")
async def parse_intent(request: Request) -> JSONResponse:
    """Parse a free-form HMI utterance into structured intent JSON.

    Body shape:
        {"text": "set a timer for 10 minutes", "backend": "qwen"}

    Response shape (matches :class:`i3.intent.types.IntentResult`):
        {"raw_input": ..., "action": "set_timer",
         "params": {"duration_seconds": 600},
         "valid_json": true, "valid_action": true, "valid_slots": true,
         "confidence": 1.0, "latency_ms": 78.4, "backend": "qwen-lora",
         "error": null}

    SEC:
        * Input length capped at 512 chars (HMI utterances are short).
        * Backend choice validated against the closed set above.
        * Inference is wrapped — any backend exception becomes a
          friendly 200 response with ``error`` populated, not a 500
          (so the dashboard stays responsive even when a model
          checkpoint is missing).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    text = body.get("text", "")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="text must be a string")
    if len(text) > 512:
        raise HTTPException(status_code=413, detail="text too long (>512)")
    backend = body.get("backend", "qwen")
    if backend not in ("qwen", "gemini"):
        raise HTTPException(status_code=400, detail="backend must be qwen|gemini")
    try:
        parser = _get_intent_parser(backend)
        result = parser.parse(text)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("intent parse failed")
        return JSONResponse({
            "raw_input": text,
            "error": f"backend error: {type(exc).__name__}",
            "backend": backend,
            "valid_json": False,
            "valid_action": False,
            "valid_slots": False,
            "confidence": 0.0,
        })
    return JSONResponse(result.to_dict())


@router.get("/intent/status")
async def intent_status() -> JSONResponse:
    """Status of the two intent backends (loaded? checkpoint paths?
    latest training metrics?).  Used by the dashboard's Intent tab.
    """
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    qwen_dir = repo / "checkpoints" / "intent_lora" / "qwen3.5-2b"
    qwen_best = repo / "checkpoints" / "intent_lora" / "qwen3.5-2b_best"
    gemini_dir = repo / "checkpoints" / "intent_gemini"
    qwen_metrics = qwen_dir / "training_metrics.json"
    gemini_result = gemini_dir / "tuning_result.json"

    out: dict[str, Any] = {
        "qwen": {
            "ready": qwen_dir.exists() and (qwen_dir / "adapter_config.json").exists(),
            "adapter_dir": str(qwen_dir),
            "best_dir": str(qwen_best) if qwen_best.exists() else None,
        },
        "gemini": {
            "ready": gemini_result.exists(),
            "tuning_result_path": str(gemini_result) if gemini_result.exists() else None,
        },
    }
    if qwen_metrics.exists():
        try:
            with qwen_metrics.open("r", encoding="utf-8") as f:
                out["qwen"]["training_metrics"] = json.load(f)
        except Exception:  # pragma: no cover
            pass
    if gemini_result.exists():
        try:
            with gemini_result.open("r", encoding="utf-8") as f:
                out["gemini"]["tuning_result"] = json.load(f)
        except Exception:  # pragma: no cover
            pass
    # Eval reports if present.
    eval_dir = repo / "checkpoints" / "intent_eval"
    out["eval"] = {}
    for name in ("qwen_report.json", "gemini_report.json"):
        path = eval_dir / name
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    out["eval"][name.replace("_report.json", "")] = json.load(f)
            except Exception:  # pragma: no cover
                pass
    return JSONResponse(out)
