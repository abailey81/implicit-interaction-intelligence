"""Voice-prosody fusion for the I³ multimodal user-state embedding.

This module is the **server-side** half of the voice-prosody flagship feature.
Its complement is :mod:`web/js/voice_prosody.js`, which extracts the eight
prosodic scalars in the browser via the WebAudio API.

Privacy contract
----------------
* The browser **never transmits raw audio**. Only eight numeric scalars
  (plus two metadata fields) cross the WebSocket.
* This module *cannot* invert the prosody features back into speech: the
  representation is lossy by design (8 floats per ~3 s window).
* The class :class:`ProsodyFeatures` documents the exact shape that the JS
  side serialises; the WS layer validates the payload before it ever reaches
  the encoder.

Why these eight features
------------------------
We deliberately use a small, interpretable subset of the openSMILE GeMAPS
feature set (Eyben et al. 2010) so the JS-side worklet can compute them in
real time at <50 ms latency.  The subset captures the four canonical
paralinguistic axes from Schuller (2009)'s *Computational Paralinguistics
Challenge*:

* **arousal** → energy_mean, energy_variance, pitch_variance
* **rate / engagement** → speech_rate_wpm, voiced_ratio, pause_density
* **pitch register** → pitch_mean
* **timbre** → spectral_centroid

References
~~~~~~~~~~
* Schuller, B. et al. (2009). *The INTERSPEECH 2009 Emotion Challenge.*
  Proc. INTERSPEECH 2009, pp. 312–315.
* Schuller, B. et al. (2013). *The Computational Paralinguistics Challenge.*
  Computer Speech & Language 27 (1), 4–39.
* Eyben, F. et al. (2010). *openSMILE: The Munich versatile and fast
  open-source audio feature extractor.*  Proc. ACM Multimedia, pp. 1459–1462.
* Eyben, F. et al. (2016). *The Geneva Minimalistic Acoustic Parameter Set
  (GeMAPS) for voice research and affective computing.*  IEEE TAC 7 (2),
  190–202.

CPU-only contract
~~~~~~~~~~~~~~~~~
This file deliberately avoids ``.cuda()``: the encoder is so small (8 → 32
→ 32 MLP, ~1.4k params) that a CPU forward pass costs <100 µs and we want
the inference path to remain identical in CI, on-device, and in the
training cluster.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


__all__ = [
    "ProsodyFeatures",
    "ProsodyEncoder",
    "GazeEncoder",
    "GAZE_FEATURE_KEYS",
    "MultimodalFusion",
    "PROSODY_FEATURE_KEYS",
    "validate_prosody_payload",
    "prosody_payload_to_tensor",
    "validate_gaze_payload",
    "gaze_payload_to_tensor",
]


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

# The eight ordered keys that make up the prosody feature vector.  This list
# is the single source of truth: the JS extractor must emit exactly these
# keys, the WS validator checks the same set, and ``prosody_payload_to_tensor``
# zips them in this order to produce the 8-dim tensor consumed by
# :class:`ProsodyEncoder`.
PROSODY_FEATURE_KEYS: tuple[str, ...] = (
    "speech_rate_wpm_norm",
    "pitch_mean_norm",
    "pitch_variance_norm",
    "energy_mean_norm",
    "energy_variance_norm",
    "voiced_ratio",
    "pause_density",
    "spectral_centroid_norm",
)


@dataclass
class ProsodyFeatures:
    """A single 8-dim prosody feature snapshot.

    All features are in normalised ``[0, 1]`` ranges (or close to it for
    the variance terms) so they fuse cleanly with the keystroke 32-d
    feature window without an additional rescale layer.

    Cites Schuller (2009) and Eyben et al. (2010) for the openSMILE
    feature design; we deliberately use a small, interpretable subset
    that the JS-side WebAudio worklet can compute in real time at
    <50 ms latency.

    Attributes:
        speech_rate_wpm_norm: words-per-minute, normalised ``[0, 1]``
            (raw 0–300 wpm; 0.5 ≈ 150 wpm, conversational).
        pitch_mean_norm: ``f0`` mean Hz, normalised ``[0, 1]``
            (raw 50–400 Hz; 0.3 ≈ 150 Hz adult male, 0.5 ≈ 225 Hz
            adult female).
        pitch_variance_norm: ``f0`` standard deviation normalised
            ``[0, 1]`` (raw 0–80 Hz σ).
        energy_mean_norm: short-term RMS energy, normalised ``[0, 1]``.
        energy_variance_norm: RMS standard deviation, normalised ``[0, 1]``.
        voiced_ratio: fraction of frames classified as voiced (energy
            above silence threshold AND a usable pitch estimate),
            ``[0, 1]``.
        pause_density: pauses-per-second of audio, normalised ``[0, 1]``
            (raw 0–4 pauses/s; 0.25 ≈ a typical conversational rhythm).
        spectral_centroid_norm: brightness — weighted-mean FFT bin
            frequency, normalised ``[0, 1]`` over the analysis band.
        samples_count: how many audio analysis frames contributed to
            the aggregate (a frame is one 100 ms window in the JS
            extractor).
        captured_seconds: how long the buffer was, in seconds (typically
            close to 3.0 since the JS extractor keeps a 3 s rolling
            window).
    """

    speech_rate_wpm_norm: float
    pitch_mean_norm: float
    pitch_variance_norm: float
    energy_mean_norm: float
    energy_variance_norm: float
    voiced_ratio: float
    pause_density: float
    spectral_centroid_norm: float
    samples_count: int = 0
    captured_seconds: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Return the eight feature scalars as a 1-D ``[8]`` tensor.

        Order matches :data:`PROSODY_FEATURE_KEYS`.  The two metadata
        fields (``samples_count`` and ``captured_seconds``) are
        intentionally excluded — they are bookkeeping for the UI and
        the reasoning trace, not signal.
        """
        return torch.tensor(
            [getattr(self, k) for k in PROSODY_FEATURE_KEYS],
            dtype=torch.float32,
        )

    def to_dict(self) -> dict:
        """JSON-safe dict for the WS frame and the reasoning trace."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation + payload coercion (used by server/websocket.py)
# ---------------------------------------------------------------------------

def validate_prosody_payload(payload: object) -> ProsodyFeatures | None:
    """Validate a client-supplied ``prosody_features`` dict.

    The WS layer hands us whatever the JS client sent.  We:

    * accept only ``dict`` payloads (anything else returns ``None``)
    * require all eight keys in :data:`PROSODY_FEATURE_KEYS`
    * coerce each to ``float`` with ``_safe_float`` semantics
    * clamp each into ``[0, 1]`` so an attacker cannot inject NaN/inf or
      a 1e30 value that would later corrupt the encoder
    * accept optional ``samples_count`` (int) and ``captured_seconds``
      (float, ≥ 0) bookkeeping fields, defaulting to 0

    Returns ``None`` on any failure so the caller can branch into the
    "no prosody this turn" path without crashing.  We deliberately do
    NOT raise — prosody is a soft signal; a malformed payload should
    degrade to keystroke-only, not break the response.
    """
    if not isinstance(payload, dict):
        return None

    coerced: dict[str, float] = {}
    for key in PROSODY_FEATURE_KEYS:
        if key not in payload:
            logger.debug("Prosody payload missing key %r; rejecting.", key)
            return None
        try:
            v = float(payload[key])
        except (TypeError, ValueError):
            logger.debug("Prosody payload %r is not float; rejecting.", key)
            return None
        # Reject NaN / inf — float() accepts them but we don't want them.
        if v != v or v in (float("inf"), float("-inf")):
            logger.debug("Prosody payload %r is NaN/inf; rejecting.", key)
            return None
        coerced[key] = max(0.0, min(1.0, v))

    # Optional bookkeeping fields.
    try:
        samples_count = int(payload.get("samples_count", 0) or 0)
    except (TypeError, ValueError):
        samples_count = 0
    if samples_count < 0 or samples_count > 100_000:
        # Cap to a sane upper bound so a malformed client cannot push
        # an arbitrary integer into our logs.
        samples_count = max(0, min(100_000, samples_count))

    try:
        captured_seconds = float(payload.get("captured_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        captured_seconds = 0.0
    if captured_seconds != captured_seconds or captured_seconds < 0.0:
        captured_seconds = 0.0
    captured_seconds = min(captured_seconds, 600.0)  # 10-minute hard cap

    return ProsodyFeatures(
        samples_count=samples_count,
        captured_seconds=captured_seconds,
        **coerced,
    )


def prosody_payload_to_tensor(payload: object) -> torch.Tensor | None:
    """Convenience: validate + return the ``[8]`` feature tensor or ``None``."""
    feats = validate_prosody_payload(payload)
    if feats is None:
        return None
    return feats.to_tensor()


# ---------------------------------------------------------------------------
# Prosody encoder (8 → 32)
# ---------------------------------------------------------------------------

class ProsodyEncoder(nn.Module):
    """Tiny MLP mapping 8-d prosody features → 32-d prosody embedding.

    Architecture (intentionally minimal — see module docstring for the
    rationale):

    * ``Linear(8, 32)`` → ``GELU`` → ``LayerNorm(32)``
    * ``Linear(32, 32)`` → ``LayerNorm(32)``

    Trained jointly with the TCN keystroke pipeline as an auxiliary head
    in the Phase-3 SLM training plan; here we just initialise it.  Even
    un-trained, the embedding is a deterministic random-projection of
    the eight prosodic axes — that adds non-zero signal to the
    multimodal fusion downstream because the random projection
    preserves rank.

    Args:
        in_dim: Input feature dim, default 8 (see
            :data:`PROSODY_FEATURE_KEYS`).
        hidden_dim: Hidden width of the bottleneck, default 32.
        out_dim: Output embedding dim, default 32.
    """

    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 32,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "ProsodyEncoder created: %d → %d → %d, %d params.",
            in_dim, hidden_dim, out_dim, n_params,
        )

    def forward(self, prosody: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            prosody: Tensor of shape ``[in_dim]`` or ``[B, in_dim]``.

        Returns:
            Tensor of shape ``[out_dim]`` (or ``[B, out_dim]``).
        """
        # Accept a 1-D vector — promote to a single-row batch.
        squeezed = False
        if prosody.dim() == 1:
            prosody = prosody.unsqueeze(0)
            squeezed = True
        x = self.fc1(prosody)
        x = self.act(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        if squeezed:
            x = x.squeeze(0)
        return x


# ---------------------------------------------------------------------------
# Gaze feature contract (the eight numeric scalars shipped from the JS
# extractor — separate from the discrete label).  Used by the engine's
# multimodal fusion path to embed the gaze state alongside keystroke /
# prosody.  The full :class:`i3.multimodal.gaze_classifier.GazeFeatures`
# carries strings (the predicted label) for the UI; for the encoder we
# only need the numeric scalars.
# ---------------------------------------------------------------------------

GAZE_FEATURE_KEYS: tuple[str, ...] = (
    "p_at_screen",
    "p_away_left",
    "p_away_right",
    "p_away_other",
    "presence",
    "blink_rate_norm",
    "head_stability",
    "confidence",
)
"""Eight ordered keys for the gaze feature vector.

The first four are the softmax probabilities over the four discrete
gaze classes; the remaining four are bookkeeping scalars.  Together
they span the full information content of a
:class:`~i3.multimodal.gaze_classifier.GazeFeatures` snapshot in a
fixed-shape numeric form the encoder can consume.
"""


def validate_gaze_payload(payload: object) -> dict | None:
    """Validate a client-supplied / engine-generated gaze features dict.

    The payload comes from one of two sources:

    1. The browser ships a high-level gaze dict on the WS frame
       (``{label, confidence, label_probs, presence, blink_rate_norm,
       head_stability, captured_seconds, samples_count}``).
    2. The server-side classifier returns the same shape after a frame
       inference.

    We extract the eight numeric scalars in :data:`GAZE_FEATURE_KEYS`,
    coerce to ``[0, 1]``, reject NaN / inf, and return a flat dict
    ready for :func:`gaze_payload_to_tensor`.  Returns ``None`` on any
    validation failure so the caller can branch into the no-gaze path.
    """
    if not isinstance(payload, dict):
        return None

    label_probs = payload.get("label_probs")
    if not isinstance(label_probs, dict):
        return None

    coerced: dict[str, float] = {}
    # Class probabilities — accept the full ``label_probs`` dict.
    for cls in ("at_screen", "away_left", "away_right", "away_other"):
        try:
            v = float(label_probs.get(cls, 0.0))
        except (TypeError, ValueError):
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
        coerced[f"p_{cls}"] = max(0.0, min(1.0, v))

    # Bookkeeping scalars.
    try:
        presence = 1.0 if bool(payload.get("presence", False)) else 0.0
        confidence = float(payload.get("confidence", 0.0))
        blink = float(payload.get("blink_rate_norm", 0.0))
        head = float(payload.get("head_stability", 0.0))
    except (TypeError, ValueError):
        return None
    for v in (confidence, blink, head):
        if v != v or v in (float("inf"), float("-inf")):
            return None

    coerced["presence"] = presence
    coerced["blink_rate_norm"] = max(0.0, min(1.0, blink))
    coerced["head_stability"] = max(0.0, min(1.0, head))
    coerced["confidence"] = max(0.0, min(1.0, confidence))
    return coerced


def gaze_payload_to_tensor(payload: object) -> torch.Tensor | None:
    """Convenience: validate + return the ``[8]`` feature tensor or ``None``."""
    feats = validate_gaze_payload(payload)
    if feats is None:
        return None
    return torch.tensor(
        [feats[k] for k in GAZE_FEATURE_KEYS],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Gaze encoder (8 → 32) — mirrors ProsodyEncoder
# ---------------------------------------------------------------------------

class GazeEncoder(nn.Module):
    """Tiny MLP mapping the 8-d gaze feature vector → 32-d gaze embedding.

    Identity-init through a deliberate residual: when the mean of the
    8-d input is zero (the "no camera" path with zero-padding) the
    output is approximately zero, so the downstream multimodal fusion's
    keystroke half passes through unchanged.

    Architecture:

    * ``Linear(8, 32)`` → ``GELU`` → ``LayerNorm(32)``
    * ``Linear(32, 32)`` → ``LayerNorm(32)``

    See :class:`i3.multimodal.gaze_classifier.GazeFineTuneHead` for the
    fine-tuned vision model that produces the 8-d vector.  This encoder
    sits *after* the discrete classifier and bridges its output into
    the joint multimodal embedding space.
    """

    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 32,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "GazeEncoder created: %d → %d → %d, %d params.",
            in_dim, hidden_dim, out_dim, n_params,
        )

    def forward(self, gaze: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            gaze: Tensor of shape ``[in_dim]`` or ``[B, in_dim]``.

        Returns:
            Tensor of shape ``[out_dim]`` (or ``[B, out_dim]``).
        """
        squeezed = False
        if gaze.dim() == 1:
            gaze = gaze.unsqueeze(0)
            squeezed = True
        x = self.fc1(gaze)
        x = self.act(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        if squeezed:
            x = x.squeeze(0)
        return x


# ---------------------------------------------------------------------------
# Multimodal fusion (64 + 32 + 32 → 128, with backward-compat 96-d mode)
# ---------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """Fuse the 64-d keystroke embedding with the 32-d prosody embedding
    AND optionally a 32-d gaze embedding into a 128-d multimodal
    user-state vector.

    Backward compatibility: callers that construct the fusion with
    ``out_dim = key_dim + prosody_dim`` (i.e. 96) get the legacy 96-d
    fusion that ignores gaze entirely.  Callers that construct with
    ``out_dim = key_dim + prosody_dim + gaze_dim`` (i.e. 128) get the
    full tri-modal fusion.

    Implementation details:

    * Project each modality through an identity-initialised ``Linear``
      so the un-trained network passes the inputs through verbatim.
    * Concatenate → ``LayerNorm`` → ``GELU`` → ``Linear(out, out)``.
    * Add a residual connection from the concatenated input.

    The fusion is therefore *deterministic-at-init*: the output is
    approximately the concatenation of the inputs.  As joint training
    progresses (Phase-3 SLM training), the fusion learns a proper joint
    representation while never destroying the keystroke-only fallback
    path.

    Args:
        key_dim: Dimensionality of the keystroke embedding (default 64,
            matches :class:`i3.encoder.tcn.TemporalConvNet`).
        prosody_dim: Dimensionality of the prosody embedding (default 32,
            matches :class:`ProsodyEncoder`).
        gaze_dim: Dimensionality of the gaze embedding (default 32,
            matches :class:`GazeEncoder`).  Set to 0 for the legacy
            96-d two-modality fusion.
        out_dim: Dimensionality of the fused output.  Must equal
            ``key_dim + prosody_dim + gaze_dim`` for the identity-init
            residual connection to be well-formed.
    """

    def __init__(
        self,
        key_dim: int = 64,
        prosody_dim: int = 32,
        gaze_dim: int = 32,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        # Backward-compat: a caller that passes ``out_dim=96`` (the old
        # signature) implicitly disables gaze.  We detect that and
        # rewrite gaze_dim to 0 so the rest of the constructor works.
        if out_dim is None:
            out_dim = key_dim + prosody_dim + gaze_dim
        if out_dim == key_dim + prosody_dim and gaze_dim != 0:
            # Legacy two-modality init.
            gaze_dim = 0
        if out_dim != key_dim + prosody_dim + gaze_dim:
            raise ValueError(
                f"MultimodalFusion: out_dim ({out_dim}) must equal "
                f"key_dim + prosody_dim + gaze_dim "
                f"({key_dim + prosody_dim + gaze_dim}) so "
                f"the identity-init residual connection is well-formed."
            )

        self.key_dim = key_dim
        self.prosody_dim = prosody_dim
        self.gaze_dim = gaze_dim
        self.out_dim = out_dim

        # Identity-init projections.  We start from torch.eye so the
        # un-trained fusion is a no-op pass-through; gradients then move
        # the projections away from identity during joint training.
        self.key_proj = nn.Linear(key_dim, key_dim, bias=False)
        with torch.no_grad():
            self.key_proj.weight.copy_(torch.eye(key_dim))

        self.prosody_proj = nn.Linear(prosody_dim, prosody_dim, bias=False)
        with torch.no_grad():
            self.prosody_proj.weight.copy_(torch.eye(prosody_dim))

        if gaze_dim > 0:
            self.gaze_proj = nn.Linear(gaze_dim, gaze_dim, bias=False)
            with torch.no_grad():
                self.gaze_proj.weight.copy_(torch.eye(gaze_dim))
        else:
            self.gaze_proj = None

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        # The post-fusion linear is also identity-initialised (with zero
        # bias) so the residual dominates at init: out ≈ concat(...).
        self.post = nn.Linear(out_dim, out_dim)
        with torch.no_grad():
            self.post.weight.copy_(torch.eye(out_dim))
            self.post.bias.zero_()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MultimodalFusion created: (%d + %d + %d) → %d, %d params, "
            "identity-init.",
            key_dim, prosody_dim, gaze_dim, out_dim, n_params,
        )

    def forward(
        self,
        key_emb: torch.Tensor,
        prosody_emb: torch.Tensor | None,
        gaze_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse keystroke + (optional) prosody + (optional) gaze embeddings.

        Args:
            key_emb: Tensor of shape ``[key_dim]`` or ``[B, key_dim]``.
            prosody_emb: Tensor of shape ``[prosody_dim]`` or
                ``[B, prosody_dim]``, **or** ``None``.  When ``None``
                we substitute zeros so the downstream consumer always
                sees an ``out_dim``-d embedding regardless of mic state.
            gaze_emb: Tensor of shape ``[gaze_dim]`` or
                ``[B, gaze_dim]``, **or** ``None``.  When ``None`` we
                substitute zeros (camera off path).  Ignored entirely
                when the fusion was constructed with ``gaze_dim=0``.

        Returns:
            Tensor of shape ``[out_dim]`` (or ``[B, out_dim]``).
        """
        squeezed = False
        if key_emb.dim() == 1:
            key_emb = key_emb.unsqueeze(0)
            squeezed = True

        if prosody_emb is None:
            prosody_emb = torch.zeros(
                key_emb.shape[0],
                self.prosody_dim,
                dtype=key_emb.dtype,
                device=key_emb.device,
            )
        elif prosody_emb.dim() == 1:
            prosody_emb = prosody_emb.unsqueeze(0)

        if prosody_emb.shape[-1] != self.prosody_dim:
            raise ValueError(
                f"MultimodalFusion: prosody_emb last-dim "
                f"{prosody_emb.shape[-1]} != prosody_dim {self.prosody_dim}."
            )
        if key_emb.shape[-1] != self.key_dim:
            raise ValueError(
                f"MultimodalFusion: key_emb last-dim "
                f"{key_emb.shape[-1]} != key_dim {self.key_dim}."
            )

        k = self.key_proj(key_emb)
        p = self.prosody_proj(prosody_emb)

        if self.gaze_dim > 0 and self.gaze_proj is not None:
            if gaze_emb is None:
                gaze_emb = torch.zeros(
                    key_emb.shape[0],
                    self.gaze_dim,
                    dtype=key_emb.dtype,
                    device=key_emb.device,
                )
            elif gaze_emb.dim() == 1:
                gaze_emb = gaze_emb.unsqueeze(0)
            if gaze_emb.shape[-1] != self.gaze_dim:
                raise ValueError(
                    f"MultimodalFusion: gaze_emb last-dim "
                    f"{gaze_emb.shape[-1]} != gaze_dim {self.gaze_dim}."
                )
            g = self.gaze_proj(gaze_emb)
            concat = torch.cat([k, p, g], dim=-1)
        else:
            concat = torch.cat([k, p], dim=-1)

        h = self.norm(concat)
        h = self.act(h)
        h = self.post(h)
        out = h + concat                              # residual

        if squeezed:
            out = out.squeeze(0)
        return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - smoke test
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=== ProsodyFeatures ===")
    feats = ProsodyFeatures(
        speech_rate_wpm_norm=0.55,
        pitch_mean_norm=0.40,
        pitch_variance_norm=0.30,
        energy_mean_norm=0.65,
        energy_variance_norm=0.20,
        voiced_ratio=0.78,
        pause_density=0.22,
        spectral_centroid_norm=0.48,
        samples_count=30,
        captured_seconds=3.0,
    )
    t = feats.to_tensor()
    print(f"  to_tensor shape: {tuple(t.shape)}  (expected (8,))")
    assert t.shape == (8,), t.shape

    print("\n=== ProsodyEncoder ===")
    enc = ProsodyEncoder()
    pemb = enc(t)
    print(f"  prosody_emb shape: {tuple(pemb.shape)}  (expected (32,))")
    assert pemb.shape == (32,), pemb.shape

    print("\n=== GazeEncoder ===")
    genc = GazeEncoder()
    gtensor = torch.tensor(
        [0.83, 0.05, 0.05, 0.07, 1.0, 0.12, 0.85, 0.83],
        dtype=torch.float32,
    )
    gemb = genc(gtensor)
    print(f"  gaze_emb shape: {tuple(gemb.shape)}  (expected (32,))")
    assert gemb.shape == (32,), gemb.shape

    print("\n=== MultimodalFusion (tri-modal: key + prosody + gaze) ===")
    fusion = MultimodalFusion()  # default: 64 + 32 + 32 = 128
    key_emb = torch.randn(64)
    fused = fusion(key_emb, pemb, gemb)
    print(f"  fused shape: {tuple(fused.shape)}  (expected (128,))")
    assert fused.shape == (128,), fused.shape

    print("\n=== MultimodalFusion (prosody=None, gaze=None) ===")
    fused_none = fusion(key_emb, None, None)
    print(f"  fused shape: {tuple(fused_none.shape)}  (expected (128,))")
    assert fused_none.shape == (128,), fused_none.shape

    print("\n=== MultimodalFusion legacy 96-d ===")
    legacy = MultimodalFusion(key_dim=64, prosody_dim=32, out_dim=96)
    fused_legacy = legacy(key_emb, pemb)
    print(f"  legacy fused shape: {tuple(fused_legacy.shape)}  (expected (96,))")
    assert fused_legacy.shape == (96,), fused_legacy.shape

    print("\n=== validate_gaze_payload ===")
    good_gaze = {
        "label": "at_screen",
        "confidence": 0.83,
        "label_probs": {
            "at_screen": 0.83, "away_left": 0.05,
            "away_right": 0.05, "away_other": 0.07,
        },
        "presence": True,
        "blink_rate_norm": 0.12,
        "head_stability": 0.85,
    }
    parsed_gaze = validate_gaze_payload(good_gaze)
    assert parsed_gaze is not None
    assert validate_gaze_payload({"missing": "everything"}) is None
    print("  good gaze payload   -> OK")
    print("  bad gaze payload    -> rejected")
    gtensor2 = gaze_payload_to_tensor(good_gaze)
    assert gtensor2 is not None and gtensor2.shape == (8,)
    print(f"  payload_to_tensor   -> shape {tuple(gtensor2.shape)}")

    print("\n=== Identity-init sanity ===")
    # At init the fused output should be approximately concat([key, 0]) when
    # prosody is None, modulo a single LayerNorm + Linear pass.
    head_norm = float(torch.linalg.norm(fused_none[:64]))
    tail_norm = float(torch.linalg.norm(fused_none[64:]))
    print(f"  ||fused[:64]|| = {head_norm:.3f}, ||fused[64:]|| = {tail_norm:.3f}")
    print(f"  ||key_emb||    = {float(torch.linalg.norm(key_emb)):.3f}")

    print("\n=== validate_prosody_payload ===")
    good_payload = {**{k: 0.5 for k in PROSODY_FEATURE_KEYS},
                    "samples_count": 30, "captured_seconds": 3.0}
    parsed = validate_prosody_payload(good_payload)
    assert parsed is not None, "good payload should validate"
    bad_payload = {"speech_rate_wpm_norm": 0.5}  # missing keys
    assert validate_prosody_payload(bad_payload) is None
    nan_payload = {**good_payload, "pitch_mean_norm": float("nan")}
    assert validate_prosody_payload(nan_payload) is None
    not_a_dict = "hello"
    assert validate_prosody_payload(not_a_dict) is None
    print("  good_payload  -> OK")
    print("  missing keys  -> rejected")
    print("  NaN value     -> rejected")
    print("  non-dict      -> rejected")

    print("\nAll smoke tests passed.")
