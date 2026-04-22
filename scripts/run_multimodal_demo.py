"""Live multimodal demo (Batch F-1).

Captures webcam video (OpenCV, soft-imported) + microphone audio
(sounddevice, soft-imported) + simulated keystroke timings for ~10 seconds,
runs the three Batch-F-1 extractors, and prints the resulting 64-dim fused
user-state embedding alongside a per-modality feature breakdown.

Each capture device is **optional**; when any of the heavy libraries
(``opencv-python``, ``sounddevice``, ``librosa``, ``mediapipe``) is missing
the demo simply flags that modality as "unavailable" and continues.

Usage::

    python scripts/run_multimodal_demo.py [--duration 10] [--strategy late_concat]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from typing import Any

import numpy as np

logger = logging.getLogger("i3.multimodal.demo")


# ---------------------------------------------------------------------------
# Soft-import optional runtime deps
# ---------------------------------------------------------------------------

def _try_import(name: str) -> Any | None:
    """Import a module by name, returning ``None`` if unavailable."""
    try:
        return __import__(name)
    except ImportError:
        return None


cv2 = _try_import("cv2")
sounddevice = _try_import("sounddevice")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argv slice.  Defaults to :data:`sys.argv[1:]`.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Capture multimodal signals and run Batch F-1 fusion."
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Capture duration in seconds."
    )
    parser.add_argument(
        "--strategy",
        choices=["late_concat", "late_gated", "attention"],
        default="late_concat",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["zero_fill", "mask_drop"],
        default="zero_fill",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16_000, help="Microphone sample rate."
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index."
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level."
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _record_audio(duration_s: float, sample_rate: int) -> np.ndarray | None:
    """Capture a mono waveform of length *duration_s*.

    Returns ``None`` when the ``sounddevice`` library is unavailable or the
    backend raises on start-up (e.g. no microphone on CI).
    """
    if sounddevice is None:
        return None
    try:
        logger.info("Recording %.1f s of audio at %d Hz...", duration_s, sample_rate)
        buf = sounddevice.rec(
            int(duration_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sounddevice.wait()
        return np.asarray(buf, dtype=np.float32).flatten()
    except Exception as exc:  # noqa: BLE001 - hardware failures are expected
        logger.warning("Audio capture failed: %s", exc)
        return None


def _capture_video_frames(duration_s: float, camera_index: int) -> list[np.ndarray]:
    """Capture RGB frames from the webcam for *duration_s*.

    Returns an empty list when OpenCV is missing or the camera cannot be
    opened.  Frames are returned in RGB (not BGR) for direct MediaPipe use.
    """
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.warning("Could not open camera index %d", camera_index)
        return []
    frames: list[np.ndarray] = []
    end = time.monotonic() + duration_s
    try:
        while time.monotonic() < end:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    return frames


def _simulate_keystroke_features() -> np.ndarray:
    """Return a plausible keystroke-dynamics vector for the demo.

    Real keystroke capture is outside the scope of this script — the demo
    fills the slot with a fixed "baseline user" vector so the fusion head
    always has something to work with.
    """
    return np.asarray(
        [0.40, 0.20, 0.10, 0.15, 0.05, 0.50, 0.30, 0.10],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Demo main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    """Execute the demo; return a POSIX-style exit code."""
    # Soft-import the project modules lazily so the script can still help with
    # its ``--help`` message even on a minimal install.
    from i3.multimodal.fusion_real import MultimodalFusion

    try:
        from i3.multimodal.voice_real import VoiceProsodyExtractor

        voice_available = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("voice_real unavailable: %s", exc)
        voice_available = False

    try:
        from i3.multimodal.vision import FacialAffectExtractor

        vision_available = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision unavailable: %s", exc)
        vision_available = False

    # ---------------------------------------------------------------- capture
    keystroke_vec = _simulate_keystroke_features()

    waveform = _record_audio(args.duration, args.sample_rate)
    frames = _capture_video_frames(args.duration, args.camera_index)

    # --------------------------------------------------------- voice features
    voice_vec: np.ndarray | None = None
    if voice_available and waveform is not None and waveform.size > 0:
        try:
            extractor = VoiceProsodyExtractor()
            vec = extractor.extract(waveform, args.sample_rate)
            voice_vec = vec.to_array()
            print("\n[voice] features:")
            for k, v in vec.model_dump().items():
                print(f"  {k}: {v:.4f}")
        except RuntimeError as exc:
            logger.warning("Voice extraction skipped: %s", exc)
    else:
        logger.info("Voice modality unavailable (no audio or librosa missing).")

    # -------------------------------------------------------- vision features
    vision_vec: np.ndarray | None = None
    if vision_available and frames:
        try:
            with FacialAffectExtractor() as extractor:
                # Pool features across captured frames.
                collected: list[np.ndarray] = []
                for frame in frames[:: max(1, len(frames) // 30)]:
                    vec = extractor.extract(frame)
                    if vec is not None:
                        collected.append(vec.to_array())
                if collected:
                    vision_vec = np.mean(
                        np.stack(collected, axis=0), axis=0
                    ).astype(np.float32)
                    print("\n[vision] features (mean over frames):")
                    names = [
                        "eye_aspect_ratio",
                        "mouth_aspect_ratio",
                        "gaze_direction_x",
                        "gaze_direction_y",
                        "head_pose_pitch_deg",
                        "head_pose_yaw_deg",
                        "brow_furrow_au4",
                        "smile_au12",
                    ]
                    for k, v in zip(names, vision_vec):
                        print(f"  {k}: {float(v):.4f}")
                else:
                    logger.info("No face detected across captured frames.")
        except RuntimeError as exc:
            logger.warning("Vision extraction skipped: %s", exc)
    else:
        logger.info("Vision modality unavailable (no frames or mediapipe missing).")

    # ----------------------------------------------------------------- fusion
    fusion = MultimodalFusion(
        modality_dim_map={"keystroke": 8, "voice": 8, "vision": 8, "accelerometer": 8},
        fusion_strategy=args.strategy,
        missing_modality_policy=args.missing_policy,
    )
    fusion.eval()

    embedding = await fusion.fuse(
        keystroke_features=keystroke_vec,
        voice_features=voice_vec,
        vision_features=vision_vec,
        accelerometer_features=None,
    )

    print("\n=== Fused user-state embedding ===")
    print(f"strategy={args.strategy} missing_policy={args.missing_policy}")
    print(f"shape={tuple(embedding.shape)}")
    emb_list = embedding.detach().cpu().tolist()
    print("values (first 16):", [f"{x:.4f}" for x in emb_list[:16]])
    print("values (full):", [f"{x:.4f}" for x in emb_list])

    print("\nModality availability:")
    print(f"  keystroke: simulated")
    print(f"  voice:     {'ok' if voice_vec is not None else 'unavailable'}")
    print(f"  vision:    {'ok' if vision_vec is not None else 'unavailable'}")
    print("  accelerometer: not captured in this demo")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point compatible with ``python -m`` and ``console_scripts``."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
