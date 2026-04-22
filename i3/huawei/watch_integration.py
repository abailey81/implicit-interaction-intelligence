"""Huawei Watch 5 HRV integration stub (Batch F-2).

The Huawei Watch 5 (launched 2025) is the first consumer wearable to ship
a three-in-one ECG + PPG + pressure sensor and runs a 30-minute stress
estimation loop based on resting-state HRV.  In a production HarmonyOS
deployment the I³ runtime would consume HRV readings in one of three
ways:

1.  **Native HarmonyOS**: subscribe to ``health.service.hrv`` events via
    the Huawei Health Kit SDK and receive push-style callbacks.
2.  **DDM (Distributed Data Manager) sync**: mirror the watch's HRV table
    onto the phone via HarmonyOS distributed data management; the phone-
    side I³ process reads the synchronised KV-store partition.
3.  **Cloud pull**: poll Huawei Health Cloud Kit's HRV endpoint from a
    gateway service that forwards encrypted payloads into I³.

This module provides a **reference implementation** suitable for unit
testing and for the agentic-core-runtime integration point — a mocked
source that reads the DDM-style payload from a local JSON file.  The
public API (:class:`HuaweiWatchHRVSource` with ``subscribe`` /
``unsubscribe`` / ``latest_hrv``) matches the surface the native binding
would expose so the rest of I³ can ingest HRV without code changes when
the real SDK is wired up.

References
----------
* Huawei Consumer Business (2025).  *Huawei Watch 5 launch press
  release — three-in-one ECG + PPG + pressure sensor, continuous HRV
  monitoring.*
* Huawei (2024).  *Health Kit developer documentation — HRV streaming
  API and HarmonyOS DDM synchronization of health records.*
* HarmonyOS (2024).  *Distributed Data Management (DDM) developer
  guide — KV store + cross-device consistency model.*
* Nelson, B. W. et al. (2020).  *Guidelines for wrist-worn consumer
  wearable assessment of heart rate in cardiology.*  npj Digital
  Medicine 3, 90.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from i3.multimodal.ppg_hrv import PPGFeatureVector

logger = logging.getLogger(__name__)


HRVCallback = Callable[[PPGFeatureVector], None]
"""Signature for callbacks registered with :class:`HuaweiWatchHRVSource`."""


# ---------------------------------------------------------------------------
# Expected DDM-synchronised payload format
# ---------------------------------------------------------------------------


_EXPECTED_FIELDS: frozenset[str] = frozenset(
    {
        "hr_bpm",
        "rmssd_ms",
        "sdnn_ms",
        "pnn50_percent",
        "lf_power",
        "hf_power",
        "lf_hf_ratio",
        "sample_entropy",
    }
)


@dataclass
class HuaweiWatchHRVPayload:
    """Single HRV reading as materialised from the DDM KV-store.

    Attributes:
        timestamp: Unix epoch seconds when the watch finished the HRV
            estimation window (Huawei Watch 5 runs this every 30 minutes
            at rest).
        device_id: Stable HarmonyOS device identifier, for multi-watch
            households.
        feature_vector: The decoded 8-dim HRV vector.
        raw_stress_level: Huawei's native stress label (``"low"``,
            ``"medium"``, ``"high"``, ``"very_high"``) when provided.
    """

    timestamp: float
    device_id: str
    feature_vector: PPGFeatureVector
    raw_stress_level: str | None = None


def _payload_from_dict(obj: dict) -> HuaweiWatchHRVPayload:
    """Decode one entry from the DDM-synchronised JSON payload.

    Args:
        obj: Parsed JSON dict.

    Returns:
        A :class:`HuaweiWatchHRVPayload`.

    Raises:
        ValueError: If the dict is missing required keys or contains
            invalid types.
    """
    missing = _EXPECTED_FIELDS - set(obj.keys())
    if missing:
        raise ValueError(
            f"HuaweiWatchHRVPayload missing required fields: {sorted(missing)!r}"
        )
    ts_raw = obj.get("timestamp")
    if ts_raw is None:
        raise ValueError("HuaweiWatchHRVPayload: 'timestamp' is required")
    try:
        ts = float(ts_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp: {ts_raw!r}") from exc

    device_id = str(obj.get("device_id", "huawei-watch-5"))
    stress = obj.get("stress_level")
    stress_str = str(stress) if stress is not None else None

    vec = PPGFeatureVector(
        hr_bpm=float(obj["hr_bpm"]),
        rmssd_ms=float(obj["rmssd_ms"]),
        sdnn_ms=float(obj["sdnn_ms"]),
        pnn50_percent=float(obj["pnn50_percent"]),
        lf_power=float(obj["lf_power"]),
        hf_power=float(obj["hf_power"]),
        lf_hf_ratio=float(obj["lf_hf_ratio"]),
        sample_entropy=float(obj["sample_entropy"]),
    )
    return HuaweiWatchHRVPayload(
        timestamp=ts,
        device_id=device_id,
        feature_vector=vec,
        raw_stress_level=stress_str,
    )


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


@dataclass
class HuaweiWatchHRVSource:
    """Reference HRV source for Huawei Watch 5 DDM-synced payloads.

    This class simulates the Huawei Health SDK surface area that a
    production HarmonyOS build would use.  In production, the JSON file
    would be replaced by:

    * Native binding to ``@ohos.health.hrv`` via the HarmonyOS SDK.
    * Or a DDM KV-store watcher using ``@ohos.data.distributedKVStore``.

    Args:
        ddm_payload_path: Filesystem path to a JSON file containing an
            array of HRV payload objects (one per reading).  In a real
            deployment this is the DDM-synchronised phone-side mirror.
        device_id: Expected HarmonyOS device identifier — used for
            logging and filtering.

    Attributes:
        callbacks: Registered :data:`HRVCallback` instances.
    """

    ddm_payload_path: Path
    device_id: str = "huawei-watch-5"
    callbacks: list[HRVCallback] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _latest: HuaweiWatchHRVPayload | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.ddm_payload_path = Path(self.ddm_payload_path)

    # ------------------------------------------------------------------
    def subscribe(self, callback: HRVCallback) -> None:
        """Register a callback for newly-observed HRV payloads.

        In production this maps to ``healthKit.subscribe('hrv', cb)``.
        Calling :meth:`refresh` after subscribing replays the most recent
        payload to all callbacks.

        Args:
            callback: Function receiving a :class:`PPGFeatureVector`.

        Raises:
            ValueError: If ``callback`` is not callable.
        """
        if not callable(callback):
            raise ValueError("callback must be callable")
        with self._lock:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

    # ------------------------------------------------------------------
    def unsubscribe(self, callback: HRVCallback) -> None:
        """Remove a previously-registered callback.

        Args:
            callback: The callback to remove.  Silently ignores callbacks
                that were never subscribed.
        """
        with self._lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    # ------------------------------------------------------------------
    def latest_hrv(self) -> PPGFeatureVector | None:
        """Return the most recently observed HRV feature vector.

        Returns:
            The latest :class:`PPGFeatureVector`, or ``None`` when no
            payload has been loaded yet.
        """
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.feature_vector

    # ------------------------------------------------------------------
    def latest_payload(self) -> HuaweiWatchHRVPayload | None:
        """Return the most recent full payload including device metadata.

        Returns:
            A :class:`HuaweiWatchHRVPayload` or ``None``.
        """
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    def refresh(self) -> list[HuaweiWatchHRVPayload]:
        """Re-read the DDM JSON file and dispatch new payloads.

        In production this is driven by HarmonyOS DDM change events; in
        this mock it is an explicit pull operation useful for tests and
        batch backfills.

        Returns:
            The list of payloads loaded (empty if the file is absent or
            contains no applicable entries).

        Raises:
            ValueError: If the payload file exists but is not a JSON array
                of objects.
        """
        if not self.ddm_payload_path.is_file():
            logger.debug(
                "HuaweiWatchHRVSource: DDM payload not yet available at %s",
                self.ddm_payload_path,
            )
            return []
        try:
            with self.ddm_payload_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid DDM payload JSON at {self.ddm_payload_path!s}: {exc}"
            ) from exc
        if not isinstance(raw, list):
            raise ValueError(
                "DDM payload must be a top-level array of HRV entries."
            )
        payloads: list[HuaweiWatchHRVPayload] = []
        for i, obj in enumerate(raw):
            if not isinstance(obj, dict):
                logger.warning(
                    "HuaweiWatchHRVSource: DDM entry %d is not an object, skipping",
                    i,
                )
                continue
            if obj.get("device_id", self.device_id) != self.device_id:
                continue
            try:
                payload = _payload_from_dict(obj)
            except ValueError as exc:
                logger.warning(
                    "HuaweiWatchHRVSource: skipping malformed DDM entry %d: %s",
                    i,
                    exc,
                )
                continue
            payloads.append(payload)

        if not payloads:
            return []

        payloads.sort(key=lambda p: p.timestamp)
        with self._lock:
            self._latest = payloads[-1]
            callbacks_snapshot = list(self.callbacks)

        for cb in callbacks_snapshot:
            try:
                cb(payloads[-1].feature_vector)
            except Exception as exc:  # noqa: BLE001
                # A user callback must never break the ingest loop; log & continue.
                logger.exception(
                    "HuaweiWatchHRVSource: subscriber callback raised %s",
                    type(exc).__name__,
                )
        return payloads


# ---------------------------------------------------------------------------
# Public module API
# ---------------------------------------------------------------------------

__all__ = [
    "HRVCallback",
    "HuaweiWatchHRVPayload",
    "HuaweiWatchHRVSource",
]
