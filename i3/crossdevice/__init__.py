"""Cross-device state-synchronisation sketches for I³.

This package mocks the HarmonyOS Distributed Data Management (DDM) sync
surface defined in ``docs/huawei/harmony_hmaf_integration.md §4``.  It is
the L3 step on the L1–L5 ladder (see ``docs/huawei/l1_l5_framework.md``):
single-device persistence stays as-is; a thin sync layer makes the long-term
user profile visible across a paired device constellation.

Everything here is a **pure-Python mock**.  Production should target
HarmonyOS's distributed KV store and the
``@ohos.data.distributedDataObject`` subsystem; nothing in this sketch
performs device discovery, pairing, or on-wire transport in the way a real
HarmonyOS deployment would.
"""

from __future__ import annotations

from i3.crossdevice.ai_glasses_arm import PairedDeviceRouter, paired_phone_inference_arm
from i3.crossdevice.device_registry import DeviceClass, DeviceInfo, DeviceRegistry
from i3.crossdevice.hmos_ddm_sync import (
    AdaptationVectorPayload,
    DDMSyncClient,
    I3UserStateSync,
)

__all__ = [
    "AdaptationVectorPayload",
    "DDMSyncClient",
    "DeviceClass",
    "DeviceInfo",
    "DeviceRegistry",
    "I3UserStateSync",
    "PairedDeviceRouter",
    "paired_phone_inference_arm",
]
