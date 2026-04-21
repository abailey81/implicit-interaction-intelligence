"""Mock HarmonyOS pairing registry for the cross-device sketches.

The registry models a user's device constellation — phone, AI Glasses,
wearable, Smart Hanhan — and acts as the lookup table the DDM sync client
hits for peer discovery.  In production, this would be backed by HarmonyOS's
``@ohos.distributedDeviceManager`` service.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceClass(str, Enum):
    """Classes of devices a user may have in their I³ constellation.

    Mirrors the taxonomy from the L1–L5 framework
    (``docs/huawei/l1_l5_framework.md``) and the Huawei device list in the
    brief: Kirin 9000 phones → Kirin 820 tablets → Kirin A2 wearables →
    Smart Hanhan class.
    """

    PHONE = "phone"                     # Kirin 9000 class
    AI_GLASSES = "ai_glasses"           # Huawei AI Glasses, Apr 2026
    WEARABLE = "wearable"               # Watches, bands, Kirin A2
    SMART_HANHAN = "smart_hanhan"       # ~64 MB RAM companion toy
    HOME_HUB = "home_hub"               # Speakers, hubs, TV
    TABLET = "tablet"


@dataclass(frozen=True)
class DeviceInfo:
    """Metadata for a paired device in the I³ constellation.

    Attributes:
        device_id: Stable 64-bit identifier.
        device_class: Coarse device class.
        display_name: Human-readable name; set by the user at pairing time.
        owner_user_id: The user that owns this device.
        paired_at: Unix epoch seconds when pairing completed.
        ram_mb: Reported RAM budget for I³; used by the router to decide
            whether to offload to this peer.
        supports_slm: True if the device can run the local SLM itself.
        supports_encoder: True if the device can run the TCN encoder.
    """

    device_id: int
    device_class: DeviceClass
    display_name: str
    owner_user_id: str
    paired_at: float = field(default_factory=time.time)
    ram_mb: int = 512
    supports_slm: bool = True
    supports_encoder: bool = True


class DeviceRegistry:
    """Thread-safe registry of paired devices for a single user.

    Example:
        >>> reg = DeviceRegistry()
        >>> reg.register(DeviceInfo(
        ...     device_id=1, device_class=DeviceClass.PHONE,
        ...     display_name="Phone", owner_user_id="alice",
        ... ))
        >>> len(reg.list_devices()) == 1
        True
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: dict[int, DeviceInfo] = {}

    def register(self, device: DeviceInfo) -> None:
        """Add or update a device in the registry.

        Args:
            device: The device info to register.  An existing entry with the
                same ``device_id`` is replaced.
        """
        with self._lock:
            self._devices[device.device_id] = device
        logger.info(
            "DeviceRegistry: registered %s (%s, class=%s)",
            device.display_name,
            device.device_id,
            device.device_class.value,
        )

    def unregister(self, device_id: int) -> None:
        """Remove a device from the registry.

        Silently no-ops if the device is absent.
        """
        with self._lock:
            self._devices.pop(device_id, None)

    def get(self, device_id: int) -> DeviceInfo | None:
        """Return the device record for *device_id*, or ``None``."""
        with self._lock:
            return self._devices.get(device_id)

    def list_devices(self) -> list[DeviceInfo]:
        """Return an immutable snapshot of all registered devices."""
        with self._lock:
            return list(self._devices.values())

    def list_by_class(self, device_class: DeviceClass) -> list[DeviceInfo]:
        """Return all registered devices of a given class."""
        with self._lock:
            return [d for d in self._devices.values() if d.device_class == device_class]

    def find_paired_phone(self) -> DeviceInfo | None:
        """Return the first registered phone, if one exists.

        Used by the AI-Glasses routing arm to locate a peer capable of
        running the full SLM stack.
        """
        phones = self.list_by_class(DeviceClass.PHONE)
        return phones[0] if phones else None
