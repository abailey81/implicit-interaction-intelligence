"""Kirin-class target-device specifications for I³ deployment.

This module declares frozen Pydantic v2 models for each Kirin-class
target device I³ has been profiled against (Kirin 9000, Kirin 9010,
Kirin A2, Smart Hanhan), plus a :func:`select_deployment_profile`
helper that picks the most constrained device that still meets a
model's memory and compute requirements.

Canonical facts:

- **Kirin 9000** uses HiSilicon Da Vinci with 1 Big NPU core +
  1 Tiny NPU core.
- **Kirin 9010** launched Q2 2024 with a 2+6+4 big-LITTLE-LITTLE
  configuration, Taishan big core up to 2.3 GHz. Da Vinci Lite has
  2048 FP16 MACs + 4096 INT8 MACs.
- **Kirin A2** is a wearable-class chip with a Lite NPU.
- **Smart Hanhan** is Huawei's smart-companion IoT device category;
  sub-0.3 W envelope, MCU-class compute.

See :doc:`docs/huawei/kirin_deployment.md` for the engineering
context these numbers come from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DeviceClass = Literal["phone", "tablet", "wearable", "iot"]


class DeviceProfile(BaseModel):
    """Frozen Pydantic v2 model describing one Kirin-class deployment target.

    All memory figures are in **megabytes**; TOPS figures are INT8
    tera-operations per second; latency figures are milliseconds.

    Attributes:
        name: Human-readable device label.
        device_class: Broad category for deployment policy decisions.
        npu_architecture: Descriptor of the NPU block (e.g. ``"Da Vinci"``).
        ram_mb: Total on-device RAM available (MB).
        model_budget_mb: The 50% budget rule applied to ``ram_mb``.
        int8_tops: Peak INT8 TOPS available to ML workloads.
        cpu_tdp_w: Typical CPU big-cluster sustained power (watts).
        npu_tdp_w: Typical NPU sustained power (watts).
        est_encoder_latency_ms: Estimated TCN encoder P50 latency (ms).
        est_slm_latency_ms: Estimated AdaptiveSLM 32-token generate (ms).
        supports_executorch: Whether a working ExecuTorch backend is
            expected for this class.
        notes: Free-form caveats or deployment guidance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    device_class: DeviceClass
    npu_architecture: str
    ram_mb: float = Field(gt=0)
    model_budget_mb: float = Field(gt=0)
    int8_tops: float = Field(ge=0)
    cpu_tdp_w: float = Field(ge=0)
    npu_tdp_w: float = Field(ge=0)
    est_encoder_latency_ms: float = Field(ge=0)
    est_slm_latency_ms: float = Field(ge=0)
    supports_executorch: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Canonical device registry
# ---------------------------------------------------------------------------

# The numbers below mirror ``docs/huawei/kirin_deployment.md``. They are
# scaled from an Apple M2 CPU INT8 baseline by the TOPS ratio and are
# conservative — NPUs are typically more efficient at small-matmul
# workloads than a pure TOPS ratio predicts.

KIRIN_9000: DeviceProfile = DeviceProfile(
    name="Kirin 9000 (Phone)",
    device_class="phone",
    npu_architecture="HiSilicon Da Vinci — 1 Big NPU core + 1 Tiny NPU core",
    ram_mb=12_288.0,
    model_budget_mb=256.0,
    int8_tops=2.0,
    cpu_tdp_w=3.0,
    npu_tdp_w=1.0,
    est_encoder_latency_ms=1.5,
    est_slm_latency_ms=75.0,
    supports_executorch=True,
    notes="Flagship. Full I³ stack comfortable on NPU.",
)

KIRIN_9010: DeviceProfile = DeviceProfile(
    name="Kirin 9010 (Flagship)",
    device_class="phone",
    npu_architecture=(
        "Da Vinci Lite — 2048 FP16 MACs + 4096 INT8 MACs; "
        "2+6+4 big-LITTLE-LITTLE; Taishan big core to 2.3 GHz"
    ),
    ram_mb=16_384.0,
    model_budget_mb=256.0,
    int8_tops=2.5,
    cpu_tdp_w=3.0,
    npu_tdp_w=1.0,
    est_encoder_latency_ms=1.3,
    est_slm_latency_ms=70.0,
    supports_executorch=True,
    notes="Q2 2024 launch. Memory-bandwidth-bound, not TOPS-bound.",
)

KIRIN_A2: DeviceProfile = DeviceProfile(
    name="Kirin A2 (Wearable)",
    device_class="wearable",
    npu_architecture="Da Vinci Lite (wearable tier)",
    ram_mb=128.0,
    model_budget_mb=64.0,
    int8_tops=0.5,
    cpu_tdp_w=0.8,
    npu_tdp_w=0.2,
    est_encoder_latency_ms=12.0,
    est_slm_latency_ms=600.0,
    supports_executorch=True,
    notes=(
        "Tight. Recommended: encoder on-NPU, SLM delegated to phone via "
        "HMAF databus; INT4 (torchao) brings local SLM to ~350 ms."
    ),
)

SMART_HANHAN: DeviceProfile = DeviceProfile(
    name="Smart Hanhan (IoT)",
    device_class="iot",
    npu_architecture="MCU + DSP (implementation-dependent)",
    ram_mb=64.0,
    model_budget_mb=32.0,
    int8_tops=0.1,
    cpu_tdp_w=0.3,
    npu_tdp_w=0.15,
    est_encoder_latency_ms=8.0,
    est_slm_latency_ms=0.0,  # SLM not deployed on this class
    supports_executorch=False,
    notes=(
        "Encoder-only deployment. See docs/huawei/smart_hanhan.md. "
        "SLM is delegated to the phone-class owner device."
    ),
)


ALL_PROFILES: tuple[DeviceProfile, ...] = (
    KIRIN_9000,
    KIRIN_9010,
    KIRIN_A2,
    SMART_HANHAN,
)


# ---------------------------------------------------------------------------
# Deployment-profile selector
# ---------------------------------------------------------------------------


def select_deployment_profile(
    model_size_mb: float,
    tops_required: float,
) -> DeviceProfile | None:
    """Pick the most constrained device that can still run a model.

    The selector walks the canonical registry from *most constrained*
    (Smart Hanhan) to *least constrained* (Kirin 9010) and returns the
    first profile whose memory budget and INT8 TOPS meet the given
    requirements. This encodes the engineering preference for
    deploying on the smallest device that fits — for power, thermal,
    and fleet-reach reasons.

    Args:
        model_size_mb: INT8 deployed size of the model in MB.
        tops_required: Minimum INT8 TOPS required for interactive
            latency. Typically ~0.1 for the TCN encoder alone, ~1.0
            for the full Adaptive SLM at 32-token generation.

    Returns:
        The most constrained :class:`DeviceProfile` that satisfies both
        constraints, or ``None`` if no registered device fits.

    Example:
        >>> profile = select_deployment_profile(model_size_mb=0.05, tops_required=0.05)
        >>> profile.name
        'Smart Hanhan (IoT)'
        >>> profile = select_deployment_profile(model_size_mb=6.4, tops_required=1.0)
        >>> profile.name
        'Kirin 9000 (Phone)'
        >>> select_deployment_profile(model_size_mb=500.0, tops_required=100.0) is None
        True
    """
    if model_size_mb <= 0:
        raise ValueError(
            f"model_size_mb must be positive, got {model_size_mb}."
        )
    if tops_required < 0:
        raise ValueError(
            f"tops_required must be non-negative, got {tops_required}."
        )

    # Registry is ordered least-constrained to most-constrained.
    # We want most-constrained first, so we walk in reverse.
    candidates = sorted(ALL_PROFILES, key=lambda p: p.model_budget_mb)
    for profile in candidates:
        if (
            profile.model_budget_mb >= model_size_mb
            and profile.int8_tops >= tops_required
        ):
            return profile
    return None


__all__ = [
    "ALL_PROFILES",
    "DeviceClass",
    "DeviceProfile",
    "KIRIN_9000",
    "KIRIN_9010",
    "KIRIN_A2",
    "SMART_HANHAN",
    "select_deployment_profile",
]
