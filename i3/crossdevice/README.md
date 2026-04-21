# Cross-Device Sketches (HarmonyOS DDM)

> *"Cross-device HarmonyOS Distributed Data Management sync of the 64-dim
> embedding."* — THE_COMPLETE_BRIEF §12
>
> *"AI Glasses arm extension: 'route to paired smartphone' as a third arm —
> config change, not arch change."* — THE_COMPLETE_BRIEF §11

This package implements the **L3 step** on the L1–L5 framework (see
`docs/huawei/l1_l5_framework.md`). It is an in-memory mock; production
would target HarmonyOS's distributed KV store (`@ohos.data.distributedKVStore`)
and `@ohos.distributedDeviceManager`.

## Contents

| File | Purpose |
|:---|:---|
| `hmos_ddm_sync.py` | `DDMSyncClient` implementing the `I3UserStateSync` wire contract (§4.3 of `docs/huawei/harmony_hmaf_integration.md`). Push / pull / subscribe / list_peers. |
| `device_registry.py` | `DeviceRegistry` — an in-memory pairing list covering phone / AI Glasses / wearable / Smart Hanhan / home hub / tablet. |
| `ai_glasses_arm.py` | `PairedDeviceRouter` — an **outer wrapper** around the existing 2-arm router that adds a third `paired_phone_inference` arm without modifying the bandit. |

## Threat model

The sync fabric expands the privacy surface — data now traverses the device
constellation — which **increases** the pressure on the guarantees that are
already load-bearing at L1–L2:

| Threat | Mitigation |
|:---|:---|
| Passive eavesdrop on the databus | Fernet-wrap every payload before push; HarmonyOS transport already encrypts, we layer on top for in-app confidentiality. |
| Tampering in transit | SHA-256 integrity tag appended to every encrypted payload; pull rejects records whose tag does not match. |
| Replay of an old record | `monotonic_seq` + `ttl_s` on each record; expired records return `None` from `pull`. |
| Raw user text on the wire | **Architectural guarantee.** The `I3UserStateSync` struct contains only scalars and embeddings — no text field exists. |
| Untrusted peer in the constellation | `DeviceRegistry` is per-user and pairing-gated; the sync client never reaches beyond registered peers. |
| Compromised peer decrypting past state | Forward-secret key rotation is called out in the L3 privacy notes (`docs/huawei/l1_l5_framework.md §3.4`); revocation triggers a re-key and old records become inaccessible. |
| Server-side observability leak | The adapter emits text-free telemetry events only (§3.3 of `harmony_hmaf_integration.md`). |

## Why this preserves I³'s privacy guarantees

The sync layer is **additive** — it doesn't weaken anything that holds in
the single-device case. Specifically:

1. **No raw text ever enters the sync layer.** The `I3UserStateSync` struct
   has no text field; there is no way to accidentally serialise a message
   body into it.
2. **Every payload is Fernet-encrypted end-to-end** using the existing
   `ModelEncryptor`.
3. **Integrity tags prevent silent tampering.** Fernet is authenticated, but
   we add an explicit SHA-256 tag for callers that decrypt on a device they
   don't fully trust (e.g. a shared Smart Hanhan).
4. **The AI-Glasses extension arm respects the privacy override.** When the
   sensitivity classifier has vetoed cloud routing, the paired-phone arm
   *also* declines — the glasses must process locally rather than transmit
   even to a trusted peer.

## The AI-Glasses third arm

`PairedDeviceRouter` is an **outer decorator** around the existing
`IntelligentRouter`. On phone-class devices it is a transparent pass-through.
On AI-Glasses-class devices, it:

1. Runs the inner 2-arm bandit normally.
2. Computes an arm score for `paired_phone_inference` via
   `paired_phone_inference_arm(context, peer)`.
3. If that score exceeds the best inner-arm score and a threshold, returns
   an `ExtendedRoutingDecision` with `final_route="paired_phone_inference"`.
4. Never feeds extension-arm rewards back into the inner bandit's posterior
   — those are handled separately (see L4 handover policy in
   `docs/huawei/l1_l5_framework.md §4`).

This is deliberately **a config change, not an architecture change**, which
is how the brief frames it. No existing code is touched.
