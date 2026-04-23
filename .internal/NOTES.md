# NOTES — Deviations from the Brief

Engineering disclosure for the Huawei London HMI Lab interview. Anything here is a
conscious departure from `THE_COMPLETE_BRIEF.md`; rationale in one paragraph each.

## 1. Directory layout: `src/` (brief) -> `i3/` (actual)

The brief sketches `src/...`; the repo ships a flat top-level `i3/`. Modern
Python accepts both. A `src/i3/` layout would only matter if we needed strict
import-shadowing during tests; the flat layout yields cleaner imports and
simpler Poetry/Ruff/Mypy configs. Poetry is told the truth via `pyproject.toml`:
```toml
packages = [ { include = "i3" }, { include = "server" },
             { include = "training" }, { include = "demo" } ]
```

## 2. Tokenizer scope

Word-level `SimpleTokenizer`, 8 K vocabulary (matches spec). BPE was rejected:
at ~8 M parameters, interpretability of OOV behaviour > BPE's compression ratio.
The tokenizer is also simpler to round-trip-test (§18.2 Day 5).

## 3. Cross-attention conditioning

Four conditioning tokens are projected from `concat(AdaptationVector[8],
UserStateEmbedding[64]) = 72-dim` and passed as K/V into a dedicated
cross-attention block inside every PreLN transformer block. Conditioning is
**structural, not prompt-based** — no "[COGLOAD=HIGH]" tokens injected into the
stream. See `docs/ARCHITECTURE.md §8`.

## 4. `cryptography.Fernet` as a placeholder for TrustZone

Embeddings at rest use `cryptography.Fernet` (AES-128-CBC + HMAC-SHA256 + random
IV = authenticated encryption). Production migrates to Huawei **TrustZone / SE**
for hardware-rooted key storage. The abstraction boundary
(`i3.privacy.encryption`) is already shaped to accept that swap.

## 5. INT8 dynamic quantisation on laptop only

**INT8 dynamic quantisation** is applied to `Linear` layers only — attention
softmax stays fp32 for numerical stability. Production converts via **MindSpore
Lite** and targets the Kirin NPU **Da Vinci** architecture (Big + Tiny cores).
The laptop path demonstrates the weights *could* convert cleanly.

## 6. Anthropic Claude `claude-sonnet-4-5`

Cloud model ID is locked at `configs/default.yaml:cloud.model`. Chosen because
the brief names it explicitly; not silently swapped for a placeholder.

## 7. Synthetic data for TCN; curated corpora for SLM

TCN: 8 synthetic archetypes following **Epp 2011 / Vizer 2009 / Zimmermann
2014**. SLM: **DailyDialog + EmpatheticDialogues** (public, consented corpora,
standard research licences). A Datasheet-for-Datasets-style entry lives at
`docs/data_card.md` when generated.

## 8. Laptop-only live demo — Kirin numbers extrapolated

Deliberate reversal from the brief's Raspberry Pi suggestion: laptop is the
demo substrate for reliability. `docs/edge_profiling_report.md` flags all Kirin
9000 / A2 / Smart Hanhan figures as **extrapolations** from the laptop INT8
footprint × Kirin TOPS ratios. Nothing measured on Huawei silicon; the honesty
slide repeats this.

## 9. What is NOT in the prototype

- Real federated updates (averaging stub only).
- On-device continual learning (session-only EMA).
- Touch / voice / gaze modalities (TCN is modality-agnostic; only keystrokes wired).
- Multi-user keystroke biometric identification.
- Cross-device HarmonyOS distributed databus sync of the long-term profile.
- Interpretability panel (per-feature contribution heatmaps).
- Auxiliary conditioning-consistency loss during SLM training.

All listed on the slides' future-work slide.

## 10. Commit discipline

Every commit message is written as if the panel will read it.
