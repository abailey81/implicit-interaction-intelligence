# EU AI Act — scope declaration for I³

**Declaration date:** 2026-04-23
**Applicable regulation:** Regulation (EU) 2024/1689 (the "AI Act"),
which entered into force on 1 August 2024 with a phased application
timeline.  Prohibited-practices articles (Article 5) applied from
2 February 2025.  Obligations for high-risk AI systems apply from
2 August 2026 (most categories) and 2 August 2027 (some embedded
systems).

This document states, for every AI-Act category that *could*
plausibly apply to I³, exactly where this system sits and why.  It
is a working compliance artefact — not legal advice — and will be
re-reviewed at every minor release.

---

## Summary

| Article / Annex | Prohibited / high-risk category | I³'s posture |
|---|---|---|
| Article 5(1)(f) | Emotion-recognition systems in the workplace and in educational institutions | **OUT OF SCOPE** — not deployed in workplace or education settings; the demo is a research / interview artefact. |
| Article 5(1)(g) | Biometric categorisation to infer protected characteristics (race, political opinions, trade-union membership, religion, philosophical beliefs, sex life, sexual orientation) | **NEVER** — the AdaptationVector has no such dimension; the sanitiser explicitly strips PII before any cloud leg. |
| Article 5(1)(h) | Real-time remote biometric identification in publicly accessible spaces for law enforcement | **OUT OF SCOPE** — not a law-enforcement system. |
| Annex III §1 | Biometrics (remote biometric identification, biometric categorisation, emotion recognition in contexts other than those prohibited by Article 5) | **VOLUNTARILY HIGH-RISK-COMPLIANT** — the project ships the high-risk obligations that would apply if this were productised into a consumer device: logging, transparency to users, human oversight points, robustness and cybersecurity. |
| Annex III §4 | Employment, workers management | **OUT OF SCOPE**. |
| Annex III §5 | Access to essential services | **OUT OF SCOPE**. |

---

## 1. Emotion recognition in the workplace — Article 5(1)(f)

Article 5 of the AI Act prohibits placing on the market or using AI
systems that infer emotions of a natural person **in the areas of
workplace and education institutions**, with a narrow exception for
medical or safety purposes (e.g. a driver-fatigue monitor).

**I³'s posture.** The system infers a `cognitive_load`,
`emotional_tone`, `accessibility`, and four `style_mirror`
dimensions from keystroke dynamics and linguistic complexity.
Whether any of these count as "emotion recognition" under Article 5
depends on deployment context:

- **The shipped demo** is a single-user research / interview artefact.
  It is *not* placed on the EU market as a workplace or education
  product.  Article 5(1)(f) does not apply to the demo.
- **A hypothetical productisation** that targeted workplace or
  education settings would need to either (a) rewrite the
  AdaptationVector to exclude the `emotional_tone` dimension in
  those contexts, or (b) fit the medical / safety exception.  The
  safer posture is (a).

See
[Wolters-Kluwer Global Workplace Law blog: *The prohibition of AI
emotion-recognition technologies in the workplace under the AI
Act*](https://legalblogs.wolterskluwer.com/global-workplace-law-and-policy/the-prohibition-of-ai-emotion-recognition-technologies-in-the-workplace-under-the-ai-act/).

## 2. Biometric categorisation — Article 5(1)(g)

Article 5(1)(g) prohibits biometric categorisation systems that
categorise individuals "on the basis of their biometric data to
deduce or infer their race, political opinions, trade-union
membership, religious or philosophical beliefs, sex life or sexual
orientation".

**I³'s posture.** The AdaptationVector is defined in
[`i3/adaptation/types.py`](../../i3/adaptation/types.py) as an
8-dimensional vector spanning cognitive load, style mirror,
emotional tone, and accessibility.  **None of those dimensions map
onto a protected characteristic.**  Beyond that:

- No training objective conditions on a protected characteristic.
- The diary schema (`i3/diary/store.py`) stores only embeddings,
  scalar metrics, and TF-IDF topic keywords — never a protected-
  characteristic label.
- The sanitiser (`i3/privacy/sanitizer.py`) strips 10 categories of
  PII from every payload before any cloud leg.
- The PDDL safety planner (`i3/safety/pddl_planner.py`) enforces a
  *privacy-override* invariant — sensitive topics (health, finance,
  credentials, security) are force-routed to the local SLM rather
  than the cloud, with a machine-checkable `SafetyCertificate`.

## 3. Transparency — Article 50

Article 50 imposes transparency obligations on providers and
deployers of certain AI systems, including emotion-recognition and
biometric-categorisation systems.  Even though I³ is *out of scope*
for the Article 5 prohibitions, the project applies Article 50-style
transparency by default:

- The demo UI surfaces every adaptation decision with its confidence
  interval and a natural-language counterfactual explanation
  (`/api/explain/adaptation`, returning a per-dimension
  `DimensionConfidence`).
- Every cloud route carries a `SafetyCertificate` showing the
  planner's justification; every local route is similarly logged.
- The interaction diary can be exported per-user
  (`/admin/export/{user_id}` — GDPR right-to-export parity).
- The system can be fully reset per-user
  (`/admin/delete/{user_id}` — GDPR right-to-erase parity).

## 4. High-risk voluntary compliance (Annex III §1)

Even though the demo is not placed on the EU market as a biometric-
categorisation or emotion-recognition product, the project
deliberately ships the obligations it *would* carry if it were:

| Obligation | Where it is implemented |
|---|---|
| Risk-management system | [`docs/security/policy_as_code.md`](../security/policy_as_code.md) — NIST 800-53 + CIS K8s matrix, T1–T13 threat model. |
| Data governance | [`i3/data/`](../../i3/data/) pipeline (cleaning, quality, dedup, provenance lineage, deterministic splits). |
| Technical documentation | This doc-site, plus 10 ADRs and the research paper at [`docs/paper/`](../paper/). |
| Record-keeping | Structured JSON logs via `i3/observability/` — every inference + every route decision is logged. |
| Transparency | `/api/explain/*` endpoints + the counterfactual panel in the advanced UI. |
| Human oversight | Admin router + privacy-override + PDDL-grounded safety certificates. |
| Accuracy, robustness, cybersecurity | 46-check verification harness + 55-attack red-team corpus + SBOM + SLSA L3 build provenance. |

## 5. GDPR interaction

The AI Act complements the GDPR.  Where GDPR applies, this project's
commitments are documented at
[`docs/architecture/privacy.md`](../architecture/privacy.md) and
[`SECURITY.md`](../../SECURITY.md).  Summary:

- Raw user text is never persisted (privacy-by-architecture).
- The user-state embedding is Fernet-encrypted at rest with
  configurable `MultiFernet` key rotation.
- The `/admin/export/{user_id}` and `/admin/delete/{user_id}`
  endpoints provide right-to-export and right-to-erase parity.

---

## References

- [EU Digital Strategy — AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) (European Commission).
- [Article 5 — Prohibited AI Practices](https://artificialintelligenceact.eu/article/5/).
- [Annex III — High-risk AI systems](https://artificialintelligenceact.eu/annex/3/).
- [Article 50 — Transparency obligations](https://artificialintelligenceact.eu/article/50/).
- [IAPP: *Biometrics in the EU — navigating the GDPR and the AI Act*](https://iapp.org/news/a/biometrics-in-the-eu-navigating-the-gdpr-ai-act).
- [William Fry: *The Time to (AI) Act is Now — a practical guide to emotion-recognition systems under the AI Act*](https://www.williamfry.com/knowledge/the-time-to-ai-act-is-now-a-practical-guide-to-emotion-recognition-systems-under-the-ai-act/).
- [Wolters Kluwer Global Workplace Law Policy blog: *The prohibition of AI emotion-recognition technologies in the workplace under the AI Act*](https://legalblogs.wolterskluwer.com/global-workplace-law-and-policy/the-prohibition-of-ai-emotion-recognition-technologies-in-the-workplace-under-the-ai-act/).
