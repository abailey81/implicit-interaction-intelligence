# Responsible AI Documentation

This directory collects the Responsible AI artefacts for the Implicit
Interaction Intelligence (I³) prototype. They are written for a technical
reviewer who wants to know, in concrete and traceable terms, what the
system does, what it does *not* do, what data it was trained on, and how
its claims are calibrated.

## Contents

| File                                                    | Purpose                                                                 |
|:--------------------------------------------------------|:------------------------------------------------------------------------|
| [`model_card_slm.md`](model_card_slm.md)                | Model card for the ~6.3 M-parameter Adaptive SLM, following the Mitchell et al. 2019 template. |
| [`model_card_tcn.md`](model_card_tcn.md)                | Model card for the TCN user-state encoder, same template.               |
| [`data_card.md`](data_card.md)                          | Datasheets-for-Datasets-style card (Gebru et al. 2021) covering the synthetic interaction set, the DailyDialog / EmpatheticDialogues subsets, and the valence lexicon. |
| [`accessibility_statement.md`](accessibility_statement.md) | Accessibility statement covering the implicit-signal detection stack, its relationship to WCAG 2.2 and ARIA, and the opt-out guarantees. |

## Who this is for

The reviewer for this material is the interviewer panel at the Huawei
London HMI Lab, and — transitively — anyone considering running a
descendant of this code on real user data. The voice is intentionally
plain and engineer-facing; it is not a marketing document. Claims are
calibrated: the prototype is labelled a prototype, the training data
are labelled synthetic or small-scale, and every honest limitation of
the approach is surfaced rather than buried.

## What is *not* here, deliberately

- A compliance checklist against a specific regulation (GDPR, EU AI Act,
  UK Data Protection Act). The prototype does not operate on real user
  data in production; the architectural properties (no raw text
  persisted, PII sanitisation before cloud calls, Fernet encryption at
  rest) are documented in `SECURITY.md` and `docs/architecture/full-reference.md` §9
  and are the raw material for such a checklist, not a substitute for
  one.
- A fairness / bias audit with subgroup metrics. I have not collected
  the demographic ground truth needed to run such an audit honestly;
  what I have instead is an explicit statement in the model cards and
  the accessibility statement about the populations the training data
  under-represents.
- A red-team report. The sensitive-topic classifier (`i3/router/sensitivity.py`)
  and the privacy override in `i3/router/router.py` are the code-level
  mitigations; they have unit tests but not adversarial testing.

## How to read these documents

Start with `model_card_slm.md` and `model_card_tcn.md` if you want to
understand what the two learned components are and are not. Read
`data_card.md` if you want to evaluate the training data honestly —
including the parts that are synthetic and the bias inherited from the
public dialogue corpora. Read `accessibility_statement.md` last; it is
the document that puts the whole system in its actual context, which is
one signal among many rather than a replacement for explicit user
control.
