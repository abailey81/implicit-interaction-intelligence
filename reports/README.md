# Reports

Machine-readable and narrative artefacts from the three verification layers
the project runs on every change.

## Layout

| Directory | Purpose |
|---|---|
| [`audits/`](audits/) | Narrative review reports (security, robustness, code quality, completeness, documentation) with per-finding severity and evidence. |
| [`verification/`](verification/) | Output of the 46-check automated verification harness — `latest.{json,md}` plus historical passes under `history/`. |
| [`redteam/`](redteam/) | Output of the 55-attack adversarial harness — `latest.{json,md}`. |

## How to regenerate

```bash
# 46-check verification
python scripts/verify_all.py --strict \
    --out reports/verification/latest.json \
    --out-md reports/verification/latest.md

# 55-attack red-team (sanitizer / PDDL / guardrails surfaces)
python scripts/security/run_redteam.py --targets sanitizer,pddl,guardrails \
    --out reports/redteam/latest.json \
    --out-md reports/redteam/latest.md
```

## Latest results (see `audits/2026-04-23-index.md`)

- **Verification harness**: 27 pass / 0 fail / 19 skip (skips are environment-gated).
- **Red-team invariants**: 3 / 4 pass.
- **Manual audits**: 0 blocker · 0 high open · all listed highs fixed in
  [`audits/2026-04-23-fixes-applied.md`](audits/2026-04-23-fixes-applied.md).
